from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_explainer(model):
    """Create a fast TreeSHAP explainer for a fitted tree model."""
    import shap

    return shap.TreeExplainer(model)


def explain_rows(explainer, rows: pd.DataFrame):
    """Return positive-class SHAP explanations with a consistent 2-D shape."""
    explanation = explainer(rows)
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]
    if explanation.values.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP values, got shape {explanation.values.shape}")
    return explanation


def top_contributions(explanation, row: int = 0, limit: int = 8) -> list[dict[str, Any]]:
    """Serialize the strongest local contributions for an API or dashboard."""
    values = np.asarray(explanation.values[row], dtype=float)
    data = np.asarray(explanation.data[row], dtype=float)
    names = list(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {
            "feature": names[index],
            "feature_value": float(data[index]),
            "shap_value": float(values[index]),
            "direction": "increases_risk" if values[index] > 0 else "decreases_risk",
        }
        for index in order
    ]


def _base_value(explanation, row: int = 0) -> float:
    values = np.asarray(explanation.base_values)
    return float(values if values.ndim == 0 else values[row])


def generate_shap_artifacts(
    model,
    rows: pd.DataFrame,
    output_dir: str | Path = "artifacts",
    prefix: str = "shap",
    sample_size: int = 1_000,
    random_state: int = 42,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Create compact global, local, and plot artifacts from a bounded sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if rows.empty:
        raise ValueError("Cannot explain an empty dataset")
    sample = rows.sample(min(sample_size, len(rows)), random_state=random_state)
    explanation = explain_rows(create_explainer(model), sample)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_path = output_dir / f"{prefix}_global_importance.csv"
    local_path = output_dir / f"{prefix}_local_explanations.csv"
    beeswarm_path = output_dir / f"{prefix}_beeswarm.png"
    bar_path = output_dir / f"{prefix}_bar.png"

    importance = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(global_path, index=False)

    local_rows = []
    probabilities = model.predict_proba(sample)[:, 1]
    for row_number, source_index in enumerate(sample.index):
        context = metadata.loc[source_index].to_dict() if metadata is not None else {}
        for rank, contribution in enumerate(top_contributions(explanation, row_number), start=1):
            local_rows.append({
                "source_index": source_index,
                "prediction_risk": float(probabilities[row_number]),
                **context,
                "rank": rank,
                **contribution,
            })
    pd.DataFrame(local_rows).to_csv(local_path, index=False)

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()
    return {
        "global_importance": global_path,
        "local_explanations": local_path,
        "beeswarm": beeswarm_path,
        "bar": bar_path,
    }


def explain_prediction(model, row: pd.DataFrame, limit: int = 8) -> dict[str, Any]:
    explanation = explain_rows(create_explainer(model), row)
    return {
        "base_value": _base_value(explanation),
        "output_space": "raw_model_score",
        "contributions": top_contributions(explanation, limit=limit),
    }
