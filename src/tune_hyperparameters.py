from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from src.config import load_config
from src.features import model_features
from src.ims_features import ims_model_features
from src.ims_train import operational_metrics, split_by_run
from src.train import calculate_operational_metrics, evaluate, select_threshold, temporal_split


SEARCH_SPACE = {
    "n_estimators": [150, 300, 600],
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [7, 15, 31],
    "max_depth": [-1, 5, 8],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def candidate_parameters(n_iter: int, random_state: int) -> list[dict[str, Any]]:
    """Return a reproducible search containing the current baseline."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    baseline = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }
    sampled = (
        list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter - 1, random_state=random_state))
        if n_iter > 1 else []
    )
    return [baseline, *[params for params in sampled if params != baseline]][:n_iter]


def tune(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_iter: int,
    random_state: int,
) -> tuple[LGBMClassifier, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fitted: list[LGBMClassifier] = []
    for trial, params in enumerate(candidate_parameters(n_iter, random_state), start=1):
        model = LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            subsample_freq=1,
            verbosity=-1,
        )
        model.fit(train[features], train["failure_within_horizon"])
        probabilities = model.predict_proba(validation[features])[:, 1]
        threshold, _ = select_threshold(model, validation[features], validation["failure_within_horizon"])
        validation_metrics = evaluate(
            model, validation[features], validation["failure_within_horizon"], threshold
        )
        rows.append({
            "trial": trial,
            "is_baseline": trial == 1,
            **params,
            "validation_pr_auc": float(average_precision_score(validation["failure_within_horizon"], probabilities)),
            "validation_roc_auc": float(roc_auc_score(validation["failure_within_horizon"], probabilities)),
            "validation_threshold": threshold,
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
        })
        fitted.append(model)
        print(f"Trial {trial}/{n_iter}: validation PR-AUC={rows[-1]['validation_pr_auc']:.4f}")

    results = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc", "validation_f1"], ascending=False
    ).reset_index(drop=True)
    best_trial = int(results.iloc[0]["trial"])
    best_model = fitted[best_trial - 1]
    best_threshold = float(results.iloc[0]["validation_threshold"])
    test_metrics = evaluate(best_model, test[features], test["failure_within_horizon"], best_threshold)
    test_metrics["threshold"] = best_threshold
    test_metrics["best_trial"] = best_trial
    test_metrics["best_params"] = {
        name: results.iloc[0][name].item() if hasattr(results.iloc[0][name], "item") else results.iloc[0][name]
        for name in SEARCH_SPACE
    }
    return best_model, results, test_metrics


def run_experiment(name: str, input_path: Path, output_dir: Path, n_iter: int, seed: int, horizon: int) -> None:
    data = pd.read_parquet(input_path)
    if name == "synthetic":
        features = model_features(data)
        train, validation, test = temporal_split(data, purge_hours=horizon)
    else:
        features = ims_model_features(data)
        train, validation, test = split_by_run(data)

    best_model, results, metrics = tune(train, validation, test, features, n_iter, seed)
    if name == "ims":
        metrics.update(operational_metrics(best_model, test, features, metrics["threshold"]))
        metrics["evaluation_note"] = "Exploratory re-evaluation: Set 3 previously influenced feature design."
    else:
        operational = test[["machine_id", "timestamp", "failure"]].copy()
        operational["alert"] = best_model.predict_proba(test[features])[:, 1] >= metrics["threshold"]
        metrics["operational_metrics"] = calculate_operational_metrics(
            operational, entity_col="machine_id", event_col="failure", horizon_hours=horizon
        )

    baseline = results.loc[results["is_baseline"]].iloc[0]
    winner = results.iloc[0]
    metrics.update({
        "experiment": name,
        "selection_metric": "validation_pr_auc",
        "n_candidates": len(results),
        "baseline_validation_pr_auc": float(baseline["validation_pr_auc"]),
        "best_validation_pr_auc": float(winner["validation_pr_auc"]),
        "validation_pr_auc_improvement": float(winner["validation_pr_auc"] - baseline["validation_pr_auc"]),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"{name}_hyperparameter_trials.csv", index=False)
    (output_dir / f"{name}_tuning_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "experiment": name,
        "best_params": metrics["best_params"],
        "validation_pr_auc": metrics["best_validation_pr_auc"],
        "test_pr_auc": metrics["pr_auc"],
        "test_f1": metrics["f1"],
        "threshold": metrics["threshold"],
    }, indent=2))


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Validation-only LightGBM hyperparameter experiments")
    parser.add_argument("--dataset", choices=("synthetic", "ims", "both"), default="both")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--output-dir", default="artifacts/tuning")
    args = parser.parse_args()
    selected = ("synthetic", "ims") if args.dataset == "both" else (args.dataset,)
    inputs = {
        "synthetic": Path("data/processed/features.parquet"),
        "ims": Path("data/processed/ims_features.parquet"),
    }
    for name in selected:
        run_experiment(
            name, inputs[name], Path(args.output_dir), args.n_iter,
            int(config.get("random_seed", 42)), int(config.get("failure_horizon_hours", 24)),
        )


if __name__ == "__main__":
    main()
