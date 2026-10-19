import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.explain import create_explainer, explain_rows, top_contributions


def test_tree_explanations_preserve_feature_alignment():
    X = pd.DataFrame({
        "temperature": [60, 65, 90, 95, 62, 92],
        "vibration": [1.0, 1.2, 5.0, 6.0, 0.9, 5.5],
    })
    y = pd.Series([0, 0, 1, 1, 0, 1])
    model = LGBMClassifier(n_estimators=10, min_child_samples=1, verbosity=-1, random_state=42).fit(X, y)

    explanation = explain_rows(create_explainer(model), X.iloc[[2]])
    contributions = top_contributions(explanation, limit=2)

    assert explanation.values.shape == (1, 2)
    assert {item["feature"] for item in contributions} == set(X.columns)
    assert all(np.isfinite(item["shap_value"]) for item in contributions)
    assert all(item["direction"] in {"increases_risk", "decreases_risk"} for item in contributions)
