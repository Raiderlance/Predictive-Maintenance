from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.scoring import risk_band
from src.explain import create_explainer, explain_rows, top_contributions

MODEL_PATH = Path("models/lightgbm_predictive_maintenance.joblib")
app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

class MachineFeatures(BaseModel):
    values: dict[str, float] = Field(..., description="Feature dictionary matching training feature names")

@lru_cache(maxsize=1)
def load_model_bundle():
    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def load_explainer():
    return create_explainer(load_model_bundle()["model"])

def validated_frame(payload: MachineFeatures, bundle) -> pd.DataFrame:
    missing = [c for c in bundle["features"] if c not in payload.values]
    extra = sorted(set(payload.values) - set(bundle["features"]))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {extra[:8]}{'...' if len(extra) > 8 else ''}")
    if any(not math.isfinite(value) for value in payload.values.values()):
        raise HTTPException(status_code=422, detail="All feature values must be finite numbers")
    return pd.DataFrame([{c: payload.values[c] for c in bundle["features"]}])

def prediction_response(bundle, X: pd.DataFrame) -> dict:
    risk = float(bundle["model"].predict_proba(X)[:, 1][0])
    threshold = float(bundle.get("threshold", 0.5))
    return {"failure_risk": risk, "risk_band": risk_band(risk, threshold), "alert": risk >= threshold,
            "threshold": threshold, "model_version": bundle.get("model_version", "unknown")}

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: MachineFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Run training first.")
    bundle = load_model_bundle()
    return prediction_response(bundle, validated_frame(payload, bundle))

@app.post("/predict/explain")
def predict_explain(payload: MachineFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Run training first.")
    bundle = load_model_bundle()
    X = validated_frame(payload, bundle)
    response = prediction_response(bundle, X)
    explanation = explain_rows(load_explainer(), X)
    response.update({
        "explanation_output": "raw_model_score",
        "base_value": float(explanation.base_values[0]),
        "contributions": top_contributions(explanation),
        "interpretation_warning": "SHAP explains model behavior, not physical root cause.",
    })
    return response
