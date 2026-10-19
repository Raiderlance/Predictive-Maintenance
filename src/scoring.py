from __future__ import annotations


def risk_band(risk: float, alert_threshold: float, high_threshold: float = 0.6) -> str:
    """Map probability to an action band consistent with the alert threshold."""
    if risk >= max(alert_threshold, high_threshold):
        return "HIGH"
    if risk >= alert_threshold:
        return "MEDIUM"
    return "LOW"
