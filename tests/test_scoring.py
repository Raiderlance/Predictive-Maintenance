from src.scoring import risk_band


def test_risk_bands_follow_selected_alert_threshold():
    assert risk_band(0.09, alert_threshold=0.10) == "LOW"
    assert risk_band(0.10, alert_threshold=0.10) == "MEDIUM"
    assert risk_band(0.60, alert_threshold=0.10) == "HIGH"
