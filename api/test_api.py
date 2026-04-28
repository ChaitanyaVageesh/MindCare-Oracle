"""API unit tests — run with: pytest test_api.py -v"""
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "Age": 30,
    "Gender": "male",
    "family_history": "No",
    "benefits": "Yes",
    "care_options": "Not sure",
    "anonymity": "Yes",
    "leave": "Somewhat easy",
    "work_interfere": "Sometimes",
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint():
    response = client.get("/ready")
    # 200 if model loaded, 503 if not — both are valid in CI
    assert response.status_code in (200, 503)


def test_metrics_endpoint_contains_prometheus_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text


def test_predict_missing_fields_returns_422():
    response = client.post("/predict", json={"Age": "invalid"})
    assert response.status_code == 422


def test_predict_missing_required_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Gender"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_drift_check_missing_baseline_returns_503_or_200():
    payload = {"records": [VALID_PAYLOAD]}
    response = client.post("/drift/check", json=payload)
    # 503 when no baseline file present, 200 when it is
    assert response.status_code in (200, 503)
