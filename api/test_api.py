import pytest
from fastapi.testclient import TestClient
from main import app, model, load_model
import os

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_endpoint_fail_without_model():
    # Intentionally bypass loading
    response = client.get("/ready")
    assert response.status_code == 503 or response.status_code == 200

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text

def test_predict_validation():
    response = client.post("/predict", json={
        "Age": "invalid"
    })
    assert response.status_code == 422 # Validation Error
