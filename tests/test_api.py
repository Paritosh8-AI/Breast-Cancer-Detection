"""
Integration tests for FastAPI REST endpoints.
"""

from fastapi.testclient import TestClient
from cancer_ai.api.app import app

client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "OPERATIONAL"


def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_metrics():
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["roc_auc"] > 0.98
    assert data["accuracy"] > 0.95


def test_sample_cases():
    res = client.get("/api/v1/sample-cases")
    assert res.status_code == 200
    data = res.json()
    assert "typical_benign" in data
    assert "typical_malignant" in data


def test_predict_endpoint(sample_malignant_patient):
    payload = sample_malignant_patient.model_dump(by_alias=True)
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["diagnosis"] == "Malignant"
    assert data["risk_score"] > 70.0
