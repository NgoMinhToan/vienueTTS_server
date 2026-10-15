import os
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "vtts_test_secret_123"

from vienueTTS_server.main import app

client = TestClient(app)

def test_health_check_open_without_api_key():
    response = client.get("/internal/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_synthesize_rejected_without_api_key():
    response = client.post("/internal/v1/synthesize", json={"text": "Xin chào", "voice": "minh_duc"})
    assert response.status_code == 401
    assert "Invalid or missing worker API key" in response.json()["detail"]

def test_synthesize_rejected_with_invalid_api_key():
    response = client.post(
        "/internal/v1/synthesize",
        headers={"X-API-Key": "wrong_key"},
        json={"text": "Xin chào", "voice": "minh_duc"}
    )
    assert response.status_code == 401
    assert "Invalid or missing worker API key" in response.json()["detail"]

def test_synthesize_accepted_with_valid_api_key(monkeypatch):
    from vienueTTS_server.core import engine
    monkeypatch.setattr(engine.engine_service, "synthesize", lambda **kwargs: (b"RIFFmockWAVdata", 0.05))

    response = client.post(
        "/internal/v1/synthesize",
        headers={"X-API-Key": "vtts_test_secret_123"},
        json={"text": "Xin chào", "voice": "minh_duc"}
    )
    assert response.status_code == 200
    assert response.content == b"RIFFmockWAVdata"
    assert response.headers["content-type"] == "audio/wav"
