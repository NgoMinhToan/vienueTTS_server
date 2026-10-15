import base64
import io
import os
import numpy as np
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
    assert "total_voices" in data
    assert data["total_voices"] >= 23


def test_root_health_open():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_voices_list_rejected_without_api_key():
    res = client.get("/internal/v1/voices")
    assert res.status_code == 401


def test_voices_list_with_api_key():
    res = client.get("/internal/v1/voices", headers={"X-API-Key": "vtts_test_secret_123"})
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] >= 23
    assert data["default_voice"] == "Minh Đức"
    voices = data["voices"]
    assert len(voices) >= 23
    ids = [v["id"] for v in voices]
    assert "minh_duc" in ids
    assert "quynh_anh" in ids


def test_voice_detail_endpoint():
    res = client.get("/internal/v1/voices/minh_duc", headers={"X-API-Key": "vtts_test_secret_123"})
    assert res.status_code == 200
    voice = res.json()
    assert voice["name"] == "Minh Đức"
    assert voice["sample_rate"] == 48000


def test_v1_alias_routes():
    res = client.get("/v1/voices", headers={"X-API-Key": "vtts_test_secret_123"})
    assert res.status_code == 200
    assert res.json()["total_count"] >= 23


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
    monkeypatch.setattr(engine.engine_service, "synthesize", lambda **kwargs: (b"RIFFmockWAVdata", 48000))

    response = client.post(
        "/internal/v1/synthesize",
        headers={"X-API-Key": "vtts_test_secret_123"},
        json={"text": "Xin chào", "voice": "minh_duc"}
    )
    assert response.status_code == 200
    assert response.content == b"RIFFmockWAVdata"
    assert response.headers["content-type"] == "audio/wav"


def test_synthesize_with_cloned_voice_npz_base64(monkeypatch):
    from vienueTTS_server.core import engine

    received_kwargs = {}
    def mock_synthesize(**kwargs):
        received_kwargs.update(kwargs)
        return (b"RIFFmockClonedWAVdata", 48000)

    monkeypatch.setattr(engine.engine_service, "synthesize", mock_synthesize)

    # Create dummy .npz in memory
    buf = io.BytesIO()
    np.savez(buf, speaker_emb=np.ones(192, dtype=np.float32), codes=np.zeros((1, 256), dtype=np.int64))
    npz_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    res = client.post(
        "/internal/v1/synthesize",
        headers={"X-API-Key": "vtts_test_secret_123"},
        json={"text": "Giọng clone thử nghiệm", "ref_npz_base64": npz_b64}
    )
    assert res.status_code == 200
    assert res.content == b"RIFFmockClonedWAVdata"
    assert received_kwargs.get("ref_npz_bytes") is not None


def test_extract_speaker_embedding_embed_endpoint():
    import soundfile as sf
    buf = io.BytesIO()
    sample_audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000))).astype(np.float32)
    sf.write(buf, sample_audio, 16000, format="WAV")
    valid_wav = buf.getvalue()
    files = {"audio_file": ("test.wav", valid_wav, "audio/wav")}

    res = client.post(
        "/internal/v1/embed",
        headers={"X-API-Key": "vtts_test_secret_123"},
        files=files,
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    # Verify returning valid .npz
    npz = np.load(io.BytesIO(res.content))
    assert "speaker_emb" in npz


def test_pna_options_header():
    headers = {
        "Origin": "https://ngominhtoan.github.io",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Private-Network": "true",
    }
    res = client.options("/internal/v1/synthesize", headers=headers)
    assert res.headers.get("access-control-allow-private-network") == "true" or res.headers.get("Access-Control-Allow-Private-Network") == "true"
