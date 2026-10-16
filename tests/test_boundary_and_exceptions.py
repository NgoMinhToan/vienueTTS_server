import io
import os
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "vtts_test_secret_123"

from vienueTTS_server.main import app

client = TestClient(app)
VALID_HEADERS = {"X-API-Key": "vtts_test_secret_123"}


def test_synthesize_empty_text_rejected():
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "", "voice": "minh_duc"}
    )
    assert res.status_code == 422


def test_synthesize_oversized_text_rejected():
    oversized = "A" * 5001
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": oversized, "voice": "minh_duc"}
    )
    assert res.status_code == 422


def test_synthesize_speed_out_of_bounds():
    # speed < 0.5
    res_low = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "speed": 0.2}
    )
    assert res_low.status_code == 422

    # speed > 2.0
    res_high = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "speed": 3.0}
    )
    assert res_high.status_code == 422


def test_synthesize_pitch_out_of_bounds():
    # pitch < -12
    res_low = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "pitch": -15}
    )
    assert res_low.status_code == 422

    # pitch > 12
    res_high = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "pitch": 15}
    )
    assert res_high.status_code == 422


def test_synthesize_emotion_out_of_bounds():
    res_low = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "emotion": -5}
    )
    assert res_low.status_code == 422

    res_high = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "emotion": 105}
    )
    assert res_high.status_code == 422


def test_synthesize_path_traversal_ref_audio_rejected():
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "ref_audio_path": "../../etc/passwd.wav"}
    )
    assert res.status_code == 400
    assert "traversal" in res.json()["detail"].lower()


def test_synthesize_invalid_audio_extension_rejected():
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "ref_audio_path": "malicious_script.exe"}
    )
    assert res.status_code == 400
    assert "không hợp lệ" in res.json()["detail"]


def test_synthesize_invalid_base64_npz_rejected():
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={"text": "Xin chào", "voice": "minh_duc", "ref_npz_base64": "!@#$%^&*()not_valid_base64"}
    )
    assert res.status_code == 400
    assert "Base64" in res.json()["detail"]


def test_get_non_existent_voice_detail_returns_404():
    res = client.get(
        "/internal/v1/voices/non_existent_voice_999",
        headers=VALID_HEADERS,
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()
