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


def test_dereference_model_symlinks(tmp_path):
    from pathlib import Path
    from vienueTTS_server.core.engine import engine_service

    blob_dir = tmp_path / "hub" / "blobs" / "5c"
    blob_dir.mkdir(parents=True)
    real_data = blob_dir / "target_weights.data"
    real_data.write_bytes(b"ONNX_WEIGHTS_DATA_123456789")

    snapshot_dir = tmp_path / "hub" / "models--test" / "snapshots" / "rev1"
    snapshot_dir.mkdir(parents=True)
    symlink_file = snapshot_dir / "vieneu_backbone_shared.data"

    try:
        symlink_file.symlink_to(real_data)
    except (OSError, NotImplementedError):
        # On Windows without developer mode/admin privileges, symlink creation may require privilege
        pytest.skip("Symlink creation not permitted in this environment")

    assert symlink_file.is_symlink()

    engine_service._dereference_model_symlinks(tmp_path)

    assert not symlink_file.is_symlink()
    assert symlink_file.is_file()
    assert symlink_file.read_bytes() == b"ONNX_WEIGHTS_DATA_123456789"

def test_synthesize_missing_ref_audio_path_returns_400():
    res = client.post(
        "/internal/v1/synthesize",
        headers=VALID_HEADERS,
        json={
            "text": "Xin chào",
            "voice": "minh_duc",
            "ref_audio_path": "/app/VoiceClonerAppServer/storage/cloned_voices/does_not_exist.wav",
        },
    )
    assert res.status_code == 400
    assert "không tồn tại trên worker" in res.json()["detail"]
