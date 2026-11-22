import sys
import types

import pytest

from vienueTTS_server.core import engine, runtime


def test_auto_without_torch_is_cpu():
    assert runtime.resolve_device(
        "auto",
        torch_installed=lambda: False,
        cuda_info=lambda: (False, 0),
    ) == "cpu"


def test_auto_with_torch_but_no_cuda_is_cpu():
    assert runtime.resolve_device(
        "auto",
        torch_installed=lambda: True,
        cuda_info=lambda: (False, 0),
    ) == "cpu"


def test_auto_with_cuda_is_cuda():
    assert runtime.resolve_device(
        "auto",
        torch_installed=lambda: True,
        cuda_info=lambda: (True, 1),
    ) == "cuda"


def test_explicit_cpu_wins_even_when_cuda_exists():
    assert runtime.resolve_device(
        "cpu",
        torch_installed=lambda: True,
        cuda_info=lambda: (True, 4),
    ) == "cpu"


def test_explicit_cuda_without_torch_raises():
    with pytest.raises(runtime.DeviceConfigurationError, match="-cuda image"):
        runtime.resolve_device(
            "cuda",
            torch_installed=lambda: False,
            cuda_info=lambda: (False, 0),
        )


def test_explicit_cuda_without_device_raises():
    with pytest.raises(runtime.DeviceConfigurationError, match="no usable CUDA device"):
        runtime.resolve_device(
            "cuda",
            torch_installed=lambda: True,
            cuda_info=lambda: (False, 0),
        )


def test_cuda_index_validation():
    assert runtime.resolve_device(
        "cuda:1",
        torch_installed=lambda: True,
        cuda_info=lambda: (True, 2),
    ) == "cuda:1"
    with pytest.raises(runtime.DeviceConfigurationError):
        runtime.resolve_device(
            "cuda:2",
            torch_installed=lambda: True,
            cuda_info=lambda: (True, 2),
        )
    with pytest.raises(runtime.DeviceConfigurationError):
        runtime.resolve_device(
            "cuda:-1",
            torch_installed=lambda: True,
            cuda_info=lambda: (True, 2),
        )


@pytest.mark.parametrize("device", ["mps", "xpu", "gpu", "banana"])
def test_unsupported_devices_raise(device):
    with pytest.raises(runtime.DeviceConfigurationError, match="Supported values"):
        runtime.resolve_device(
            device,
            torch_installed=lambda: True,
            cuda_info=lambda: (True, 1),
        )


def test_build_engine_kwargs_cpu():
    assert runtime.build_engine_kwargs("cpu", "int8", 4) == {
        "mode": "v3turbo",
        "device": "cpu",
        "precision": "int8",
        "threads": 4,
    }


def test_build_engine_kwargs_cuda():
    assert runtime.build_engine_kwargs("cuda:1", "int8", 4) == {
        "mode": "v3turbo",
        "device": "cuda:1",
        "dtype": "auto",
    }


def test_invalid_env_values_fall_back_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("VIENEU_THREADS", "abc")
    monkeypatch.setenv("VIENEU_PRECISION", "weird")
    monkeypatch.setenv("VIENEU_MAX_CONCURRENCY", "-1")
    monkeypatch.setenv("VIENEU_DEVICE", "cpu")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    service = engine.ModelEngineService()
    assert service.threads == 8
    assert service.precision == "fp32"
    assert service.max_concurrency == 0
    assert service.device == "cpu"
    assert service.device_error is None


def test_engine_forces_v3turbo_and_wires_cpu_kwargs(monkeypatch, tmp_path):
    captured = {}

    class FakeVieneu:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._preset_voices = {}

    monkeypatch.setenv("VIENEU_DEVICE", "cpu")
    monkeypatch.setenv("VIENEU_MODE", "v3nano")
    monkeypatch.setenv("VIENEU_PRECISION", "int8")
    monkeypatch.setenv("VIENEU_THREADS", "4")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=FakeVieneu))

    service = engine.ModelEngineService()
    loaded = service._load_engine()

    assert loaded is not None
    assert captured["mode"] == "v3turbo"
    assert captured["device"] == "cpu"
    assert captured["precision"] == "int8"
    assert captured["threads"] == 4
    assert service.runtime == "onnx"


def test_engine_auto_cuda_uses_pytorch_kwargs(monkeypatch, tmp_path):
    captured = {}

    class FakeVieneu:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._preset_voices = {}

    monkeypatch.setenv("VIENEU_DEVICE", "auto")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "_torch_installed", lambda: True)
    monkeypatch.setattr(runtime, "_cuda_info", lambda: (True, 1))
    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=FakeVieneu))

    service = engine.ModelEngineService()
    assert service.device == "cuda"
    assert service.runtime == "pytorch"

    loaded = service._load_engine()
    assert loaded is not None
    assert captured["mode"] == "v3turbo"
    assert captured["device"] == "cuda"
    assert captured["dtype"] == "auto"
    assert "precision" not in captured
    assert "threads" not in captured


def test_explicit_cuda_without_torch_marks_engine_unready(monkeypatch, tmp_path):
    monkeypatch.setenv("VIENEU_DEVICE", "cuda")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "_torch_installed", lambda: False)

    service = engine.ModelEngineService()

    assert service.device_error is not None
    assert service.device == "cpu"
    assert service._load_engine() is None
