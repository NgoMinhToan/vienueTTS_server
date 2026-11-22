"""Runtime configuration helpers for CPU/CUDA device selection.

This module stays free of ``torch`` and ``vieneu`` imports at module load time so
the API and tests can use it in a CPU-only environment.  Torch is only imported
when the caller asks the default probe to inspect CUDA.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Callable, Optional, Tuple

logger = logging.getLogger("vienueTTS_server.runtime")

SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "cuda:N")


class DeviceConfigurationError(RuntimeError):
    """Raised when ``VIENEU_DEVICE`` requests an unavailable runtime."""


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cuda_info() -> Tuple[bool, int]:
    """Return ``(cuda_available, device_count)`` without failing on broken drivers."""
    try:
        import torch
    except Exception:
        return False, 0
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
    except Exception:
        return False, 0
    return available, count


def parse_int_env(name: str, default: int, minimum: int = 0) -> int:
    """Parse an integer env var without ever crashing process import."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r; using %s.", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Ignoring out-of-range %s=%r; using %s.", name, raw, default)
        return default
    return value


def parse_precision_env() -> str:
    raw = os.getenv("VIENEU_PRECISION", "fp32")
    precision = (raw or "").strip().lower()
    if precision not in ("fp32", "int8"):
        logger.warning("Ignoring invalid VIENEU_PRECISION=%r; using fp32.", raw)
        return "fp32"
    return precision


def resolve_device(
    requested: str,
    *,
    torch_installed: Optional[Callable[[], bool]] = None,
    cuda_info: Optional[Callable[[], Tuple[bool, int]]] = None,
) -> str:
    """Resolve ``VIENEU_DEVICE`` to a concrete device string.

    ``auto`` falls back to CPU when torch/CUDA is unavailable.  An explicit
    ``cuda``/``cuda:N`` request is strict: it raises when the runtime cannot
    honour it, so a misconfigured GPU container fails at startup instead of
    silently running on CPU.
    """
    torch_installed = torch_installed or _torch_installed
    cuda_info = cuda_info or _cuda_info

    value = (requested or "auto").strip().lower()

    if value in ("", "auto"):
        if not torch_installed():
            return "cpu"
        available, count = cuda_info()
        return "cuda" if available and count > 0 else "cpu"

    if value == "cpu":
        return "cpu"

    if value == "cuda":
        if not torch_installed():
            raise DeviceConfigurationError(
                "VIENEU_DEVICE=cuda requires the CUDA image (torch with CUDA). "
                "Use the -cuda image, or set VIENEU_DEVICE=auto for CPU fallback."
            )
        available, count = cuda_info()
        if not available or count < 1:
            raise DeviceConfigurationError(
                "VIENEU_DEVICE=cuda requested, but torch reports no usable CUDA device. "
                "Check the NVIDIA driver, nvidia-container-toolkit, and --gpus all."
            )
        return "cuda"

    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if not suffix.isdigit():
            raise DeviceConfigurationError(
                f"Invalid CUDA index in VIENEU_DEVICE={value!r}; use cuda or cuda:N."
            )
        index = int(suffix)
        if not torch_installed():
            raise DeviceConfigurationError(
                f"VIENEU_DEVICE={value!r} requires the CUDA image (torch with CUDA)."
            )
        available, count = cuda_info()
        if not available or index >= count:
            raise DeviceConfigurationError(
                f"VIENEU_DEVICE={value!r} is not available; torch sees {count} CUDA device(s)."
            )
        return f"cuda:{index}"

    raise DeviceConfigurationError(
        f"Unsupported VIENEU_DEVICE={value!r}. Supported values: auto, cpu, cuda, cuda:N. "
        "Apple MPS / Intel XPU / AMD ROCm are not supported by this worker."
    )


def build_engine_kwargs(device: str, precision: str, threads: int) -> dict:
    """Build the ``Vieneu()`` kwargs for the requested runtime.

    v3 Turbo uses ONNX on CPU and PyTorch on CUDA.  Precision/threads only apply
    to the ONNX CPU path; the PyTorch path uses its dtype auto-selection.
    """
    kwargs = {"mode": "v3turbo", "device": device}
    if device == "cpu":
        kwargs["precision"] = precision
        kwargs["threads"] = threads
    else:
        kwargs["dtype"] = "auto"
    return kwargs
