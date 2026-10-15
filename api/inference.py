import asyncio
import base64
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from vienueTTS_server.core.engine import (
    SAMPLE_RATE,
    engine_service,
    float32_to_pcm16,
    wav_from_float32,
)
from vienueTTS_server.core.security import verify_worker_api_key
from vienueTTS_server.models.schemas import (
    HealthResponse,
    MetricsResponse,
    SynthesizeRequest,
    VoiceItem,
    VoiceListResponse,
)

logger = logging.getLogger("vienueTTS_server.inference")

router = APIRouter(tags=["Inference"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".ogg", ".flac"}


def validate_file_path(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    path = Path(file_path)
    # Check directory traversal
    if ".." in path.parts:
        raise HTTPException(
            status_code=400,
            detail="Đường dẫn tệp không hợp lệ (path traversal attempt).",
        )
    return str(path.resolve())


def validate_ref_audio_path(ref_audio_path: Optional[str]) -> Optional[str]:
    if not ref_audio_path:
        return None
    path = Path(ref_audio_path)
    if path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tệp âm thanh tham chiếu không hợp lệ. Đuôi tệp cho phép: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    return validate_file_path(ref_audio_path)


class InferenceMetricsTracker:
    def __init__(self):
        self.total_requests: int = 0
        self.active_jobs: int = 0
        self.total_latency_ms: float = 0.0
        self._lock = asyncio.Lock()

    async def record_start(self):
        async with self._lock:
            self.active_jobs += 1
            self.total_requests += 1

    async def record_finish(self, duration_ms: float):
        async with self._lock:
            self.active_jobs = max(0, self.active_jobs - 1)
            self.total_latency_ms += duration_ms

    async def get_metrics(self) -> dict:
        async with self._lock:
            completed = self.total_requests - self.active_jobs
            avg_latency = (self.total_latency_ms / completed) if completed > 0 else 0.0
            return {
                "total_requests": self.total_requests,
                "active_jobs": self.active_jobs,
                "avg_latency_ms": round(avg_latency, 2),
            }

    async def get_active_jobs(self) -> int:
        async with self._lock:
            return self.active_jobs


metrics_tracker = InferenceMetricsTracker()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Worker health status, voice count and engine configuration."""
    active = await metrics_tracker.get_active_jobs()
    is_loaded = engine_service._engine is not None
    return HealthResponse(
        status="healthy",
        service="vienueTTS_server",
        model_loaded=is_loaded,
        engine_mode=engine_service.engine_mode,
        precision=engine_service.precision,
        device="cpu",
        threads=engine_service.threads,
        active_jobs=active,
        total_voices=engine_service.get_voice_count(),
    )


@router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(verify_worker_api_key)])
async def get_metrics():
    """Performance telemetry and inference latency."""
    metrics = await metrics_tracker.get_metrics()
    return MetricsResponse(**metrics)


@router.get("/voices", response_model=VoiceListResponse, dependencies=[Depends(verify_worker_api_key)])
async def list_voices():
    """
    List all available preset voices with metadata and total count.
    Protected by worker API key.
    """
    raw_voices = engine_service.list_voices()
    voices = [VoiceItem(**v) for v in raw_voices]
    return VoiceListResponse(
        total_count=len(voices),
        default_voice="Minh Đức",
        voices=voices,
    )


@router.get("/voices/{voice_id}", response_model=VoiceItem, dependencies=[Depends(verify_worker_api_key)])
async def get_voice_detail(voice_id: str):
    """Retrieve details for a specific voice by slug or name."""
    raw_voices = engine_service.list_voices()
    resolved_name = engine_service.resolve_voice_name(voice_id)
    for v in raw_voices:
        if v["id"] == voice_id or v["name"].lower() == resolved_name.lower():
            return VoiceItem(**v)
    raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")


@router.post("/synthesize", dependencies=[Depends(verify_worker_api_key)])
async def synthesize(request: SynthesizeRequest):
    """
    Stateless AI text-to-speech synthesis endpoint.
    Protected by worker API key.
    Supports preset voices and cloned voices via ref_npz_path, ref_npz_base64, or ref_audio_path.
    Returns standard 48kHz WAV audio bytes (or PCM).
    """
    validated_ref_path = validate_ref_audio_path(request.ref_audio_path)
    validated_npz_path = validate_file_path(request.ref_npz_path)

    ref_npz_bytes = None
    if request.ref_npz_base64:
        try:
            ref_npz_bytes = base64.b64decode(request.ref_npz_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 ref_npz_base64: {e}")

    await metrics_tracker.record_start()
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        audio, sample_rate = await loop.run_in_executor(
            None,
            lambda: engine_service.synthesize(
                text=request.text,
                voice_id=request.voice_id or request.voice or "minh_duc",
                speed=request.speed,
                pitch=request.pitch,
                emotion_level=request.emotion_level or request.emotion or 0,
                denoise=request.denoise,
                smart_pause=request.smart_pause,
                ref_audio_path=validated_ref_path,
                ref_npz_path=validated_npz_path,
                ref_npz_bytes=ref_npz_bytes,
            ),
        )

        duration_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(audio, (bytes, bytearray)):
            audio_bytes = bytes(audio)
            content_type = "audio/wav"
        elif (request.response_format or request.format or "wav").lower() == "pcm":
            audio_bytes = float32_to_pcm16(audio, sample_rate=sample_rate)
            content_type = "audio/pcm"
        else:
            audio_bytes = wav_from_float32(audio, sample_rate=sample_rate)
            content_type = "audio/wav"

        headers = {
            "Content-Type": content_type,
            "X-Sample-Rate": str(sample_rate),
            "X-Execution-Time-Ms": f"{duration_ms:.2f}",
        }
        return Response(content=audio_bytes, media_type=content_type, headers=headers)
    except ValueError as e:
        logger.warning(f"Validation error in synthesize: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        await metrics_tracker.record_finish(duration_ms)


@router.post("/embed", dependencies=[Depends(verify_worker_api_key)])
async def extract_speaker_embedding(audio_file: UploadFile = File(...)):
    """
    Extract speaker embedding codes from reference audio file.
    Returns binary application/octet-stream representing NumPy .npz file.
    """
    await metrics_tracker.record_start()
    t0 = time.perf_counter()
    temp_in = None
    temp_out = None
    try:
        contents = await audio_file.read()
        suffix = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
            f_in.write(contents)
            temp_in = Path(f_in.name)

        temp_out = temp_in.with_suffix(".npz")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: engine_service.precompute_voice_embeddings(str(temp_in), temp_out),
        )

        if temp_out.exists() and temp_out.stat().st_size > 0:
            npz_bytes = temp_out.read_bytes()
        else:
            buf = io.BytesIO()
            np.savez(buf, speaker_emb=np.zeros(192, dtype=np.float32), codes=np.zeros((1, 256), dtype=np.int64))
            npz_bytes = buf.getvalue()

        duration_ms = (time.perf_counter() - t0) * 1000.0
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Execution-Time-Ms": f"{duration_ms:.2f}",
        }
        return Response(content=npz_bytes, media_type="application/octet-stream", headers=headers)
    except Exception as e:
        logger.error(f"Inference embedding extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        await metrics_tracker.record_finish(duration_ms)
        if temp_in and temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass
        if temp_out and temp_out.exists():
            try:
                temp_out.unlink()
            except Exception:
                pass
