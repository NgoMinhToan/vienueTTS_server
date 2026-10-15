import io
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger("vienueTTS_server.engine")

SAMPLE_RATE = 48000

# Audio conversion utilities
def float32_to_pcm16(audio: np.ndarray, sample_rate: int = 48000) -> bytes:
    if audio is None or len(audio) == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_arr = (clipped * 32767.0).astype(np.int16)
    return int16_arr.tobytes()


def wav_from_float32(audio: np.ndarray, sample_rate: int = 48000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class ModelEngineService:
    def __init__(self):
        self._engine = None
        self.engine_mode = os.getenv("VIENEU_MODE", "v3turbo")
        self.precision = os.getenv("VIENEU_PRECISION", "fp32")
        self.threads = int(os.getenv("VIENEU_THREADS", "8"))
        self._init_env_paths()

    def _init_env_paths(self):
        models_dir = Path(os.getenv("VIENEU_MODELS_DIR", "D:/Project/vieneu-dubbling/data/models"))
        if models_dir.exists():
            os.environ.setdefault("HF_HOME", str(models_dir.resolve()))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        src_dir_env = os.getenv("VIENEU_SRC_DIR")
        if src_dir_env:
            src_path = Path(src_dir_env)
        elif Path("/app/vieneu_src").exists():
            src_path = Path("/app/vieneu_src")
        else:
            src_path = Path("D:/Project/vieneu-dubbling/src")

        if src_path.exists() and str(src_path.resolve()) not in sys.path:
            sys.path.insert(0, str(src_path.resolve()))

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from vieneu import Vieneu
            logger.info("Initializing core Vieneu inference engine...")
            self._engine = Vieneu()
            logger.info("Vieneu inference engine loaded successfully.")
            return self._engine
        except Exception as e:
            logger.warning(f"Could not load native Vieneu engine ({e}). Fallback to mock synthesis.")
            return None

    def synthesize(
        self,
        text: str,
        voice_id: str = "minh_duc",
        speed: float = 1.0,
        pitch: int = 0,
        emotion_level: int = 0,
        denoise: bool = True,
        smart_pause: bool = True,
        ref_audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        engine = self._load_engine()
        if engine is not None:
            # Native model call
            prompt = text
            # Format audio via Vieneu infer
            audio = engine.infer(
                text=prompt,
                voice=voice_id,
                speed=speed,
                pitch=pitch,
                emotion=emotion_level,
                ref_audio=ref_audio_path,
            )
            if isinstance(audio, tuple):
                audio = audio[0]
            return audio.astype(np.float32), SAMPLE_RATE

        # Fallback sine wave tone for unit test/dev mode without model checkpoint
        duration = min(3.0, max(0.5, len(text) * 0.05 / max(0.5, speed)))
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
        base_freq = 220.0 * (2.0 ** (pitch / 12.0))
        audio = 0.25 * np.sin(2.0 * np.pi * base_freq * t)
        return audio, SAMPLE_RATE

    def precompute_voice_embeddings(self, audio_path: str, output_npz_path: Path) -> bool:
        engine = self._load_engine()
        if engine is not None and hasattr(engine, "extract_features"):
            return engine.extract_features(audio_path, str(output_npz_path))

        # Default fallback embedding arrays
        speaker_emb = np.zeros(256, dtype=np.float32)
        codes = np.zeros((1, 256), dtype=np.int32)
        np.savez(output_npz_path, speaker_emb=speaker_emb, codes=codes)
        return True


engine_service = ModelEngineService()
