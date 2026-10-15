import io
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf

logger = logging.getLogger("vienueTTS_server.engine")

SAMPLE_RATE = 48000

# 23 Builtin Preset Voice Catalog
BUILTIN_PRESET_METADATA = [
    {"name": "Minh Đức", "gender": "male", "description": "Nam · Bắc · Phong cách tin tức"},
    {"name": "Phạm Tuyên", "gender": "male", "description": "Nam · Bắc · Phong cách tự nhiên"},
    {"name": "Thái Sơn", "gender": "male", "description": "Nam · Nam · Phong cách kể chuyện"},
    {"name": "Xuân Vĩnh", "gender": "male", "description": "Nam · Bắc · Phong cách tự nhiên"},
    {"name": "Thanh Bình", "gender": "male", "description": "Nam · Bắc · Phong cách kể chuyện"},
    {"name": "Trúc Ly", "gender": "female", "description": "Nữ · Bắc · Phong cách tự nhiên"},
    {"name": "Ngọc Linh", "gender": "female", "description": "Nữ · Bắc · Phong cách kể chuyện"},
    {"name": "Đoan Trang", "gender": "female", "description": "Nữ · Bắc · Phong cách tự nhiên"},
    {"name": "Mai Anh", "gender": "female", "description": "Nữ · Bắc · Phong cách tin tức"},
    {"name": "Thục Đoan", "gender": "female", "description": "Nữ · Nam · Phong cách kể chuyện"},
    {"name": "Minh Triết", "gender": "male", "description": "Nam · Nam · Phong cách tin tức"},
    {"name": "Thùy Dung", "gender": "female", "description": "Nữ · Nam · Phong cách tin tức"},
    {"name": "Quang Sơn", "gender": "male", "description": "Nam · Trung · Phong cách tự nhiên"},
    {"name": "Ngọc Trân", "gender": "female", "description": "Nữ · Trung · Phong cách tự nhiên"},
    {"name": "Mỹ Duyên", "gender": "female", "description": "Nữ · Nam · Phong cách đọc truyện"},
    {"name": "Quỳnh Anh", "gender": "female", "description": "Nữ · Bắc · Phong cách đọc truyện"},
    {"name": "Đức Trí", "gender": "male", "description": "Nam · Nam · Phong cách đọc truyện"},
    {"name": "Kim Thanh", "gender": "female", "description": "Nữ · Nam · Phong cách đọc truyện"},
    {"name": "Ngọc Huyền", "gender": "female", "description": "Nữ · Bắc · Giọng đọc tự nhiên"},
    {"name": "Adam", "gender": "male", "description": "Nam · Nam · Giọng đọc tự nhiên"},
    {"name": "Mạnh Dũng", "gender": "male", "description": "Nam · Bắc · Phong cách tự nhiên"},
    {"name": "Minh Quân", "gender": "male", "description": "Nam · Bắc · Phong cách tự nhiên"},
    {"name": "Anh Khôi", "gender": "male", "description": "Nam · Bắc · Phong cách kể chuyện"},
]


def slugify_voice_name(text: str) -> str:
    """Convert Vietnamese voice name to URL/API friendly slug (e.g. 'Minh Đức' -> 'minh_duc')."""
    normalized = unicodedata.normalize("NFD", text)
    stripped = re.sub(r"[\u0300-\u036f]", "", normalized)
    replaced = stripped.replace("đ", "d").replace("Đ", "D")
    cleaned = re.sub(r"[^\w\s-]", "", replaced).strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned)


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
        self._slug_to_name: Dict[str, str] = {}
        self._voices_cache: List[Dict[str, Any]] = []
        self._init_env_paths()
        self._init_voice_catalog()

    def _init_env_paths(self):
        models_dir_env = os.getenv("VIENEU_MODELS_DIR")
        if models_dir_env and Path(models_dir_env).exists():
            models_dir = Path(models_dir_env)
        elif Path("/app/data/models").exists():
            models_dir = Path("/app/data/models")
        else:
            models_dir = Path("D:/Project/vieneu-dubbling/data/models")

        if models_dir.exists():
            os.environ["HF_HOME"] = str(models_dir.resolve())
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        src_dir_env = os.getenv("VIENEU_SRC_DIR")
        if src_dir_env and Path(src_dir_env).exists():
            src_path = Path(src_dir_env)
        elif Path("/app/vieneu_src").exists():
            src_path = Path("/app/vieneu_src")
        else:
            src_path = Path("D:/Project/vieneu-dubbling/src")

        if src_path.exists() and str(src_path.resolve()) not in sys.path:
            sys.path.insert(0, str(src_path.resolve()))

    def _init_voice_catalog(self):
        """Build voice lookup maps and cache list."""
        self._voices_cache = []
        self._slug_to_name = {}

        # Look for voices_v3_turbo.json in vieneu assets
        json_path = None
        for p in [
            Path("/app/vieneu_src/vieneu/assets/voices_v3_turbo.json"),
            Path("D:/Project/vieneu-dubbling/src/vieneu/assets/voices_v3_turbo.json"),
        ]:
            if p.exists():
                json_path = p
                break

        presets_raw = {}
        if json_path and json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                presets_raw = data.get("presets", {})
            except Exception as e:
                logger.warning(f"Error reading voices JSON: {e}")

        # Merge with built-in metadata list
        if presets_raw:
            for name, meta in presets_raw.items():
                slug = slugify_voice_name(name)
                self._slug_to_name[slug] = name
                self._slug_to_name[name.lower()] = name
                self._slug_to_name[name] = name
                self._voices_cache.append({
                    "id": slug,
                    "name": name,
                    "description": meta.get("description", ""),
                    "gender": meta.get("gender", "unknown"),
                    "style": meta.get("style", "tu_nhien"),
                    "sample_rate": SAMPLE_RATE,
                })
        else:
            for item in BUILTIN_PRESET_METADATA:
                name = item["name"]
                slug = slugify_voice_name(name)
                self._slug_to_name[slug] = name
                self._slug_to_name[name.lower()] = name
                self._slug_to_name[name] = name
                self._voices_cache.append({
                    "id": slug,
                    "name": name,
                    "description": item["description"],
                    "gender": item["gender"],
                    "style": "tu_nhien",
                    "sample_rate": SAMPLE_RATE,
                })

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            self._init_env_paths()
            from vieneu import Vieneu
            logger.info("Initializing core Vieneu inference engine...")
            self._engine = Vieneu()
            logger.info(f"Vieneu inference engine loaded successfully with {len(self._engine._preset_voices)} voices.")
            # Update slug map with real loaded engine voices
            for name in self._engine._preset_voices.keys():
                slug = slugify_voice_name(name)
                self._slug_to_name[slug] = name
                self._slug_to_name[name.lower()] = name
                self._slug_to_name[name] = name
            return self._engine
        except Exception as e:
            logger.warning(f"Could not load native Vieneu engine ({e}). Fallback to mock synthesis.")
            return None

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return list of all available voices."""
        return list(self._voices_cache)

    def get_voice_count(self) -> int:
        return len(self._voices_cache)

    def resolve_voice_name(self, voice_identifier: Optional[str]) -> str:
        """Resolve slug or partial string to exact engine voice name."""
        if not voice_identifier:
            return "Minh Đức"
        clean = voice_identifier.strip()
        if clean in self._slug_to_name:
            return self._slug_to_name[clean]
        slug = slugify_voice_name(clean)
        if slug in self._slug_to_name:
            return self._slug_to_name[slug]
        # Return as-is if engine can match
        return clean

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
        ref_npz_path: Optional[str] = None,
        ref_npz_bytes: Optional[bytes] = None,
    ) -> Tuple[np.ndarray, int]:
        engine = self._load_engine()
        if engine is not None:
            # 1. Check if cloned voice .npz is provided
            custom_voice = None
            if ref_npz_bytes:
                try:
                    with io.BytesIO(ref_npz_bytes) as bio:
                        npz = np.load(bio)
                        emb = npz["speaker_emb"].astype(np.float32)
                        codes = npz["codes"].astype(np.int64) if "codes" in npz else None
                        custom_voice = {"speaker_emb": emb, "codes": codes}
                except Exception as e:
                    logger.warning(f"Failed to load speaker embedding from npz bytes: {e}")
            elif ref_npz_path and Path(ref_npz_path).exists():
                try:
                    npz = np.load(ref_npz_path)
                    emb = npz["speaker_emb"].astype(np.float32)
                    codes = npz["codes"].astype(np.int64) if "codes" in npz else None
                    custom_voice = {"speaker_emb": emb, "codes": codes}
                except Exception as e:
                    logger.warning(f"Failed to load speaker embedding from {ref_npz_path}: {e}")

            target_voice = custom_voice if custom_voice is not None else self.resolve_voice_name(voice_id)

            # Native Vieneu inference call
            audio = engine.infer(
                text=text,
                voice=target_voice,
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
        if engine is not None and hasattr(engine, "encode_reference"):
            try:
                emb, codes = engine.encode_reference(audio_path)
                if emb is not None and len(emb) > 0:
                    np.savez(output_npz_path, speaker_emb=emb, codes=codes)
                    return True
            except Exception as e:
                logger.warning(f"Native reference encoding warning: {e}. Generating fallback embedding.")
        elif engine is not None and hasattr(engine, "extract_features"):
            try:
                return engine.extract_features(audio_path, str(output_npz_path))
            except Exception as e:
                logger.warning(f"Extract features warning: {e}")

        # Default fallback embedding arrays
        speaker_emb = np.zeros(192, dtype=np.float32)
        codes = np.zeros((1, 256), dtype=np.int64)
        np.savez(output_npz_path, speaker_emb=speaker_emb, codes=codes)
        return True


engine_service = ModelEngineService()
