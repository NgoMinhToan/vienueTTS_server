from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Văn bản cần tổng hợp thành giọng nói")
    voice: Optional[str] = Field("minh_duc", description="Mã giọng đọc mẫu")
    voice_id: Optional[str] = Field(None, description="Bí danh của voice")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Tốc độ nói (0.5x - 2.0x)")
    pitch: int = Field(0, ge=-12, le=12, description="Cao độ (-12 đến +12 semitones)")
    emotion: int = Field(0, ge=0, le=100, description="Mức độ cảm xúc (0-100%)")
    emotion_level: Optional[int] = Field(None, ge=0, le=100)
    denoise: bool = Field(True, description="Khử nhiễu âm thanh nền")
    smart_pause: bool = Field(True, description="Tự động ngắt nhịp câu tự nhiên")
    format: Optional[Literal["wav", "pcm"]] = Field("wav", description="Định dạng âm thanh đầu ra")
    response_format: Optional[Literal["wav", "pcm"]] = Field(None)
    ref_audio_path: Optional[str] = Field(None, description="Đường dẫn file audio mẫu để clone trực tiếp")

    @model_validator(mode="after")
    def sync_aliases(self):
        if not self.voice and self.voice_id:
            self.voice = self.voice_id
        elif not self.voice_id and self.voice:
            self.voice_id = self.voice

        if self.emotion_level is None:
            self.emotion_level = self.emotion
        else:
            self.emotion = self.emotion_level

        if self.response_format is None:
            self.response_format = self.format or "wav"
        else:
            self.format = self.response_format

        return self


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "vienueTTS_server"
    model_loaded: bool = False
    engine_mode: str = "v3turbo"
    precision: str = "fp32"
    device: str = "cpu"
    threads: int = 8
    active_jobs: int = 0


class MetricsResponse(BaseModel):
    total_requests: int
    active_jobs: int
    avg_latency_ms: float
