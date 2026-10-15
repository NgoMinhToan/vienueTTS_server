from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Văn bản cần tổng hợp thành giọng nói")
    voice: Optional[str] = Field("minh_duc", description="Mã hoặc tên giọng đọc mẫu")
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
    ref_npz_path: Optional[str] = Field(None, description="Đường dẫn file .npz speaker embedding trên server")
    ref_npz_base64: Optional[str] = Field(None, description="Dữ liệu file .npz speaker embedding dạng Base64")

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


class VoiceItem(BaseModel):
    id: str = Field(..., description="Mã định danh rút gọn (slug) của giọng đọc")
    name: str = Field(..., description="Tên đầy đủ của giọng đọc")
    description: str = Field("", description="Mô tả đặc trưng giọng đọc")
    gender: str = Field("unknown", description="Giới tính: male/female")
    style: str = Field("tu_nhien", description="Phong cách thể hiện")
    sample_rate: int = Field(48000, description="Tần số lấy mẫu chuẩn (Hz)")


class VoiceListResponse(BaseModel):
    total_count: int = Field(..., description="Tổng số giọng đọc khả dụng")
    default_voice: str = Field("Minh Đức", description="Giọng đọc mặc định")
    voices: List[VoiceItem] = Field(default_factory=list, description="Danh sách chi tiết các giọng đọc")


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "vienueTTS_server"
    model_loaded: bool = False
    engine_mode: str = "v3turbo"
    precision: str = "fp32"
    device: str = "cpu"
    threads: int = 8
    active_jobs: int = 0
    total_voices: int = 23


class MetricsResponse(BaseModel):
    total_requests: int
    active_jobs: int
    avg_latency_ms: float
