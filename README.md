# vienueTTS_server

Stateless AI text-to-speech inference and voice embedding microservice powered by **VieNeu-TTS v3 Turbo/Nano** (Vietnamese and English, 48kHz WAV audio standard).

---

## 1. Overview & Architecture

`vienueTTS_server` is designed as a lightweight, stateless microservice dedicated solely to AI inference tasks:
- **TTS Synthesis (`POST /internal/v1/synthesize`)**: Synthesizes speech from text at 48kHz sampling rate with configurable speed, pitch, and emotion levels.
- **Voice Embedding Extraction (`POST /internal/v1/embed`)**: Extracts speaker embedding features (`.npz`) from reference audio files for instant voice cloning.
- **Health & Metrics Probes (`GET /internal/v1/health`, `GET /internal/v1/metrics`)**: Exposes engine readiness, active jobs, and inference latency telemetry.
- **Security**: Authenticates requests using a shared API Key via header `X-API-Key`.

This microservice acts as an upstream worker node for [VoiceClonerAppServer](https://github.com/NgoMinhToan/VoiceClonerAppServer).

---

## 2. Configuration & Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

| Biến | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `PORT` | `8001` | Cổng HTTP lắng nghe của microservice |
| `HOST` | `0.0.0.0` | Địa chỉ mạng lắng nghe |
| `API_KEY` | `""` | API Key bảo vệ endpoint. Yêu cầu Header `X-API-Key` khi gọi |
| `VIENEU_MODE` | `v3turbo` | Chế độ engine (`v3turbo` hoặc `v3nano`) |
| `VIENEU_PRECISION` | `fp32` | Định dạng tensor (`fp32` hoặc `int8`) |
| `VIENEU_THREADS` | `8` | Số luồng tính toán CPU ONNX Runtime |
| `VIENEU_MODELS_DIR` | `/app/data/models` | Thư mục chứa trọng số mô hình VieNeu-TTS |
| `VIENEU_SRC_DIR` | `/app/vieneu_src` | Đường dẫn mã nguồn VieNeu core |

---

## 3. Chạy Server cục bộ

### Chạy trực tiếp với Python:
```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Khởi chạy server
python main.py
```

### Chạy với Docker:
```bash
# Build image
docker build -t vienuetts_server:latest .

# Run container
docker run -d \
  -p 8001:8001 \
  -e PORT=8001 \
  -e API_KEY=vtts_worker_secret_key_123 \
  -v /path/to/models:/app/data/models:ro \
  --name vienuetts-worker \
  vienuetts_server:latest
```

---

## 4. API Endpoints

### 4.1 Health Check (Mở, không cần API Key)
- **URL**: `GET /internal/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "vienueTTS_server",
  "model_loaded": true,
  "engine_mode": "v3turbo",
  "precision": "fp32",
  "device": "cpu",
  "threads": 8,
  "active_jobs": 0
}
```

### 4.2 Tổng Hợp Âm Thanh (Yêu cầu `X-API-Key`)
- **URL**: `POST /internal/v1/synthesize`
- **Headers**:
  - `X-API-Key: vtts_worker_secret_key_123`
  - `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "Xin chào, đây là giọng đọc thử nghiệm.",
  "voice": "minh_duc",
  "speed": 1.0,
  "pitch": 0,
  "emotion": 0,
  "format": "wav"
}
```
- **Response**: Binary 48kHz WAV audio (`Content-Type: audio/wav`).

### 4.3 Trích Xuất Embedding Giọng Nói (Yêu cầu `X-API-Key`)
- **URL**: `POST /internal/v1/embed`
- **Headers**: `X-API-Key: vtts_worker_secret_key_123`
- **Body**: `multipart/form-data` chứa file `audio_file` (WAV/M4A/MP3).
- **Response**: Binary `.npz` chứa `speaker_emb` và `codes`.

---

## 5. CI/CD & GitHub Container Registry (GHCR)

Dự án tích hợp GitHub Actions workflow tự động:
- Khi push vào branch `master` hoặc `main`, workflow sẽ đếm số lượng commit `v{count}`.
- Tự động build và push image lên GHCR:
  - `ghcr.io/ngominhtoan/vienuetts_server:latest`
  - `ghcr.io/ngominhtoan/vienuetts_server:v{commit_count}`
