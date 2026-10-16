# vienueTTS_server

Stateless AI text-to-speech inference and voice embedding microservice powered by **VieNeu-TTS v3 Turbo/Nano** (Vietnamese and English, standard 48kHz WAV audio output).

---

## 1. Overview & Architecture

`vienueTTS_server` is designed as a lightweight, stateless microservice dedicated solely to AI inference tasks:
- **TTS Synthesis (`POST /internal/v1/synthesize`)**: Synthesizes speech from text at 48kHz sampling rate. Supports 23 built-in preset voices and instant cloned voices (via `.npz` speaker embedding or audio reference).
- **Preset Voices Catalog (`GET /internal/v1/voices`)**: Exposes list of 23 built-in voices with descriptions, regional accents (Bắc, Trung, Nam), genders, and metadata.
- **Voice Embedding Extraction (`POST /internal/v1/embed`)**: Extracts speaker embedding features (`.npz` with 192-d embedding and reference codes) from reference audio files for voice cloning.
- **Health & Metrics Probes (`GET /internal/v1/health`, `GET /internal/v1/metrics`)**: Exposes engine readiness, total voices, active jobs, and inference latency telemetry.
- **Security**: Authenticates requests using a shared API Key via header `X-API-Key`.
- **Private Network Access (PNA)**: Built-in middleware supporting secure HTTPS web clients (e.g., GitHub Pages) connecting to local Voice Server instances.

This microservice acts as an upstream worker node for [VoiceClonerAppServer](https://github.com/NgoMinhToan/VoiceClonerAppServer) and can be accessed directly by testing clients like [VoiceClonerWebDemo](https://github.com/NgoMinhToan/VoiceClonerWebDemo).

---

## 2. Configuration & Environment Variables

Copy `.env.example` to `.env` or set environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8001` | HTTP port listened by the microservice |
| `HOST` | `0.0.0.0` | Network interface to bind |
| `API_KEY` | `""` | Worker API key. If set, incoming requests must supply header `X-API-Key` |
| `VIENEU_MODE` | `v3turbo` | Engine mode: `v3turbo` (48kHz HQ, default) or `v3nano` (24kHz lightweight) |
| `VIENEU_PRECISION` | `fp32` | Tensor precision: `fp32` (maximum quality) or `int8` (accelerated) |
| `VIENEU_THREADS` | `8` | Number of intra-op threads for ONNX Runtime inference |
| `VIENEU_MODELS_DIR` | `/app/data/models` | Local directory for model checkpoint cache |
| `HF_HOME` | `/app/data/models` | Hugging Face cache root directory |
| `HF_HUB_OFFLINE` | `0` | Set to `1` only to strictly block internet access and force local cache |

---

## 3. Model Weights: Auto-Download vs Volume Mount

`vienueTTS_server` is fully self-sufficient and does not require local host source code mounts:

### Option A: Automatic Download (Zero-Setup, Recommended for VPS)
If you start the server on a fresh VPS without any pre-existing models:
1. The server detects that the cache directory is empty.
2. It automatically downloads the ONNX graphs and codec from Hugging Face:
   - `pnnbao-ump/VieNeu-TTS-v3-Turbo` (subfolder `onnx_update`)
   - `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`
3. If volume `-v ./models:/app/data/models` is mounted, the downloaded files persist on the host disk so subsequent container restarts will not re-download.

### Option B: Offline / Pre-downloaded Models
If you already have downloaded model checkpoints on your host disk:
```bash
docker run -d \
  -p 8001:8001 \
  -e PORT=8001 \
  -e API_KEY=vtts_worker_secret_key_123 \
  -v /path/to/models:/app/data/models \
  --name vienuetts-worker \
  ghcr.io/ngominhtoan/vienuetts_server:latest
```

---

## 4. API Endpoints Specification

### 4.1 Health Check (Open probe, no key required)
- **Endpoint**: `GET /health` or `GET /internal/v1/health`
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
  "active_jobs": 0,
  "total_voices": 23
}
```

### 4.2 List Preset Voices (Requires `X-API-Key`)
- **Endpoint**: `GET /internal/v1/voices` (or `/v1/voices`)
- **Headers**: `X-API-Key: <API_KEY>`
- **Response**:
```json
{
  "total_count": 23,
  "default_voice": "Minh Đức",
  "voices": [
    {
      "id": "minh_duc",
      "name": "Minh Đức",
      "description": "Nam · Bắc · Phong cách tin tức",
      "gender": "male",
      "style": "tu_nhien",
      "sample_rate": 48000
    },
    {
      "id": "quynh_anh",
      "name": "Quỳnh Anh",
      "description": "Nữ · Bắc · Phong cách đọc truyện",
      "gender": "female",
      "style": "tu_nhien",
      "sample_rate": 48000
    }
  ]
}
```

### 4.3 Synthesize Speech (Requires `X-API-Key`)
- **Endpoint**: `POST /internal/v1/synthesize` (or `/v1/synthesize`)
- **Headers**:
  - `X-API-Key: <API_KEY>`
  - `Content-Type: application/json`
- **Request Body (Preset Voice)**:
```json
{
  "text": "Xin chào, đây là giọng đọc chuẩn AI VieNeu-TTS 48kHz.",
  "voice": "minh_duc",
  "speed": 1.0,
  "pitch": 0,
  "emotion": 0,
  "denoise": true,
  "smart_pause": true,
  "format": "wav"
}
```
- **Request Body (Cloned Voice with `.npz` Base64)**:
```json
{
  "text": "Xin chào, đây là giọng đọc nhân bản từ file embedding .npz.",
  "ref_npz_base64": "<base64_encoded_npz_data>",
  "speed": 1.0,
  "format": "wav"
}
```
- **Response**: Binary 48kHz WAV audio (`Content-Type: audio/wav`) with headers `X-Sample-Rate: 48000` and `X-Execution-Time-Ms`.

### 4.4 Extract Speaker Embedding (Requires `X-API-Key`)
- **Endpoint**: `POST /internal/v1/embed` (or `/v1/embed`)
- **Headers**: `X-API-Key: <API_KEY>`
- **Body**: `multipart/form-data` with `audio_file` (WAV/M4A/MP3, 3 - 15 seconds clip).
- **Response**: Binary `application/octet-stream` representing NumPy `.npz` file containing `speaker_emb` (192-d) and `codes`.

### 4.5 Metrics Telemetry (Requires `X-API-Key`)
- **Endpoint**: `GET /internal/v1/metrics`
- **Headers**: `X-API-Key: <API_KEY>`
- **Response**:
```json
{
  "total_requests": 142,
  "active_jobs": 0,
  "avg_latency_ms": 3840.5
}
```

---

## 5. Testing & Verification

Run the comprehensive pytest suite:
```bash
pytest tests/ -v
```
All 21 test cases verify health probes, voice catalog, synthesis with preset and cloned `.npz`, boundary value constraints, path traversal defense, and speaker embedding extraction.

### Enable Local Git Pre-Push Hook:
To automatically run tests and block pushing if any test fails:
```bash
git config core.hooksPath .githooks
```
