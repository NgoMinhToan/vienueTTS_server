# vienueTTS_server

Stateless AI inference worker for the Voice Cloner platform. It runs VieNeu-TTS v3 Turbo/Nano through ONNX Runtime and returns **48 kHz WAV** audio for Vietnamese and English text.

## Endpoints

Both the internal and client prefixes are mounted:

| Method | Path | Auth | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/health`, `/internal/v1/health` | none | Liveness, engine mode, voice count, active jobs |
| `GET` | `/internal/v1/voices`, `/v1/voices` | `X-API-Key` | Dynamic preset voice catalog |
| `GET` | `/internal/v1/voices/{id}` | `X-API-Key` | One preset by id or name |
| `POST` | `/internal/v1/synthesize`, `/v1/synthesize` | `X-API-Key` | Text → 48 kHz WAV (preset or cloned voice) |
| `POST` | `/internal/v1/embed`, `/v1/embed` | `X-API-Key` | Reference audio → `.npz` speaker embedding |
| `GET` | `/internal/v1/metrics` | `X-API-Key` | Request count, active jobs, average latency |

### Synthesize request

```json
{
  "text": "Xin chào, đây là giọng đọc 48kHz.",
  "voice": "minh_duc",
  "speed": 1.0,
  "pitch": 0,
  "emotion": 0,
  "denoise": true,
  "smart_pause": true,
  "format": "wav"
}
```

Cloned voices use `ref_npz_base64` (preferred) or `ref_npz_path`. `ref_audio_path` is resolved on the worker's own filesystem only.

Response: binary `audio/wav` with `X-Sample-Rate: 48000` and `X-Execution-Time-Ms`.

Error contract: `400` for invalid input, including a missing `ref_audio_path`; `503` when the engine is not ready or inference fails. Gateways should treat 4xx as input errors and 5xx as worker faults.

## Configuration

Copy `.env.example` to `.env` or pass environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8001` | HTTP port |
| `HOST` | `0.0.0.0` | Bind address |
| `API_KEY` | `""` | Worker key; when set, all protected routes require `X-API-Key` |
| `VIENEU_MODE` | `v3turbo` | `v3turbo` (48 kHz HQ) or `v3nano` (lightweight) |
| `VIENEU_PRECISION` | `fp32` | `fp32` or `int8` |
| `VIENEU_THREADS` | `8` | ONNX Runtime intra-op threads |
| `VIENEU_MODELS_DIR` | `/app/data/models` | Checkpoint cache directory |
| `HF_HOME` | `/app/data/models` | Hugging Face cache root |
| `HF_HUB_OFFLINE` | `0` | `1` forbids downloads; boot fails with 503 if weights are missing |
| `SKIP_MODEL_PRELOAD` | `0` | `1` skips eager preload at boot |
| `ALLOW_MOCK_SYNTHESIS` | `0` | `1` enables the 220 Hz mock tone (development only) |

Model directory resolution order: `HF_HOME` → `VIENEU_MODELS_DIR` → `/app/data/models` → `./models` → `./data/models` → Hugging Face cache.

## Model Weights

**Automatic download (default).** On first boot the worker downloads the ONNX graphs and audio tokenizer from Hugging Face. Mount `./models` to persist them:

```bash
docker run -d \
  -p 8001:8001 \
  -e API_KEY=vtts_worker_secret_key_123 \
  -v /path/to/models:/app/data/models \
  --name vienuetts-worker \
  ghcr.io/ngominhtoan/vienuetts_server:latest
```

**Offline.** Pre-populate the mounted directory and set `HF_HUB_OFFLINE=1`.

**Versioned builds.** The automatic workflow builds the `vieneu` version in `requirements.txt` and publishes `:latest` plus `:v<commit-count>`. The manual workflow `docker-publish-vieneu.yml` accepts one of `3.7.1`, `3.8.0`, `3.8.1`, `3.8.2`, `3.8.3` and publishes `vieneu-<version>` and `vieneu-<version>-v<commit-count>` without overwriting `:latest`.

Each version has its own preset catalog: 3.7.1 ships 23 voices, while 3.8.x ships 25 with renamed entries. Run one model volume per container when mixing generations.

## Running

Root stack:

```bash
docker compose up -d --build model-service
```

Standalone (from this directory):

```bash
docker build -t vienuetts-worker .
docker run -p 8001:8001 --env-file .env vienuetts-worker
```

## Testing

```bash
pytest tests/ -v
```

The suite covers auth, health, voice catalog and lookup, preset and `.npz` synthesis, boundary values, path traversal, invalid extensions, bad Base64, missing reference paths, and unready-engine responses.

Enable the pre-push hook once:

```bash
git config core.hooksPath .githooks
```
