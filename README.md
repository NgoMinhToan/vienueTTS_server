# vienueTTS_server

Stateless AI inference worker for the Voice Cloner platform. It runs VieNeu-TTS v3 Turbo through ONNX Runtime on CPU (PyTorch/CUDA in the `-cuda` image) and returns **48 kHz WAV** audio for Vietnamese and English text.

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
| `VIENEU_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`. `auto` uses CUDA only when the image has CUDA torch and a GPU is available; explicit `cuda` fails boot if unavailable |
| `VIENEU_PRECISION` | `fp32` | CPU/ONNX only: `fp32` or `int8`; ignored on CUDA |
| `VIENEU_THREADS` | `8` | CPU/ONNX only: ONNX Runtime intra-op threads; ignored on CUDA |
| `VIENEU_MAX_CONCURRENCY` | `0` | `0` = unlimited; the CUDA image defaults to `1` |
| `VIENEU_MODELS_DIR` | `/app/data/models` | Checkpoint cache directory |
| `HF_HOME` | `/app/data/models` | Hugging Face cache root |
| `HF_HUB_OFFLINE` | `0` | `1` forbids downloads; boot fails with 503 if weights are missing |
| `SKIP_MODEL_PRELOAD` | `0` | `1` skips eager preload at boot |
| `ALLOW_MOCK_SYNTHESIS` | `0` | `1` enables the 220 Hz mock tone (development only) |

Model directory resolution order: `HF_HOME` → `VIENEU_MODELS_DIR` → `/app/data/models` → `./models` → `./data/models` → Hugging Face cache.

## Model Weights

**Automatic download (default).** On first boot the worker downloads its runtime artifacts from Hugging Face (ONNX graphs on CPU, PyTorch weights on CUDA) plus the audio tokenizer. Mount `./models` to persist them:

```bash
docker run -d \
  -p 8001:8001 \
  -e API_KEY=vtts_worker_secret_key_123 \
  -v /path/to/models:/app/data/models \
  --name vienuetts-worker \
  ghcr.io/ngominhtoan/vienuetts_server:latest
```

**Offline.** Pre-populate the mounted directory and set `HF_HUB_OFFLINE=1`. CPU and CUDA warm different artifacts: CPU caches the ONNX graphs, while CUDA caches the PyTorch `model.safetensors` and MOSS tokenizer. Warm each variant with its own volume, or warm one shared volume with both variants, before going offline.

**Versioned builds.** The automatic workflow builds the `vieneu` version in `requirements.txt` and publishes CPU `:latest` plus `:v<commit-count>`. The manual workflow `docker-publish-vieneu.yml` accepts one of `3.7.1`, `3.8.0`, `3.8.1`, `3.8.2`, `3.8.3`, plus an `image_variant` choice of `cpu`, `gpu`, or `both`.

| Variant | Dockerfile | Platforms | Tags |
| :--- | :--- | :--- | :--- |
| CPU | `Dockerfile` | `linux/amd64,linux/arm64` | `vieneu-<version>`, `vieneu-<version>-v<count>` |
| CUDA | `Dockerfile.gpu` | `linux/amd64` | `vieneu-<version>-cuda`, `vieneu-<version>-cuda-v<count>` |

Unsuffixed tags always mean CPU, so existing deployments keep working. The CUDA image is amd64-only and requires an NVIDIA driver compatible with CUDA 12.8 plus `nvidia-container-toolkit`.

Each version has its own preset catalog: 3.7.1 ships 23 voices, while 3.8.x ships 25 with renamed entries. Run one model volume per container when mixing generations.

## Running

Root stack:

```bash
docker compose up -d --build model-service
```

Standalone CPU image (from this directory):

```bash
docker build -t vienuetts-worker:cpu .
docker run -p 8001:8001 --env-file .env vienuetts-worker:cpu
```

Standalone CUDA image (linux/amd64 host with NVIDIA driver + `nvidia-container-toolkit`):

```bash
docker build -f Dockerfile.gpu -t vienuetts-worker:cuda .
docker run --gpus all -p 8001:8001 --env-file .env \
  -e VIENEU_MAX_CONCURRENCY=1 \
  vienuetts-worker:cuda
```

An env file copied from `.env.example` sets `VIENEU_MAX_CONCURRENCY=0`, which overrides the CUDA image's built-in `1`; the explicit `-e` above keeps GPU inference serialized.

A GPU mounted into the CPU image is ignored: that image has no CUDA torch. Use the `-cuda` image or the image built from `Dockerfile.gpu` for GPU inference. Apple Silicon MPS, Intel XPU/iGPU, and AMD ROCm are not supported by this worker; the `linux/arm64` image runs on CPU.

## Testing

```bash
pytest tests/ -v
```

The suite covers auth, health, voice catalog and lookup, preset and `.npz` synthesis, boundary values, path traversal, invalid extensions, bad Base64, missing reference paths, device resolution, 48 kHz normalization, and unready-engine responses.

Enable the pre-push hook once:

```bash
git config core.hooksPath .githooks
```
