# Project Context: vienueTTS_server

Current state of the inference worker. Git history is the changelog.

## 1. Role

`vienueTTS_server` is a stateless FastAPI worker. It owns no users, database, or persistent state beyond the model cache. It exposes synthesis, speaker-embedding extraction, a dynamic voice catalog, health, and metrics.

- Port: `8001`
- Auth: `X-API-Key` compared against `API_KEY`; health probes are open.
- Audio contract: 48 kHz WAV (`X-Sample-Rate: 48000`), or PCM when requested.
- Both `/internal/v1/*` and `/v1/*` prefixes are mounted for gateway and direct-client compatibility.

## 2. Engine

- `core/engine.py` owns model resolution, the voice catalog, ONNX (CPU) / PyTorch (CUDA) inference, and WAV encoding.
- The engine is always `v3turbo` at 48 kHz. `VIENEU_MODE` is deprecated and ignored. `VIENEU_PRECISION` (`fp32`/`int8`) and `VIENEU_THREADS` apply only to CPU/ONNX.
- `VIENEU_DEVICE` selects `auto` (default), `cpu`, `cuda`, or `cuda:N`. `auto` falls back to CPU when CUDA torch is absent; explicit CUDA fails startup when unavailable.
- Eager preload runs in the FastAPI lifespan. `SKIP_MODEL_PRELOAD=1` defers loading for fast test boots.
- An unready engine returns 503. `ALLOW_MOCK_SYNTHESIS=1` produces a 220 Hz tone and must never be used in production.
- Model path resolution: `HF_HOME` → `VIENEU_MODELS_DIR` → `/app/data/models` → `./models` → `./data/models` → Hugging Face cache. `HF_HUB_DISABLE_SYMLINKS` is forced on so split `.onnx`/`.data` files stay co-located.
- `_dereference_model_symlinks()` self-heals symlinked weight files copied into the model directory.

## 3. Voice Catalog

- Loaded from the installed `vieneu` package's `voices_v3_turbo.json`; if unavailable, the built-in metadata list is used.
- Slugs and names are both accepted for lookup. The catalog is exposed dynamically; clients must not hardcode counts.
- Catalog size depends on the installed version: 3.7.1 → 23 voices; 3.8.x → 25 voices with renamed entries.

## 4. API Behavior

- Health reports the resolved runtime: `engine_mode`, `device`, `device_requested`, `runtime`, `sample_rate`, `device_error`, and `max_concurrency`.
- `POST /synthesize` accepts preset voices or cloned voices via `ref_npz_base64`, `ref_npz_path`, or `ref_audio_path`.
- `ref_audio_path` is worker-local. A missing file returns 400 (input error), not 503, so gateways do not penalize a healthy worker.
- `POST /embed` accepts multipart `audio_file` and returns a NumPy `.npz` with `speaker_emb` (192-d) and `codes`.
- Path traversal and forbidden extensions are rejected before file access.
- Text limit: 5,000 characters. Bounds: speed 0.5–2.0, pitch −12…+12, emotion 0–100.

## 5. Packaging and CI

- Multi-stage `Dockerfile` (CPU): builder compiles into `/opt/venv`; runtime keeps only `curl` and `libsndfile1`.
- `Dockerfile.gpu` (CUDA): base `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`, installs `vieneu[cuda]` (torch 2.8.0), amd64-only, sets `VIENEU_MAX_CONCURRENCY=1`.
- GUI/test bloat pulled by `vieneu` (`gradio`, `matplotlib`, etc.) is uninstalled during both builds.
- `docker-publish.yml`: automatic multi-arch CPU build from `requirements.txt`; tags `latest` and `v<commit-count>` with GitHub Actions layer cache.
- `docker-publish-vieneu.yml`: manual dispatch with a fixed version dropdown (`3.7.1`, `3.8.0`, `3.8.1`, `3.8.2`, `3.8.3`) and `image_variant` (`cpu`, `gpu`, `both`). CPU tags keep `vieneu-<version>[-v<count>]`; CUDA tags use `vieneu-<version>-cuda[-v<count>]` and never overwrite `latest`.
- Docker build arg `VIENEU_VERSION` rewrites only the `vieneu` line in `requirements.txt` (CPU) or selects `vieneu[cuda]==<version>` (CUDA).

## 6. Known Constraints

- CPU inference is the default. The optional CUDA image is amd64-only, multi-GB, and needs an NVIDIA driver compatible with CUDA 12.8 plus `nvidia-container-toolkit`. Apple MPS, Intel XPU/iGPU, and AMD ROCm are unsupported.
- A GPU mounted into the CPU image is ignored; only the `-cuda` image contains CUDA torch.
- Long CPU paragraphs are slow; the gateway uses a 300 s worker read timeout for this reason.
- Audio reference paths cannot cross hosts; use `.npz` payloads.
- Voice naming differs across `vieneu` generations; verify the catalog after changing the pinned version.
- The pre-push hook calls `../.venv/Scripts/python.exe` on Windows, falling back to `python3`/`python`; run the suite manually if the interpreter is not on PATH.

## 7. Verification

```bash
pytest tests/ -v
```

The suite covers auth, catalog, synthesis, embeddings, boundaries, security, device resolution, 48 kHz normalization, and unready-engine behavior.
