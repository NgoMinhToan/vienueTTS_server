# CLAUDE.md - vienueTTS_server

Guidance for AI agents working in the inference worker repository.

## Architecture

- `main.py` — FastAPI app, lifespan preload, PNA middleware, mounts the router at `/internal/v1` and `/v1`.
- `api/inference.py` — synthesis, embedding extraction, voice catalog, health, metrics.
- `core/engine.py` — model path resolution, ONNX inference, voice catalog, 48 kHz WAV encoding.
- `core/security.py` — `X-API-Key` verification against `API_KEY`.
- `models/schemas.py` — request/response contracts and validation bounds.
- `tests/` — auth, boundary, and security suite.

## Commands

```bash
pytest tests/ -v
pytest tests/test_inference_auth.py -v
pytest tests/test_boundary_and_exceptions.py -v
docker build -t vienuetts-worker .
```

On Windows, run tests with the repository virtual environment if `python` is not on PATH: `..\.venv\Scripts\python.exe -m pytest tests/ -v`.

## Invariants

- Output is always 48 kHz WAV unless `format=pcm` is requested; include `X-Sample-Rate`.
- The service is stateless. No database, user accounts, or persisted jobs.
- Preset and cloned voices share the same endpoint. Cloned voices use `ref_npz_base64` (preferred), `ref_npz_path`, or worker-local `ref_audio_path`.
- A missing `ref_audio_path` is a 400 input error. Engine failures are 503.
- Voice catalogs are version-dependent; never hardcode names or counts.
- Keep path traversal and extension checks before any filesystem access.

## Repository Rules

- Run `pytest tests/ -v` before committing; never push a failing suite.
- Update `README.md` when endpoints, env vars, or model management change; update `CONTEXT.md` for architecture or constraints.
- Activate hooks once per clone: `git config core.hooksPath .githooks`.
- Develop on a feature branch; do not push directly to `master` without confirmation.
