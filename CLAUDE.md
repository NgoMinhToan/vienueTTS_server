# CLAUDE.md - vienueTTS_server

Guidance for Claude Code and AI agents working on `vienueTTS_server`.

## Project Overview
`vienueTTS_server` is a standalone, stateless AI text-to-speech inference microservice powered by VieNeu-TTS v3 Turbo/Nano (Vietnamese & English, 48kHz WAV audio standard).

- **Port**: 8001
- **Authentication**: `X-API-Key` header matching env `API_KEY`.
- **Core Endpoints**:
  - `GET /health` and `GET /internal/v1/health`: Worker status and total voice count.
  - `GET /internal/v1/voices` (or `/v1/voices`): Exposes dynamic voice catalog.
  - `POST /internal/v1/synthesize`: 48kHz WAV speech synthesis. Supports preset voices and cloned voices via `ref_npz_base64` or `ref_audio_path`.
  - `POST /internal/v1/embed`: Extracts speaker embedding features (`.npz`) from uploaded audio.
  - `GET /internal/v1/metrics`: Latency and active jobs telemetry.

---

## Mandatory Developer & AI Agent Rules

### 0. Initial Repository Setup
When opening or cloning this repository, ensure pre-push verification hooks are active:
```bash
git config core.hooksPath .githooks
```

### 1. Test Verification Before Push
**Rule:** You MUST run and pass all unit and boundary tests before making any git commit or push:
```bash
pytest tests/ -v
```
Never push code that breaks any test case.

### 2. Documentation Synchronization
**Rule:** Whenever modifying API endpoints, model parameters, or inference behavior, you MUST update:
- `PLAN.md`: Keep architectural plan and test checklists current.
- `README.md`: Keep API contract and configuration guides current.

### 3. Git Branching
- Always develop on a feature branch (`refactor/...` or `feature/...`).
- Never push directly to `main` or `master` without explicit user confirmation.

---

## Testing Commands
```bash
# Run full pytest suite
pytest tests/ -v

# Run boundary & exception tests
pytest tests/test_boundary_and_exceptions.py -v

# Run auth & synthesis tests
pytest tests/test_inference_auth.py -v
```
