# Plan: vienueTTS_server

Open work only. Git history and `CONTEXT.md` record what is already implemented.

## Current Focus

- Keep the worker compatible with the `vieneu` versions published by the manual workflow (`3.7.1`–`3.8.3`).
- Preserve the stable 48 kHz contract and the 4xx-input / 5xx-worker error split.

## Backlog (candidates, not committed)

- Add a version-matrix check that verifies voice-catalog parity for each published image.
- Benchmark `v3turbo` vs `v3nano` on CPU and document expected latency and memory.
- Harden embedding extraction failure reporting with structured error details.
- Evaluate streaming or chunked synthesis for long paragraphs (the current API is non-streaming).
- Add readiness/latency metrics suitable for gateway circuit-breaker decisions.

## Verification Checklist

```bash
pytest tests/ -v
docker build -t vienuetts-worker .
```

When changing the pinned `vieneu` version, verify the catalog size, renamed voices, and a full synthesize/embed round trip.
