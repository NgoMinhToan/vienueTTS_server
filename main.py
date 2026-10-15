import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also allow vienueTTS_server directory itself to be run as root
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vienueTTS_server.api.inference import router as inference_router
from vienueTTS_server.core.engine import engine_service

logger = logging.getLogger("vienueTTS_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = int(os.getenv("PORT", "8001"))
    has_key = bool(os.getenv("API_KEY", "").strip())
    logger.info(f"Starting vienueTTS_server on port {port} (API_KEY protected: {has_key})...")
    logger.info(f"Engine mode: {engine_service.engine_mode}, Threads: {engine_service.threads}")
    yield
    logger.info("Shutting down vienueTTS_server.")


app = FastAPI(
    title="vienueTTS_server",
    description="Stateless VieNeu-TTS AI Voice Inference Microservice",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router, prefix="/internal/v1")


@app.get("/health", tags=["System"])
async def root_health():
    return {"status": "healthy", "service": "vienueTTS_server"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("vienueTTS_server.main:app", host=host, port=port, reload=False)
