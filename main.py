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

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from vienueTTS_server.api.inference import router as inference_router
from vienueTTS_server.core.engine import engine_service

logger = logging.getLogger("vienueTTS_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """
    Handle Chrome Private Network Access (PNA) preflight and response headers.
    Enables secure HTTPS Web Clients (e.g., GitHub Pages) to access local Voice Server.
    """
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" and (
            request.headers.get("access-control-request-private-network") == "true"
            or request.headers.get("Access-Control-Request-Private-Network") == "true"
        ):
            headers = {
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*"),
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Private-Network": "true",
            }
            return Response(status_code=200, headers=headers)

        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = int(os.getenv("PORT", "8001"))
    has_key = bool(os.getenv("API_KEY", "").strip())
    logger.info(f"Starting vienueTTS_server on port {port} (API_KEY protected: {has_key})...")
    logger.info(f"Engine mode: {engine_service.engine_mode}, Threads: {engine_service.threads}, Voices: {engine_service.get_voice_count()}")
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
app.add_middleware(PrivateNetworkAccessMiddleware)

# Expose under both /internal/v1 (internal cluster convention) and /v1 (standard client convention)
app.include_router(inference_router, prefix="/internal/v1")
app.include_router(inference_router, prefix="/v1")


@app.get("/health", tags=["System"])
async def root_health():
    return {
        "status": "healthy",
        "service": "vienueTTS_server",
        "total_voices": engine_service.get_voice_count(),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("vienueTTS_server.main:app", host=host, port=port, reload=False)
