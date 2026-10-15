import os
import secrets
from fastapi import Header, HTTPException, status


def get_expected_api_key() -> str:
    return os.getenv("API_KEY", "").strip()


async def verify_worker_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    expected = get_expected_api_key()
    if not expected:
        # Open access when API_KEY is unset in development
        return
    if not x_api_key or not secrets.compare_digest(x_api_key.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing worker API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
