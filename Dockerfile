# Optimized Python 3.12 Image for vienueTTS_server AI inference
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for audio decoding and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service codebase
COPY . /app/vienueTTS_server/
WORKDIR /app/vienueTTS_server

# Default port
EXPOSE 8001

# Healthcheck probe
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/internal/v1/health || exit 1

# Start FastAPI server
CMD ["python", "main.py"]
