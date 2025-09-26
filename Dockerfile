# Seed-VC V1/V2 - Python 3.13 CPU (cross-platform)
# Base: Debian slim with Python 3.13
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_CACHE=/app/checkpoints/hf_cache

WORKDIR /app

# System dependencies
# - ffmpeg: audio conversion
# - libsndfile1: soundfile backend
# - libsoxr-dev: high-quality resampling backend used by librosa/soxr
# - git, build-essential: build some wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libsoxr-dev git build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies for Python 3.13
# We reuse the repo's cross-platform CPU requirements for 3.13
# Note: requirements-py313.txt uses PyTorch nightly CPU index to support Python 3.13
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements-py313.txt

# Default command prints help; override in `docker run` to execute inference
CMD ["python", "-c", "print('Seed-VC container ready. Override CMD to run inference.')"]
