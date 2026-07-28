# =============================================================================
# TELOS-S Backend — Dockerfile
# Located in the root of the telos-s-backend repository (along with backend.py and requirements.txt)
# Python 3.11 slim · FastAPI · ESM-2 via HuggingFace · Biopython
# =============================================================================
#
# The ESM-2 model (2.5GB) is not included in the image.
# It is downloaded during the first analysis and remains in the huggingface_cache volume.
# This keeps the image size at approximately 1.5GB instead of ~4GB.
# =============================================================================
 
FROM python:3.11-slim
 
# Metadata
LABEL maintainer="Telos Genomics"
LABEL description="Telos-S API — Predictive Intelligence Engine for Protein Evolution"
LABEL version="0.1.1"
 
# Build variables
ARG DEBIAN_FRONTEND=noninteractive
 
# System dependencies required for Biopython and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# ---------------------------------------------------------------------------
# Python Dependencies
# We first copy the requirements file to take advantage of Docker's caching mechanism.
# ---------------------------------------------------------------------------
COPY requirements.txt .
 
RUN pip install --no-cache-dir -r requirements.txt
 
# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . .
 
# Starting point: If Wuhan doesn't exist, download it first, then start the server.
RUN chmod +x entrypoint.sh
 
# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TELOS_FORCE_CPU=true
ENV ESM_2_SIZE=facebook/esm2_t33_650M_UR50D
ENV BATCH_SIZE=16
 
# Exposed port
EXPOSE 6002
 
# ---------------------------------------------------------------------------
# Entrypoint
# Download NC_045512.2 and extract the spike if it doesn't exist, then start uvicorn
# ---------------------------------------------------------------------------
ENTRYPOINT ["./entrypoint.sh"]