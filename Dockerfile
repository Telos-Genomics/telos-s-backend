# =============================================================================
# TELOS-S Backend — Dockerfile
# Va en la RAÍZ del repo telos-s-backend (junto a backend.py y requirements.txt)
# Python 3.11 slim · FastAPI · ESM-2 via HuggingFace · Biopython
# =============================================================================
#
# El modelo ESM-2 (2.5GB) NO se incluye en la imagen.
# Se descarga en el primer análisis y queda en el volumen huggingface_cache.
# Esto mantiene la imagen en ~1.5GB en lugar de ~4GB.
# =============================================================================
 
FROM python:3.11-slim
 
# Metadata
LABEL maintainer="Telos Genomics"
LABEL description="Telos-S API — Predictive Intelligence Engine for Protein Evolution"
LABEL version="1.0.0"
 
# Variables de build
ARG DEBIAN_FRONTEND=noninteractive
 
# Dependencias del sistema necesarias para Biopython y Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# ---------------------------------------------------------------------------
# Dependencias Python
# Copiamos requirements primero para aprovechar el cache de Docker
# ---------------------------------------------------------------------------
COPY requirements.txt .
 
# Instalamos PyTorch CPU (compatible con cualquier arquitectura)
# En producción con GPU NVIDIA, cambiar a la versión con CUDA
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu
 
RUN pip install --no-cache-dir -r requirements.txt
 
# ---------------------------------------------------------------------------
# Código de la aplicación
# ---------------------------------------------------------------------------
COPY . .
 
# Entrypoint: descarga Wuhan si no existe, luego arranca el servidor
RUN chmod +x entrypoint.sh
 
# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TELOS_FORCE_CPU=true
 
# Puerto expuesto
EXPOSE 6002
 
# ---------------------------------------------------------------------------
# Entrypoint
# Descarga NC_045512.2 y extrae la Spike si no existe, luego arranca uvicorn
# ---------------------------------------------------------------------------
ENTRYPOINT ["./entrypoint.sh"]