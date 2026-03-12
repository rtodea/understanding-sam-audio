# Build context: repo root (so we can install sam_audio + webrtc_server in one image)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3       /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch with CUDA 12.1 (large layer — install before local code) ───────
RUN pip install --no-cache-dir \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── sam_audio package (repo root) ──────────────────────────────────────────
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/
RUN pip install --no-cache-dir .

# ── webrtc_server package ──────────────────────────────────────────────────
COPY webrtc-server/ ./webrtc-server/
RUN pip install --no-cache-dir ./webrtc-server/

# ── Runtime config ─────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache
EXPOSE 8000

# --workers 1 is required: the GPU model is not safe across multiple processes.
CMD ["uvicorn", "webrtc_server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
