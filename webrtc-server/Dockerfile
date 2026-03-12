# Build context: repo root (so we can install sam_audio + webrtc_server in one image)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────
# Note: do NOT install python3-pip via apt — it targets Python 3.10 on Ubuntu 22.04.
# Bootstrap pip for Python 3.11 via ensurepip instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv git ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── PyTorch with CUDA 12.1 (large layer — install before local code) ───────
RUN python3.11 -m pip install --no-cache-dir \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── sam_audio package (repo root) ──────────────────────────────────────────
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/
RUN python3.11 -m pip install --no-cache-dir .

# ── webrtc_server package ──────────────────────────────────────────────────
COPY webrtc-server/ ./webrtc-server/
RUN python3.11 -m pip install --no-cache-dir ./webrtc-server/

# ── Runtime config ─────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache
EXPOSE 8000

# --workers 1 is required: the GPU model is not safe across multiple processes.
CMD ["python3.11", "-m", "uvicorn", "webrtc_server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
