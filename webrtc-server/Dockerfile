# Build context: repo root (so we can install sam_audio + webrtc_server in one image)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────
# Ubuntu 22.04 disables ensurepip and python3-pip targets Python 3.10.
# Bootstrap pip for Python 3.11 via get-pip.py instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev curl git ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch + CUDA-dependent packages ─────────────────────────────────────
# torchcodec MUST come from the same cu121 index as torch — installing it
# from PyPI produces a binary built against a different torch ABI, causing
# "undefined symbol" errors at runtime.
RUN python3.11 -m pip install --no-cache-dir \
    torch torchaudio torchvision torchcodec \
    --index-url https://download.pytorch.org/whl/cu121

# ── sam_audio package (repo root) ─────────────────────────────────────────
# --extra-index-url ensures any remaining deps that are also on the cu121
# index (e.g. torchdiffeq) are resolved against the already-installed
# cu121 builds rather than being pulled from PyPI and overwriting them.
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/
RUN python3.11 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    .

# ── webrtc_server package ──────────────────────────────────────────────────
COPY webrtc-server/ ./webrtc-server/
RUN python3.11 -m pip install --no-cache-dir ./webrtc-server/

# ── Runtime config ─────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache
EXPOSE 8000

# --workers 1 is required: the GPU model is not safe across multiple processes.
CMD ["python3.11", "-m", "uvicorn", "webrtc_server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
