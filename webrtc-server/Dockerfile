# Build context: repo root (so we can install sam_audio + webrtc_server in one image)
#
# Why Ubuntu 24.04 + CUDA 12.4:
#   - Ubuntu 22.04 ships FFmpeg 4 (libavutil.so.56); current torchcodec only
#     supports FFmpeg 5-7, so it fails to load on 22.04.
#   - Ubuntu 24.04 ships FFmpeg 6 (libavutil.so.58) — supported by torchcodec.
#   - nvidia/cuda:12.1 is deprecated; 12.4 is the current stable release.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────
# Ubuntu 24.04: python3 = 3.12, ffmpeg = 6.x (libavutil.so.58).
# PEP 668 blocks system pip installs; --break-system-packages is correct
# and safe inside an isolated Docker container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip curl git ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# ── PyTorch + CUDA-dependent packages ──────────────────────────────────────
# torchcodec MUST come from the same cu124 index as torch — the PyPI binary
# is built against a different ABI and will fail with "undefined symbol".
RUN pip install --no-cache-dir \
    torch torchaudio torchvision torchcodec \
    --index-url https://download.pytorch.org/whl/cu124

# ── sam_audio package (repo root) ──────────────────────────────────────────
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    .

# ── webrtc_server package ──────────────────────────────────────────────────
COPY webrtc-server/ ./webrtc-server/
RUN pip install --no-cache-dir ./webrtc-server/

# ── Runtime config ─────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache
EXPOSE 8000

# --workers 1 is required: the GPU model is not safe across multiple processes.
CMD ["uvicorn", "webrtc_server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
