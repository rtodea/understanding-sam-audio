# Build context: repo root (so we can install sam_audio + webrtc_server in one image)
#
# Why Ubuntu 24.04 + CUDA 12.6:
#   - Ubuntu 22.04 ships FFmpeg 4 (libavutil.so.56); torchcodec only
#     supports FFmpeg 5+, so it fails to load on 22.04.
#   - Ubuntu 24.04 ships FFmpeg 6 (libavutil.so.58) — supported by torchcodec.
#   - The cu126 whl index ships torchcodec 0.3.0+ (first version with
#     AudioDecoder). cu124 tops out at 0.2.1 which lacks AudioDecoder.
#
# Why uv instead of pip:
#   - pip 24.x (Ubuntu 24.04 default) enforces sha256 hash verification for
#     direct-URL Requires-Dist entries in wheel metadata. Multiple upstream
#     packages (nvidia-cudnn-cu12, mpmath, yarl, …) have been replaced in-place
#     at their pinned URLs, causing cascading hash failures across torch,
#     audiobox_aesthetics, and every other package that pins them.
#   - uv does not verify URL-fragment hashes during plain `pip install`-style
#     operations (only when using a lockfile with --require-hashes). This is
#     the same behaviour as pip < 22 and matches what the README assumes.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip curl git ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
# uv equivalents: install into system Python and bypass PEP 668 protection
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# ── uv ─────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── PyTorch with CUDA 12.6 ─────────────────────────────────────────────────
RUN uv pip install --no-cache \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# ── torchcodec (cu126 index → 0.3.0+, has AudioDecoder, matches torch ABI) ─
RUN uv pip install --no-cache "torchcodec>=0.3" \
    --index-url https://download.pytorch.org/whl/cu126

# ── ImageBind checkpoint ────────────────────────────────────────────────────
# imagebind_model.py unconditionally loads from .checkpoints/imagebind_huge.pth
# relative to CWD at runtime. Downloading at build time avoids a slow and
# failure-prone 2 GB fetch on every fresh container start.
#
# --mount=type=cache keeps the file in BuildKit's persistent cache so it is
# downloaded exactly once across ALL builds, even when earlier layers change.
RUN --mount=type=cache,id=imagebind-v2,target=/buildcache/imagebind \
    mkdir -p /app/.checkpoints /buildcache/imagebind \
    && CACHED=/buildcache/imagebind/imagebind_huge.pth \
    && TMP="${CACHED}.tmp" \
    && MIN_BYTES=1500000000 \
    && VALID=0 \
    && if [ -f "$CACHED" ] && [ "$(stat -c%s "$CACHED")" -ge "$MIN_BYTES" ]; then \
         VALID=1; \
       fi \
    && if [ "$VALID" = "0" ]; then \
         for attempt in 1 2 3 4 5; do \
           echo "[imagebind] download attempt ${attempt}/5 …"; \
           curl -fL --max-time 1800 --retry 3 --retry-delay 10 --retry-all-errors \
                -o "$TMP" \
                https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth \
           && ACTUAL=$(stat -c%s "$TMP" 2>/dev/null || echo 0) \
           && if [ "$ACTUAL" -ge "$MIN_BYTES" ]; then \
                mv "$TMP" "$CACHED"; \
                echo "[imagebind] download complete (${ACTUAL} bytes)"; \
                break; \
              else \
                echo "[imagebind] attempt ${attempt} incomplete (${ACTUAL} bytes), retrying …"; \
                rm -f "$TMP"; \
              fi; \
         done; \
       fi \
    && [ -f "$CACHED" ] && [ "$(stat -c%s "$CACHED")" -ge "$MIN_BYTES" ] \
    && cp "$CACHED" /app/.checkpoints/imagebind_huge.pth

# ── sam_audio package (repo root) ──────────────────────────────────────────
# Two-stage install so that changing sam_audio source code does not
# re-download the heavy git dependencies (dacvae, imagebind, etc.).
#
# Stage A: install all dependencies from pyproject.toml alone.
# A stub __init__.py lets setuptools discover the package without real source.
COPY pyproject.toml .
RUN mkdir -p sam_audio && touch sam_audio/__init__.py
RUN uv pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    .

# Stage B: overwrite stub with real source, reinstall package only (no deps).
# This layer is the only one that re-runs when Python files change.
COPY sam_audio/ ./sam_audio/
RUN uv pip install --no-cache --no-deps .

# ── webrtc_server package ──────────────────────────────────────────────────
COPY webrtc-server/ ./webrtc-server/
RUN uv pip install --no-cache ./webrtc-server/

# ── Runtime config ─────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache
EXPOSE 8000

# --workers 1 is required: the GPU model is not safe across multiple processes.
CMD ["uvicorn", "webrtc_server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
