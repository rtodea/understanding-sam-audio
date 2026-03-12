"""
WebSocket handler — orchestration only, no business logic.

Receives:
  1. One JSON text frame:  {"description": "person speaking"}
  2. N binary frames:      WebM/Opus audio chunks from MediaRecorder

Sends:
  N binary frames: WAV audio chunks (separated target audio)
"""

import asyncio
import logging

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio_utils import crossfade, decode_webm_chunk, encode_wav_chunk
from .config import settings
from .model_registry import get_model
from .overlap_buffer import OverlapBuffer

router = APIRouter()
logger = logging.getLogger(__name__)


def _separate(
    chunk: torch.Tensor,
    description: str,
) -> torch.Tensor:
    """
    Run SAM-Audio separation synchronously (called via run_in_executor
    so it does not block the event loop).
    """
    model, processor = get_model()
    device = settings.effective_device
    batch = processor(audios=[chunk], descriptions=[description]).to(device)
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=settings.predict_spans,
            reranking_candidates=settings.sam_reranking_candidates,
        )
    return result.target.squeeze(0).cpu()


@router.websocket("/ws/separate")
async def websocket_separate(ws: WebSocket) -> None:
    await ws.accept()

    chunk_samples   = int(settings.chunk_seconds * settings.sample_rate)
    overlap_samples = int(settings.overlap_seconds * settings.sample_rate)
    buf             = OverlapBuffer(chunk_samples, overlap_samples)
    prev_output: torch.Tensor | None = None
    loop = asyncio.get_event_loop()

    try:
        # ── Handshake ────────────────────────────────────────────────────────
        config      = await ws.receive_json()
        description = config.get("description", "person speaking")
        logger.info("Session opened — description=%r", description)

        # ── Streaming loop ───────────────────────────────────────────────────
        while True:
            raw = await ws.receive_bytes()

            pcm    = await loop.run_in_executor(None, decode_webm_chunk, raw)
            chunks = buf.push(pcm)

            for chunk in chunks:
                separated = await loop.run_in_executor(
                    None, _separate, chunk, description
                )

                if prev_output is not None:
                    output  = crossfade(prev_output, separated, overlap_samples)
                    to_send = output[:-overlap_samples]
                else:
                    # Skip the very first overlap window (it's leading silence).
                    to_send = separated[overlap_samples:]

                prev_output = separated

                wav_bytes = encode_wav_chunk(to_send, settings.sample_rate)
                await ws.send_bytes(wav_bytes)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Unhandled error in WebSocket session")
