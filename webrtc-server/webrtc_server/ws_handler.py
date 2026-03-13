"""
WebSocket handler — orchestration only, no business logic.

Receives:
  1. One JSON text frame:  {"description": "person speaking"}
  2. N binary frames:      raw PCM float32 frames

Sends:
  N binary frames: WAV audio chunks (separated target audio)
"""

import asyncio
import json
import logging

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio_utils import (
    crossfade,
    decode_pcm_chunk,
    encode_wav_chunk,
)
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
    if device == "cuda":
        # The model is converted to fp16 during CUDA startup, so keep the audio
        # tensor dtype aligned with the model weights.
        batch.audios = batch.audios.half()
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=settings.predict_spans,
            reranking_candidates=settings.sam_reranking_candidates,
        )

    target = result.target
    if isinstance(target, list):
        if not target:
            raise ValueError("Model returned an empty target list")
        target = target[0]

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Unexpected target type: {type(target)!r}")

    return target.detach().float().reshape(-1).cpu()


async def _flush_tail(
    *,
    ws: WebSocket,
    buf: OverlapBuffer,
    prev_output: torch.Tensor | None,
    chunk_samples: int,
    overlap_samples: int,
    description: str,
    loop: asyncio.AbstractEventLoop,
) -> torch.Tensor | None:
    tail = buf.flush()
    if tail is None or tail.numel() <= overlap_samples:
        return prev_output

    padded = torch.nn.functional.pad(tail, (0, max(0, chunk_samples - tail.numel())))
    padded = padded[:chunk_samples]
    separated = await loop.run_in_executor(None, _separate, padded, description)
    if prev_output is not None:
        output = crossfade(prev_output, separated, overlap_samples)
        to_send = output[:-overlap_samples]
    else:
        to_send = separated[overlap_samples:]

    if to_send.numel() > 0:
        wav_bytes = encode_wav_chunk(to_send, settings.sample_rate)
        await ws.send_bytes(wav_bytes)
    return separated


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
        encoding = config.get("encoding", "pcm_f32le")
        source_sample_rate = int(config.get("sampleRate", settings.sample_rate))
        if encoding != "pcm_f32le":
            raise ValueError(f"Unsupported audio encoding: {encoding}")
        logger.info(
            "Session opened — description=%r encoding=%s sample_rate=%d",
            description,
            encoding,
            source_sample_rate,
        )

        # ── Streaming loop ───────────────────────────────────────────────────
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect

            if "text" in message and message["text"] is not None:
                control = message["text"]
                if control:
                    try:
                        payload = json.loads(control)
                    except Exception:
                        logger.warning("Ignoring invalid text frame: %r", control[:80])
                        continue
                    if payload.get("event") == "stop":
                        prev_output = await _flush_tail(
                            ws=ws,
                            buf=buf,
                            prev_output=prev_output,
                            chunk_samples=chunk_samples,
                            overlap_samples=overlap_samples,
                            description=description,
                            loop=loop,
                        )
                        await ws.close()
                        break
                continue

            raw = message.get("bytes")

            if not raw:
                logger.warning("Skipping empty websocket audio frame")
                continue

            try:
                pcm = await loop.run_in_executor(
                    None,
                    decode_pcm_chunk,
                    raw,
                    source_sample_rate,
                    settings.sample_rate,
                )
            except Exception:
                logger.exception(
                    "Failed to decode audio frame (encoding=%s bytes=%d)",
                    encoding,
                    len(raw),
                )
                continue

            if pcm.ndim != 1 or pcm.numel() == 0:
                logger.warning(
                    "Skipping invalid decoded audio frame (encoding=%s bytes=%d, shape=%s)",
                    encoding,
                    len(raw),
                    tuple(pcm.shape),
                )
                continue

            chunks = buf.push(pcm)

            for chunk in chunks:
                if chunk.ndim != 1 or chunk.numel() == 0:
                    logger.warning("Skipping invalid buffered chunk shape=%s", tuple(chunk.shape))
                    continue

                try:
                    separated = await loop.run_in_executor(
                        None, _separate, chunk, description
                    )
                except Exception:
                    logger.exception(
                        "Failed to separate audio chunk shape=%s",
                        tuple(chunk.shape),
                    )
                    continue

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
