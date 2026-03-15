"""
WebSocket handler — orchestration only, no business logic.

Receives:
  1. One JSON text frame:  {"description": "person speaking"}
  2. N binary frames:      raw PCM float32 frames

Sends:
  N binary frames:  WAV audio chunks (separated target audio)
  N JSON text frames: {"event":"stt","stream":"raw"|"separated","type":"recognizing"|"recognized","text":"..."}

Overlap-add assembly:
  Each input chunk advances by `advance_samples`.  After separation we emit
  exactly `advance_samples` per step:
    - First chunk: emit separated[:advance_samples] (no prior context to blend).
    - Subsequent: crossfade the stored tail of the previous output with the
      overlap prefix of the current output, emit the blended region.
  This guarantees each real-time sample appears in the output exactly once,
  with a smooth fade at chunk boundaries.

STT streams:
  Two parallel AzureSttStream instances run for the session lifetime:
    - "raw":       fed with the decoded mic PCM before the overlap buffer.
    - "separated": fed with the overlap-add output after each SAM inference.
  Both fire callbacks on Azure background threads; results are posted back to
  the asyncio event loop via asyncio.run_coroutine_threadsafe().
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


def _blend_and_advance(
    prev_tail: torch.Tensor | None,
    separated: torch.Tensor,
    overlap_samples: int,
    advance_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (to_send, new_tail).

    to_send is exactly `advance_samples` of non-duplicated output:
      - First chunk (prev_tail is None): the leading advance_samples of `separated`.
      - Subsequent chunks: crossfade of prev_tail with separated[:overlap_samples],
        giving a smooth blend over the shared overlap region.

    new_tail is the last `overlap_samples` of `separated`, retained for the
    next call.
    """
    if prev_tail is not None:
        output  = crossfade(prev_tail, separated, overlap_samples)
        to_send = output[:advance_samples]
    else:
        to_send = separated[:advance_samples]

    new_tail = separated[advance_samples:]
    return to_send, new_tail


async def _flush_tail(
    *,
    ws: WebSocket,
    buf: "OverlapBuffer",
    prev_tail: torch.Tensor | None,
    chunk_samples: int,
    overlap_samples: int,
    advance_samples: int,
    description: str,
    loop: asyncio.AbstractEventLoop,
    stt_separated,
) -> None:
    """Drain the overlap buffer and send any remaining audio to the client."""
    tail = buf.flush()

    if tail is None or tail.numel() == 0:
        if prev_tail is not None and prev_tail.numel() > 0:
            await ws.send_bytes(encode_wav_chunk(prev_tail, settings.sample_rate))
        return

    actual_samples = tail.numel()
    remaining_new  = actual_samples - overlap_samples

    if remaining_new <= 0:
        if prev_tail is not None and prev_tail.numel() > 0:
            await ws.send_bytes(encode_wav_chunk(prev_tail, settings.sample_rate))
        return

    padded    = torch.nn.functional.pad(tail, (0, max(0, chunk_samples - actual_samples)))[:chunk_samples]
    separated = await loop.run_in_executor(None, _separate, padded, description)

    if prev_tail is not None:
        output   = crossfade(prev_tail, separated, overlap_samples)
        blend    = output[:overlap_samples]
        new_part = separated[overlap_samples:actual_samples]
        to_send  = torch.cat([blend, new_part]) if new_part.numel() > 0 else blend
    else:
        to_send = separated[:actual_samples]

    if to_send.numel() > 0:
        if stt_separated is not None:
            stt_separated.push(to_send, settings.sample_rate)
        await ws.send_bytes(encode_wav_chunk(to_send, settings.sample_rate))


def _make_stt_streams(ws: WebSocket, loop: asyncio.AbstractEventLoop):
    """
    Create and start two AzureSttStream instances if STT is configured.
    Returns (stt_raw, stt_separated) or (None, None) when STT is disabled.
    """
    if not settings.stt_enabled:
        return None, None

    from .stt_service import AzureSttStream

    def _post(stream_label: str, msg_type: str, text: str) -> None:
        """Thread-safe WebSocket JSON send from Azure callback thread."""
        async def _send():
            try:
                await ws.send_json({
                    "event": "stt",
                    "stream": stream_label,
                    "type":   msg_type,
                    "text":   text,
                })
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(_send(), loop)

    def on_recognizing(label, text):
        _post(label, "recognizing", text)

    def on_recognized(label, text):
        _post(label, "recognized", text)

    stt_raw = AzureSttStream(
        "raw",
        settings.azure_stt_key,
        settings.azure_stt_region,
        settings.azure_stt_language,
        silence_timeout_ms=300,
    )
    stt_sep = AzureSttStream(
        "separated",
        settings.azure_stt_key,
        settings.azure_stt_region,
        settings.azure_stt_language,
    )

    stt_raw.start(on_recognizing, on_recognized)
    stt_sep.start(on_recognizing, on_recognized)
    logger.info("STT streams started (key=%.4s***)", settings.azure_stt_key)
    return stt_raw, stt_sep


@router.websocket("/ws/separate")
async def websocket_separate(ws: WebSocket) -> None:
    await ws.accept()

    chunk_samples   = int(settings.chunk_seconds * settings.sample_rate)
    overlap_samples = int(settings.overlap_seconds * settings.sample_rate)
    advance_samples = chunk_samples - overlap_samples
    buf             = OverlapBuffer(chunk_samples, overlap_samples)
    prev_tail: torch.Tensor | None = None
    loop = asyncio.get_event_loop()

    stt_raw, stt_separated = _make_stt_streams(ws, loop)

    try:
        # ── Handshake ────────────────────────────────────────────────────────
        config      = await ws.receive_json()
        description = config.get("description", "person speaking")
        encoding    = config.get("encoding", "pcm_f32le")
        source_sample_rate = int(config.get("sampleRate", settings.sample_rate))
        if encoding != "pcm_f32le":
            raise ValueError(f"Unsupported audio encoding: {encoding}")
        logger.info(
            "Session opened — description=%r encoding=%s sample_rate=%d stt=%s",
            description, encoding, source_sample_rate,
            "enabled" if settings.stt_enabled else "disabled",
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
                        # Stop STT streams before final flush so silence-flush
                        # fires while the connection is still open.
                        if stt_raw is not None:
                            await loop.run_in_executor(None, stt_raw.stop)
                        if stt_separated is not None:
                            await loop.run_in_executor(None, stt_separated.stop)
                        await _flush_tail(
                            ws=ws,
                            buf=buf,
                            prev_tail=prev_tail,
                            chunk_samples=chunk_samples,
                            overlap_samples=overlap_samples,
                            advance_samples=advance_samples,
                            description=description,
                            loop=loop,
                            stt_separated=None,  # already stopped
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
                    encoding, len(raw),
                )
                continue

            if pcm.ndim != 1 or pcm.numel() == 0:
                logger.warning(
                    "Skipping invalid decoded audio frame (encoding=%s bytes=%d, shape=%s)",
                    encoding, len(raw), tuple(pcm.shape),
                )
                continue

            # ── Feed raw mic PCM to STT (pre-separation, real-time) ──────────
            if stt_raw is not None:
                stt_raw.push(pcm, settings.sample_rate)

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
                        "Failed to separate audio chunk shape=%s", tuple(chunk.shape)
                    )
                    continue

                to_send, prev_tail = _blend_and_advance(
                    prev_tail, separated, overlap_samples, advance_samples
                )

                # ── Feed separated output to STT ─────────────────────────────
                if stt_separated is not None:
                    stt_separated.push(to_send, settings.sample_rate)

                await ws.send_bytes(encode_wav_chunk(to_send, settings.sample_rate))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Unhandled error in WebSocket session")
    finally:
        # Safety net: stop STT streams if they weren't stopped cleanly above.
        if stt_raw is not None:
            try:
                await loop.run_in_executor(None, stt_raw.stop)
            except Exception:
                pass
        if stt_separated is not None:
            try:
                await loop.run_in_executor(None, stt_separated.stop)
            except Exception:
                pass
