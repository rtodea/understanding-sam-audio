"""
WebSocket handler — orchestration only, no business logic.

Receives:
  1. One JSON text frame:  {"description": "person speaking"}
  2. N binary frames:      raw PCM float32 frames

Sends:
  N binary frames:    WAV audio chunks (separated target audio)
  N JSON text frames: {"event":"stt","stream":"raw"|"separated",
                       "type":"recognizing"|"recognized","text":"..."}

Architecture — two concurrent asyncio tasks per session:

  receiver   Reads WebSocket messages as fast as they arrive.
             Decodes each PCM chunk and immediately pushes it to the raw
             STT stream so Azure always gets audio at real-time pace,
             regardless of how long SAM inference takes.
             Puts decoded chunks into an asyncio.Queue for the processor.
             On stop/disconnect: stops the raw STT stream and sends the
             sentinel (None) to the queue.

  processor  Consumes the queue, feeds the OverlapBuffer, runs SAM
             inference (in a thread-pool executor), applies overlap-add,
             pushes separated audio to the separated STT stream, and sends
             WAV chunks back to the client.
             On sentinel: flushes remaining audio, stops the separated STT
             stream, and closes the WebSocket.

This decoupling ensures that the raw STT stream receives a smooth,
real-time audio feed even when SAM inference takes 3–5 s per chunk.
"""

import asyncio
import json
import logging

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio_utils import crossfade, decode_pcm_chunk, encode_wav_chunk
from .config import settings
from .model_registry import get_model
from .overlap_buffer import OverlapBuffer

router = APIRouter()
logger = logging.getLogger(__name__)


# ── SAM-Audio inference ───────────────────────────────────────────────────────

def _separate(chunk: torch.Tensor, description: str) -> torch.Tensor:
    """
    Run SAM-Audio separation synchronously (called via run_in_executor
    so it does not block the event loop).
    """
    model, processor = get_model()
    device = settings.effective_device
    batch = processor(audios=[chunk], descriptions=[description]).to(device)
    if device == "cuda":
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


# ── Overlap-add helpers ───────────────────────────────────────────────────────

def _blend_and_advance(
    prev_tail: torch.Tensor | None,
    separated: torch.Tensor,
    overlap_samples: int,
    advance_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (to_send, new_tail) — exactly advance_samples of non-duplicated
    output per call.
    """
    if prev_tail is not None:
        output  = crossfade(prev_tail, separated, overlap_samples)
        to_send = output[:advance_samples]
    else:
        to_send = separated[:advance_samples]
    return to_send, separated[advance_samples:]


async def _flush_tail(
    *,
    ws: WebSocket,
    buf: OverlapBuffer,
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


# ── STT stream factory ────────────────────────────────────────────────────────

def _make_stt_streams(ws: WebSocket, loop: asyncio.AbstractEventLoop):
    """
    Create and start two AzureSttStream instances if STT is configured.
    Returns (stt_raw, stt_separated) or (None, None) when disabled.
    """
    if not settings.stt_enabled:
        return None, None

    from .stt_service import AzureSttStream

    def _post(stream_label: str, msg_type: str, text: str) -> None:
        async def _send():
            try:
                await ws.send_json({
                    "event":  "stt",
                    "stream": stream_label,
                    "type":   msg_type,
                    "text":   text,
                })
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(_send(), loop)

    def on_recognizing(label, text): _post(label, "recognizing", text)
    def on_recognized(label, text):  _post(label, "recognized",  text)

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


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/separate")
async def websocket_separate(ws: WebSocket) -> None:
    await ws.accept()

    chunk_samples   = int(settings.chunk_seconds * settings.sample_rate)
    overlap_samples = int(settings.overlap_seconds * settings.sample_rate)
    advance_samples = chunk_samples - overlap_samples
    loop            = asyncio.get_event_loop()

    stt_raw, stt_separated = _make_stt_streams(ws, loop)

    try:
        # ── Handshake ────────────────────────────────────────────────────────
        config             = await ws.receive_json()
        description        = config.get("description", "person speaking")
        encoding           = config.get("encoding", "pcm_f32le")
        source_sample_rate = int(config.get("sampleRate", settings.sample_rate))
        if encoding != "pcm_f32le":
            raise ValueError(f"Unsupported audio encoding: {encoding}")
        logger.info(
            "Session opened — description=%r encoding=%s sample_rate=%d stt=%s",
            description, encoding, source_sample_rate,
            "enabled" if settings.stt_enabled else "disabled",
        )

        # Queue carries decoded PCM tensors; None is the end-of-stream sentinel.
        pcm_queue: asyncio.Queue[torch.Tensor | None] = asyncio.Queue()

        # ── Receiver ─────────────────────────────────────────────────────────
        async def receiver() -> None:
            try:
                while True:
                    message = await ws.receive()

                    if message["type"] == "websocket.disconnect":
                        break

                    if "text" in message and message["text"]:
                        try:
                            payload = json.loads(message["text"])
                        except Exception:
                            logger.warning("Ignoring invalid text frame: %r", message["text"][:80])
                            continue
                        if payload.get("event") == "stop":
                            break
                        continue

                    raw = message.get("bytes")
                    if not raw:
                        logger.warning("Skipping empty WebSocket audio frame")
                        continue

                    try:
                        pcm = await loop.run_in_executor(
                            None, decode_pcm_chunk, raw, source_sample_rate, settings.sample_rate
                        )
                    except Exception:
                        logger.exception("Failed to decode audio frame (bytes=%d)", len(raw))
                        continue

                    if pcm.ndim != 1 or pcm.numel() == 0:
                        logger.warning("Skipping invalid decoded frame shape=%s", tuple(pcm.shape))
                        continue

                    # Push to raw STT immediately — real-time, independent of SAM.
                    if stt_raw is not None:
                        stt_raw.push(pcm, settings.sample_rate)

                    await pcm_queue.put(pcm)

            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("Receiver error")
            finally:
                # Signal processor that the stream is done.
                await pcm_queue.put(None)
                # Silence-flush + teardown (blocking — run in executor).
                if stt_raw is not None:
                    await loop.run_in_executor(None, stt_raw.stop)

        # ── Processor ────────────────────────────────────────────────────────
        async def processor() -> None:
            buf       = OverlapBuffer(chunk_samples, overlap_samples)
            prev_tail: torch.Tensor | None = None

            try:
                while True:
                    pcm = await pcm_queue.get()
                    if pcm is None:
                        break  # end-of-stream sentinel

                    for chunk in buf.push(pcm):
                        if chunk.ndim != 1 or chunk.numel() == 0:
                            logger.warning("Skipping invalid chunk shape=%s", tuple(chunk.shape))
                            continue
                        try:
                            separated = await loop.run_in_executor(
                                None, _separate, chunk, description
                            )
                        except Exception:
                            logger.exception("Failed to separate chunk shape=%s", tuple(chunk.shape))
                            continue

                        to_send, prev_tail = _blend_and_advance(
                            prev_tail, separated, overlap_samples, advance_samples
                        )
                        if stt_separated is not None:
                            stt_separated.push(to_send, settings.sample_rate)
                        try:
                            await ws.send_bytes(encode_wav_chunk(to_send, settings.sample_rate))
                        except Exception:
                            logger.warning("Failed to send WAV chunk — client may have disconnected")
                            return

                # Flush remaining overlap buffer.
                await _flush_tail(
                    ws=ws,
                    buf=buf,
                    prev_tail=prev_tail,
                    chunk_samples=chunk_samples,
                    overlap_samples=overlap_samples,
                    advance_samples=advance_samples,
                    description=description,
                    loop=loop,
                    stt_separated=stt_separated,
                )

            except Exception:
                logger.exception("Processor error")
            finally:
                if stt_separated is not None:
                    await loop.run_in_executor(None, stt_separated.stop)
                try:
                    await ws.close()
                except Exception:
                    pass

        await asyncio.gather(receiver(), processor())

    except WebSocketDisconnect:
        logger.info("Client disconnected during handshake")
    except Exception:
        logger.exception("Unhandled error in WebSocket session")
    finally:
        # Safety net: ensure STT streams are stopped even on unexpected exit.
        for stt in (stt_raw, stt_separated):
            if stt is not None:
                try:
                    await loop.run_in_executor(None, stt.stop)
                except Exception:
                    pass
