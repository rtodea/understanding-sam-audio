"""
Headless batch processor — SAM-Audio source separation + Azure STT.

Reads any audio/video file, runs the same overlap-add + SAM inference
pipeline as the WebSocket server, optionally transcribes both the raw
and separated streams via Azure STT, and writes:

  <stem>_transcript.txt    — aligned, human-readable transcript
  <stem>_transcript.csv    — machine-readable (time_sec, stream, delta_sec, text)
  <stem>_separated.wav     — separated audio  (only with --save-audio)
  <stem>_original.wav      — mono original    (only with --save-audio)

Run inside the Docker container where all Python deps are present:

  docker compose exec webrtc-server \\
      python -m webrtc_server.cli /data/recording.webm "person speaking"

See --help for full usage.
"""

import argparse
import csv
import io
import logging
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sam-cli")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SttEvent:
    audio_sec: float   # audio-stream position when the callback fired
    stream: str        # "raw" | "separated"
    type: str          # "recognizing" | "recognized"
    text: str


# ── Audio I/O helpers ─────────────────────────────────────────────────────────

def _decode_file(path: Path, target_sr: int) -> torch.Tensor:
    """Load any audio/video file → mono float32 tensor at target_sr."""
    logger.info("Decoding %s …", path)
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        # Some builds of torchaudio need the backend specified explicitly.
        waveform, sr = torchaudio.load(str(path), backend="ffmpeg")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        logger.info("Resampling %d Hz → %d Hz …", sr, target_sr)
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sr, target_sr
        ).squeeze(0)

    logger.info(
        "Decoded: %.2f s  (%d samples @ %d Hz)",
        waveform.shape[0] / target_sr,
        waveform.shape[0],
        target_sr,
    )
    return waveform.float().contiguous()


def _save_wav(tensor: torch.Tensor, sample_rate: int, path: Path) -> None:
    pcm16 = (
        tensor.cpu().float().clamp(-1.0, 1.0)
        .mul(32767.0).to(torch.int16).contiguous()
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.numpy().tobytes())
    path.write_bytes(buf.getvalue())
    logger.info("Saved: %s", path)


# ── Output formatting ─────────────────────────────────────────────────────────

def _compute_deltas(events: list[SttEvent]) -> list[float | None]:
    """
    For each "separated/recognized" event: delta = its audio_sec minus the
    audio_sec of the most recent "raw/recognized" event before it.
    All other events get None.
    """
    last_raw_sec: float | None = None
    deltas: list[float | None] = []
    for ev in events:
        if ev.stream == "raw" and ev.type == "recognized":
            last_raw_sec = ev.audio_sec
        if ev.stream == "separated" and ev.type == "recognized" and last_raw_sec is not None:
            deltas.append(ev.audio_sec - last_raw_sec)
        else:
            deltas.append(None)
    return deltas


def _write_txt(events: list[SttEvent], deltas: list[float | None], path: Path) -> None:
    lines = ["SAM-Audio CLI Transcript", "─" * 62, ""]
    for ev, delta in zip(events, deltas):
        if ev.type != "recognized":
            continue
        ts    = f"{ev.audio_sec:7.2f}s"
        badge = f"[{ev.stream:<9}]"
        d_str = f"  Δ{delta:+.2f}s" if delta is not None else ""
        lines.append(f"{ts}  {badge}  {ev.text}{d_str}")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", path)


def _write_csv(events: list[SttEvent], deltas: list[float | None], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "stream", "type", "delta_sec", "text"])
        for ev, delta in zip(events, deltas):
            writer.writerow([
                f"{ev.audio_sec:.3f}",
                ev.stream,
                ev.type,
                f"{delta:.3f}" if delta is not None else "",
                ev.text,
            ])
    logger.info("Saved: %s", path)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: Path,
    description: str,
    out_dir: Path,
    save_audio: bool,
    no_stt: bool,
) -> None:
    from .audio_utils import crossfade
    from .config import settings
    from .model_registry import get_model
    from .overlap_buffer import OverlapBuffer

    chunk_samples   = int(settings.chunk_seconds   * settings.sample_rate)
    overlap_samples = int(settings.overlap_seconds * settings.sample_rate)
    advance_samples = chunk_samples - overlap_samples

    # ── Decode ────────────────────────────────────────────────────────────────
    pcm_full = _decode_file(input_path, settings.sample_rate)
    duration = pcm_full.shape[0] / settings.sample_rate

    # ── STT setup ─────────────────────────────────────────────────────────────
    # audio_pos is a mutable container so Azure background-thread callbacks can
    # read the current feed position without needing asyncio.
    audio_pos  = [0.0]
    events: list[SttEvent] = []
    ev_lock    = threading.Lock()
    stt_raw = stt_sep = None

    if not no_stt:
        if not settings.stt_enabled:
            logger.warning("AZURE_STT_KEY is not set — skipping STT.")
        else:
            from .stt_service import AzureSttStream

            def _make_cb(stream_label: str):
                def on_recognizing(label: str, text: str) -> None:
                    with ev_lock:
                        events.append(SttEvent(audio_pos[0], label, "recognizing", text))
                    logger.debug("[STT:%s] recognizing: %s", label, text)

                def on_recognized(label: str, text: str) -> None:
                    with ev_lock:
                        events.append(SttEvent(audio_pos[0], label, "recognized", text))
                    logger.info("[STT:%s] ✓ %s", label, text)

                stream = AzureSttStream(
                    stream_label,
                    settings.azure_stt_key,
                    settings.azure_stt_region,
                    settings.azure_stt_language,
                    silence_timeout_ms=300 if stream_label == "raw" else None,
                )
                stream.start(on_recognizing, on_recognized)
                return stream

            stt_raw = _make_cb("raw")
            stt_sep = _make_cb("separated")
            logger.info("STT streams started.")

    # ── Model load ────────────────────────────────────────────────────────────
    logger.info("Loading SAM model (%s) …", settings.sam_model)
    get_model()
    device = settings.effective_device

    # ── Separation helpers (inline to avoid re-importing) ─────────────────────
    def _separate(chunk: torch.Tensor) -> torch.Tensor:
        model, processor = get_model()
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
            target = target[0]
        return target.detach().float().reshape(-1).cpu()

    def _blend(
        prev_tail: torch.Tensor | None,
        separated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prev_tail is not None:
            output  = crossfade(prev_tail, separated, overlap_samples)
            to_send = output[:advance_samples]
        else:
            to_send = separated[:advance_samples]
        return to_send, separated[advance_samples:]

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Feed PCM in 250 ms increments — same granularity as the browser client.
    feed_samples      = int(0.25 * settings.sample_rate)
    buf               = OverlapBuffer(chunk_samples, overlap_samples)
    prev_tail: torch.Tensor | None = None
    separated_chunks: list[torch.Tensor] = []
    total_fed   = 0
    total_samp  = pcm_full.shape[0]
    chunk_idx   = 0
    t_start     = time.monotonic()

    logger.info(
        "Processing %.2f s  |  model=%s  |  device=%s  |  chunk=%.1fs  |  overlap=%.1fs",
        duration,
        settings.sam_model,
        device,
        settings.chunk_seconds,
        settings.overlap_seconds,
    )

    while total_fed < total_samp:
        end   = min(total_fed + feed_samples, total_samp)
        frame = pcm_full[total_fed:end]
        total_fed   = end
        audio_pos[0] = total_fed / settings.sample_rate

        if stt_raw is not None:
            stt_raw.push(frame, settings.sample_rate)

        for chunk in buf.push(frame):
            chunk_idx += 1
            pct = total_fed / total_samp * 100
            logger.info(
                "[chunk %d]  %.0f%%  audio=%.2fs / %.2fs",
                chunk_idx, pct, audio_pos[0], duration,
            )
            separated = _separate(chunk)
            to_send, prev_tail = _blend(prev_tail, separated)

            if stt_sep is not None:
                stt_sep.push(to_send, settings.sample_rate)
            separated_chunks.append(to_send)

    # ── Flush overlap tail ────────────────────────────────────────────────────
    # Before the tail, push prev_tail to stt_sep so Azure receives the overlap
    # region at full-chunk quality (not as part of a padded/crossfaded blend).
    # This ensures any speech that straddles the last chunk boundary arrives
    # cleanly, without duplicating audio already sent via the main loop advances.
    if stt_sep is not None and prev_tail is not None:
        stt_sep.push(prev_tail, settings.sample_rate)

    tail = buf.flush()
    if tail is not None and tail.numel() > 0:
        actual = tail.numel()
        padded = torch.nn.functional.pad(
            tail, (0, max(0, chunk_samples - actual))
        )[:chunk_samples]
        separated = _separate(padded)

        if prev_tail is not None:
            output   = crossfade(prev_tail, separated, overlap_samples)
            blend    = output[:overlap_samples]
            new_part = separated[overlap_samples:actual]
            to_send  = torch.cat([blend, new_part]) if new_part.numel() > 0 else blend
        else:
            to_send = separated[:actual]

        if to_send.numel() > 0:
            if stt_sep is not None:
                # Push only the truly new samples (beyond the overlap region
                # already sent via prev_tail above). No duplicates, no crossfade.
                if new_part.numel() > 0:
                    stt_sep.push(new_part, settings.sample_rate)
            separated_chunks.append(to_send)

    elapsed = time.monotonic() - t_start
    logger.info(
        "Separation done in %.1f s  (%.1fx real-time)",
        elapsed,
        duration / elapsed if elapsed > 0 else 0,
    )

    # ── Tear down STT (silence-flush blocks briefly) ──────────────────────────
    if stt_raw is not None:
        logger.info("Flushing raw STT …")
        stt_raw.stop()
    if stt_sep is not None:
        logger.info("Flushing separated STT …")
        stt_sep.stop()

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    stem = f"{input_path.stem}_{ts}"

    if save_audio:
        if separated_chunks:
            _save_wav(
                torch.cat(separated_chunks),
                settings.sample_rate,
                out_dir / f"{stem}_separated.wav",
            )
        _save_wav(pcm_full, settings.sample_rate, out_dir / f"{stem}_original.wav")

    with ev_lock:
        snapshot = list(events)

    if snapshot:
        # Recognized-only view for the human-readable TXT
        final_only  = [e for e in snapshot if e.type == "recognized"]
        fin_deltas  = _compute_deltas(final_only)
        _write_txt(final_only, fin_deltas, out_dir / f"{stem}_transcript.txt")
        _write_csv(final_only, fin_deltas, out_dir / f"{stem}_transcript.csv")

        # Full log (including partials) — separate CSV
        all_deltas = _compute_deltas(snapshot)
        _write_csv(snapshot, all_deltas, out_dir / f"{stem}_transcript_full.csv")

        logger.info(
            "Transcript: %d recognized events  (%d total incl. partials)",
            len(final_only),
            len(snapshot),
        )
    else:
        logger.warning("No STT events collected — transcript files not written.")

    logger.info("Done. Outputs in %s/", out_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m webrtc_server.cli",
        description="Headless SAM-Audio CLI: separate + transcribe an audio/video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Basic usage (runs STT if AZURE_STT_KEY is set in environment):
  python -m webrtc_server.cli recording.webm "person speaking"

# Save separated audio alongside transcripts:
  python -m webrtc_server.cli recording.webm "person speaking" --save-audio

# Custom output directory:
  python -m webrtc_server.cli recording.webm "guitar" --out-dir results/guitar/

# Skip STT — test separation pipeline only:
  python -m webrtc_server.cli recording.webm "person speaking" --no-stt --save-audio

# Run via Docker (most common):
  docker compose exec webrtc-server \\
      python -m webrtc_server.cli /data/recording.webm "person speaking" --save-audio
""",
    )
    parser.add_argument(
        "input",
        help="Input audio/video file (WebM, WAV, MP4, OGG, …)",
    )
    parser.add_argument(
        "description",
        help="Audio source to isolate, e.g. 'person speaking' or 'guitar'",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        metavar="DIR",
        help="Directory for output files (default: current directory)",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save <stem>_separated.wav and <stem>_original.wav",
    )
    parser.add_argument(
        "--no-stt",
        action="store_true",
        help="Skip speech recognition (separation pipeline only)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    run(
        input_path=Path(args.input),
        description=args.description,
        out_dir=Path(args.out_dir),
        save_audio=args.save_audio,
        no_stt=args.no_stt,
    )


if __name__ == "__main__":
    main()
