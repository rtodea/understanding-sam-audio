# WebRTC Debug TODOs

## Current Status

- The browser-to-backend transport now uses raw `pcm_f32le` audio frames from Web Audio instead of `MediaRecorder` WebM chunks.
- Live separated playback is working in the browser.
- `Download separated.wav` is working.
- `Download original` is working and provides the untouched camera+microphone recording.
- Dev hot reload works for:
  - `webrtc-client/` via browser refresh
  - `webrtc-server/webrtc_server/` via `uvicorn --reload`
  - `sam_audio/` via `uvicorn --reload`
- Changes to `webrtc-docker-compose.dev.yml` still require container recreate.

## Main Conclusions

### 1. Old WebM chunk transport was unreliable

- `MediaRecorder` chunks were not consistently decodable as standalone WebM/Opus files on the backend.
- Moving to raw PCM streaming removed the repeated decode failures and unlocked stable end-to-end behavior.

### 2. Several backend integration bugs were fixed

- Mono PCM tensors from the streaming path were incorrectly collapsed to scalars in `sam_audio/processor.py`.
- CUDA inference required converting `batch.audios` to `fp16` to match the model weights.
- The separation result shape was normalized because `result.target` could be a list, not just a tensor.
- WAV serialization was switched away from `torchaudio.save(..., BytesIO)` to a Python `wave` writer.
- Session stop was made graceful so the client can receive final chunks before download assembly.

### 3. Model quality is still mixed

- The pipeline now works, but output quality is not yet ideal.
- Example observed issue:
  - repeated phrases like `Hello... Hello... My name is... Robert...`
  - some kid speech still leaking into the separated output

## Most Likely Remaining Issues

### A. Overlap / crossfade duplication

- Repeated words are likely caused by the current overlap-buffer and crossfade emission logic.
- This should be investigated before judging the model too harshly.
- Likely file to inspect:
  - `webrtc-server/webrtc_server/ws_handler.py`

### B. Small model quality limits

- `facebook/sam-audio-small` is useful for fast iteration, but likely worsens separation quality.
- Leakage from the kid voice may improve when switching back to `facebook/sam-audio-base`.

### C. Prompt phrasing matters

- Negative prompts such as:
  - `only the man speaking (not the kid)`
  may be weaker than positive target prompts.
- Try prompts like:
  - `adult man speaking`
  - `male adult voice`
  - `the adult male speaker`

## Next Steps

1. Investigate overlap/crossfade logic to remove repeated words and duplicate fragments.
2. Compare `sam-audio-small` vs `sam-audio-base` once stitching is fixed.
3. Try prompt tuning with positive target descriptions instead of negation.
4. Consider testing `SAM_RERANKING_CANDIDATES=2` after basic streaming quality is stable.
5. Hook in a speech-to-text service such as Azure Speech and display partial recognized text live on screen.
6. Use those partial STT updates to estimate end-to-end latency across capture, transport, separation, and UI rendering.
7. Update `doc/realtime-webrtc.md` so it no longer describes the old WebM transport as the active implementation.

## Useful Commands

### Start dev stack

```bash
docker compose -f webrtc-docker-compose.yml -f webrtc-docker-compose.dev.yml up
```

### Recreate backend after compose changes

```bash
docker compose -f webrtc-docker-compose.yml -f webrtc-docker-compose.dev.yml up -d --force-recreate webrtc-server
```
