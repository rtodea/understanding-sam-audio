"""
Azure Cognitive Services Speech-to-Text via PushAudioInputStream.

Mirrors the C++ AzureSpeechToText pattern:
  - push raw PCM in via push()
  - receive Recognizing (partial) and Recognized (final) callbacks

Callbacks fire on Azure's background threads — callers must use
asyncio.run_coroutine_threadsafe() to post results back to the event loop.
"""

import logging
import threading

import torch
import torchaudio

logger = logging.getLogger(__name__)

_AZURE_SR = 16_000  # Azure PushAudioInputStream default: 16 kHz 16-bit mono


def _to_azure_pcm(pcm: torch.Tensor, source_sr: int) -> bytes:
    """Resample float32 tensor to 16 kHz int16 PCM bytes."""
    if source_sr != _AZURE_SR:
        pcm = torchaudio.functional.resample(
            pcm.unsqueeze(0), source_sr, _AZURE_SR
        ).squeeze(0)
    return (
        pcm.clamp(-1.0, 1.0)
        .mul(32767.0)
        .to(torch.int16)
        .contiguous()
        .numpy()
        .tobytes()
    )


class AzureSttStream:
    """
    Continuous speech recognition for a single audio stream.

    Usage:
        stream = AzureSttStream("raw", key, region)
        stream.start(on_recognizing, on_recognized)   # callbacks: (label, text)
        stream.push(pcm_tensor, source_sample_rate)   # call for each audio chunk
        stream.stop()                                  # silence-flush + teardown
    """

    def __init__(
        self,
        label: str,
        key: str,
        region: str,
        language: str = "en-US",
        silence_timeout_ms: int | None = None,
    ):
        self._label               = label
        self._key                 = key
        self._region              = region
        self._language            = language
        self._silence_timeout_ms  = silence_timeout_ms
        self._push_stream  = None
        self._recognizer   = None
        self._accumulated  = ""
        self._on_recognizing = None
        self._on_recognized  = None
        self._session_done   = threading.Event()

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self, on_recognizing, on_recognized) -> None:
        """
        Start continuous recognition.

        on_recognizing(label, text) — called with partial (interim) results.
        on_recognized(label, text)  — called when a sentence is finalized.
        Both are called from Azure's background thread.
        """
        import azure.cognitiveservices.speech as speechsdk

        self._on_recognizing = on_recognizing
        self._on_recognized  = on_recognized
        self._accumulated    = ""

        speech_config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        speech_config.speech_recognition_language = self._language
        if self._silence_timeout_ms is not None:
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                str(self._silence_timeout_ms),
            )
            logger.info("[STT:%s] segmentation silence timeout = %d ms", self._label, self._silence_timeout_ms)

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=_AZURE_SR, bits_per_sample=16, channels=1
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=audio_format
        )
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        self._recognizer.recognizing.connect(self._handle_recognizing)
        self._recognizer.recognized.connect(self._handle_recognized)
        self._recognizer.session_started.connect(
            lambda e: logger.info("[STT:%s] session started", self._label)
        )
        self._recognizer.session_stopped.connect(
            lambda e: (logger.info("[STT:%s] session stopped", self._label), self._session_done.set())
        )
        self._recognizer.canceled.connect(
            lambda e: logger.warning(
                "[STT:%s] canceled: %s", self._label, e.cancellation_details
            )
        )

        self._recognizer.start_continuous_recognition_async().get()
        logger.info("[STT:%s] continuous recognition started", self._label)

    def push(self, pcm: torch.Tensor, source_sr: int) -> None:
        """Push a float32 mono PCM tensor into Azure's recognition stream."""
        if self._push_stream is None:
            return
        self._push_stream.write(_to_azure_pcm(pcm.cpu().float(), source_sr))

    def stop(self) -> None:
        """
        Flush with ~2 s of silence to force Azure to finalize any in-progress
        utterance (mirrors the C++ stopListening() silence-flush trick), then
        tear down the recognizer.

        Waits for the session_stopped event before returning so that all
        recognized callbacks have fired — important for the CLI where processing
        finishes faster than real-time and the last sentence would otherwise be lost.

        Blocks briefly — call from a thread pool, not from the event loop.
        """
        if self._push_stream is None:
            return
        self._session_done.clear()
        # 2 s × 16-bit samples at 16 kHz
        self._push_stream.write(bytes(_AZURE_SR * 2 * 2))
        self._push_stream.close()
        if self._recognizer:
            self._recognizer.stop_continuous_recognition_async().get()
        # Wait for session_stopped — Azure fires this after all recognized
        # callbacks have completed, guaranteeing no events are dropped.
        if not self._session_done.wait(timeout=15):
            logger.warning("[STT:%s] timed out waiting for session_stopped", self._label)
        self._push_stream = None
        self._recognizer  = None
        self._accumulated = ""
        logger.info("[STT:%s] stopped", self._label)

    # ── Azure event handlers (Azure background thread) ───────────────────────

    def _handle_recognizing(self, evt) -> None:
        text = evt.result.text
        if not text:
            return
        partial = (self._accumulated + " " + text).strip() if self._accumulated else text
        logger.debug("[STT:%s] recognizing: %s", self._label, partial)
        if self._on_recognizing:
            self._on_recognizing(self._label, partial)

    def _handle_recognized(self, evt) -> None:
        text = evt.result.text
        if not text:
            return
        self._accumulated = (
            (self._accumulated + " " + text).strip()
            if self._accumulated
            else text
        )
        logger.info("[STT:%s] recognized: %s", self._label, self._accumulated)
        if self._on_recognized:
            self._on_recognized(self._label, self._accumulated)
