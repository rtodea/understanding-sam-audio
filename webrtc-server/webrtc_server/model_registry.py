"""
Singleton model loader.

get_model() is safe to call from multiple coroutines: the threading.Lock
ensures the model is loaded exactly once. It is NOT safe across multiple
OS processes — always run uvicorn with --workers 1.
"""

import logging
import threading

import torch

from sam_audio import SAMAudio, SAMAudioProcessor

from .config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: SAMAudio | None = None
_processor: SAMAudioProcessor | None = None


def get_model() -> tuple[SAMAudio, SAMAudioProcessor]:
    global _model, _processor

    if _model is not None:
        return _model, _processor

    with _lock:
        if _model is not None:
            return _model, _processor

        device = settings.effective_device
        logger.info("Loading %s on %s …", settings.sam_model, device)

        if device == "cpu":
            logger.warning(
                "CUDA not available — running on CPU. "
                "Latency will be 50–100× higher than on a GPU."
            )

        _processor = SAMAudioProcessor.from_pretrained(settings.sam_model)
        print("[registry] Processor loaded. Loading model weights …", flush=True)
        _model = SAMAudio.from_pretrained(settings.sam_model).eval()
        print(f"[registry] Model loaded to CPU RAM. Moving to {device} …", flush=True)

        if device == "cuda":
            # fp16 is only safe on CUDA; CPU does not support float16 inference.
            print("[registry] Converting to fp16 …", flush=True)
            _model = _model.half()
            print("[registry] fp16 done. Calling .cuda() …", flush=True)
            _model = _model.cuda()
            print("[registry] .cuda() complete.", flush=True)

        logger.info("Model ready.")
        return _model, _processor
