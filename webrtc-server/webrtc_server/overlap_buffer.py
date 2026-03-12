"""
OverlapBuffer — accumulates PCM samples and yields fixed-size chunks
with a configurable overlap window.

Pure PyTorch — no I/O, fully unit-testable.
"""

import torch


class OverlapBuffer:
    """
    Push arbitrary-length PCM tensors in, get back a list of fixed-size
    chunks ready to be sent to SAM-Audio.

    Overlap logic:
        After yielding a chunk of `chunk_samples`, the buffer retains
        the last `overlap_samples` so the next chunk shares that region.
        SAM-Audio's crossfade then smooths the boundary artefacts.
    """

    def __init__(self, chunk_samples: int, overlap_samples: int) -> None:
        assert overlap_samples < chunk_samples, (
            "overlap_samples must be strictly less than chunk_samples"
        )
        self._chunk_samples   = chunk_samples
        self._overlap_samples = overlap_samples
        self._advance         = chunk_samples - overlap_samples
        self._buf             = torch.zeros(0)

    def push(self, pcm: torch.Tensor) -> list[torch.Tensor]:
        """
        Append `pcm` to the internal buffer.

        Returns:
            A (possibly empty) list of chunks, each of length `chunk_samples`.
            Call this every time a new decoded audio blob arrives.
        """
        self._buf = torch.cat([self._buf, pcm])
        chunks: list[torch.Tensor] = []
        while len(self._buf) >= self._chunk_samples:
            chunks.append(self._buf[: self._chunk_samples].clone())
            self._buf = self._buf[self._advance :]
        return chunks

    def flush(self) -> torch.Tensor | None:
        """
        Return any remaining samples at end-of-session, or None if empty.
        The tail may be shorter than chunk_samples.
        """
        if len(self._buf) > 0:
            tail = self._buf.clone()
            self._buf = torch.zeros(0)
            return tail
        return None
