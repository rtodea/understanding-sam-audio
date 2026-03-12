"""
Pure audio utility functions — no I/O side effects, easy to unit-test.
"""

import io

import torch
import torchaudio


def decode_webm_chunk(raw: bytes, target_sr: int = 48_000) -> torch.Tensor:
    """
    Decode a WebM/Opus binary blob to a mono float32 tensor at target_sr.

    torchaudio.load() requires ffmpeg (libavcodec) to decode WebM.
    Ensure the container image has ffmpeg installed.

    Returns:
        Tensor of shape [T] — mono, float32.
    """
    buf = io.BytesIO(raw)
    waveform, sr = torchaudio.load(buf)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.mean(dim=0)  # stereo → mono


def encode_wav_chunk(tensor: torch.Tensor, sample_rate: int = 48_000) -> bytes:
    """
    Encode a mono float32 tensor to WAV bytes.

    Args:
        tensor: shape [T], float32, values in [-1, 1].
    """
    buf = io.BytesIO()
    torchaudio.save(buf, tensor.unsqueeze(0).cpu().float(), sample_rate, format="wav")
    return buf.getvalue()


def crossfade(prev: torch.Tensor, curr: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Overlap-add two consecutive separated chunks with a linear crossfade.

    The output is: prev[:-overlap] + blended + curr[overlap:]
    where blended is prev[-overlap:] fading out and curr[:overlap] fading in.
    """
    if overlap == 0 or len(prev) < overlap or len(curr) < overlap:
        return torch.cat([prev, curr])

    fade_out = torch.linspace(1.0, 0.0, overlap, device=prev.device)
    fade_in  = torch.linspace(0.0, 1.0, overlap, device=curr.device)
    blended  = prev[-overlap:] * fade_out + curr[:overlap] * fade_in

    return torch.cat([prev[:-overlap], blended, curr[overlap:]])
