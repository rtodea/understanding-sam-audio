"""
Pure audio utility functions — no I/O side effects, easy to unit-test.
"""

import wave

import torch
import torchaudio


def decode_pcm_chunk(
    raw: bytes,
    source_sr: int = 48_000,
    target_sr: int = 48_000,
) -> torch.Tensor:
    """
    Decode little-endian float32 PCM bytes to a mono tensor at target_sr.
    """
    if not raw:
        return torch.zeros(0, dtype=torch.float32)
    if len(raw) % 4 != 0:
        raise ValueError(f"PCM payload has invalid byte length: {len(raw)}")
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers")

    waveform = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    if waveform.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), source_sr, target_sr
        ).squeeze(0)
    return waveform.contiguous().float()


def encode_wav_chunk(tensor: torch.Tensor, sample_rate: int = 48_000) -> bytes:
    """
    Encode a mono PCM16 WAV chunk to bytes.

    Args:
        tensor: shape [T], float32, values in [-1, 1].
    """
    buf = io.BytesIO()
    pcm16 = (
        tensor.detach()
        .cpu()
        .float()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .to(torch.int16)
        .contiguous()
    )
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.numpy().tobytes())
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
