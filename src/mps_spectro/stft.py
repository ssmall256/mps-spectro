import math
from collections import OrderedDict
from typing import Optional

import torch

from mps_spectro import compiler

_WINDOW_CACHE_MAX = 64
_WINDOW_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def _should_use_metal_stft(
    *,
    batch_size: int,
    output_bytes: int,
) -> bool:
    # Small single-item workloads still favor torch.stft on MPS, but medium
    # batched workloads cross over much earlier than the old global 5 MB rule.
    if batch_size >= 4:
        return output_bytes >= 4 * 1024 * 1024
    return output_bytes >= 5 * 1024 * 1024


def _validate_input(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    orig_1d = (x.dim() == 1)
    if orig_1d:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError(f"stft expects input shape [T] or [B, T], got {tuple(x.shape)}")
    if not x.device.type == "mps":
        raise ValueError("Input must be on MPS device")
    if not x.is_floating_point():
        raise TypeError("Expected floating-point input waveform")
    return x, orig_1d


def _window_cache_get(key: tuple) -> Optional[torch.Tensor]:
    value = _WINDOW_CACHE.get(key)
    if value is None:
        return None
    _WINDOW_CACHE.move_to_end(key)
    return value


def _window_cache_set(key: tuple, value: torch.Tensor) -> torch.Tensor:
    _WINDOW_CACHE[key] = value
    _WINDOW_CACHE.move_to_end(key)
    if len(_WINDOW_CACHE) > _WINDOW_CACHE_MAX:
        _WINDOW_CACHE.popitem(last=False)
    return value


def _resolve_window(
    window: Optional[torch.Tensor],
    *,
    win_length: int,
    n_fft: int,
    device: torch.device,
) -> torch.Tensor:
    if window is None:
        key = ("hann", int(win_length), int(n_fft), device.type, device.index)
        cached = _window_cache_get(key)
        if cached is not None:
            return cached
        w = torch.hann_window(win_length, periodic=True, device=device, dtype=torch.float32)
    else:
        if window.dim() != 1:
            raise ValueError(f"window must be 1D, got shape {tuple(window.shape)}")
        w = window.to(device=device, dtype=torch.float32)

    if int(w.numel()) == int(n_fft):
        out = w.contiguous()
        if window is None:
            return _window_cache_set(key, out)
        return out
    if int(w.numel()) != int(win_length):
        raise ValueError(
            f"window length must be win_length ({win_length}) or n_fft ({n_fft}), got {int(w.numel())}"
        )

    left = (n_fft - win_length) // 2
    right = (n_fft - win_length + 1) // 2
    out = torch.nn.functional.pad(w, (left, right)).contiguous()
    if window is None:
        return _window_cache_set(key, out)
    return out


def mps_stft_forward(
    x: torch.Tensor,
    *,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
) -> torch.Tensor:
    """Fast MPS STFT using a fused Metal kernel for frame extraction + windowing."""
    x, orig_1d = _validate_input(x)

    n_fft = int(n_fft)
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if hop_length is None:
        hop_length = n_fft // 4
    hop_length = int(hop_length)
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if win_length is None:
        win_length = n_fft
    win_length = int(win_length)
    if win_length <= 0:
        raise ValueError("win_length must be positive")
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

    x = x.to(dtype=torch.float32).contiguous()
    window_nfft = _resolve_window(
        window,
        win_length=win_length,
        n_fft=n_fft,
        device=x.device,
    )

    batch_size = x.size(0)
    input_length = x.size(1)
    pad_amount = n_fft // 2 if center else 0
    padded_length = input_length + 2 * pad_amount
    n_frames = (padded_length - n_fft) // hop_length + 1

    # Use the custom Metal path once the workload is large enough to amortize
    # the Python-side dispatch and extra FFT plumbing. Single-item workloads
    # still favor torch.stft until the output gets fairly large, but medium
    # batched jobs cross over much earlier.
    output_bytes = batch_size * n_frames * n_fft * 4  # float32

    if not _should_use_metal_stft(batch_size=batch_size, output_bytes=output_bytes):
        out = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window_nfft,
            center=center,
            normalized=normalized,
            onesided=onesided,
            return_complex=True,
        )
    else:
        # Fused Metal kernel: reflect-pad + strided frame extraction + windowing
        # in a single GPU pass (no intermediate padded or strided tensors).
        frames = torch.ops.mps_spectro.stft_extract_frames(
            x, window_nfft, int(hop_length), int(n_fft), bool(center),
        )

        if onesided:
            spec = torch.fft.rfft(frames, n=n_fft, dim=-1, norm="backward")
        else:
            spec = torch.fft.fft(frames, n=n_fft, dim=-1, norm="backward")
        if normalized:
            spec = spec * (1.0 / math.sqrt(n_fft))

        out = spec.transpose(1, 2).contiguous()

    if orig_1d:
        return out.squeeze(0)
    return out


__all__ = ["mps_stft_forward"]
