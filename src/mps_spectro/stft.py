import math
from typing import Optional

import torch

from mps_spectro.compiler import compiled_lib


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


def _resolve_window(
    window: Optional[torch.Tensor],
    *,
    win_length: int,
    n_fft: int,
    device: torch.device,
) -> torch.Tensor:
    if window is None:
        w = torch.hann_window(win_length, periodic=True, device=device, dtype=torch.float32)
    else:
        if window.dim() != 1:
            raise ValueError(f"window must be 1D, got shape {tuple(window.shape)}")
        w = window.to(device=device, dtype=torch.float32)

    if int(w.numel()) == int(n_fft):
        return w.contiguous()
    if int(w.numel()) != int(win_length):
        raise ValueError(
            f"window length must be win_length ({win_length}) or n_fft ({n_fft}), got {int(w.numel())}"
        )

    left = (n_fft - win_length) // 2
    right = (n_fft - win_length + 1) // 2
    return torch.nn.functional.pad(w, (left, right)).contiguous()


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

    # Use Metal kernel when the output tensor is large enough to be
    # bandwidth-bound, where the tiled shared-memory kernel amortises the
    # Python 3-op dispatch overhead.  For smaller workloads torch.stft's
    # single fused C++ call is faster.  Empirical crossover is ~5 MB on
    # Apple Silicon (M-series).
    _METAL_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5 MB
    output_bytes = batch_size * n_frames * n_fft * 4  # float32

    if output_bytes < _METAL_THRESHOLD_BYTES:
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
        frames = compiled_lib.mps_stft_extract_frames(
            x, window_nfft, int(hop_length), int(n_fft), bool(center)
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
