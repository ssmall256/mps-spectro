"""Metal kernel compilation and dispatch via ``torch.mps.compile_shader``.

Lazily compiles the Metal shader source on first use and provides dispatch
wrapper functions that match the previous C++ extension API.
"""

from __future__ import annotations

import torch

from mps_spectro.metal._shaders import METAL_SOURCE

# ---------------------------------------------------------------------------
# Lazy shader compilation (singleton)
# ---------------------------------------------------------------------------

_LIB = None


def _get_library():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(METAL_SOURCE)
    return _LIB


# ---------------------------------------------------------------------------
# Shared memory limit for tiled STFT kernel (32KB = 8192 floats).
# ---------------------------------------------------------------------------

_STFT_THREADGROUP_MEM_LIMIT = 32768


# ---------------------------------------------------------------------------
# ISTFT overlap-add dispatch
# ---------------------------------------------------------------------------


def mps_istft_overlap_add_div_envelope(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    """ISTFT overlap-add with envelope normalization — native layout [B, W, N]."""
    lib = _get_library()
    B, W, N = frames.shape
    output = torch.zeros(B, output_length, device=frames.device, dtype=frames.dtype)
    kernel_name = "istft_overlap_add_div_envelope_half" if frames.dtype == torch.float16 else "istft_overlap_add_div_envelope_float"
    kernel = getattr(lib, kernel_name)
    kernel(frames, window, window_sq, output, B, N, W, hop_length, output_length,
           threads=(B * output_length,))
    return output


def mps_istft_overlap_add_div_envelope_mixed(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    """ISTFT overlap-add — mixed precision (half frames, float window/output)."""
    lib = _get_library()
    B, W, N = frames.shape
    output = torch.zeros(B, output_length, device=frames.device, dtype=torch.float32)
    lib.istft_overlap_add_div_envelope_mixed(
        frames, window, window_sq, output, B, N, W, hop_length, output_length,
        threads=(B * output_length,))
    return output


def mps_istft_overlap_add_div_envelope_transposed(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    """ISTFT overlap-add — transposed layout [B, N, W]."""
    lib = _get_library()
    B, N, W = frames.shape
    output = torch.zeros(B, output_length, device=frames.device, dtype=frames.dtype)
    kernel_name = "istft_overlap_add_div_envelope_half_t" if frames.dtype == torch.float16 else "istft_overlap_add_div_envelope_float_t"
    kernel = getattr(lib, kernel_name)
    kernel(frames, window, window_sq, output, B, N, W, hop_length, output_length,
           threads=(B * output_length,))
    return output


def mps_istft_overlap_add_div_envelope_mixed_transposed(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    """ISTFT overlap-add — mixed precision, transposed layout [B, N, W]."""
    lib = _get_library()
    B, N, W = frames.shape
    output = torch.zeros(B, output_length, device=frames.device, dtype=torch.float32)
    lib.istft_overlap_add_div_envelope_mixed_t(
        frames, window, window_sq, output, B, N, W, hop_length, output_length,
        threads=(B * output_length,))
    return output


# ---------------------------------------------------------------------------
# STFT extract frames dispatch
# ---------------------------------------------------------------------------


def mps_stft_extract_frames(
    input: torch.Tensor,
    window: torch.Tensor,
    hop_length: int,
    n_fft: int,
    center: bool,
) -> torch.Tensor:
    """Fused reflect-pad + windowed frame extraction."""
    lib = _get_library()
    batch_size = input.size(0)
    input_length = input.size(1)
    pad = n_fft // 2 if center else 0
    padded_length = input_length + 2 * pad
    n_frames = (padded_length - n_fft) // hop_length + 1

    output = torch.empty(batch_size, n_frames, n_fft, device=input.device, dtype=input.dtype)

    # Decide whether to use the tiled (shared memory) kernel.
    max_tile_frames = (_STFT_THREADGROUP_MEM_LIMIT // 4 - n_fft) // hop_length + 1
    use_tiled = (max_tile_frames >= 2) and (n_frames >= 4)

    if use_tiled:
        tile_frames = min(max_tile_frames, n_frames)
        shared_span = hop_length * (tile_frames - 1) + n_fft
        n_tile_groups = (n_frames + tile_frames - 1) // tile_frames

        lib.stft_extract_frames_tiled_float(
            input, window, output,
            batch_size, input_length, n_fft, hop_length, n_frames, pad,
            tile_frames, shared_span,
            threads=(n_fft, n_tile_groups, batch_size),
        )
    else:
        lib.stft_extract_frames_float(
            input, window, output,
            batch_size, input_length, n_fft, hop_length, n_frames, pad,
            threads=(n_fft, n_frames, batch_size),
        )

    return output


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------


def mps_stft_backward_input(
    grad_frames: torch.Tensor,
    window: torch.Tensor,
    input_length: int,
    hop_length: int,
    n_fft: int,
    center: bool,
) -> torch.Tensor:
    """STFT backward: grad_input from grad_frames."""
    lib = _get_library()
    batch_size = grad_frames.size(0)
    n_frames = grad_frames.size(1)
    pad = n_fft // 2 if center else 0

    grad_input = torch.zeros(batch_size, input_length, device=grad_frames.device, dtype=grad_frames.dtype)
    lib.stft_backward_input_float(
        grad_frames, window, grad_input,
        batch_size, input_length, n_fft, hop_length, n_frames, pad,
        threads=(batch_size * input_length,),
    )
    return grad_input


def mps_istft_backward_frames(
    grad_output: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    n_frames: int,
    hop_length: int,
) -> torch.Tensor:
    """ISTFT backward: grad_frames from grad_output."""
    lib = _get_library()
    batch_size = grad_output.size(0)
    output_length = grad_output.size(1)
    win_length = window.size(0)

    grad_frames = torch.empty(batch_size, win_length, n_frames, device=grad_output.device, dtype=grad_output.dtype)
    lib.istft_backward_frames_float(
        grad_output, window, window_sq, grad_frames,
        batch_size, n_frames, win_length, hop_length, output_length,
        threads=(win_length, n_frames, batch_size),
    )
    return grad_frames
