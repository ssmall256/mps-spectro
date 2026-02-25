"""Register mps-spectro operations as torch.library custom ops.

This enables torch.compile to trace through STFT/iSTFT calls by providing:
  1. Custom op definitions with proper schemas
  2. Fake (meta) tensor implementations for shape inference
  3. Autograd formulas for backward pass

The registered ops are the two Metal kernel primitives:
  - stft_extract_frames: fused reflect-pad + strided frame extraction + windowing
  - istft_overlap_add: fused overlap-add + envelope normalization

Limitations (carried from autograd.py):
  - Window gradients are not computed (returns None).
  - Only float32 is supported for autograd backward.
  - ISTFT autograd forces native layout [B, W, N].
"""

from __future__ import annotations

import torch

from mps_spectro import compiler

# ---------------------------------------------------------------------------
# Library definition
# ---------------------------------------------------------------------------

MPS_SPECTRO_LIB = torch.library.Library("mps_spectro", "DEF")

# ---------------------------------------------------------------------------
# stft_extract_frames
# ---------------------------------------------------------------------------

MPS_SPECTRO_LIB.define(
    "stft_extract_frames(Tensor input, Tensor window, int hop_length, "
    "int n_fft, bool center) -> Tensor"
)


@torch.library.impl(MPS_SPECTRO_LIB, "stft_extract_frames", "MPS")
def _stft_fwd_mps(
    input: torch.Tensor,
    window: torch.Tensor,
    hop_length: int,
    n_fft: int,
    center: bool,
) -> torch.Tensor:
    return compiler.mps_stft_extract_frames(input, window, hop_length, n_fft, center)


@torch.library.impl(MPS_SPECTRO_LIB, "stft_extract_frames", "Meta")
def _stft_fwd_meta(
    input: torch.Tensor,
    window: torch.Tensor,
    hop_length: int,
    n_fft: int,
    center: bool,
) -> torch.Tensor:
    B, T = input.shape
    pad = n_fft // 2 if center else 0
    n_frames = (T + 2 * pad - n_fft) // hop_length + 1
    return input.new_empty(B, n_frames, n_fft)


# ---------------------------------------------------------------------------
# istft_overlap_add
# ---------------------------------------------------------------------------

MPS_SPECTRO_LIB.define(
    "istft_overlap_add(Tensor frames, Tensor window, Tensor window_sq, "
    "int hop_length, int output_length) -> Tensor"
)


@torch.library.impl(MPS_SPECTRO_LIB, "istft_overlap_add", "MPS")
def _istft_fwd_mps(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    return compiler.mps_istft_overlap_add_div_envelope(
        frames, window, window_sq, hop_length, output_length,
    )


@torch.library.impl(MPS_SPECTRO_LIB, "istft_overlap_add", "Meta")
def _istft_fwd_meta(
    frames: torch.Tensor,
    window: torch.Tensor,
    window_sq: torch.Tensor,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    B = frames.shape[0]
    return frames.new_empty(B, output_length)


# ---------------------------------------------------------------------------
# Autograd: stft_extract_frames
# ---------------------------------------------------------------------------


def _setup_stft_ctx(ctx, inputs, output):
    input, window, hop_length, n_fft, center = inputs
    ctx.save_for_backward(window)
    ctx.input_length = input.size(1)
    ctx.hop_length = hop_length
    ctx.n_fft = n_fft
    ctx.center = center


def _stft_backward(ctx, grad_frames):
    (window,) = ctx.saved_tensors
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = compiler.mps_stft_backward_input(
            grad_frames.contiguous(),
            window,
            ctx.input_length,
            ctx.hop_length,
            ctx.n_fft,
            ctx.center,
        )
    # No grad for window, hop_length, n_fft, center.
    return grad_input, None, None, None, None


torch.library.register_autograd(
    "mps_spectro::stft_extract_frames",
    _stft_backward,
    setup_context=_setup_stft_ctx,
)


# ---------------------------------------------------------------------------
# Autograd: istft_overlap_add
# ---------------------------------------------------------------------------


def _setup_istft_ctx(ctx, inputs, output):
    frames, window, window_sq, hop_length, output_length = inputs
    ctx.save_for_backward(window, window_sq)
    ctx.hop_length = hop_length
    ctx.output_length = output_length
    ctx.frames_shape = frames.shape  # (B, W, N)


def _istft_backward(ctx, grad_output):
    window, window_sq = ctx.saved_tensors
    B, W, N = ctx.frames_shape

    grad_frames = None
    if ctx.needs_input_grad[0]:
        grad_frames = compiler.mps_istft_backward_frames(
            grad_output.contiguous(),
            window,
            window_sq,
            N,
            ctx.hop_length,
        )
    # No grad for window, window_sq, hop_length, output_length.
    return grad_frames, None, None, None, None


torch.library.register_autograd(
    "mps_spectro::istft_overlap_add",
    _istft_backward,
    setup_context=_setup_istft_ctx,
)
