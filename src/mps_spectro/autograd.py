"""Autograd support for mps-spectro custom Metal kernels.

Wraps the forward Metal kernels with torch.autograd.Function so that
gradients flow through stft() and istft().  Backward passes use custom
Metal kernels for GPU-accelerated gradient computation.

Limitations (v1):
  - Window gradients are not computed (returns None).  Windows are
    almost always frozen in practice.
  - Only float32 is supported for autograd.
  - ISTFT autograd forces native layout.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# STFT autograd
# ---------------------------------------------------------------------------


class STFTExtractFrames(torch.autograd.Function):
    """Differentiable wrapper around the Metal ``mps_stft_extract_frames`` kernel.

    Forward: calls the Metal kernel (fast).
    Backward w.r.t. input: Metal kernel that gathers from grad_frames for each
      input position, handling reflect-pad contributions.
    Backward w.r.t. window: not implemented (returns None).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,     # [B, T]
        window: torch.Tensor,    # [n_fft]
        hop_length: int,
        n_fft: int,
        center: bool,
    ) -> torch.Tensor:
        from mps_spectro import compiler

        frames = compiler.mps_stft_extract_frames(
            input, window, int(hop_length), int(n_fft), bool(center),
        )
        # Save what we need for backward.
        ctx.save_for_backward(window)
        ctx.input_length = input.size(1)
        ctx.hop_length = hop_length
        ctx.n_fft = n_fft
        ctx.center = center
        return frames  # [B, n_frames, n_fft]

    @staticmethod
    def backward(ctx, grad_frames: torch.Tensor):
        (window,) = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            from mps_spectro import compiler

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


# ---------------------------------------------------------------------------
# ISTFT autograd
# ---------------------------------------------------------------------------


class ISTFTOverlapAdd(torch.autograd.Function):
    """Differentiable wrapper around the Metal ISTFT overlap-add kernel.

    Forward:
        y[b, t] = (Σ_f frames[b, j, f] * window[j]) / wsum[t]
        where j = t - f * hop,  wsum[t] = Σ_f window_sq[j]

    Backward w.r.t. frames: Metal kernel that computes
        grad_frames[b,j,f] = grad_output[b, f*hop+j] * window[j] / wsum[f*hop+j]
    Backward w.r.t. window / window_sq: not implemented (returns None).
    """

    @staticmethod
    def forward(
        ctx,
        frames: torch.Tensor,      # [B, W, N]  (native layout)
        window: torch.Tensor,       # [W]
        window_sq: torch.Tensor,    # [W]
        hop_length: int,
        output_length: int,
    ) -> torch.Tensor:
        from mps_spectro import compiler

        y = compiler.mps_istft_overlap_add_div_envelope(
            frames, window, window_sq, int(hop_length), int(output_length),
        )
        ctx.save_for_backward(window, window_sq)
        ctx.hop_length = hop_length
        ctx.output_length = output_length
        ctx.frames_shape = frames.shape  # (B, W, N)
        return y  # [B, output_length]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        window, window_sq = ctx.saved_tensors
        hop = ctx.hop_length
        output_length = ctx.output_length
        B, W, N = ctx.frames_shape

        grad_frames = None
        if ctx.needs_input_grad[0]:
            from mps_spectro import compiler

            grad_frames = compiler.mps_istft_backward_frames(
                grad_output.contiguous(),
                window,
                window_sq,
                N,
                hop,
            )

        # No grad for window, window_sq, hop_length, output_length.
        return grad_frames, None, None, None, None
