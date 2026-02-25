"""Tests that mps-spectro custom ops work under torch.compile(backend='aot_eager').

Verifies that:
  1. The Meta (FakeTensor) kernels produce correct output shapes for tracing.
  2. Forward execution through the compiled graph produces valid results.
  3. Backward execution through the compiled graph produces valid gradients.
"""

import pytest
import torch

import mps_spectro

N_FFT = 2048
HOP = 512
DEVICE = "mps"

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS device required",
)


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------


def test_compile_stft_forward():
    """torch.compile traces through stft_extract_frames forward."""

    @torch.compile(backend="aot_eager")
    def f(x, window):
        return torch.ops.mps_spectro.stft_extract_frames(x, window, HOP, N_FFT, True)

    x = torch.randn(2, 44100, device=DEVICE)
    window = torch.hann_window(N_FFT, device=DEVICE)
    out = f(x, window)
    expected_frames = (44100 + 2 * (N_FFT // 2) - N_FFT) // HOP + 1
    assert out.shape == (2, expected_frames, N_FFT)
    assert out.isfinite().all()


def test_compile_stft_backward():
    """torch.compile backward traces through stft_extract_frames autograd."""

    @torch.compile(backend="aot_eager")
    def f(x, window):
        frames = torch.ops.mps_spectro.stft_extract_frames(x, window, HOP, N_FFT, True)
        return frames.sum()

    x = torch.randn(2, 44100, device=DEVICE, requires_grad=True)
    window = torch.hann_window(N_FFT, device=DEVICE)
    loss = f(x, window)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.isfinite().all()


def test_compile_stft_center_false():
    """torch.compile traces through stft_extract_frames with center=False."""

    @torch.compile(backend="aot_eager")
    def f(x, window):
        return torch.ops.mps_spectro.stft_extract_frames(x, window, HOP, N_FFT, False)

    x = torch.randn(2, 44100, device=DEVICE)
    window = torch.hann_window(N_FFT, device=DEVICE)
    out = f(x, window)
    expected_frames = (44100 - N_FFT) // HOP + 1
    assert out.shape == (2, expected_frames, N_FFT)


# ---------------------------------------------------------------------------
# ISTFT
# ---------------------------------------------------------------------------


def test_compile_istft_forward():
    """torch.compile traces through istft_overlap_add forward."""
    n_frames = 87

    @torch.compile(backend="aot_eager")
    def f(frames, window, window_sq):
        return torch.ops.mps_spectro.istft_overlap_add(frames, window, window_sq, HOP, 44100)

    frames = torch.randn(2, N_FFT, n_frames, device=DEVICE)
    window = torch.hann_window(N_FFT, device=DEVICE)
    window_sq = window * window
    out = f(frames, window, window_sq)
    assert out.shape == (2, 44100)
    assert out.isfinite().all()


def test_compile_istft_backward():
    """torch.compile backward traces through istft_overlap_add autograd."""
    n_frames = 87

    @torch.compile(backend="aot_eager")
    def f(frames, window, window_sq):
        y = torch.ops.mps_spectro.istft_overlap_add(frames, window, window_sq, HOP, 44100)
        return y.sum()

    frames = torch.randn(2, N_FFT, n_frames, device=DEVICE, requires_grad=True)
    window = torch.hann_window(N_FFT, device=DEVICE)
    window_sq = window * window
    loss = f(frames, window, window_sq)
    loss.backward()
    assert frames.grad is not None
    assert frames.grad.shape == frames.shape
    assert frames.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Full public API under torch.compile
# ---------------------------------------------------------------------------


def test_compile_public_stft():
    """torch.compile through the public mps_spectro.stft() including FFT."""

    @torch.compile(backend="aot_eager")
    def f(x):
        return mps_spectro.stft(x, n_fft=N_FFT, hop_length=HOP, center=True)

    x = torch.randn(2, 44100, device=DEVICE)
    out = f(x)
    assert out.is_complex()
    assert out.dim() == 3


def test_ops_registered():
    """Verify custom ops are registered in the mps_spectro namespace."""
    assert hasattr(torch.ops, "mps_spectro")
    assert hasattr(torch.ops.mps_spectro, "stft_extract_frames")
    assert hasattr(torch.ops.mps_spectro, "istft_overlap_add")
