"""Tests for autograd support in mps-spectro STFT and ISTFT."""

import pytest
import torch

from mps_spectro.stft import mps_stft_forward
from mps_spectro.istft import mps_istft_forward


def _assert_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


# ---------------------------------------------------------------------------
# STFT backward
# ---------------------------------------------------------------------------


class TestSTFTBackward:
    """Basic backward-pass tests for stft."""

    def test_stft_backward_exists(self) -> None:
        """Gradient is non-None and finite after backward through stft."""
        _assert_mps()
        x = torch.randn(2, 4096, device="mps", requires_grad=True)
        spec = mps_stft_forward(x, n_fft=512, hop_length=128)
        loss = spec.abs().pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0  # non-trivial gradient

    def test_stft_backward_small_signal(self) -> None:
        """Backward works for small signals (torch.stft fallback path)."""
        _assert_mps()
        x = torch.randn(1, 512, device="mps", requires_grad=True)
        spec = mps_stft_forward(x, n_fft=256, hop_length=64)
        loss = spec.abs().pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_stft_backward_no_center(self) -> None:
        """Backward works with center=False (no reflect padding)."""
        _assert_mps()
        x = torch.randn(2, 4096, device="mps", requires_grad=True)
        spec = mps_stft_forward(x, n_fft=512, hop_length=128, center=False)
        loss = spec.abs().pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# ISTFT backward
# ---------------------------------------------------------------------------


class TestISTFTBackward:
    """Basic backward-pass tests for istft."""

    def test_istft_backward_exists(self) -> None:
        """Gradient is non-None and finite after backward through istft."""
        _assert_mps()
        # Create a spectrogram with requires_grad.
        x = torch.randn(2, 4096, device="mps")
        spec = torch.stft(
            x,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window=torch.hann_window(512, device="mps"),
            center=True,
            return_complex=True,
        )
        spec = spec.detach().requires_grad_(True)

        y = mps_istft_forward(spec, n_fft=512, hop_length=128, center=True, length=4096)
        loss = y.pow(2).sum()
        loss.backward()

        assert spec.grad is not None
        assert torch.isfinite(spec.grad).all()
        assert spec.grad.abs().sum() > 0

    def test_istft_backward_no_center(self) -> None:
        """Backward works with center=False."""
        _assert_mps()
        spec = torch.randn(2, 257, 16, device="mps", dtype=torch.cfloat, requires_grad=True)
        y = mps_istft_forward(spec, n_fft=512, hop_length=128, center=False)
        loss = y.pow(2).sum()
        loss.backward()

        assert spec.grad is not None
        assert torch.isfinite(spec.grad).all()


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtripGrad:
    """Test gradient flow through STFT → ISTFT roundtrip."""

    def test_roundtrip_grad(self) -> None:
        """Full roundtrip: x → stft → istft → loss → backward."""
        _assert_mps()
        x = torch.randn(2, 8000, device="mps", requires_grad=True)

        spec = mps_stft_forward(x, n_fft=512, hop_length=128)
        y = mps_istft_forward(spec, n_fft=512, hop_length=128, center=True, length=8000)

        loss = (y - x.detach()).pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    def test_roundtrip_grad_large_batch(self) -> None:
        """Roundtrip with B=4 (exercises Metal kernel path for STFT)."""
        _assert_mps()
        # B=4 * 160k * 1024 * 4 bytes = ~2.5 GB output → well above 5MB threshold
        x = torch.randn(4, 16000, device="mps", requires_grad=True)

        spec = mps_stft_forward(x, n_fft=512, hop_length=128)
        y = mps_istft_forward(spec, n_fft=512, hop_length=128, center=True, length=16000)

        loss = y.pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Zero overhead when requires_grad=False
# ---------------------------------------------------------------------------


class TestNoGradOverhead:
    """Verify that requires_grad=False doesn't change behaviour."""

    def test_stft_no_grad(self) -> None:
        _assert_mps()
        x = torch.randn(2, 4096, device="mps", requires_grad=False)
        spec = mps_stft_forward(x, n_fft=512, hop_length=128)
        assert spec.grad_fn is None  # no autograd graph

    def test_istft_no_grad(self) -> None:
        _assert_mps()
        spec = torch.randn(2, 257, 16, device="mps", dtype=torch.cfloat, requires_grad=False)
        y = mps_istft_forward(spec, n_fft=512, hop_length=128, center=True)
        assert y.grad_fn is None


# ---------------------------------------------------------------------------
# Numerical gradient check
# ---------------------------------------------------------------------------


class TestGradcheck:
    """Use torch.autograd.gradcheck to verify backward correctness."""

    @pytest.mark.parametrize("center", [True, False])
    def test_stft_gradcheck(self, center: bool) -> None:
        """Numerical gradient check for STFT."""
        _assert_mps()
        # Small problem size for speed; float32 with reduced tolerance
        # (Metal kernels are float32-only so we can't use float64).
        x = torch.randn(1, 512, device="mps", dtype=torch.float32, requires_grad=True)

        def fn(inp):
            return mps_stft_forward(inp, n_fft=128, hop_length=32, center=center).abs()

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-3, atol=2e-2, rtol=2e-2,
            nondet_tol=2e-3,
        )

    def test_istft_gradcheck(self) -> None:
        """Numerical gradient check for ISTFT."""
        _assert_mps()
        # Small spectrogram.
        spec = torch.randn(1, 65, 8, device="mps", dtype=torch.cfloat)
        spec = spec.detach().requires_grad_(True)

        def fn(s):
            return mps_istft_forward(s, n_fft=128, hop_length=32, center=True)

        assert torch.autograd.gradcheck(
            fn, (spec,), eps=1e-3, atol=1e-2, rtol=1e-2,
            nondet_tol=1e-3,
        )
