import pytest
import torch

from mps_spectro.stft import mps_stft_forward


def _assert_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")


@pytest.mark.parametrize(
    "batch,n_fft,hop,n_frames,center,normalized,onesided",
    [
        (1, 512, 256, 64, True, False, True),
        (1, 512, 256, 64, True, False, False),
        (4, 1024, 256, 48, False, True, True),
        (4, 1024, 256, 48, False, True, False),
    ],
)
def test_mps_stft_matches_torch(
    batch: int,
    n_fft: int,
    hop: int,
    n_frames: int,
    center: bool,
    normalized: bool,
    onesided: bool,
) -> None:
    _assert_mps()
    device = torch.device("mps")
    if center:
        wav_len = max(1, hop * (n_frames - 1))
    else:
        wav_len = max(n_fft, n_fft + hop * (n_frames - 1))
    wav = torch.randn(batch, wav_len, device=device, dtype=torch.float32)
    window = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)

    ref = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    out = mps_stft_forward(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )

    diff = out - ref
    assert float(diff.abs().max().item()) <= 1.0e-5
    assert float(torch.mean(diff.abs() ** 2).item()) <= 1.0e-8


@pytest.mark.parametrize(
    "batch,n_fft,hop,wav_len,center",
    [
        (1, 256, 64, 8000, True),
        (1, 512, 128, 16000, True),
        (1, 1024, 256, 160000, True),
        (2, 2048, 512, 160000, True),
        (4, 1024, 256, 160000, False),
        (1, 512, 256, 600, True),   # short signal, tests reflect pad edge
    ],
)
def test_mps_stft_metal_kernel_parity(
    batch: int,
    n_fft: int,
    hop: int,
    wav_len: int,
    center: bool,
) -> None:
    """Verify the fused Metal frame-extraction kernel matches torch.stft."""
    _assert_mps()
    device = torch.device("mps")
    wav = torch.randn(batch, wav_len, device=device, dtype=torch.float32)
    window = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)

    ref = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    out = mps_stft_forward(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
    )

    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    max_abs = float((out - ref).abs().max().item())
    assert max_abs <= 1.0e-5, f"max abs error {max_abs}"


def test_mps_stft_preserves_1d_shape() -> None:
    _assert_mps()
    device = torch.device("mps")
    n_fft = 256
    hop = 128
    wav = torch.randn(hop * 16, device=device, dtype=torch.float32)
    out = mps_stft_forward(wav, n_fft=n_fft, hop_length=hop)
    assert out.dim() == 2
