import pytest
import torch

from mps_spectro.istft import mps_istft_forward


def _assert_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")


def _make_valid_spec(
    *,
    batch: int,
    n_fft: int,
    hop: int,
    n_frames: int,
    center: bool,
    device: torch.device,
    window_kind: str = "hann",
) -> tuple[torch.Tensor, torch.Tensor]:
    length = hop * (n_frames - 1) + n_fft
    x = torch.randn(batch, length, device=device, dtype=torch.float32)
    if window_kind == "hann":
        window = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)
    elif window_kind == "ones":
        window = torch.ones(n_fft, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported window_kind: {window_kind}")
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spec, window


def _tolerances(kernel_dtype: str) -> tuple[float, float]:
    if kernel_dtype == "float32":
        return (1.0e-5, 1.0e-7)
    if kernel_dtype == "mixed":
        return (2.0e-3, 3.0e-7)
    return (9.0e-3, 1.5e-6)


@pytest.mark.parametrize("kernel_dtype", ["float32", "float16", "mixed"])
@pytest.mark.parametrize("kernel_layout", ["native", "transposed", "auto"])
def test_parity_center_true_trimmed(kernel_dtype: str, kernel_layout: str) -> None:
    _assert_mps()
    device = torch.device("mps")
    batch, n_fft, hop, n_frames = 1, 512, 256, 64
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=True,
        device=device,
    )

    ref = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    out = mps_istft_forward(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        kernel_dtype=kernel_dtype,
        kernel_layout=kernel_layout,
    ).to(torch.float32)

    atol, mse_max = _tolerances(kernel_dtype)
    max_abs = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()
    assert max_abs <= atol
    assert mse <= mse_max


@pytest.mark.parametrize("kernel_dtype", ["float32", "float16", "mixed"])
@pytest.mark.parametrize("kernel_layout", ["native", "transposed", "auto"])
def test_parity_center_true_long_length(kernel_dtype: str, kernel_layout: str) -> None:
    _assert_mps()
    device = torch.device("mps")
    batch, n_fft, hop, n_frames = 1, 512, 256, 64
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=True,
        device=device,
    )
    trimmed = hop * (n_frames - 1)
    target_length = trimmed + hop

    ref = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
        length=target_length,
    )
    out = mps_istft_forward(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=target_length,
        kernel_dtype=kernel_dtype,
        kernel_layout=kernel_layout,
    ).to(torch.float32)

    atol, mse_max = _tolerances(kernel_dtype)
    max_abs = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()
    assert max_abs <= atol
    assert mse <= mse_max


@pytest.mark.parametrize("kernel_dtype", ["float32", "float16", "mixed"])
@pytest.mark.parametrize("kernel_layout", ["native", "transposed", "auto"])
def test_parity_center_false_length_clip(kernel_dtype: str, kernel_layout: str) -> None:
    _assert_mps()
    device = torch.device("mps")
    batch, n_fft, hop, n_frames = 1, 512, 256, 64
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=False,
        device=device,
        window_kind="ones",
    )
    full = hop * (n_frames - 1) + n_fft
    target_length = full - hop

    ref = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False,
        length=target_length,
    )
    out = mps_istft_forward(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        length=target_length,
        kernel_dtype=kernel_dtype,
        kernel_layout=kernel_layout,
    ).to(torch.float32)

    atol, mse_max = _tolerances(kernel_dtype)
    max_abs = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()
    assert max_abs <= atol
    assert mse <= mse_max
