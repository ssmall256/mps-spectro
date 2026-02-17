import os
import time

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
    onesided: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if center:
        length = max(1, hop * (n_frames - 1))
    else:
        length = max(n_fft, hop * (n_frames - 1) + n_fft)
    x = torch.randn(batch, length, device=device, dtype=torch.float32)
    window = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )
    return spec, window


def _length_for_mode(*, mode: str, hop: int, n_fft: int, n_frames: int) -> int | None:
    if mode == "none":
        return None
    trimmed = hop * (n_frames - 1)
    if mode == "trimmed":
        return trimmed
    if mode == "long":
        return trimmed + hop
    raise ValueError(f"Unsupported mode: {mode}")


def _atol_for_dtype(kernel_dtype: str) -> float:
    if kernel_dtype == "float32":
        return 1.0e-5
    if kernel_dtype == "mixed":
        return 3.0e-3
    raise ValueError(f"Unsupported kernel dtype: {kernel_dtype}")


@pytest.mark.parametrize(
    "batch,n_fft,hop,n_frames,length_mode,kernel_dtype",
    [
        (1, 512, 256, 64, "none", "float32"),
        (1, 512, 256, 64, "trimmed", "float32"),
        (1, 512, 256, 64, "long", "float32"),
        (4, 1024, 512, 64, "trimmed", "mixed"),
    ],
)
def test_fused_auto_regression_matrix(
    batch: int,
    n_fft: int,
    hop: int,
    n_frames: int,
    length_mode: str,
    kernel_dtype: str,
) -> None:
    _assert_mps()
    device = torch.device("mps")
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=True,
        device=device,
    )
    length = _length_for_mode(mode=length_mode, hop=hop, n_fft=n_fft, n_frames=n_frames)

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
        length=length,
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
        length=length,
        allow_fused=True,
        kernel_layout="auto",
        kernel_dtype=kernel_dtype,
    ).to(torch.float32)

    max_abs = (out - ref).abs().max().item()
    assert max_abs <= _atol_for_dtype(kernel_dtype)


def test_true_long_mode_matches_torch() -> None:
    _assert_mps()
    device = torch.device("mps")
    batch, n_fft, hop, n_frames = 1, 512, 128, 65
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=True,
        device=device,
    )

    trimmed = hop * (int(spec.size(-1)) - 1)
    length = trimmed + hop
    assert length > trimmed

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
        length=length,
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
        length=length,
        allow_fused=True,
        kernel_layout="auto",
        kernel_dtype="float32",
        long_mode_strategy="custom",
    )

    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-5


def test_dualsided_matches_torch() -> None:
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
        onesided=False,
    )
    target_length = hop * (n_frames - 1)

    ref = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=False,
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
        onesided=False,
        length=target_length,
        allow_fused=True,
        kernel_layout="auto",
        kernel_dtype="float32",
    )

    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-5


def test_dualsided_infers_nfft_matches_torch() -> None:
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
        onesided=False,
    )
    target_length = hop * (n_frames - 1)

    ref = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=False,
        return_complex=False,
        length=target_length,
    )
    out = mps_istft_forward(
        spec,
        hop_length=hop,
        win_length=None,
        window=window,
        center=True,
        normalized=False,
        onesided=False,
        length=target_length,
        allow_fused=True,
        kernel_layout="auto",
        kernel_dtype="float32",
    )

    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-5


@pytest.mark.perf
def test_perf_smoke_fused_auto_nonblocking() -> None:
    _assert_mps()
    if os.getenv("MPS_ISTFT_RUN_PERF_SMOKE", "0") != "1":
        pytest.skip("Set MPS_ISTFT_RUN_PERF_SMOKE=1 to run perf smoke")

    device = torch.device("mps")
    batch, n_fft, hop, n_frames = 1, 1024, 512, 128
    spec, window = _make_valid_spec(
        batch=batch,
        n_fft=n_fft,
        hop=hop,
        n_frames=n_frames,
        center=True,
        device=device,
    )
    target_length = hop * (n_frames - 1)

    def _run_custom() -> None:
        _ = mps_istft_forward(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=target_length,
            allow_fused=True,
            kernel_layout="auto",
            kernel_dtype="float32",
        )

    def _run_torch() -> None:
        _ = torch.istft(
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

    for _ in range(4):
        _run_custom()
        _run_torch()
    torch.mps.synchronize()

    def _median_ms(fn) -> float:
        trials_ms: list[float] = []
        for _ in range(12):
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            trials_ms.append((time.perf_counter() - t0) * 1e3)
        trials_ms.sort()
        return trials_ms[len(trials_ms) // 2]

    custom_ms = _median_ms(_run_custom)
    torch_ms = _median_ms(_run_torch)

    max_slowdown = float(os.getenv("MPS_ISTFT_PERF_MAX_SLOWDOWN", "1.35"))
    slowdown = custom_ms / max(torch_ms, 1e-9)
    if slowdown > max_slowdown:
        pytest.xfail(
            f"Perf smoke threshold miss: custom={custom_ms:.3f}ms torch={torch_ms:.3f}ms slowdown={slowdown:.3f}x"
        )
