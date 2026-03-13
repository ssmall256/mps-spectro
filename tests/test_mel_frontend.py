from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torchaudio

from mps_spectro import (
    CompatMelSpectrogramTransform,
    DynamicMelSpectrogramTransform,
    LogMelSpectrogramTransform,
    MelSpectrogramTransform,
    amplitude_to_db,
    dynamic_mel_spectrogram,
    dynamic_spectrogram,
    mel_spectrogram,
)


def _mamba_amt_reference(device: torch.device) -> tuple[torch.nn.Module, LogMelSpectrogramTransform]:
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        pad_mode="constant",
        n_mels=256,
        f_min=30.0,
        f_max=8000.0,
        window_fn=torch.hann_window,
        power=1.0,
        center=True,
        norm="slaney",
        mel_scale="htk",
    ).to(device)
    ours = LogMelSpectrogramTransform(
        sample_rate=16000,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=256,
        f_min=30.0,
        f_max=8000.0,
        center=True,
        pad_mode="constant",
        power=1.0,
        norm="slaney",
        mel_scale="htk",
        log_amin=1e-5,
        log_mode="clamp",
    ).to(device)
    return ref, ours


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
def test_log_mel_matches_torchaudio_for_mamba_amt_settings(device_name: str) -> None:
    device = torch.device(device_name)
    ref, ours = _mamba_amt_reference(device)
    wav = torch.randn(2, 16000 * 3, device=device, dtype=torch.float32)

    expected = torch.log(torch.clamp(ref(wav), min=1e-5))
    actual = ours(wav)

    assert actual.shape == expected.shape
    max_abs = float((actual - expected).abs().max().item())
    mean_abs = float((actual - expected).abs().mean().item())
    assert max_abs <= 5.0e-4
    assert mean_abs <= 1.0e-5


def test_log_output_modes() -> None:
    mel = torch.tensor([[0.0, 1.0, 10.0]], dtype=torch.float32)
    linear_out = mel_spectrogram(
        torch.randn(1, 64),
        sample_rate=16000,
        n_fft=16,
        hop_length=4,
        n_mels=8,
        output_scale="linear",
    )
    assert linear_out.shape == (1, 8, 17)
    expected_clamp = torch.log(torch.clamp(mel, min=1e-5))
    expected_add = torch.log(mel + 1e-5)
    from mps_spectro.mel import _apply_log_output

    torch.testing.assert_close(_apply_log_output(mel, log_amin=1e-5, log_mode="clamp"), expected_clamp)
    torch.testing.assert_close(_apply_log_output(mel, log_amin=1e-5, log_mode="add"), expected_add)


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
def test_db_output_matches_torchaudio(device_name: str) -> None:
    device = torch.device(device_name)
    wav = torch.randn(2, 24000 * 2, device=device, dtype=torch.float32)
    ours = MelSpectrogramTransform(
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
        power=2.0,
        pad_mode="reflect",
        output_scale="db",
    ).to(device)
    ref_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
    ).to(device)
    ref_db = torchaudio.transforms.AmplitudeToDB().to(device)

    expected = ref_db(ref_mel(wav))
    actual = ours(wav)

    assert actual.shape == expected.shape
    max_abs = float((actual - expected).abs().max().item())
    mean_abs = float((actual - expected).abs().mean().item())
    assert max_abs <= 5.0e-4
    assert mean_abs <= 5.0e-5


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
def test_linear_output_matches_torchaudio_linkseg_like_config(device_name: str) -> None:
    device = torch.device(device_name)
    wav = torch.randn(1, 22050 * 4, device=device, dtype=torch.float32)
    ours = MelSpectrogramTransform(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=0.0,
        f_max=11025.0,
        power=2.0,
        pad_mode="reflect",
        output_scale="linear",
    ).to(device)
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=0.0,
        f_max=11025.0,
        power=2.0,
        pad_mode="reflect",
    ).to(device)

    expected = ref(wav)
    actual = ours(wav)

    assert actual.shape == expected.shape
    rel = (actual - expected).abs() / (expected.abs() + 1e-8)
    max_rel = float(rel.max().item())
    mean_rel = float(rel.mean().item())
    if device_name == "mps":
        assert max_rel <= 5.0e-5
        assert mean_rel <= 5.0e-6
    else:
        assert max_rel <= 1.0e-6
        assert mean_rel <= 1.0e-7


def test_amplitude_to_db_matches_torchaudio() -> None:
    x = torch.rand(2, 128, 100, dtype=torch.float32) * 1e4
    expected = torchaudio.transforms.AmplitudeToDB()(x)
    actual = amplitude_to_db(x, stype="power", top_db=80.0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
def test_compat_matches_linkseg_style_db_frontend(device_name: str) -> None:
    device = torch.device(device_name)
    wav = torch.randn(1, 22050 * 4, device=device, dtype=torch.float32)
    ours = CompatMelSpectrogramTransform(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=64,
        f_min=0.0,
        f_max=11025.0,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
        output_scale="db",
        top_db=None,
    ).to(device)
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=11025.0,
        n_mels=64,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
    ).to(device)
    ref_db = torchaudio.transforms.AmplitudeToDB(top_db=None).to(device)

    expected = ref_db(ref(wav))
    actual = ours(wav)

    assert actual.shape == expected.shape
    max_abs = float((actual - expected).abs().max().item())
    mean_abs = float((actual - expected).abs().mean().item())
    assert max_abs <= 5.0e-4
    assert mean_abs <= 5.0e-5


def _load_rvmpe_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "RVMPE" / "rmvpe.py"
    spec = importlib.util.spec_from_file_location("rvmpe_local_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fcpe_like_reference(
    y: torch.Tensor,
    *,
    mel_basis: torch.Tensor,
    n_fft: int,
    win_size: int,
    hop_length: int,
    key_shift: float,
    speed: float,
) -> torch.Tensor:
    factor = 2 ** (key_shift / 12)
    n_fft_new = int(round(n_fft * factor))
    win_size_new = int(round(win_size * factor))
    hop_length_new = int(round(hop_length * speed))

    pad_left = (win_size_new - hop_length_new) // 2
    pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
    mode = "reflect" if pad_right < y.size(-1) else "constant"
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

    spec = torch.stft(
        y,
        n_fft=n_fft_new,
        hop_length=hop_length_new,
        win_length=win_size_new,
        window=torch.hann_window(win_size_new, device=y.device, dtype=y.dtype),
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    if key_shift != 0:
        size = mel_basis.shape[1]
        resize = spec.size(1)
        if resize < size:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, size - resize))
        spec = spec[:, :size, :] * (win_size / win_size_new)
    return torch.matmul(mel_basis, spec)


def test_dynamic_mel_matches_rvmpe_semantics() -> None:
    module = _load_rvmpe_module()
    reference = module.MelSpectrogram(
        is_half=False,
        n_mel_channels=128,
        sampling_rate=16000,
        win_length=1024,
        hop_length=160,
        n_fft=None,
        mel_fmin=30,
        mel_fmax=8000,
        clamp=1e-5,
    ).cpu()
    audio = torch.randn(2, 16000 * 3, dtype=torch.float32)
    expected = reference(audio, keyshift=3, speed=1.2, center=True)
    actual = dynamic_mel_spectrogram(
        audio,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        center=True,
        pad_mode="reflect",
        power=1.0,
        output_scale="log",
        log_amin=1e-5,
        log_mode="clamp",
        keyshift=3,
        speed=1.2,
        mel_basis=reference.mel_basis,
        use_mps_kernels=False,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_dynamic_spectrogram_matches_fcpe_like_semantics() -> None:
    audio = torch.randn(2, 16000 * 2, dtype=torch.float32)
    mel_basis = torch.randn(128, 1025, dtype=torch.float32)
    expected = _fcpe_like_reference(
        audio,
        mel_basis=mel_basis,
        n_fft=2048,
        win_size=2048,
        hop_length=160,
        key_shift=2,
        speed=1.0,
    )

    factor = 2 ** (2 / 12)
    win_size_new = int(round(2048 * factor))
    hop_length_new = int(round(160 * 1.0))
    pad_left = (win_size_new - hop_length_new) // 2
    pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - audio.size(-1) - pad_left)
    mode = "reflect" if pad_right < audio.size(-1) else "constant"
    padded = torch.nn.functional.pad(audio.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

    actual = dynamic_spectrogram(
        padded,
        n_fft=2048,
        hop_length=160,
        win_length=2048,
        center=False,
        pad_mode="constant",
        power=1.0,
        keyshift=2,
        speed=1.0,
        target_n_freqs=mel_basis.shape[1],
        magnitude_eps=1e-9,
        use_mps_kernels=False,
    )
    actual = torch.matmul(mel_basis, actual)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_dynamic_transform_matches_functional() -> None:
    audio = torch.randn(1, 16000, dtype=torch.float32)
    mel_basis = torch.randn(128, 513, dtype=torch.float32)
    transform = DynamicMelSpectrogramTransform(
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        output_scale="log",
        log_amin=1e-5,
        mel_basis=mel_basis,
        use_mps_kernels=False,
    )
    expected = dynamic_mel_spectrogram(
        audio,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        output_scale="log",
        log_amin=1e-5,
        keyshift=-2,
        speed=1.1,
        mel_basis=mel_basis,
        use_mps_kernels=False,
    )
    actual = transform(audio, keyshift=-2, speed=1.1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
