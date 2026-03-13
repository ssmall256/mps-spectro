from __future__ import annotations

import pytest
import torch
import torchaudio

from mps_spectro import LogMelSpectrogramTransform, MelSpectrogramTransform, amplitude_to_db, mel_spectrogram


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
