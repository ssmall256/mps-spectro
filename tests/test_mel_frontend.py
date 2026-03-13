from __future__ import annotations

import pytest
import torch
import torchaudio

from mps_spectro import LogMelSpectrogramTransform, MelSpectrogramTransform, mel_spectrogram


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
