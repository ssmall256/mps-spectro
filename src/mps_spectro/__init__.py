"""mps-spectro: Fast torch-compatible STFT/ISTFT on Apple MPS via custom Metal kernels."""

import mps_spectro._torch_ops as _torch_ops  # noqa: F401 — register custom ops

from mps_spectro.mel import (
    CompatMelSpectrogramTransform,
    DynamicMelSpectrogramTransform,
    LogMelSpectrogramTransform,
    MelSpectrogramTransform,
    amplitude_to_db,
    dynamic_mel_spectrogram,
    dynamic_spectrogram,
    mel_spectrogram,
    melscale_fbanks,
)
from mps_spectro.stft import mps_stft_forward as stft
from mps_spectro.istft import mps_istft_forward as istft

__version__ = "0.3.0"
__all__ = [
    "stft",
    "istft",
    "CompatMelSpectrogramTransform",
    "DynamicMelSpectrogramTransform",
    "MelSpectrogramTransform",
    "LogMelSpectrogramTransform",
    "dynamic_spectrogram",
    "dynamic_mel_spectrogram",
    "mel_spectrogram",
    "melscale_fbanks",
    "amplitude_to_db",
]
