"""mps-spectro: Fast torch-compatible STFT/ISTFT on Apple MPS via custom Metal kernels."""

from mps_spectro.stft import mps_stft_forward as stft
from mps_spectro.istft import mps_istft_forward as istft

__version__ = "0.2.0rc1"
__all__ = ["stft", "istft"]
