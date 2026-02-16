# mps-spectro

Fast `torch.stft` / `torch.istft` replacements for Apple MPS, powered by custom Metal kernels.

- Fused overlap-add with optimized Metal kernels
- PyTorch-compatible STFT/ISTFT semantics
- Autograd support with custom Metal backward kernels
- Drop-in replacement for `torch.stft` / `torch.istft`

## Install

```bash
pip install -e .
```

## Quick Start

```python
import torch
from mps_spectro import stft, istft

x = torch.randn(1, 16000, device="mps")

spec = stft(x, n_fft=1024, hop_length=256)
y = istft(spec, n_fft=1024, hop_length=256, center=True)
```

`stft` and `istft` accept the same parameters as `torch.stft` / `torch.istft` (n_fft, hop_length, win_length, window, center, normalized, onesided, length).

### Autograd

Both `stft` and `istft` support PyTorch autograd when inputs have `requires_grad=True`:

```python
x = torch.randn(4, 16000, device="mps", requires_grad=True)

spec = stft(x, n_fft=1024, hop_length=256)
y = istft(spec, n_fft=1024, hop_length=256, center=True, length=16000)

loss = y.pow(2).sum()
loss.backward()
print(x.grad.shape)  # torch.Size([4, 16000])
```

When `requires_grad=False` (the default), zero overhead -- the original Metal kernel path is used directly. Backward passes use custom Metal kernels for GPU-accelerated gradient computation. Window gradients are not computed (returns `None`) since windows are almost always frozen in practice.

### ISTFT extras

`istft` also supports:

- `torch_like=True` -- raise on NOLA violations like `torch.istft`
- `safety="auto"|"always"|"off"` -- NOLA envelope safety checking
- `kernel_dtype="float32"|"float16"|"mixed"` -- Metal kernel precision
- `kernel_layout="auto"|"native"|"transposed"` -- memory layout selection

## Benchmarks

Apple M4 Max, macOS 26.3, PyTorch 2.10.0, 20 iterations (5 warmup).

### STFT Forward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.08 ms | 0.29 ms | 3.7x |
| B=4 T=160k nfft=2048 | 1.06 ms | 0.29 ms | 3.6x |
| B=8 T=160k nfft=1024 | 0.55 ms | 0.41 ms | 1.4x |
| B=4 T=1.3M nfft=1024 | 1.80 ms | 1.37 ms | 1.3x |

### ISTFT Forward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.11 ms | 0.50 ms | 2.2x |
| B=8 T=160k nfft=1024 | 1.66 ms | 0.74 ms | 2.3x |
| B=4 T=1.3M nfft=1024 | 5.65 ms | 2.36 ms | 2.4x |
| B=1 T=1.3M nfft=1024 | 1.74 ms | 0.75 ms | 2.3x |

### Differentiable STFT + ISTFT (Forward + Backward)

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.37 ms | 1.04 ms | 1.3x |
| B=8 T=160k nfft=1024 | 2.72 ms | 1.81 ms | 1.5x |
| B=4 T=1.3M nfft=1024 | 12.20 ms | 8.88 ms | 1.4x |
| B=1 T=1.3M nfft=1024 | 2.66 ms | 1.78 ms | 1.5x |

### Roundtrip (STFT -> ISTFT) Forward + Backward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 2.33 ms | 1.27 ms | 1.8x |
| B=8 T=160k nfft=1024 | 4.28 ms | 2.38 ms | 1.8x |
| B=4 T=1.3M nfft=1024 | 17.56 ms | 10.43 ms | 1.7x |
| B=1 T=1.3M nfft=1024 | 4.25 ms | 2.36 ms | 1.8x |

To reproduce: `python scripts/benchmark.py`

## How it works

1. **STFT**: a tiled Metal kernel loads overlapping signal chunks into threadgroup shared memory (~3x data reuse for typical n_fft/hop ratios), applies reflect-padding and windowing in one pass, then `torch.fft.rfft` for the FFT
2. **ISTFT**: `torch.fft.irfft` on MPS, then a fused Metal kernel for synthesis-window multiply + overlap-add + envelope normalization

## Requirements

- macOS with Apple Silicon (MPS)
- Python 3.12+
- PyTorch 2.10+
- Xcode command-line tools (for JIT Metal kernel compilation)

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
