# mps-spectro

Fast `torch.stft` / `torch.istft` replacements for Apple MPS, powered by custom Metal kernels.

## Performance

### ISTFT

The fused Metal kernel for overlap-add + envelope normalization gives a consistent **~2x speedup** over `torch.istft`:

| Machine | `torch.istft` | `mps_spectro.istft` | Speedup |
|---|--:|--:|--:|
| MacBook Pro (M4 Max) | 0.70 ms | 0.36 ms | **1.97x** |
| MacBook Pro (M1 Max) | 1.66 ms | 0.83 ms | **1.89x** |

Median latency across 288 parameter configurations (varying n_fft, hop_length, n_frames, length mode), 3 replicates each.

### STFT

A tiled Metal kernel with threadgroup shared memory fuses reflect-padding, strided frame extraction, and windowing into a single GPU pass. Automatically selected for bandwidth-bound workloads (output ≥ 5 MB); smaller workloads delegate to `torch.stft`.

| Config | `torch.stft` | `mps_spectro.stft` | Speedup |
|---|--:|--:|--:|
| B=4, 160k samples, n_fft=1024 | 0.185 ms | 0.100 ms | **1.86x** |
| B=4, 160k samples, n_fft=2048 | 0.186 ms | 0.099 ms | **1.87x** |
| B=4, 1.3M samples, n_fft=1024 | 1.526 ms | 1.035 ms | **1.47x** |
| B=1, 1.3M samples, n_fft=1024 | 0.275 ms | 0.218 ms | **1.26x** |

Measured on M4 Max. For small/single-batch workloads the overhead of Python dispatch across three ops (Metal kernel → rfft → transpose) exceeds the kernel savings, so `torch.stft` is used directly with no regression.

## Requirements

- macOS with Apple Silicon (MPS)
- Python 3.12+
- PyTorch 2.10+
- Xcode command-line tools (for JIT Metal kernel compilation)

## Install

```bash
pip install -e .
```

## Usage

```python
import torch
from mps_spectro import stft, istft

x = torch.randn(1, 16000, device="mps")

spec = stft(x, n_fft=1024, hop_length=256)
y = istft(spec, n_fft=1024, hop_length=256, center=True)
```

`stft` and `istft` accept the same parameters as `torch.stft` / `torch.istft` (n_fft, hop_length, win_length, window, center, normalized, onesided, length).

### ISTFT extras

`istft` also supports:

- `torch_like=True` -- raise on NOLA violations like `torch.istft`
- `safety="auto"|"always"|"off"` -- NOLA envelope safety checking
- `kernel_dtype="float32"|"float16"|"mixed"` -- Metal kernel precision
- `kernel_layout="auto"|"native"|"transposed"` -- memory layout selection

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

When `requires_grad=False` (the default), zero overhead — the original Metal kernel path is used directly. Backward passes use custom Metal kernels for GPU-accelerated gradient computation (~1.4–1.9× faster than `torch.istft` backward). Window gradients are not computed (returns `None`) since windows are almost always frozen in practice.

## How it works

1. **STFT**: a tiled Metal kernel loads overlapping signal chunks into threadgroup shared memory (~3× data reuse for typical n_fft/hop ratios), applies reflect-padding and windowing in one pass, then `torch.fft.rfft` for the FFT
2. **ISTFT**: `torch.fft.irfft` on MPS, then a fused Metal kernel for synthesis-window multiply + overlap-add + envelope normalization

## Tests

```bash
pip install -e ".[dev]"
pytest
```
