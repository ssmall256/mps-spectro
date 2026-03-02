# mps-spectro

Drop-in `torch.stft` / `torch.istft` replacements for Apple Silicon — **1.4–3x faster** on MPS via custom Metal kernels.

```python
# before
spec = torch.stft(x, n_fft=2048, hop_length=512, window=window, center=True, return_complex=True)
y = torch.istft(spec, n_fft=2048, hop_length=512, window=window, center=True, length=T)

# after
from mps_spectro import stft, istft

spec = stft(x, n_fft=2048, hop_length=512)
y = istft(spec, n_fft=2048, hop_length=512, center=True, length=T)
```

Drop-in compatible with [python-audio-separator](https://github.com/karaokenerds/python-audio-separator) (MDX, Roformer, Demucs) — **1.4x faster STFT** and **2x faster iSTFT** on stereo 44.1 kHz audio with no model changes. See [benchmarks](#stftistft-in-audio-separator-workloads) below.

## Install

```bash
pip install mps-spectro
```

## Features

- PyTorch-compatible STFT/ISTFT semantics (same parameters as `torch.stft` / `torch.istft`)
- Fused overlap-add with optimized Metal compute shaders
- Autograd support with custom Metal backward kernels
- `torch.compile` compatible (`aot_eager` backend) via `torch.library` custom ops
- Pure Python — no C++ build step, no Xcode CLI tools

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

### torch.compile

Custom ops are registered via `torch.library` with Meta (FakeTensor) kernels, so `torch.compile` can trace through both forward and backward:

```python
@torch.compile(backend="aot_eager")
def f(x):
    return stft(x, n_fft=2048, hop_length=512)

f(torch.randn(4, 160000, device="mps"))  # works
```

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
| B=4 T=160k nfft=1024 | 0.51 ms | 0.35 ms | 1.5x |
| B=4 T=160k nfft=2048 | 0.53 ms | 0.31 ms | 1.7x |
| B=8 T=160k nfft=1024 | 0.78 ms | 0.46 ms | 1.7x |
| B=4 T=1.3M nfft=1024 | 1.93 ms | 1.38 ms | 1.4x |

### ISTFT Forward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.10 ms | 0.34 ms | 3.2x |
| B=8 T=160k nfft=1024 | 1.70 ms | 0.63 ms | 2.7x |
| B=4 T=1.3M nfft=1024 | 6.01 ms | 2.30 ms | 2.6x |
| B=1 T=1.3M nfft=1024 | 1.76 ms | 0.61 ms | 2.9x |

### STFT Forward + Backward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.51 ms | 1.05 ms | 1.4x |
| B=8 T=160k nfft=1024 | 2.96 ms | 2.08 ms | 1.4x |
| B=4 T=1.3M nfft=1024 | 12.75 ms | 9.73 ms | 1.3x |
| B=1 T=1.3M nfft=1024 | 2.95 ms | 2.16 ms | 1.4x |

### ISTFT Forward + Backward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 1.91 ms | 0.98 ms | 1.9x |
| B=8 T=160k nfft=1024 | 2.95 ms | 1.62 ms | 1.8x |
| B=4 T=1.3M nfft=1024 | 9.95 ms | 5.71 ms | 1.7x |
| B=1 T=1.3M nfft=1024 | 2.95 ms | 1.56 ms | 1.9x |

### Roundtrip (STFT -> ISTFT) Forward + Backward

| Config | torch MPS | mps_spectro | Speedup |
|---|--:|--:|--:|
| B=4 T=160k nfft=1024 | 2.52 ms | 1.47 ms | 1.7x |
| B=8 T=160k nfft=1024 | 4.71 ms | 2.55 ms | 1.8x |
| B=4 T=1.3M nfft=1024 | 18.42 ms | 11.07 ms | 1.7x |
| B=1 T=1.3M nfft=1024 | 4.60 ms | 2.39 ms | 1.9x |

To reproduce:
- Full suite: `python scripts/benchmark.py`
- Dispatch overhead profile: `python scripts/benchmark.py --dispatch-profile`

### STFT/iSTFT in audio-separator workloads

[python-audio-separator](https://github.com/karaokenerds/python-audio-separator) uses `torch.stft`/`torch.istft` in its MDX, Roformer, and Demucs model pipelines. We swapped in `mps_spectro` via a [compatibility layer](https://github.com/karaokenerds/python-audio-separator/blob/main/audio_separator/separator/stft_compat.py) and measured the STFT/iSTFT portion of each pipeline with two real stereo 44.1 kHz tracks (267s and 195s). Apple M4 Max, PyTorch 2.10.0, 20 iterations, 5 warmup, 5s cooldown.

| Model config | STFT speedup | iSTFT speedup |
|---|--:|--:|
| MDX (n_fft=2048, hop=512) | **1.40x** | **2.03x** |
| Roformer (n_fft=2048, hop=512) | **1.40x** | **2.01x** |
| Demucs (n_fft=4096, hop=1024) | **1.28x** | **1.87x** |

Note: total separation wall time is dominated by model inference, so E2E speedup is modest. The gains above apply to the STFT/iSTFT calls themselves.

To reproduce: `python scripts/benchmark_audio_separator.py`

**Numerical parity.** Output stems are perceptually identical — maximum float32 difference per sample is ≤ 1.83 × 10⁻⁴ (≤ 6 int16 LSBs) across all architectures:

| Model | Max abs diff (float32) | SNR (dB) | Int16 sample match |
|---|--:|--:|--:|
| BS-Roformer-SW (6-stem) | 3.05e-05 | 91 – 100 | ≥ 99.98% |
| Mel-Roformer Karaoke | 3.05e-05 | 89 – 91 | ≥ 99.84% |
| MDX-NET Inst HQ 5 | 1.83e-04 | 55 – 64 | ≥ 99%\* |
| hdemucs_mmi (shifts=0) | 4.27e-04 | 44 – 52 | ≥ 71% |

\* MDX int16 diffs are symmetric ±1 LSB rounding noise with zero bias and max ±6 LSBs.

## Using MLX instead of PyTorch?

See [mlx-spectro](https://github.com/ssmall256/mlx-spectro) — same idea, built natively on MLX with even faster kernels (2–8x vs torch).

## How it works

Metal shader source is compiled at runtime via `torch.mps.compile_shader` (pure Python, no C++ build step).

1. **STFT**: a tiled Metal kernel loads overlapping signal chunks into threadgroup shared memory (~3x data reuse for typical n_fft/hop ratios), applies reflect-padding and windowing in one pass, then `torch.fft.rfft` for the FFT
2. **ISTFT**: `torch.fft.irfft` on MPS, then a fused Metal kernel for synthesis-window multiply + overlap-add + envelope normalization

## Requirements

- macOS with Apple Silicon (MPS)
- Python 3.12+
- PyTorch 2.10+

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
