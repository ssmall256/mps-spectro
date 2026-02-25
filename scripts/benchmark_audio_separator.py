#!/usr/bin/env python3
"""Benchmark STFT/ISTFT latency for real-world audio-separator workloads.

Measures mps_spectro vs torch.stft/torch.istft using the same tensor shapes
and calling conventions as python-audio-separator's MDX, Roformer, and Demucs
model pipelines.

The benchmark uses two representative stereo 44.1 kHz signal lengths matching
the tracks used in the original evaluation (267s and 195s).

Usage:
    python scripts/benchmark_audio_separator.py
    python scripts/benchmark_audio_separator.py --quick
"""

from __future__ import annotations

import argparse
import platform
import time

import torch

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench_mps(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in ms for an MPS workload."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    torch.mps.empty_cache()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Model configurations (matching audio-separator defaults)
# ---------------------------------------------------------------------------

# MDX / Roformer: n_fft=2048, hop=512, stereo 44.1kHz
# Demucs: n_fft=4096, hop=1024, stereo 44.1kHz
# Audio-separator reshapes [B, channels, T] -> [-1, T] for STFT, so the
# effective batch size is B * channels. For stereo (2ch) with B=1 chunk:
#   MDX:  batch=2 (1 chunk × 2 channels)
#   Roformer: batch=2
#   Demucs: batch=2

CONFIGS = [
    {
        "label": "MDX (n_fft=2048, hop=512)",
        "n_fft": 2048,
        "hop_length": 512,
        "batch": 2,  # stereo
        "lengths": [267 * 44100, 195 * 44100],  # two tracks
    },
    {
        "label": "Roformer (n_fft=2048, hop=512)",
        "n_fft": 2048,
        "hop_length": 512,
        "batch": 2,
        "lengths": [267 * 44100, 195 * 44100],
    },
    {
        "label": "Demucs (n_fft=4096, hop=1024)",
        "n_fft": 4096,
        "hop_length": 1024,
        "batch": 2,
        "lengths": [267 * 44100, 195 * 44100],
    },
]

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="audio-separator STFT/ISTFT benchmark")
    parser.add_argument("--quick", action="store_true", help="Reduced iterations")
    args = parser.parse_args()

    warmup = 3 if args.quick else 5
    iters = 5 if args.quick else 20
    cooldown_s = 2 if args.quick else 5

    from mps_spectro import stft as mps_stft, istft as mps_istft

    chip = platform.processor() or "unknown"
    mac = platform.mac_ver()[0]
    print(f"## audio-separator STFT/ISTFT benchmark")
    print(f"Machine: macOS {mac}, {chip}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iterations: {iters} (warmup: {warmup}, cooldown: {cooldown_s}s)")
    print()

    print("| Model config | STFT speedup | iSTFT speedup |")
    print("|---|--:|--:|")

    for cfg in CONFIGS:
        label = cfg["label"]
        n_fft = cfg["n_fft"]
        hop = cfg["hop_length"]
        batch = cfg["batch"]
        lengths = cfg["lengths"]

        stft_speedups = []
        istft_speedups = []

        for T in lengths:
            device = "mps"
            win = torch.hann_window(n_fft, device=device)
            x = torch.randn(batch, T, device=device)

            # --- STFT benchmark ---
            def torch_stft(x=x, n_fft=n_fft, hop=hop, win=win):
                torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                           center=True, return_complex=True)

            def mps_stft_fn(x=x, n_fft=n_fft, hop=hop):
                mps_stft(x, n_fft=n_fft, hop_length=hop)

            t_torch_stft = bench_mps(torch_stft, warmup=warmup, iters=iters)
            time.sleep(cooldown_s)
            t_mps_stft = bench_mps(mps_stft_fn, warmup=warmup, iters=iters)
            time.sleep(cooldown_s)

            stft_speedups.append(t_torch_stft / max(t_mps_stft, 1e-6))

            # --- ISTFT benchmark ---
            spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                              center=True, return_complex=True)

            def torch_istft(spec=spec, n_fft=n_fft, hop=hop, win=win, T=T):
                torch.istft(spec, n_fft=n_fft, hop_length=hop, window=win,
                            center=True, length=T)

            def mps_istft_fn(spec=spec, n_fft=n_fft, hop=hop, T=T):
                mps_istft(spec, n_fft=n_fft, hop_length=hop, center=True,
                          length=T)

            t_torch_istft = bench_mps(torch_istft, warmup=warmup, iters=iters)
            time.sleep(cooldown_s)
            t_mps_istft = bench_mps(mps_istft_fn, warmup=warmup, iters=iters)
            time.sleep(cooldown_s)

            istft_speedups.append(t_torch_istft / max(t_mps_istft, 1e-6))

            # Free memory
            del x, spec
            torch.mps.empty_cache()

        # Average speedup across the two tracks
        avg_stft = sum(stft_speedups) / len(stft_speedups)
        avg_istft = sum(istft_speedups) / len(istft_speedups)

        print(f"| {label} | **{avg_stft:.2f}x** | **{avg_istft:.2f}x** |")

    print()
    print("Stereo 44.1 kHz, two tracks (267s + 195s), median of "
          f"{iters} iterations, {warmup} warmup, {cooldown_s}s cooldown.")


if __name__ == "__main__":
    main()
