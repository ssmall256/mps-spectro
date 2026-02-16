#!/usr/bin/env python3
"""Benchmark suite for mps-spectro.

Compares mps_spectro.stft / mps_spectro.istft against torch.stft / torch.istft
on both MPS and CPU. Prints Markdown tables suitable for copy-pasting into the
README.

Usage:
    python scripts/benchmark.py              # full suite
    python scripts/benchmark.py --quick      # reduced iterations
    python scripts/benchmark.py --forward    # forward-only
    python scripts/benchmark.py --backward   # backward-only
"""

from __future__ import annotations

import argparse
import platform
import time
import warnings

import torch

warnings.filterwarnings("ignore", message=".*was resized.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

from mps_spectro.stft import mps_stft_forward
from mps_spectro.istft import mps_istft_forward

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
    return times[len(times) // 2]


def bench_cpu(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in ms for a CPU workload."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

# (batch, signal_length, n_fft, label)
CONFIGS = [
    # Small workloads (below 5 MB threshold → torch.stft fallback)
    (1, 16000, 512, "B=1 T=16k nfft=512"),
    (4, 16000, 512, "B=4 T=16k nfft=512"),
    # Medium workloads
    (1, 160000, 1024, "B=1 T=160k nfft=1024"),
    (4, 160000, 1024, "B=4 T=160k nfft=1024"),
    (4, 160000, 2048, "B=4 T=160k nfft=2048"),
    # Large workloads
    (8, 160000, 1024, "B=8 T=160k nfft=1024"),
    (4, 1320000, 1024, "B=4 T=1.3M nfft=1024"),
    (1, 1320000, 1024, "B=1 T=1.3M nfft=1024"),
]

# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_table(
    title: str,
    rows: list[tuple[str, float, float, float]],
) -> None:
    """Print a Markdown table from (label, cpu_ms, torch_mps_ms, mps_spectro_ms) rows."""
    print(f"\n### {title}\n")
    print(
        f"| {'Config':<28} "
        f"| {'CPU (ms)':>10} "
        f"| {'torch MPS (ms)':>15} "
        f"| {'mps_spectro (ms)':>17} "
        f"| {'vs torch MPS':>13} "
        f"| {'vs CPU':>8} |"
    )
    print(f"|{'-'*30}|{'-'*12}|{'-'*17}|{'-'*19}|{'-'*15}|{'-'*10}|")
    for label, t_cpu, t_torch_mps, t_mps in rows:
        vs_torch = t_torch_mps / max(t_mps, 1e-6)
        vs_cpu = t_cpu / max(t_mps, 1e-6)
        print(
            f"| {label:<28} "
            f"| {t_cpu:>8.3f}ms "
            f"| {t_torch_mps:>13.3f}ms "
            f"| {t_mps:>15.3f}ms "
            f"| **{vs_torch:.2f}x** "
            f"| **{vs_cpu:.1f}x** |"
        )

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_stft_forward(warmup: int, iters: int) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        win_mps = torch.hann_window(nfft, device="mps")
        win_cpu = torch.hann_window(nfft, device="cpu")

        def cpu_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_cpu):
            x = torch.randn(B, T)
            torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                       center=True, return_complex=True)

        def torch_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_mps):
            x = torch.randn(B, T, device="mps")
            torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                       center=True, return_complex=True)

        def mps_fn(B=B, T=T, nfft=nfft, hop=hop):
            x = torch.randn(B, T, device="mps")
            mps_stft_forward(x, n_fft=nfft, hop_length=hop)

        t_cpu = bench_cpu(cpu_fn, warmup=warmup, iters=iters)
        t_torch = bench_mps(torch_fn, warmup=warmup, iters=iters)
        t_mps = bench_mps(mps_fn, warmup=warmup, iters=iters)
        rows.append((label, t_cpu, t_torch, t_mps))
    _print_table("STFT Forward", rows)


def bench_istft_forward(warmup: int, iters: int) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        win_mps = torch.hann_window(nfft, device="mps")
        win_cpu = torch.hann_window(nfft, device="cpu")
        x0 = torch.randn(B, T, device="mps")
        spec_mps = torch.stft(x0, n_fft=nfft, hop_length=hop, window=win_mps,
                               center=True, return_complex=True)
        spec_cpu = spec_mps.detach().cpu()

        def cpu_fn(spec=spec_cpu, nfft=nfft, hop=hop, win=win_cpu, T=T):
            torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                        center=True, length=T)

        def torch_fn(spec=spec_mps, nfft=nfft, hop=hop, win=win_mps, T=T):
            torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                        center=True, length=T)

        def mps_fn(spec=spec_mps, nfft=nfft, hop=hop, T=T):
            mps_istft_forward(spec, n_fft=nfft, hop_length=hop, center=True,
                              length=T)

        t_cpu = bench_cpu(cpu_fn, warmup=warmup, iters=iters)
        t_torch = bench_mps(torch_fn, warmup=warmup, iters=iters)
        t_mps = bench_mps(mps_fn, warmup=warmup, iters=iters)
        rows.append((label, t_cpu, t_torch, t_mps))
    _print_table("ISTFT Forward", rows)


def bench_stft_backward(warmup: int, iters: int) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        win_mps = torch.hann_window(nfft, device="mps")
        win_cpu = torch.hann_window(nfft, device="cpu")

        def cpu_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_cpu):
            x = torch.randn(B, T, requires_grad=True)
            s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                           center=True, return_complex=True)
            s.abs().pow(2).sum().backward()

        def torch_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_mps):
            x = torch.randn(B, T, device="mps", requires_grad=True)
            s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                           center=True, return_complex=True)
            s.abs().pow(2).sum().backward()

        def mps_fn(B=B, T=T, nfft=nfft, hop=hop):
            x = torch.randn(B, T, device="mps", requires_grad=True)
            s = mps_stft_forward(x, n_fft=nfft, hop_length=hop)
            s.abs().pow(2).sum().backward()

        t_cpu = bench_cpu(cpu_fn, warmup=warmup, iters=iters)
        t_torch = bench_mps(torch_fn, warmup=warmup, iters=iters)
        t_mps = bench_mps(mps_fn, warmup=warmup, iters=iters)
        rows.append((label, t_cpu, t_torch, t_mps))
    _print_table("STFT Forward + Backward", rows)


def bench_istft_backward(warmup: int, iters: int) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        win_mps = torch.hann_window(nfft, device="mps")
        win_cpu = torch.hann_window(nfft, device="cpu")
        x0 = torch.randn(B, T, device="mps")
        spec0_mps = torch.stft(x0, n_fft=nfft, hop_length=hop, window=win_mps,
                                center=True, return_complex=True).detach()
        spec0_cpu = spec0_mps.cpu()

        def cpu_fn(spec0=spec0_cpu, nfft=nfft, hop=hop, win=win_cpu, T=T):
            spec = spec0.requires_grad_(True)
            y = torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                            center=True, length=T)
            y.pow(2).sum().backward()

        def torch_fn(spec0=spec0_mps, nfft=nfft, hop=hop, win=win_mps, T=T):
            spec = spec0.requires_grad_(True)
            y = torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                            center=True, length=T)
            y.pow(2).sum().backward()

        def mps_fn(spec0=spec0_mps, nfft=nfft, hop=hop, T=T):
            spec = spec0.requires_grad_(True)
            y = mps_istft_forward(spec, n_fft=nfft, hop_length=hop,
                                  center=True, length=T)
            y.pow(2).sum().backward()

        t_cpu = bench_cpu(cpu_fn, warmup=warmup, iters=iters)
        t_torch = bench_mps(torch_fn, warmup=warmup, iters=iters)
        t_mps = bench_mps(mps_fn, warmup=warmup, iters=iters)
        rows.append((label, t_cpu, t_torch, t_mps))
    _print_table("ISTFT Forward + Backward", rows)


def bench_roundtrip(warmup: int, iters: int) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        win_mps = torch.hann_window(nfft, device="mps")
        win_cpu = torch.hann_window(nfft, device="cpu")

        def cpu_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_cpu):
            x = torch.randn(B, T, requires_grad=True)
            s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                           center=True, return_complex=True)
            y = torch.istft(s, n_fft=nfft, hop_length=hop, window=win,
                            center=True, length=T)
            y.pow(2).sum().backward()

        def torch_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_mps):
            x = torch.randn(B, T, device="mps", requires_grad=True)
            s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                           center=True, return_complex=True)
            y = torch.istft(s, n_fft=nfft, hop_length=hop, window=win,
                            center=True, length=T)
            y.pow(2).sum().backward()

        def mps_fn(B=B, T=T, nfft=nfft, hop=hop):
            x = torch.randn(B, T, device="mps", requires_grad=True)
            s = mps_stft_forward(x, n_fft=nfft, hop_length=hop)
            y = mps_istft_forward(s, n_fft=nfft, hop_length=hop,
                                  center=True, length=T)
            y.pow(2).sum().backward()

        t_cpu = bench_cpu(cpu_fn, warmup=warmup, iters=iters)
        t_torch = bench_mps(torch_fn, warmup=warmup, iters=iters)
        t_mps = bench_mps(mps_fn, warmup=warmup, iters=iters)
        rows.append((label, t_cpu, t_torch, t_mps))
    _print_table("Roundtrip (STFT → ISTFT) Forward + Backward", rows)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="mps-spectro benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations for fast sanity checks")
    parser.add_argument("--forward", action="store_true",
                        help="Run forward benchmarks only")
    parser.add_argument("--backward", action="store_true",
                        help="Run backward benchmarks only")
    args = parser.parse_args()

    warmup = 3 if args.quick else 5
    iters = 5 if args.quick else 20

    # Header
    chip = platform.processor() or "unknown"
    mac = platform.mac_ver()[0]
    print(f"## mps-spectro benchmarks")
    print(f"Machine: macOS {mac}, {chip}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iterations: {iters} (warmup: {warmup})")

    run_forward = not args.backward
    run_backward = not args.forward

    if run_forward:
        bench_stft_forward(warmup, iters)
        bench_istft_forward(warmup, iters)

    if run_backward:
        bench_stft_backward(warmup, iters)
        bench_istft_backward(warmup, iters)
        bench_roundtrip(warmup, iters)


if __name__ == "__main__":
    main()
