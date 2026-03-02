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
from statistics import median

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
    torch.mps.empty_cache()
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


def bench_mps_trimmed(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in ms using 20% trimmed samples on MPS."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    trim = max(1, len(samples) // 5)
    trimmed = samples[trim:-trim] if len(samples) > (2 * trim) else samples
    torch.mps.empty_cache()
    return float(median(trimmed))

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

# (batch, signal_length, n_fft, label)
# Configs representative of real audio workloads:
#   - Speech/TTS training: B=4-16, 3-10s @ 16-24kHz, nfft=1024
#   - Music/enhancement: B=4-8, 5-10s @ 44.1kHz, nfft=2048
#   - Single-item inference: B=1, 10s @ 16kHz
CONFIGS = [
    # Single-batch (inference-like)
    (1, 160000, 1024, "B=1 T=160k nfft=1024"),
    # Typical training batches
    (4, 160000, 1024, "B=4 T=160k nfft=1024"),
    (4, 160000, 2048, "B=4 T=160k nfft=2048"),
    (8, 160000, 1024, "B=8 T=160k nfft=1024"),
    (16, 160000, 1024, "B=16 T=160k nfft=1024"),
    # Long-sequence / high-sample-rate
    (4, 480000, 1024, "B=4 T=480k nfft=1024"),
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
        f"| {'vs CPU':>7} |"
    )
    print(f"|{'-'*30}|{'-'*12}|{'-'*17}|{'-'*19}|{'-'*15}|{'-'*9}|")
    for label, t_cpu, t_torch_mps, t_mps in rows:
        vs_torch = t_torch_mps / max(t_mps, 1e-6)
        vs_cpu = t_cpu / max(t_mps, 1e-6)
        print(
            f"| {label:<28} "
            f"| {t_cpu:>8.3f}ms "
            f"| {t_torch_mps:>13.3f}ms "
            f"| {t_mps:>15.3f}ms "
            f"| {vs_torch:>11.2f}x  "
            f"| {vs_cpu:>5.1f}x  |"
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
# Dispatch profiling
# ---------------------------------------------------------------------------

def _stft_n_frames(*, n_fft: int, hop_length: int, length: int, center: bool) -> int:
    pad = n_fft // 2 if center else 0
    padded_length = length + (2 * pad)
    return (padded_length - n_fft) // hop_length + 1


def bench_dispatch_profile(
    warmup: int,
    iters: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> None:
    """Profile wrapper dispatch overhead against kernel and FFT floors."""
    print("\n### Dispatch Profile\n")

    configs = [
        ("B=4 T=160k", 4, 160_000),
        ("B=4 T=1.3M", 4, 1_300_000),
        ("B=8 T=480k", 8, 480_000),
    ]

    stft_rows: list[dict] = []
    istft_rows: list[dict] = []

    for label, batch, length in configs:
        x = torch.randn(batch, length, device="mps", dtype=torch.float32)
        window = torch.hann_window(n_fft, periodic=True, device="mps", dtype=torch.float32)
        window_sq = (window * window).contiguous()

        # STFT: full API path
        eager_stft_ms = bench_mps_trimmed(
            lambda: mps_stft_forward(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
            ),
            warmup=warmup,
            iters=iters,
        )

        # STFT: explicit kernel path (reflect-pad + frame extract + FFT)
        direct_stft_ms = bench_mps_trimmed(
            lambda: torch.fft.rfft(
                torch.ops.mps_spectro.stft_extract_frames(
                    x,
                    window,
                    int(hop_length),
                    int(n_fft),
                    True,
                ),
                n=n_fft,
                dim=-1,
                norm="backward",
            ),
            warmup=warmup,
            iters=iters,
        )

        n_frames = _stft_n_frames(n_fft=n_fft, hop_length=hop_length, length=length, center=True)
        frames = torch.randn(batch, n_frames, n_fft, device="mps", dtype=torch.float32)
        fft_floor_ms = bench_mps_trimmed(
            lambda: torch.fft.rfft(frames, n=n_fft, dim=-1, norm="backward"),
            warmup=warmup,
            iters=iters,
        )

        stft_rows.append(
            {
                "label": label,
                "eager": eager_stft_ms,
                "direct": direct_stft_ms,
                "fft_floor": fft_floor_ms,
                "dispatch": max(0.0, eager_stft_ms - direct_stft_ms),
            }
        )

        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            return_complex=True,
        )

        # ISTFT: full API path
        eager_istft_ms = bench_mps_trimmed(
            lambda: mps_istft_forward(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                center=True,
                length=length,
            ),
            warmup=warmup,
            iters=iters,
        )

        # ISTFT: explicit kernel path (iFFT + overlap-add kernel)
        direct_istft_ms = bench_mps_trimmed(
            lambda: torch.ops.mps_spectro.istft_overlap_add(
                torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward").transpose(-1, -2).contiguous(),
                window,
                window_sq,
                int(hop_length),
                int(hop_length * (spec.size(-1) - 1) + n_fft),
            ),
            warmup=warmup,
            iters=iters,
        )

        ifft_floor_ms = bench_mps_trimmed(
            lambda: torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward"),
            warmup=warmup,
            iters=iters,
        )

        istft_rows.append(
            {
                "label": label,
                "eager": eager_istft_ms,
                "direct": direct_istft_ms,
                "ifft_floor": ifft_floor_ms,
                "dispatch": max(0.0, eager_istft_ms - direct_istft_ms),
            }
        )

    print(f"STFT dispatch profile (n_fft={n_fft}, hop={hop_length})")
    print("| Config | Eager (ms) | Direct path (ms) | FFT floor (ms) | Dispatch est (ms) | Direct speedup |")
    print("|---|--:|--:|--:|--:|--:|")
    for row in stft_rows:
        speedup = row["eager"] / max(row["direct"], 1e-6)
        print(
            f"| {row['label']} | {row['eager']:.3f} | {row['direct']:.3f} | "
            f"{row['fft_floor']:.3f} | {row['dispatch']:.3f} | {speedup:.2f}x |"
        )

    print()
    print(f"ISTFT dispatch profile (n_fft={n_fft}, hop={hop_length})")
    print("| Config | Eager (ms) | Direct path (ms) | iFFT floor (ms) | Dispatch est (ms) | Direct speedup |")
    print("|---|--:|--:|--:|--:|--:|")
    for row in istft_rows:
        speedup = row["eager"] / max(row["direct"], 1e-6)
        print(
            f"| {row['label']} | {row['eager']:.3f} | {row['direct']:.3f} | "
            f"{row['ifft_floor']:.3f} | {row['dispatch']:.3f} | {speedup:.2f}x |"
        )

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
    parser.add_argument(
        "--dispatch-profile",
        action="store_true",
        help="Run focused wrapper-vs-kernel dispatch profiling",
    )
    parser.add_argument(
        "--dispatch-nfft",
        type=int,
        default=2048,
        help="FFT size for --dispatch-profile",
    )
    parser.add_argument(
        "--dispatch-hop",
        type=int,
        default=512,
        help="Hop length for --dispatch-profile",
    )
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

    if args.dispatch_profile:
        bench_dispatch_profile(
            warmup,
            iters,
            n_fft=args.dispatch_nfft,
            hop_length=args.dispatch_hop,
        )
        return

    if run_forward:
        bench_stft_forward(warmup, iters)
        bench_istft_forward(warmup, iters)

    if run_backward:
        bench_stft_backward(warmup, iters)
        bench_istft_backward(warmup, iters)
        bench_roundtrip(warmup, iters)


if __name__ == "__main__":
    main()
