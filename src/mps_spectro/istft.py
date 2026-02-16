import os
import hashlib
import warnings
from collections import OrderedDict
from typing import Optional

import torch

# Allow fallback when a dependent op is unavailable on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from mps_spectro.compiler import compiled_lib

_SAFETY_CACHE_MAX = 256
_SAFETY_CACHE: "OrderedDict[tuple, tuple[bool, float]]" = OrderedDict()
_LAYOUT_CACHE_MAX = 256
_LAYOUT_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_LAYOUT_TIE_MARGIN = 0.10
_LAYOUT_MAX_REL_SPREAD = 0.25
_LAYOUT_PROBE_ROUNDS = 5
_LAYOUT_PROBE_ITERS = 3
_LAYOUT_SMALL_INPUT_NUMEL_THRESHOLD = 50_000


def _validate_input(spec: torch.Tensor) -> torch.Tensor:
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)
    if spec.dim() != 3:
        raise ValueError(f"Expected spec shape [F, N] or [B, F, N], got {tuple(spec.shape)}")
    if not spec.is_complex():
        raise TypeError("Expected complex input STFT tensor")
    if not spec.device.type == "mps":
        raise ValueError("Input must be on MPS device")
    return spec


def _cache_get(key: tuple) -> Optional[tuple[bool, float]]:
    val = _SAFETY_CACHE.get(key)
    if val is None:
        return None
    _SAFETY_CACHE.move_to_end(key)
    return val


def _cache_set(key: tuple, value: tuple[bool, float]) -> None:
    _SAFETY_CACHE[key] = value
    _SAFETY_CACHE.move_to_end(key)
    if len(_SAFETY_CACHE) > _SAFETY_CACHE_MAX:
        _SAFETY_CACHE.popitem(last=False)


def _layout_cache_get(key: tuple) -> Optional[str]:
    val = _LAYOUT_CACHE.get(key)
    if val is None:
        return None
    _LAYOUT_CACHE.move_to_end(key)
    return val


def _layout_cache_set(key: tuple, value: str) -> None:
    _LAYOUT_CACHE[key] = value
    _LAYOUT_CACHE.move_to_end(key)
    if len(_LAYOUT_CACHE) > _LAYOUT_CACHE_MAX:
        _LAYOUT_CACHE.popitem(last=False)


def _choose_layout_auto(
    *,
    frames_native: torch.Tensor,
    window_for_kernel: torch.Tensor,
    window_sq_for_kernel: torch.Tensor,
    hop_length: int,
    full_length: int,
    kernel_dtype: str,
) -> str:
    key = (
        int(frames_native.size(0)),
        int(frames_native.size(1)),
        int(frames_native.size(2)),
        int(hop_length),
        int(full_length),
        str(kernel_dtype),
    )

    # Cold-start latency guardrail: skip expensive probing on small tensors.
    if int(frames_native.numel()) < _LAYOUT_SMALL_INPUT_NUMEL_THRESHOLD:
        best = "native"
        _layout_cache_set(key, best)
        return best

    def _run(layout: str) -> None:
        if kernel_dtype == "mixed":
            if layout == "native":
                _ = compiled_lib.mps_istft_overlap_add_div_envelope_mixed(
                    frames_native, window_for_kernel, window_sq_for_kernel, int(hop_length), int(full_length)
                )
            else:
                frames_transposed = frames_native.transpose(-1, -2).contiguous()
                _ = compiled_lib.mps_istft_overlap_add_div_envelope_mixed_transposed(
                    frames_transposed, window_for_kernel, window_sq_for_kernel, int(hop_length), int(full_length)
                )
        else:
            if layout == "native":
                _ = compiled_lib.mps_istft_overlap_add_div_envelope(
                    frames_native, window_for_kernel, window_sq_for_kernel, int(hop_length), int(full_length)
                )
            else:
                frames_transposed = frames_native.transpose(-1, -2).contiguous()
                _ = compiled_lib.mps_istft_overlap_add_div_envelope_transposed(
                    frames_transposed, window_for_kernel, window_sq_for_kernel, int(hop_length), int(full_length)
                )

    for _ in range(2):
        _run("native")
        _run("transposed")
    torch.mps.synchronize()

    import time

    native_trials = []
    transposed_trials = []
    # Alternate order each round to reduce temporal drift bias.
    for ridx in range(_LAYOUT_PROBE_ROUNDS):
        first = "native" if (ridx % 2 == 0) else "transposed"
        second = "transposed" if first == "native" else "native"

        start = time.perf_counter()
        for _ in range(_LAYOUT_PROBE_ITERS):
            _run(first)
        torch.mps.synchronize()
        first_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        for _ in range(_LAYOUT_PROBE_ITERS):
            _run(second)
        torch.mps.synchronize()
        second_ms = (time.perf_counter() - start) * 1e3

        if first == "native":
            native_trials.append(first_ms)
            transposed_trials.append(second_ms)
        else:
            transposed_trials.append(first_ms)
            native_trials.append(second_ms)

    def _median(xs: list[float]) -> float:
        ys = sorted(xs)
        return ys[len(ys) // 2]

    def _rel_spread(xs: list[float], med: float) -> float:
        ys = sorted(xs)
        q1 = ys[len(ys) // 4]
        q3 = ys[(3 * len(ys)) // 4]
        return (q3 - q1) / max(med, 1e-12)

    native_ms = _median(native_trials)
    transposed_ms = _median(transposed_trials)
    native_spread = _rel_spread(native_trials, native_ms)
    transposed_spread = _rel_spread(transposed_trials, transposed_ms)

    # Guardrail: if probing is noisy, keep native.
    high_jitter = (native_spread > _LAYOUT_MAX_REL_SPREAD) or (transposed_spread > _LAYOUT_MAX_REL_SPREAD)

    # Conservative policy: choose transposed only on clear, stable wins.
    if (not high_jitter) and (transposed_ms < native_ms):
        rel_win = (native_ms - transposed_ms) / max(native_ms, 1e-12)
        best = "transposed" if rel_win >= _LAYOUT_TIE_MARGIN else "native"
    else:
        best = "native"
    _layout_cache_set(key, best)
    return best


def _check_nola_safety(
    *,
    window_sq: torch.Tensor,
    hop_length: int,
    n_fft: int,
    n_frames: int,
    center: bool,
    length: Optional[int],
    safety: str,
    torch_like: bool,
) -> None:
    if safety not in {"off", "auto", "always"}:
        raise ValueError("safety must be one of {'off', 'auto', 'always'}")
    if safety == "off":
        return

    window_sq_cpu = window_sq.detach().to("cpu")
    digest = hashlib.blake2b(window_sq_cpu.numpy().tobytes(), digest_size=16).hexdigest()
    key = (
        n_fft,
        hop_length,
        int(window_sq_cpu.numel()),
        bool(center),
        int(n_frames),
        int(length) if length is not None else -1,
        digest,
    )
    if safety == "auto":
        cached = _cache_get(key)
        if cached is not None:
            ok, min_abs = cached
            if (not ok) and torch_like:
                raise RuntimeError(
                    f"istft: window overlap-add envelope is too small (cached: min={min_abs:.3e})"
                )
            return

    out_len = hop_length * (n_frames - 1) + n_fft
    denom = torch.zeros(out_len, dtype=torch.float32)
    for k in range(n_frames):
        start = k * hop_length
        denom[start : start + n_fft] += window_sq_cpu

    if center:
        pad = n_fft // 2
        if length is None:
            denom = denom[pad:-pad]
        else:
            end = min(int(denom.numel()), pad + int(length))
            denom = denom[pad:end]
    elif length is not None:
        denom = denom[: min(int(length), int(denom.numel()))]

    min_abs = float(denom.abs().min().item()) if denom.numel() > 0 else 0.0
    ok = min_abs >= 1.0e-11

    if safety == "auto":
        _cache_set(key, (ok, min_abs))

    if (not ok) and torch_like:
        raise RuntimeError(
            f"istft: window overlap-add envelope is too small (min={min_abs:.3e})"
        )


def mps_istft_forward(
    spec: torch.Tensor,
    *,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    length: Optional[int] = None,
    torch_like: bool = False,
    safety: str = "auto",
    allow_fused: bool = True,
    long_mode_strategy: str = "custom",
    kernel_dtype: str = "float32",
    kernel_layout: str = "auto",
) -> torch.Tensor:
    """
    Fast MPS ISTFT forward path matching MLX design:
    1) irfft on MPS
    2) custom Metal fused synthesis-window * overlap-add / envelope normalization
       (Torch-style masked divide at tiny envelope values)

    Current implementation requires win_length == n_fft.

    long_mode_strategy:
      - "custom": always use custom fused kernel path.
      - "torch_fallback": for center=True and extended length requests
        (length > trimmed_length), delegate to torch.istft for strict parity.
    """
    orig_2d_input = (spec.dim() == 2)
    spec = _validate_input(spec)

    freq_bins = spec.size(-2)
    n_frames = spec.size(-1)

    if n_fft is None:
        n_fft = (freq_bins - 1) * 2 if onesided else int(freq_bins)
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if win_length != n_fft:
        raise NotImplementedError("Current custom path supports only win_length == n_fft")
    if long_mode_strategy not in {"custom", "torch_fallback"}:
        raise ValueError("long_mode_strategy must be one of {'custom', 'torch_fallback'}")
    if kernel_dtype not in {"float32", "float16", "mixed"}:
        raise ValueError("kernel_dtype must be one of {'float32', 'float16', 'mixed'}")
    if kernel_layout not in {"auto", "native", "transposed"}:
        raise ValueError("kernel_layout must be one of {'auto', 'native', 'transposed'}")

    if window is None:
        window = torch.hann_window(win_length, periodic=True, device=spec.device, dtype=torch.float32)
    else:
        window = window.to(device=spec.device, dtype=torch.float32)
    window_sq = (window * window).contiguous()

    norm = "ortho" if normalized else "backward"

    trimmed_length = hop_length * (n_frames - 1) if center else (hop_length * (n_frames - 1) + n_fft)
    is_long_request = (length is not None) and (int(length) > int(trimmed_length))
    use_torch_fallback = False
    if center and is_long_request:
        if long_mode_strategy == "torch_fallback":
            use_torch_fallback = True

    if use_torch_fallback:
        try:
            y_torch = torch.istft(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                onesided=onesided,
                return_complex=False,
                length=int(length),
            )
            if orig_2d_input:
                y_torch = y_torch.squeeze(0)
            return y_torch
        except RuntimeError as err:
            warnings.warn(
                "torch.istft fallback failed; continuing with custom MPS ISTFT path "
                f"(fallback error: {err})",
                RuntimeWarning,
                stacklevel=2,
            )

    # [B, F, N] -> [B, n_fft, N]
    if onesided:
        time_frames = torch.fft.irfft(spec, n=n_fft, dim=-2, norm=norm)
    else:
        time_frames = torch.fft.ifft(spec, n=n_fft, dim=-2, norm=norm).real
    # Keep native layout [B, W, N] from irfft output.
    if kernel_dtype in {"float16", "mixed"}:
        frames_for_kernel = time_frames.to(dtype=torch.float16).contiguous()
    else:
        frames_for_kernel = time_frames.contiguous()

    if kernel_dtype == "float16":
        window_for_kernel = window.to(dtype=torch.float16)
        window_sq_for_kernel = window_sq.to(dtype=torch.float16)
    elif kernel_dtype == "mixed":
        window_for_kernel = window
        window_sq_for_kernel = window_sq
    else:
        window_for_kernel = window
        window_sq_for_kernel = window_sq

    full_length = n_fft + hop_length * (n_frames - 1)
    fuse_norm = bool(allow_fused)
    if not fuse_norm:
        raise NotImplementedError("allow_fused=False is not implemented in this Torch custom op yet")

    _check_nola_safety(
        window_sq=window_sq,
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        n_frames=int(n_frames),
        center=bool(center),
        length=length,
        safety=str(safety),
        torch_like=bool(torch_like),
    )

    layout_key = (
        int(frames_for_kernel.size(0)),
        int(frames_for_kernel.size(1)),
        int(frames_for_kernel.size(2)),
        int(hop_length),
        int(full_length),
        str(kernel_dtype),
    )
    selected_layout = kernel_layout
    frames_for_kernel_transposed = None
    if selected_layout == "auto":
        cached_layout = _layout_cache_get(layout_key)
        if cached_layout is None:
            selected_layout = _choose_layout_auto(
                frames_native=frames_for_kernel,
                window_for_kernel=window_for_kernel,
                window_sq_for_kernel=window_sq_for_kernel,
                hop_length=int(hop_length),
                full_length=int(full_length),
                kernel_dtype=str(kernel_dtype),
            )
        else:
            selected_layout = cached_layout

    if selected_layout == "transposed" and frames_for_kernel_transposed is None:
        frames_for_kernel_transposed = frames_for_kernel.transpose(-1, -2).contiguous()

    if kernel_dtype == "mixed":
        if selected_layout == "native":
            y = compiled_lib.mps_istft_overlap_add_div_envelope_mixed(
                frames_for_kernel,
                window_for_kernel,
                window_sq_for_kernel,
                int(hop_length),
                int(full_length),
            )
        else:
            y = compiled_lib.mps_istft_overlap_add_div_envelope_mixed_transposed(
                frames_for_kernel_transposed,
                window_for_kernel,
                window_sq_for_kernel,
                int(hop_length),
                int(full_length),
            )
    else:
        if selected_layout == "native":
            y = compiled_lib.mps_istft_overlap_add_div_envelope(
                frames_for_kernel,
                window_for_kernel,
                window_sq_for_kernel,
                int(hop_length),
                int(full_length),
            )
        else:
            y = compiled_lib.mps_istft_overlap_add_div_envelope_transposed(
                frames_for_kernel_transposed,
                window_for_kernel,
                window_sq_for_kernel,
                int(hop_length),
                int(full_length),
            )

    if center:
        pad = n_fft // 2
        if length is None:
            y = y[:, pad:-pad] if y.size(-1) > 2 * pad else y[:, :0]
        else:
            # Torch semantics with center=True and explicit length are equivalent to
            # taking y[pad : pad + length] (with right-pad if request exceeds data).
            start = pad
            target = int(length)
            end = min(y.size(-1), start + target)
            y = y[:, start:end]
            if y.size(-1) < target:
                y = torch.nn.functional.pad(y, (0, target - y.size(-1)))
    elif length is not None:
        if y.size(-1) >= int(length):
            y = y[:, : int(length)]
        else:
            y = torch.nn.functional.pad(y, (0, int(length) - y.size(-1)))

    if orig_2d_input:
        y = y.squeeze(0)

    return y


__all__ = ["mps_istft_forward"]
