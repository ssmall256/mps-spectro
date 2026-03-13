from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F

from mps_spectro.stft import mps_stft_forward

MelScale = Literal["htk", "slaney"]
MelNorm = Literal["slaney"] | None
MelOutputScale = Literal["linear", "log", "db"]
LogMelMode = Literal["clamp", "add", "log1p"]

_CACHE_MAX = 128
_FBANK_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_WINDOW_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def _cache_get(cache: OrderedDict, key: tuple) -> Optional[torch.Tensor]:
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _cache_set(cache: OrderedDict, key: tuple, value: torch.Tensor) -> torch.Tensor:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_MAX:
        cache.popitem(last=False)
    return value


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _hz_to_mel(freqs: torch.Tensor, mel_scale: MelScale) -> torch.Tensor:
    if mel_scale == "htk":
        return 2595.0 * torch.log10(torch.tensor(1.0, device=freqs.device, dtype=freqs.dtype) + (freqs / 700.0))
    if mel_scale == "slaney":
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        mels = freqs / f_sp
        log_region = freqs >= min_log_hz
        if torch.any(log_region):
            mels = mels.clone()
            mels[log_region] = min_log_mel + torch.log(freqs[log_region] / min_log_hz) / logstep
        return mels
    raise ValueError(f"Unsupported mel_scale: {mel_scale}")


def _mel_to_hz(mels: torch.Tensor, mel_scale: MelScale) -> torch.Tensor:
    if mel_scale == "htk":
        return 700.0 * (torch.pow(torch.tensor(10.0, device=mels.device, dtype=mels.dtype), mels / 2595.0) - 1.0)
    if mel_scale == "slaney":
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        freqs = f_sp * mels
        log_region = mels >= min_log_mel
        if torch.any(log_region):
            freqs = freqs.clone()
            freqs[log_region] = min_log_hz * torch.exp(logstep * (mels[log_region] - min_log_mel))
        return freqs
    raise ValueError(f"Unsupported mel_scale: {mel_scale}")


def melscale_fbanks(
    *,
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: MelNorm = None,
    mel_scale: MelScale = "htk",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = torch.device(device)
    key = (
        int(n_freqs),
        float(f_min),
        float(f_max),
        int(n_mels),
        int(sample_rate),
        str(norm),
        str(mel_scale),
        _device_key(device),
        str(dtype),
    )
    cached = _cache_get(_FBANK_CACHE, key)
    if cached is not None:
        return cached

    all_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, int(n_freqs), device=device, dtype=dtype)
    m_min = _hz_to_mel(torch.tensor(float(f_min), device=device, dtype=dtype), mel_scale)
    m_max = _hz_to_mel(torch.tensor(float(f_max), device=device, dtype=dtype), mel_scale)
    m_pts = torch.linspace(float(m_min), float(m_max), int(n_mels) + 2, device=device, dtype=dtype)
    f_pts = _mel_to_hz(m_pts, mel_scale)

    lower = (all_freqs.unsqueeze(0) - f_pts[:-2].unsqueeze(1)) / (f_pts[1:-1] - f_pts[:-2]).unsqueeze(1)
    upper = (f_pts[2:].unsqueeze(1) - all_freqs.unsqueeze(0)) / (f_pts[2:] - f_pts[1:-1]).unsqueeze(1)
    fbanks = torch.clamp(torch.minimum(lower, upper), min=0.0)

    if norm == "slaney":
        enorm = 2.0 / torch.clamp(f_pts[2:] - f_pts[:-2], min=torch.finfo(dtype).eps)
        fbanks = fbanks * enorm.unsqueeze(1)
    elif norm is not None:
        raise ValueError(f"Unsupported mel norm: {norm}")

    return _cache_set(_FBANK_CACHE, key, fbanks.contiguous())


def amplitude_to_db(
    x: torch.Tensor,
    *,
    stype: Literal["power", "magnitude"] = "power",
    top_db: float | None = 80.0,
    amin: float = 1e-10,
) -> torch.Tensor:
    multiplier = 10.0 if stype == "power" else 20.0
    x_db = multiplier * torch.log10(torch.clamp(x, min=float(amin)))
    if top_db is not None:
        if x_db.dim() >= 2:
            ref = torch.amax(x_db, dim=-1, keepdim=True)
            ref = torch.amax(ref, dim=-2, keepdim=True)
        else:
            ref = torch.amax(x_db, dim=0, keepdim=True)
        x_db = torch.maximum(x_db, ref - float(top_db))
    return x_db


def _apply_log_output(x: torch.Tensor, *, log_amin: float, log_mode: LogMelMode) -> torch.Tensor:
    if log_mode == "clamp":
        return torch.log(torch.clamp(x, min=float(log_amin)))
    if log_mode == "add":
        return torch.log(x + float(log_amin))
    if log_mode == "log1p":
        return torch.log1p(x / float(log_amin))
    raise ValueError(f"Unsupported log_mode: {log_mode}")


def _resolve_window(
    window: Optional[torch.Tensor],
    *,
    win_length: int,
    device: torch.device,
    dtype: torch.dtype,
    window_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    if window is not None:
        if window.dim() != 1:
            raise ValueError(f"window must be 1D, got shape {tuple(window.shape)}")
        return window.to(device=device, dtype=dtype).contiguous()

    cacheable = window_fn is torch.hann_window
    key = (int(win_length), _device_key(device), str(dtype), "hann") if cacheable else None
    if key is not None:
        cached = _cache_get(_WINDOW_CACHE, key)
        if cached is not None:
            return cached

    win = window_fn(int(win_length), periodic=True, device=device, dtype=dtype).contiguous()
    if key is not None:
        return _cache_set(_WINDOW_CACHE, key, win)
    return win


def _validate_waveform(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    orig_1d = x.dim() == 1
    if orig_1d:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"Expected waveform shape [T] or [B, T], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("Expected floating-point waveform input")
    return x, orig_1d


def _stft_for_mel(
    x: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool,
    pad_mode: Literal["constant", "reflect"],
    use_mps_kernels: bool,
) -> torch.Tensor:
    if use_mps_kernels and x.device.type == "mps":
        if center and pad_mode == "reflect":
            return mps_stft_forward(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
            )
        if center:
            pad = n_fft // 2
            x = F.pad(x.unsqueeze(1), (pad, pad), mode=pad_mode).squeeze(1)
        return mps_stft_forward(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
        )
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )


def _spectral_power(spec: torch.Tensor, power: float) -> torch.Tensor:
    if spec.is_complex():
        if power == 1.0:
            return spec.abs()
        if power == 2.0:
            return spec.real.square() + spec.imag.square()
        return spec.abs().pow(float(power))
    if spec.ndim >= 4 and spec.shape[-1] == 2:
        real = spec[..., 0]
        imag = spec[..., 1]
        if power == 1.0:
            return torch.sqrt(real.square() + imag.square())
        if power == 2.0:
            return real.square() + imag.square()
        return torch.sqrt(real.square() + imag.square()).pow(float(power))
    raise RuntimeError(f"Unsupported STFT output shape: {tuple(spec.shape)}")


def mel_spectrogram(
    x: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    window: Optional[torch.Tensor] = None,
    pad: int = 0,
    center: bool = True,
    pad_mode: Literal["constant", "reflect"] = "constant",
    power: float = 2.0,
    normalized: bool = False,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    norm: MelNorm = None,
    mel_scale: MelScale = "htk",
    output_scale: MelOutputScale = "db",
    log_amin: float = 1e-5,
    log_mode: LogMelMode = "clamp",
    top_db: float | None = 80.0,
    window_fn: Callable[..., torch.Tensor] = torch.hann_window,
    use_mps_kernels: bool = True,
    _projection_fbanks: torch.Tensor | None = None,
) -> torch.Tensor:
    x, orig_1d = _validate_waveform(x)
    device = x.device
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(dtype=dtype)
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    win_length = int(win_length or n_fft)
    if f_max is None:
        f_max = float(sample_rate) / 2.0
    if output_scale not in ("linear", "log", "db"):
        raise ValueError('output_scale must be one of {"linear", "log", "db"}')
    if pad_mode not in ("constant", "reflect"):
        raise ValueError('pad_mode must be one of {"constant", "reflect"}')
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be non-negative")

    if pad:
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="constant").squeeze(1)

    window_t = _resolve_window(window, win_length=win_length, device=device, dtype=dtype, window_fn=window_fn)
    spec = _stft_for_mel(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_t,
        center=center,
        pad_mode=pad_mode,
        use_mps_kernels=use_mps_kernels,
    )

    magnitude = _spectral_power(spec, float(power))
    if normalized:
        magnitude = magnitude * ((1.0 / math.sqrt(n_fft)) ** float(power))

    if _projection_fbanks is None:
        fbanks = melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=norm,
            mel_scale=mel_scale,
            device=device,
            dtype=dtype,
        )
        projection_fbanks = fbanks.t()
    else:
        projection_fbanks = _projection_fbanks
    # Project in [B, T, F] layout to avoid slower broadcasted [1, M, F] matmul on MPS.
    mel = torch.matmul(magnitude.transpose(-2, -1), projection_fbanks).transpose(-2, -1)
    if output_scale == "linear":
        out = mel
    elif output_scale == "log":
        out = _apply_log_output(mel, log_amin=log_amin, log_mode=log_mode)
    else:
        stype = "magnitude" if float(power) == 1.0 else "power"
        out = amplitude_to_db(mel, stype=stype, top_db=top_db)

    if orig_1d:
        return out.squeeze(0)
    return out


class MelSpectrogramTransform(torch.nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
        center: bool = True,
        pad_mode: Literal["constant", "reflect"] = "constant",
        power: float = 2.0,
        norm: MelNorm = None,
        mel_scale: MelScale = "htk",
        output_scale: MelOutputScale = "db",
        log_amin: float = 1e-5,
        log_mode: LogMelMode = "clamp",
        top_db: float | None = 80.0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        use_mps_kernels: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = None if f_max is None else float(f_max)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.power = float(power)
        self.norm = norm
        self.mel_scale = mel_scale
        self.output_scale = output_scale
        self.log_amin = float(log_amin)
        self.log_mode = log_mode
        self.top_db = None if top_db is None else float(top_db)
        self.window_fn = window_fn
        self.use_mps_kernels = bool(use_mps_kernels)
        self._resolved_window_cache_key: tuple | None = None
        self._resolved_window: torch.Tensor | None = None
        self._resolved_projection_fbanks_cache_key: tuple | None = None
        self._resolved_projection_fbanks: torch.Tensor | None = None

    def _resolve_module_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (_device_key(device), str(dtype))
        if self._resolved_window_cache_key != key or self._resolved_window is None:
            self._resolved_window = _resolve_window(
                None,
                win_length=self.win_length,
                device=device,
                dtype=dtype,
                window_fn=self.window_fn,
            )
            self._resolved_window_cache_key = key
        return self._resolved_window

    def _resolve_module_projection_fbanks(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        f_max = float(self.sample_rate) / 2.0 if self.f_max is None else self.f_max
        key = (_device_key(device), str(dtype), f_max)
        if self._resolved_projection_fbanks_cache_key != key or self._resolved_projection_fbanks is None:
            fbanks = melscale_fbanks(
                n_freqs=(self.n_fft // 2) + 1,
                f_min=self.f_min,
                f_max=f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                norm=self.norm,
                mel_scale=self.mel_scale,
                device=device,
                dtype=dtype,
            )
            self._resolved_projection_fbanks = fbanks.t().contiguous()
            self._resolved_projection_fbanks_cache_key = key
        return self._resolved_projection_fbanks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_waveform(x)
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        window = self._resolve_module_window(device=x.device, dtype=dtype)
        projection_fbanks = self._resolve_module_projection_fbanks(device=x.device, dtype=dtype)
        return mel_spectrogram(
            x,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            power=self.power,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm,
            mel_scale=self.mel_scale,
            output_scale=self.output_scale,
            log_amin=self.log_amin,
            log_mode=self.log_mode,
            top_db=self.top_db,
            window_fn=self.window_fn,
            use_mps_kernels=self.use_mps_kernels,
            _projection_fbanks=projection_fbanks,
        )

    @classmethod
    def compat(cls, **kwargs) -> "CompatMelSpectrogramTransform":
        return CompatMelSpectrogramTransform(**kwargs)


class LogMelSpectrogramTransform(MelSpectrogramTransform):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
        center: bool = True,
        pad_mode: Literal["constant", "reflect"] = "constant",
        power: float = 1.0,
        norm: MelNorm = None,
        mel_scale: MelScale = "htk",
        log_amin: float = 1e-5,
        log_mode: LogMelMode = "clamp",
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        use_mps_kernels: bool = True,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=center,
            pad_mode=pad_mode,
            power=power,
            norm=norm,
            mel_scale=mel_scale,
            output_scale="log",
            log_amin=log_amin,
            log_mode=log_mode,
            top_db=None,
            window_fn=window_fn,
            use_mps_kernels=use_mps_kernels,
        )


class CompatMelSpectrogramTransform(MelSpectrogramTransform):
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: Literal["constant", "reflect"] = "reflect",
        norm: MelNorm = None,
        mel_scale: MelScale = "htk",
        output_scale: MelOutputScale = "linear",
        log_amin: float = 1e-5,
        log_mode: LogMelMode = "clamp",
        top_db: float | None = None,
        use_mps_kernels: bool = True,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=(n_fft // 2) if hop_length is None else hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=center,
            pad_mode=pad_mode,
            power=power,
            norm=norm,
            mel_scale=mel_scale,
            output_scale=output_scale,
            log_amin=log_amin,
            log_mode=log_mode,
            top_db=top_db,
            window_fn=window_fn,
            use_mps_kernels=use_mps_kernels,
        )
        self.pad = int(pad)
        self.normalized = bool(normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_waveform(x)
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        window = self._resolve_module_window(device=x.device, dtype=dtype)
        projection_fbanks = self._resolve_module_projection_fbanks(device=x.device, dtype=dtype)
        return mel_spectrogram(
            x,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            pad=self.pad,
            center=self.center,
            pad_mode=self.pad_mode,
            power=self.power,
            normalized=self.normalized,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm,
            mel_scale=self.mel_scale,
            output_scale=self.output_scale,
            log_amin=self.log_amin,
            log_mode=self.log_mode,
            top_db=self.top_db,
            window_fn=self.window_fn,
            use_mps_kernels=self.use_mps_kernels,
            _projection_fbanks=projection_fbanks,
        )


__all__ = [
    "MelSpectrogramTransform",
    "LogMelSpectrogramTransform",
    "CompatMelSpectrogramTransform",
    "mel_spectrogram",
    "melscale_fbanks",
    "amplitude_to_db",
]
