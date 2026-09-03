"""Per-order detector noise: Poisson shot + camera floor + optional source flicker.

Zero order (m=0) and |m|>=1 are acquired on different cameras, so each channel
has its own (a, N0, readout, dark). Synthetic samples use Poisson photoelectrons
(and Poisson dark, mean-subtracted). WLS uses the matching variance.

    σ_j² = (a f_j)² + f_j / N0 + b²
    b²   = (n_pix σ_read² + n_dark) / N0²
    n_dark = dark_rate * t_exp * n_pix
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

_POISSON_GAUSS_THRESHOLD = 1.0e8


def _zero_channel() -> "ChannelNoiseConfig":
    return ChannelNoiseConfig(
        camera="DoBEAM2000-2",
        relative_a=0.01,
        n0_electrons=1.0e6,
        readout_e_rms=5.0,
        n_roi_pixels=400,
        dark_e_per_pixel_s=125.0,
        t_exp_s=0.01,
    )


def _first_channel() -> "ChannelNoiseConfig":
    return ChannelNoiseConfig(
        camera="XV4040BSI-HG",
        relative_a=0.01,
        n0_electrons=1.0e6,
        readout_e_rms=2.5,
        n_roi_pixels=400,
        dark_e_per_pixel_s=0.01,
        t_exp_s=1.0,
    )


@dataclass
class ChannelNoiseConfig:
    relative_a: float = 0.0
    n0_electrons: float = 1.0e6
    readout_e_rms: float = 0.0
    n_roi_pixels: int = 400
    dark_e_per_pixel_s: float = 0.0
    t_exp_s: float = 1.0
    camera: str = ""

    def floor_b(self, n0: float | None = None) -> float:
        n0 = float(self.n0_electrons if n0 is None else n0)
        n_pix = max(int(self.n_roi_pixels), 1)
        var_e = n_pix * (float(self.readout_e_rms) ** 2)
        n_dark = float(self.dark_e_per_pixel_s) * float(self.t_exp_s) * n_pix
        return float(np.sqrt(var_e + n_dark) / max(n0, 1.0))

    def shot_alpha(self, n0: float | None = None) -> float:
        n0 = float(self.n0_electrons if n0 is None else n0)
        return 1.0 / max(n0, 1.0)


@dataclass
class DetectorNoiseConfig:
    apply: bool = True
    zero: ChannelNoiseConfig = field(default_factory=_zero_channel)
    first: ChannelNoiseConfig = field(default_factory=_first_channel)

    def channel_for_order(self, order_m: int) -> ChannelNoiseConfig:
        return self.zero if int(order_m) == 0 else self.first

    def set_n0_electrons(self, n0: float) -> None:
        n0 = float(n0)
        self.zero.n0_electrons = n0
        self.first.n0_electrons = n0


def _channel_from_dict(raw: dict | None, default: ChannelNoiseConfig) -> ChannelNoiseConfig:
    if not raw:
        return replace(default)
    allowed = {f.name for f in fields(ChannelNoiseConfig)}
    data = {f.name: getattr(default, f.name) for f in fields(ChannelNoiseConfig)}
    data.update({k: raw[k] for k in raw if k in allowed})
    return ChannelNoiseConfig(**data)


def parse_detector_noise(raw: dict | None) -> DetectorNoiseConfig:
    if not raw:
        return DetectorNoiseConfig()
    data = dict(raw)
    apply = bool(data.pop("apply", True))
    if "zero" in data or "first" in data:
        return DetectorNoiseConfig(
            apply=apply,
            zero=_channel_from_dict(data.get("zero"), _zero_channel()),
            first=_channel_from_dict(data.get("first"), _first_channel()),
        )
    allowed = {f.name for f in fields(ChannelNoiseConfig)}
    flat = {k: data[k] for k in data if k in allowed}
    zero = _channel_from_dict(flat, _zero_channel())
    first = _channel_from_dict(flat, _first_channel())
    return DetectorNoiseConfig(apply=apply, zero=zero, first=first)


def _channel_masks(
    order_m: np.ndarray, noise: DetectorNoiseConfig
) -> list[tuple[np.ndarray, ChannelNoiseConfig]]:
    m = np.asarray(order_m, dtype=int)
    return [(m == 0, noise.zero), (m != 0, noise.first)]


def _poisson_counts(rng: np.random.Generator, mean: np.ndarray) -> np.ndarray:
    """Poisson counts; Gaussian draw only when λ is too large for integer Poisson."""
    lam = np.maximum(np.asarray(mean, dtype=float), 0.0)
    out = np.empty(lam.shape, dtype=float)
    big = lam >= _POISSON_GAUSS_THRESHOLD
    small = ~big
    if np.any(small):
        out[small] = rng.poisson(lam[small]).astype(float)
    if np.any(big):
        out[big] = rng.normal(lam[big], np.sqrt(lam[big]))
    return out


def observation_variance(
    f: np.ndarray,
    noise: DetectorNoiseConfig,
    order_m: np.ndarray,
    *,
    n0_electrons: float | None = None,
) -> np.ndarray:
    x = np.asarray(f, dtype=float)
    orders = np.asarray(order_m, dtype=int)
    if orders.shape != x.shape:
        raise ValueError(f"order_m shape {orders.shape} != f shape {x.shape}")
    var = np.zeros_like(x)
    for mask, ch in _channel_masks(orders, noise):
        if not np.any(mask):
            continue
        n0 = float(ch.n0_electrons if n0_electrons is None else n0_electrons)
        fp = np.maximum(x[mask], 0.0)
        a = float(ch.relative_a)
        b = ch.floor_b(n0)
        var[mask] = (a * fp) ** 2 + fp / max(n0, 1.0) + b * b
    return var


def observation_sigma(
    f: np.ndarray,
    noise: DetectorNoiseConfig,
    order_m: np.ndarray,
    *,
    n0_electrons: float | None = None,
) -> np.ndarray:
    return np.sqrt(observation_variance(f, noise, order_m, n0_electrons=n0_electrons))


def _realize_channel(
    f: np.ndarray,
    ch: ChannelNoiseConfig,
    rng: np.random.Generator,
    *,
    n0_electrons: float | None = None,
) -> np.ndarray:
    n0 = float(ch.n0_electrons if n0_electrons is None else n0_electrons)
    n_pix = max(int(ch.n_roi_pixels), 1)
    mean_e = np.maximum(np.asarray(f, dtype=float), 0.0) * n0
    a = float(ch.relative_a)
    if a > 0.0:
        gain = 1.0 + a * rng.normal(0.0, 1.0, size=mean_e.shape)
        mean_e = mean_e * np.clip(gain, 0.0, None)
    n_photo = _poisson_counts(rng, mean_e)
    n_dark_mean = float(ch.dark_e_per_pixel_s) * float(ch.t_exp_s) * n_pix
    n_dark = _poisson_counts(rng, np.full(mean_e.shape, n_dark_mean))
    read_rms = float(np.sqrt(n_pix) * float(ch.readout_e_rms))
    n_read = rng.normal(0.0, read_rms, size=mean_e.shape)
    y = (n_photo + (n_dark - n_dark_mean) + n_read) / max(n0, 1.0)
    return np.clip(y, 0.0, None)


def add_detector_noise(
    f: np.ndarray,
    noise: DetectorNoiseConfig,
    rng: np.random.Generator,
    order_m: np.ndarray,
    *,
    n0_electrons: float | None = None,
) -> np.ndarray:
    x = np.asarray(f, dtype=float)
    orders = np.asarray(order_m, dtype=int)
    if orders.shape != x.shape:
        raise ValueError(f"order_m shape {orders.shape} != f shape {x.shape}")
    out = np.empty_like(x)
    for mask, ch in _channel_masks(orders, noise):
        if not np.any(mask):
            continue
        out[mask] = _realize_channel(x[mask], ch, rng, n0_electrons=n0_electrons)
    return out


def inverse_sigma_weights(
    f: np.ndarray,
    noise: DetectorNoiseConfig,
    order_m: np.ndarray,
) -> np.ndarray:
    """sqrt-weights for WLS: 1/σ, then normalized to sum(w)=1."""
    sigma = observation_sigma(f, noise, order_m)
    wsqrt = 1.0 / np.maximum(sigma, 1e-30)
    w = wsqrt * wsqrt
    w /= w.sum() + 1e-30
    return np.sqrt(w)
