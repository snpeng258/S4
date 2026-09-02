#!/usr/bin/env python3
"""
Reconstruct reflected far-field |E|^2 on a 2D plane from S4 GetWaves("Air") TSV.

Floquet sum in S4 lattice-normalized coordinates, then subtract S4 planewave incident
(same convention as s4_total_to_reflected.py / run6).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_NEAR = Path(__file__).resolve().parents[1] / "nearfield" / "1d"
if str(_NEAR) not in sys.path:
    sys.path.insert(0, str(_NEAR))

from s4_total_to_reflected import s4_make_excitation_hx_hy  # noqa: E402


DEFAULT_FARFIELD_FIGSIZE = (8.0, 6.5)
DEFAULT_AXIS_HALF_MM = 50.0  # CCD 100 mm → u,v ∈ ±50 mm


@dataclass
class WaveMode:
    kx: float
    ky: float
    kz: complex
    u: tuple[float, float, float]
    cu: complex
    cv: complex


@dataclass
class SimMeta:
    wl_nm: float
    angle_deg: float
    period_nm: float
    depth_nm: float = 40.0
    duty: float = 0.5
    ng: int = 31
    azimuth_deg: float = 0.0


def load_waves_tsv(path: Path) -> tuple[SimMeta, list[WaveMode]]:
    meta = SimMeta(wl_nm=13.5, angle_deg=80.0, period_nm=80.0)
    waves: list[WaveMode] = []
    in_block = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("# wl_nm="):
                m = re.search(
                    r"wl_nm=([\d.]+).*angle_deg=([\d.]+)"
                    r"(?:.*azimuth_deg=([\d.]+))?.*period_nm=([\d.]+)"
                    r"(?:.*depth_nm=([\d.]+))?(?:.*duty=([\d.]+))?(?:.*NG=(\d+))?",
                    s,
                )
                if m:
                    meta.wl_nm = float(m.group(1))
                    meta.angle_deg = float(m.group(2))
                    if m.group(3):
                        meta.azimuth_deg = float(m.group(3))
                    meta.period_nm = float(m.group(4))
                    if m.group(5):
                        meta.depth_nm = float(m.group(5))
                    if m.group(6):
                        meta.duty = float(m.group(6))
                    if m.group(7):
                        meta.ng = int(m.group(7))
                continue
            if s == "#BEGIN_WAVES_AIR":
                in_block = True
                continue
            if s == "#END_WAVES_AIR":
                break
            if not in_block or not s or s.startswith("#") or s.startswith("idx"):
                continue
            p = s.split("\t")
            if len(p) < 12:
                continue
            try:
                vals = [float(x) for x in p[1:12]]
            except ValueError:
                continue
            kx, ky, kzr, kzi = vals[0], vals[1], vals[2], vals[3]
            ux, uy, uz = vals[4], vals[5], vals[6]
            cu = complex(vals[7], vals[8])
            cv = complex(vals[9], vals[10])
            if not (math.isfinite(cu.real) and math.isfinite(cu.imag) and math.isfinite(cv.real) and math.isfinite(cv.imag)):
                continue
            waves.append(
                WaveMode(
                    kx=kx,
                    ky=ky,
                    kz=complex(kzr, kzi),
                    u=(ux, uy, uz),
                    cu=cu,
                    cv=cv,
                )
            )
    if not waves:
        raise ValueError(f"No wave modes in {path}")
    return meta, waves


@dataclass
class OrderScreen:
    m: int
    kx: float
    ky: float
    kz: float
    x_center_mm: float
    y_center_mm: float
    r_abs: float
    amplitude: float


def order_landing_mm(z_norm: float, w: WaveMode, period_nm: float) -> tuple[float, float]:
    """Intersection of reflected ray (kx, ky, kz) with plane z = z_norm (mm via scale)."""
    scale = period_nm / 1e6
    if abs(w.kz.real) < 1e-30:
        return 0.0, 0.0
    x_mm = z_norm * w.kx / w.kz.real * scale
    y_mm = z_norm * w.ky / w.kz.real * scale
    return x_mm, y_mm


def stitched_view_bounds(
    screens: list[OrderScreen],
    *,
    half_w: float,
    axis_half_mm: float,
) -> tuple[float, float, float, float]:
    """u,v extent (mm) covering all order spots with margin."""
    margin = half_w + 5.0
    x_lo = min(s.x_center_mm for s in screens) - margin
    x_hi = max(s.x_center_mm for s in screens) + margin
    y_lo_raw = min(s.y_center_mm for s in screens) - margin
    y_hi_raw = max(s.y_center_mm for s in screens) + margin
    y_mid = 0.5 * (y_lo_raw + y_hi_raw)
    y_half = max(0.5 * (y_hi_raw - y_lo_raw), abs(axis_half_mm))
    return x_lo, x_hi, y_mid - y_half, y_mid + y_half


def farfield_angle_title(meta: SimMeta, angle_deg: float | None = None) -> str:
    theta = angle_deg if angle_deg is not None else meta.angle_deg
    if abs(meta.azimuth_deg) > 1e-9:
        return f"θ = {theta:.2f}°, φ = {meta.azimuth_deg:.2f}°"
    return f"θ = {theta:.2f}°"


def load_flux_tsv(path: Path) -> dict[int, float]:
    flux: dict[int, float] = {}
    in_block = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if s == "#BEGIN_FLUX_TSV":
                in_block = True
                continue
            if s == "#END_FLUX_TSV":
                break
            if in_block and s and not s.startswith("#") and not s.startswith("m\t"):
                parts = s.split("\t")
                if len(parts) >= 3:
                    flux[int(parts[0])] = float(parts[2])
    return flux


def _incident_kx0(waves: list[WaveMode]) -> float:
    """Forward propagating excitation kx (kz>0) from GetWaves."""
    for w in waves:
        if w.kz.real > 0 and abs(w.kz.imag) < 1e-12:
            return w.kx
    raise ValueError("No forward propagating mode found for incident kx")


def sigma_theta_rad(*, wl_nm: float, w0_um: float) -> float:
    """Angular spread (rad) of Gaussian beam: σ_θ ≈ λ/(π w0)."""
    w0_nm = w0_um * 1000.0
    if w0_nm <= 0:
        return 0.0
    return wl_nm / (math.pi * w0_nm)


def sigma_x_mm(*, z_mm: float, wl_nm: float, w0_um: float, scale: float = 1.0) -> float:
    """Far-field blur σ_x (mm) from beam angular spread at distance z."""
    return abs(z_mm) * sigma_theta_rad(wl_nm=wl_nm, w0_um=w0_um) * scale


def gaussian_angle_weights(
    theta0_deg: float,
    *,
    wl_nm: float,
    w0_um: float,
    n_theta: int,
    sigma_scale: float = 3.0,
) -> tuple[list[float], list[float]]:
    """Discrete Gaussian angular spectrum amplitudes for coherent plane-wave sum."""
    if n_theta <= 1:
        return [theta0_deg], [1.0]
    sigma_deg = math.degrees(sigma_theta_rad(wl_nm=wl_nm, w0_um=w0_um))
    if sigma_deg < 1e-15:
        return [theta0_deg], [1.0]
    span = sigma_scale * sigma_deg
    import numpy as np

    thetas = np.linspace(theta0_deg - span, theta0_deg + span, n_theta)
    thetas[n_theta // 2] = float(theta0_deg)
    weights = np.exp(-0.5 * ((thetas - theta0_deg) / sigma_deg) ** 2)
    weights = weights / np.sum(weights)
    return thetas.tolist(), weights.tolist()


def auto_unified_x_span(
    waves: list[WaveMode],
    flux: dict[int, float],
    *,
    z_mm: float,
    period_nm: float,
    margin_mm: float = 10.0,
    sigma_x_mm: float = 0.0,
    r_threshold: float = 1e-5,
) -> float:
    """Half-width (mm) for unified far-field x grid covering all propagating orders."""
    screens = compute_order_screens(
        waves,
        z_mm=z_mm,
        period_nm=period_nm,
        flux=flux,
        r_threshold=r_threshold,
    )
    if not screens:
        return 250.0
    max_abs_x = max(abs(s.x_center_mm) for s in screens)
    return max_abs_x + margin_mm + 3.0 * abs(sigma_x_mm)


def compute_order_screens(
    waves: list[WaveMode],
    *,
    z_mm: float,
    period_nm: float,
    flux: dict[int, float] | None = None,
    r_threshold: float = 1e-5,
) -> list[OrderScreen]:
    """Landing (u,v) mm on plane z=-z_mm for each propagating reflected order."""
    z_norm = (-abs(z_mm) * 1e6) / period_nm
    kx0 = _incident_kx0(waves)
    screens: list[OrderScreen] = []
    seen: set[int] = set()

    for w in waves:
        if w.kz.real >= 0 or abs(w.kz.imag) > 1e-12:
            continue
        amp = abs(w.cu) + abs(w.cv)
        if amp <= 0:
            continue
        m = int(round((kx0 - w.kx) / (2.0 * math.pi)))
        if m in seen:
            continue
        seen.add(m)
        r_abs = flux.get(m, 0.0) if flux else 0.0
        if flux and r_abs < r_threshold:
            continue
        x_center_mm, y_center_mm = order_landing_mm(z_norm, w, period_nm)
        screens.append(
            OrderScreen(
                m=m,
                kx=w.kx,
                ky=w.ky,
                kz=w.kz.real,
                x_center_mm=x_center_mm,
                y_center_mm=y_center_mm,
                r_abs=r_abs,
                amplitude=amp,
            )
        )

    screens.sort(key=lambda s: s.m)
    return screens


def waves_for_order(waves: list[WaveMode], m: int) -> list[WaveMode]:
    """Single reflected propagating Floquet mode for diffraction order m."""
    kx0 = _incident_kx0(waves)
    out: list[WaveMode] = []
    for w in waves:
        if w.kz.real >= 0 or abs(w.kz.imag) > 1e-12:
            continue
        if int(round((kx0 - w.kx) / (2.0 * math.pi))) == m:
            out.append(w)
    return out


def _single_order_field_grid(
    meta: SimMeta,
    mode_waves: list[WaveMode],
    *,
    z_mm: float,
    x_center_mm: float,
    y_center_mm: float = 0.0,
    x_half_span_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
):
    """Field from one reflected Floquet order only (discrete spot on its screen)."""
    import numpy as np

    if not mode_waves:
        raise ValueError("empty mode list")
    period = meta.period_nm
    z_nm = -abs(z_mm) * 1e6
    z_norm = z_nm / period
    x_mm = np.linspace(x_center_mm - x_half_span_mm, x_center_mm + x_half_span_mm, nx)
    y_mm = np.linspace(y_center_mm - y_half_span_mm, y_center_mm + y_half_span_mm, ny)
    xg, yg = np.meshgrid(x_mm * 1e6 / period, y_mm * 1e6 / period, indexing="xy")
    ex, ey, ez = floquet_total_field(xg, yg, z_norm, mode_waves)
    return x_mm, y_mm, ex, ey, ez


def single_order_intensity_at_screen(
    meta: SimMeta,
    waves: list[WaveMode],
    m: int,
    *,
    z_mm: float,
    x_center_mm: float,
    y_center_mm: float = 0.0,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
):
    import numpy as np

    mw = waves_for_order(waves, m)
    if not mw:
        return None
    x_mm, y_mm, ex, ey, ez = _single_order_field_grid(
        meta,
        mw,
        z_mm=z_mm,
        x_center_mm=x_center_mm,
        y_center_mm=y_center_mm,
        x_half_span_mm=order_half_width_mm,
        y_half_span_mm=y_half_span_mm,
        nx=nx,
        ny=ny,
    )
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    return x_mm, y_mm, intensity


def order_field_at_origin(
    meta: SimMeta,
    mode_waves: list[WaveMode],
    *,
    z_mm: float,
) -> tuple[complex, complex, complex]:
    """Single-order Floquet field at local origin (x=y=0) on the receiver plane."""
    z_norm = (-abs(z_mm) * 1e6) / meta.period_nm
    ex, ey, ez = floquet_total_field(0.0, 0.0, z_norm, mode_waves)
    return complex(ex), complex(ey), complex(ez)


def order_peak_intensity(meta: SimMeta, waves: list[WaveMode], m: int, *, z_mm: float) -> float:
    """|E_order|² at the order spot center (single angle)."""
    mw = waves_for_order(waves, m)
    if not mw:
        return 0.0
    ex, ey, ez = order_field_at_origin(meta, mw, z_mm=z_mm)
    return float(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2)


def spot_grid_params(
    sigma_x_mm: float,
    *,
    min_half_mm: float = 0.02,
    sigma_span: float = 5.0,
    min_nx: int = 32,
    samples_per_sigma: float = 4.0,
) -> tuple[float, int]:
    """Patch half-width (mm) and nx that resolve a Gaussian spot of width σ_x."""
    half = max(min_half_mm, sigma_span * sigma_x_mm)
    dx_target = max(sigma_x_mm / samples_per_sigma, 1e-9)
    nx = max(min_nx, int(2.0 * half / dx_target) | 1)
    return half, nx


def finite_spot_intensity_patch(
    *,
    x_center_mm: float,
    y_center_mm: float = 0.0,
    peak_intensity: float,
    sigma_x_mm: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
    sigma_y_mm: float | None = None,
    auto_grid: bool = True,
):
    """
    Gaussian far-field spot from beam angular spread (σ_x ≈ z λ/(π w0)).

    Plane-wave Floquet patches are uniform in |E|²; convolving them does not
    produce a finite spot. Use this envelope instead.
    """
    import numpy as np

    if peak_intensity <= 0 or sigma_x_mm <= 0:
        return None
    sy = sigma_y_mm if sigma_y_mm is not None else sigma_x_mm
    if auto_grid:
        spot_half, nx_eff = spot_grid_params(sigma_x_mm)
        spot_half = min(spot_half, order_half_width_mm)
        ny_eff = max(32, int(ny * spot_half / max(y_half_span_mm, 1e-9)) | 1)
        y_half = min(y_half_span_mm, max(sy * 5.0, 0.005))
    else:
        spot_half, nx_eff, ny_eff, y_half = order_half_width_mm, nx, ny, y_half_span_mm
    x_mm = np.linspace(x_center_mm - spot_half, x_center_mm + spot_half, nx_eff)
    y_mm = np.linspace(y_center_mm - y_half, y_center_mm + y_half, ny_eff)
    x_rel = x_mm[np.newaxis, :] - x_center_mm
    y_rel = y_mm[:, np.newaxis] - y_center_mm
    intensity = peak_intensity * np.exp(
        -0.5 * (x_rel / sigma_x_mm) ** 2 - 0.5 * (y_rel / sy) ** 2
    )
    return x_mm, y_mm, intensity


def _intensity_norm_for_scale(
    scale: str,
    vmax: float,
    *,
    vmin: float | None = None,
    peaks: list[float] | None = None,
):
    """Colormap normalization: linear, sqrt, log, or asinh (wide dynamic range)."""
    from matplotlib.colors import AsinhNorm, LogNorm, Normalize, PowerNorm

    if vmax <= 0:
        return Normalize(vmin=0, vmax=1.0)
    scale = scale.strip().lower()
    if vmin is None:
        if peaks:
            pos = sorted(p for p in peaks if p > 0)
            vmin = pos[0] * 0.25 if len(pos) > 1 else vmax * 1e-4
        else:
            vmin = vmax * 1e-4
    vmin = max(float(vmin), 1e-30)
    if scale == "sqrt":
        return PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    if scale == "log":
        return LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax)
    if scale == "asinh":
        return AsinhNorm(vmin=0.0, vmax=vmax, linear_width=max(vmax * 0.02, vmin * 4))
    return Normalize(vmin=0.0, vmax=vmax)


def _colorbar_label_for_scale(base: str, scale: str) -> str:
    scale = scale.strip().lower()
    if scale == "sqrt":
        return f"{base} (√)"
    if scale == "log":
        return f"{base} (log)"
    if scale == "asinh":
        return f"{base} (asinh)"
    return base


def _spot_peaks_from_patches(patches: list[tuple]) -> list[float]:
    import numpy as np

    return [float(np.max(p[2])) for p in patches if float(np.max(p[2])) > 0]


def _build_gaussian_spot_canvas(
    screens: list[OrderScreen],
    patches: list[tuple],
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    nx_g: int,
    ny_g: int,
    sigma_x_mm: float,
    min_render_sigma_mm: float,
) -> tuple:
    """Stamp Gaussian spots onto a regular grid; returns (canvas, xg, yg)."""
    import numpy as np

    canvas = np.zeros((ny_g, nx_g), dtype=float)
    xg = np.linspace(x_lo, x_hi, nx_g)
    yg = np.linspace(y_lo, y_hi, ny_g)
    if sigma_x_mm <= 0:
        return canvas, xg, yg
    canvas_dx = (x_hi - x_lo) / max(nx_g - 1, 1)
    render_sigma = max(sigma_x_mm, min_render_sigma_mm, canvas_dx * 0.5)
    for scr, (_x_loc, _y_loc, inten) in zip(screens, patches):
        peak = float(np.max(inten))
        if peak <= 0:
            continue
        stamp = peak * np.exp(
            -0.5 * ((xg[np.newaxis, :] - scr.x_center_mm) / render_sigma) ** 2
            -0.5 * ((yg[:, np.newaxis] - scr.y_center_mm) / render_sigma) ** 2
        )
        canvas = np.maximum(canvas, stamp)
    return canvas, xg, yg


def _prepare_canvas_for_display(canvas, norm, scale: str):
    import numpy as np

    data = np.asarray(canvas, dtype=float)
    if scale.strip().lower() == "log" and hasattr(norm, "vmin"):
        data = np.maximum(data, float(norm.vmin))
    return data


def _display_cmap(name: str):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    return cmap


def plot_stitched_order_spots(
    screens: list[OrderScreen],
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    z_mm: float,
    meta: SimMeta,
    out_path: Path,
    dpi: int,
    polarization: str,
    title_suffix: str = "",
    title_angle_deg: float | None = None,
    sigma_x_mm: float = 0.0,
    axis_half_mm: float = DEFAULT_AXIS_HALF_MM,
    intensity_scale: str = "log",
    cmap: str = "inferno",
) -> None:
    """Per-order Gaussian spots on receiver plane; axes in lab (u,v) mm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not screens or not patches:
        raise ValueError("no order spots to stitch")

    half_w = 0.5 * (patches[0][0][-1] - patches[0][0][0])
    x_lo, x_hi, y_lo, y_hi = stitched_view_bounds(
        screens, half_w=half_w, axis_half_mm=axis_half_mm
    )
    nx_g = 1200
    ny_g = max(64, int(nx_g * (y_hi - y_lo) / max(x_hi - x_lo, 1e-9)))
    canvas_dx = (x_hi - x_lo) / max(nx_g - 1, 1)
    min_render = canvas_dx * 2.0 if sigma_x_mm > 0 else 0.0
    canvas, _xg, _yg = _build_gaussian_spot_canvas(
        screens,
        patches,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        nx_g=nx_g,
        ny_g=ny_g,
        sigma_x_mm=sigma_x_mm,
        min_render_sigma_mm=min_render,
    )
    peaks = _spot_peaks_from_patches(patches)
    vmax = max(peaks) if peaks else 1.0
    norm = _intensity_norm_for_scale(intensity_scale, vmax, peaks=peaks)
    display = _prepare_canvas_for_display(canvas, norm, intensity_scale)
    cb_label = _colorbar_label_for_scale("|E_order|²", intensity_scale)
    cmap_obj = _display_cmap(cmap)

    pol = polarization.strip().upper() or "TE"
    suffix = f" {title_suffix}".rstrip()
    fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FARFIELD_FIGSIZE, facecolor="w")
    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=(x_lo, x_hi, y_lo, y_hi),
        cmap=cmap_obj,
        norm=norm,
        interpolation="bilinear",
    )
    ax.set_xlabel("u (mm)", fontsize=12)
    ax.set_ylabel("v (mm)", fontsize=12)
    ax.set_title(
        f"Far-field discrete orders{suffix} — "
        f"λ = {meta.wl_nm:.1f} nm, {farfield_angle_title(meta, title_angle_deg)}, "
        f"{pol}, z = {z_mm:g} mm",
        fontsize=12,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_order_spot_panels(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
    out_path: Path,
    dpi: int,
    polarization: str,
    title_suffix: str = "",
) -> None:
    """Multi-panel: one receiving screen per diffraction order (single-mode field)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not screens:
        raise ValueError("No order screens to plot")

    pol = polarization.strip().upper() or "TE"
    n = len(screens)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), facecolor="w", squeeze=False
    )

    for scr, ax in zip(screens, axes.ravel()):
        patch = single_order_intensity_at_screen(
            meta,
            waves,
            scr.m,
            z_mm=z_mm,
            x_center_mm=scr.x_center_mm,
            y_center_mm=scr.y_center_mm,
            order_half_width_mm=order_half_width_mm,
            y_half_span_mm=y_half_span_mm,
            nx=nx,
            ny=ny,
        )
        if patch is None:
            ax.set_visible(False)
            continue
        x_mm, y_mm, intensity = patch
        im = ax.imshow(
            intensity,
            origin="lower",
            aspect="auto",
            extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
            cmap="viridis",
        )
        ax.set_xlabel("x (mm)", fontsize=9)
        ax.set_ylabel("y (mm)", fontsize=9)
        ax.set_title(
            f"m = {scr.m:+d}, u = {scr.x_center_mm:.1f}, v = {scr.y_center_mm:.1f} mm\n"
            f"R = {scr.r_abs:.4g}",
            fontsize=10,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    suffix = f" {title_suffix}".rstrip()
    fig.suptitle(
        f"Per-order receiving screens{suffix} — λ = {meta.wl_nm:.1f} nm, "
        f"{farfield_angle_title(meta)}, {pol}, z = {z_mm:g} mm",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w", bbox_inches="tight")
    plt.close(fig)


def choose_farfield_screen_mode(mode: str, angle_deg: float) -> str:
    mode = mode.strip().lower()
    if mode in ("orders", "stitched", "unified"):
        return mode
    if abs(angle_deg) >= 60.0:
        return "stitched"
    return "unified"


def _v_hat_from_k_u(kx: float, ky: float, kz: complex, u: tuple[float, float, float]):
    import numpy as np

    k = np.array([kx, ky, kz], dtype=complex)
    uvec = np.array(u, dtype=float)
    kn = np.linalg.norm(k)
    if kn < 1e-30:
        return uvec, np.array([0.0, 1.0, 0.0])
    khat = k / kn
    v = np.cross(khat, uvec)
    vn = np.linalg.norm(v)
    if vn < 1e-30:
        v = np.array([0.0, 1.0, 0.0])
    else:
        v = v / vn
    un = np.linalg.norm(uvec)
    if un < 1e-30:
        uvec = np.array([1.0, 0.0, 0.0])
    else:
        uvec = uvec / un
    return uvec, v


def _reflected_field_grid(
    meta: SimMeta,
    waves: list[WaveMode],
    *,
    z_mm: float,
    x_center_mm: float,
    y_center_mm: float = 0.0,
    x_half_span_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
):
    import numpy as np

    period = meta.period_nm
    z_nm = -abs(z_mm) * 1e6
    z_norm = z_nm / period
    x_mm = np.linspace(x_center_mm - x_half_span_mm, x_center_mm + x_half_span_mm, nx)
    y_mm = np.linspace(y_center_mm - y_half_span_mm, y_center_mm + y_half_span_mm, ny)
    xg, yg = np.meshgrid(x_mm * 1e6 / period, y_mm * 1e6 / period, indexing="xy")
    ex_t, ey_t, ez_t = floquet_total_field(xg, yg, z_norm, waves)
    ex_i, ey_i, ez_i = s4_incident_grid(xg, yg, z_nm, meta=meta)
    ex_r = ex_t - ex_i
    ey_r = ey_t - ey_i
    ez_r = ez_t - ez_i
    return x_mm, y_mm, ex_r, ey_r, ez_r


def _reflected_intensity_grid(
    meta: SimMeta,
    waves: list[WaveMode],
    *,
    z_mm: float,
    x_center_mm: float,
    y_center_mm: float = 0.0,
    x_half_span_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
):
    import numpy as np

    x_mm, y_mm, ex_r, ey_r, ez_r = _reflected_field_grid(
        meta,
        waves,
        z_mm=z_mm,
        x_center_mm=x_center_mm,
        y_center_mm=y_center_mm,
        x_half_span_mm=x_half_span_mm,
        y_half_span_mm=y_half_span_mm,
        nx=nx,
        ny=ny,
    )
    intensity = np.abs(ex_r) ** 2 + np.abs(ey_r) ** 2 + np.abs(ez_r) ** 2
    return x_mm, y_mm, intensity


def plot_spot_intensity(
    x_mm,
    y_mm,
    intensity,
    *,
    out_path: Path,
    title: str,
    dpi: int = 150,
    cbar_label: str = "|E_refl|²",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.5), facecolor="w")
    im = ax.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        extent=(float(x_mm[0]), float(x_mm[-1]), float(y_mm[0]), float(y_mm[-1])),
        cmap="viridis",
    )
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_order_screens(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
    out_path: Path,
    dpi: int,
    polarization: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not screens:
        raise ValueError("No order screens to plot")

    pol = polarization.strip().upper() or "TE"
    n = len(screens)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        facecolor="w",
        squeeze=False,
    )

    for idx, (scr, ax) in enumerate(zip(screens, axes.ravel())):
        x_mm, y_mm, intensity = _reflected_intensity_grid(
            meta,
            waves,
            z_mm=z_mm,
            x_center_mm=scr.x_center_mm,
            y_center_mm=scr.y_center_mm,
            x_half_span_mm=order_half_width_mm,
            y_half_span_mm=y_half_span_mm,
            nx=nx,
            ny=ny,
        )
        im = ax.imshow(
            intensity,
            origin="lower",
            aspect="auto",
            extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
            cmap="viridis",
        )
        ax.set_xlabel("x (mm)", fontsize=9)
        ax.set_ylabel("y (mm)", fontsize=9)
        ax.set_title(
            f"m = {scr.m:+d}, u = {scr.x_center_mm:.1f}, v = {scr.y_center_mm:.1f} mm\n"
            f"R = {scr.r_abs:.4g}",
            fontsize=10,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Reflected far-field order screens — λ = {meta.wl_nm:.1f} nm, "
        f"{farfield_angle_title(meta)}, {pol}, z = {z_mm:g} mm",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w", bbox_inches="tight")
    plt.close(fig)


def plot_overview_strip(
    meta: SimMeta,
    screens: list[OrderScreen],
    *,
    z_mm: float,
    out_path: Path,
    dpi: int,
    polarization: str,
) -> None:
    """Schematic x-axis map showing where each order screen is placed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not screens:
        return

    pol = polarization.strip().upper() or "TE"
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 2.8), facecolor="w")
    rs = [max(s.r_abs, 1e-12) for s in screens]
    colors = plt.cm.viridis(np.array(rs) / max(rs))

    for scr, c in zip(screens, colors):
        ax.axvline(scr.x_center_mm, color=c, alpha=0.35, linewidth=8, zorder=1)
        ax.plot(scr.x_center_mm, scr.r_abs, "o", color=c, markersize=8, zorder=2)
        ax.annotate(
            f"m={scr.m:+d}",
            (scr.x_center_mm, scr.r_abs),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Screen center x (mm)", fontsize=11)
    ax.set_ylabel("R_abs", fontsize=11)
    ax.set_title(
        f"Order screen positions @ z = {z_mm:g} mm — λ = {meta.wl_nm:.1f} nm, "
        f"θ = {meta.angle_deg:.2f}°, {pol}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def _skip_evanescent_overflow(kz: complex, z_norm: float, *, max_log: float = 700.0) -> bool:
    """Skip modes whose exp(i kz z) would overflow at this z (40 mm >> evanescent decay length)."""
    decay_log = -kz.imag * z_norm
    return decay_log > max_log


def floquet_total_field(
    x_norm,
    y_norm,
    z_norm: float,
    waves: list[WaveMode],
):
    """Sum GetWaves modes: E = sum (cu*u + cv*v) * exp(i k·r_norm)."""
    import numpy as np

    x = np.asarray(x_norm, dtype=float)
    y = np.asarray(y_norm, dtype=float)
    ex = np.zeros(x.shape, dtype=complex)
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    for w in waves:
        if _skip_evanescent_overflow(w.kz, z_norm):
            continue
        uhat, vhat = _v_hat_from_k_u(w.kx, w.ky, w.kz, w.u)
        phase = np.exp(1j * (w.kx * x + w.ky * y + w.kz * z_norm))
        evec = w.cu * uhat + w.cv * vhat
        ex = ex + evec[0] * phase
        ey = ey + evec[1] * phase
        ez = ez + evec[2] * phase
    return ex, ey, ez


def s4_incident_grid(
    x_norm,
    y_norm,
    z_nm: float,
    *,
    meta: SimMeta,
    n_inc_real: float = 1.0,
):
    import numpy as np

    x = np.asarray(x_norm, dtype=float)
    y = np.asarray(y_norm, dtype=float)
    period = meta.period_nm
    wl = meta.wl_nm
    angle = meta.angle_deg
    k0 = 2.0 * math.pi / wl
    omega0 = 2.0 * math.pi * (period / wl)
    kz_phys = -k0 * n_inc_real * math.cos(math.radians(angle))
    hx, hy, kx0_norm, ky0_norm = s4_make_excitation_hx_hy(
        angle,
        meta.azimuth_deg,
        pol_s_amp=1.0,
        pol_s_phase_deg=0.0,
        pol_p_amp=0.0,
        pol_p_phase_deg=0.0,
        n_inc_real=n_inc_real,
    )
    phase = kx0_norm * x + ky0_norm * y + (-1.0) * kz_phys * z_nm
    pf = np.exp(1j * phase)
    Hx = hx * pf
    Hy = hy * pf
    kx_p = kx0_norm / period
    ky_p = ky0_norm / period
    ex = (ky_p * 0.0 - kz_phys * Hy) / omega0
    ey = (kz_phys * Hx - kx_p * 0.0) / omega0
    ez = (kx_p * Hy - ky_p * Hx) / omega0
    return ex, ey, ez


def plot_reflected_farfield(
    meta: SimMeta,
    waves: list[WaveMode],
    *,
    z_mm: float,
    x_span_mm: float,
    y_span_mm: float,
    nx: int,
    ny: int,
    out_path: Path,
    dpi: int,
    polarization: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_mm, y_mm, intensity = _reflected_intensity_grid(
        meta,
        waves,
        z_mm=z_mm,
        x_center_mm=0.0,
        x_half_span_mm=x_span_mm,
        y_half_span_mm=y_span_mm,
        nx=nx,
        ny=ny,
    )

    pol = polarization.strip().upper() or "TE"
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.5), facecolor="w")
    im = ax.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
        cmap="viridis",
    )
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title(
        f"Reflected far-field |E|² — λ = {meta.wl_nm:.1f} nm, "
        f"{farfield_angle_title(meta)}, {pol} polarization, z = {z_mm:g} mm",
        fontsize=12,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|E_refl|²", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def choose_screen_mode(mode: str, angle_deg: float) -> str:
    mode = mode.strip().lower()
    if mode in ("orders", "unified"):
        return mode
    # Grazing incidence: orders land far off-axis; use per-order screens by default.
    if abs(angle_deg) >= 60.0:
        return "orders"
    return "unified"


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 GetWaves far-field reflected |E|^2 reconstruction")
    ap.add_argument("waves_tsv", help="au_grating_farfield.lua output with #BEGIN_WAVES_AIR")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--z-mm", type=float, default=40.0, help="observation distance in air (mm), z<0")
    ap.add_argument("--x-span-mm", type=float, default=15.0)
    ap.add_argument("--y-span-mm", type=float, default=15.0)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--polarization", type=str, default="TE")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--screen-mode",
        choices=("auto", "orders", "unified"),
        default="auto",
        help="orders: one screen per diffraction order at its landing x; unified: centered ±x-span",
    )
    ap.add_argument(
        "--order-half-width-mm",
        type=float,
        default=8.0,
        help="half-width of each order screen in x (mm)",
    )
    ap.add_argument(
        "--order-r-thresh",
        type=float,
        default=1e-5,
        help="minimum R_abs to include an order screen",
    )
    ap.add_argument(
        "--overview-out",
        default="",
        help="optional PNG path for order-position overview strip",
    )
    args = ap.parse_args()

    waves_path = Path(args.waves_tsv)
    meta, waves = load_waves_tsv(waves_path)
    flux = load_flux_tsv(waves_path)
    out = Path(args.out)
    mode = choose_screen_mode(args.screen_mode, meta.angle_deg)

    if mode == "orders":
        screens = compute_order_screens(
            waves,
            z_mm=args.z_mm,
            period_nm=meta.period_nm,
            flux=flux,
            r_threshold=args.order_r_thresh,
        )
        if not screens:
            raise SystemExit("No propagating orders above R threshold; try lowering --order-r-thresh")
        print(f"Order screens ({len(screens)}):", file=sys.stderr)
        for scr in screens:
            print(
                f"  m={scr.m:+d}  u={scr.x_center_mm:.2f} mm  v={scr.y_center_mm:.2f} mm  "
                f"R_abs={scr.r_abs:.6g}",
                file=sys.stderr,
            )
        plot_order_screens(
            meta,
            waves,
            screens,
            z_mm=args.z_mm,
            order_half_width_mm=args.order_half_width_mm,
            y_half_span_mm=args.y_span_mm,
            nx=args.nx,
            ny=args.ny,
            out_path=out,
            dpi=args.dpi,
            polarization=args.polarization,
        )
        if args.overview_out:
            plot_overview_strip(
                meta,
                screens,
                z_mm=args.z_mm,
                out_path=Path(args.overview_out),
                dpi=args.dpi,
                polarization=args.polarization,
            )
    else:
        plot_reflected_farfield(
            meta,
            waves,
            z_mm=args.z_mm,
            x_span_mm=args.x_span_mm,
            y_span_mm=args.y_span_mm,
            nx=args.nx,
            ny=args.ny,
            out_path=out,
            dpi=args.dpi,
            polarization=args.polarization,
        )
    print(out.resolve())


if __name__ == "__main__":
    main()
