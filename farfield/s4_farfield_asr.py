#!/usr/bin/env python3
"""
run4: S4 Floquet + elliptical aperture far-field.

Default (asr): per-order k-domain aperture convolution -> propagate -> ASR
(m=0-normal plane) -> midft; incoherent sum of |E_m|² on (u,v).

Optional: fraunhofer (analytic spots), asr-legacy (spatial patch path).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_FAR = Path(__file__).resolve().parent
if str(_FAR) not in sys.path:
    sys.path.insert(0, str(_FAR))

from asr_propagate import (  # noqa: E402
    beam_divergence_info,
    build_order_propagation_k_grid,
    far_field_spot_sigma_mm,
    observation_angles_from_k,
    order_spectrum_k_gaussian,
    scalar_asr_from_spectrum,
    scalar_asr_propagate,
    vacuum_propagating_fxy_from_mode,
)
from s4_farfield_reconstruct import (  # noqa: E402
    DEFAULT_AXIS_HALF_MM,
    DEFAULT_FARFIELD_FIGSIZE,
    OrderScreen,
    SimMeta,
    WaveMode,
    _display_cmap,
    _incident_kx0,
    _intensity_norm_for_scale,
    _prepare_canvas_for_display,
    _v_hat_from_k_u,
    compute_order_screens,
    farfield_angle_title,
    finite_spot_intensity_patch,
    load_flux_tsv,
    load_waves_tsv,
    order_peak_intensity,
    plot_stitched_order_spots,
    spot_grid_params,
    stitched_view_bounds,
    waves_for_order,
)


def reflected_propagating_modes(waves: list[WaveMode]) -> list[WaveMode]:
    out: list[WaveMode] = []
    kx0 = _incident_kx0(waves)
    seen: set[int] = set()
    for w in waves:
        if w.kz.real >= 0 or abs(w.kz.imag) > 1e-12:
            continue
        if abs(w.cu) + abs(w.cv) <= 0:
            continue
        m = int(round((kx0 - w.kx) / (2.0 * math.pi)))
        if m in seen:
            continue
        seen.add(m)
        out.append(w)
    return out


def mode_cartesian_amplitude(w: WaveMode) -> tuple[complex, complex, complex]:
    uhat, vhat = _v_hat_from_k_u(w.kx, w.ky, w.kz, w.u)
    evec = w.cu * uhat + w.cv * vhat
    return complex(evec[0]), complex(evec[1]), complex(evec[2])


def k_to_um_components(w: WaveMode, period_nm: float) -> tuple[float, float, float]:
    scale = (2.0 * math.pi / period_nm) * 1000.0
    return w.kx * scale, w.ky * scale, w.kz.real * scale


def observation_angles_for_mode(
    meta: SimMeta,
    w: WaveMode,
    *,
    theta2_deg: float | None,
    phi2_deg: float | None,
) -> tuple[float, float]:
    """
    ASR (θ₂, φ₂) for k-domain propagation.

    Default θ₂=φ₂=0: fixed horizontal lab CCD; order landing is already in scr center
    and order_local_propagation_transfer handles carrier tilt. Per-order k-based θ₂
    triggers asr_rearrange_spectrum and produces multi-peak speckle at grazing incidence.
    """
    del meta, w
    if theta2_deg is not None and phi2_deg is not None:
        return math.radians(theta2_deg), math.radians(phi2_deg)
    return 0.0, 0.0


def m0_observation_angles(
    meta: SimMeta,
    waves: list[WaveMode],
    *,
    theta2_deg: float | None,
    phi2_deg: float | None,
) -> tuple[float, float]:
    """Lab-frame reference angles from m=0 (plot labels)."""
    if theta2_deg is not None and phi2_deg is not None:
        return math.radians(theta2_deg), math.radians(phi2_deg)
    mw0 = waves_for_order(waves, 0)
    if not mw0:
        raise ValueError("no m=0 mode for observation angles")
    return observation_angles_from_k(*k_to_um_components(mw0[0], meta.period_nm))


def filter_screens_for_stitch(
    screens: list[OrderScreen],
    stitch_m_max: int,
) -> list[OrderScreen]:
    if stitch_m_max < 0:
        return list(screens)
    return [s for s in screens if abs(s.m) <= stitch_m_max]


def adaptive_stitch_obs_n(
    span_mm: float,
    sigma_mm: float,
    obs_n: int,
    *,
    min_px_per_sigma: float = 4.0,
) -> int:
    """Ensure stitched grid resolves spots (≥ min_px_per_sigma pixels per σ)."""
    if span_mm <= 0 or sigma_mm <= 0:
        return max(obs_n, 256)
    need = int(math.ceil(span_mm / max(sigma_mm / min_px_per_sigma, 1e-9)))
    return max(obs_n, need, 256)


def adaptive_asr_grid_n(base_n: int, angle_deg: float) -> int:
    """Raise ASR k-grid samples at grazing incidence (midft aliasing on order patches)."""
    if angle_deg <= 55.0:
        return base_n
    scale = 1.0 + (angle_deg - 50.0) / 12.0
    return max(base_n, int(math.ceil(base_n * scale)))


def farfield_sigma_mm(*, z_mm: float, wl_nm: float, aperture_fwhm_um: float) -> float:
    return far_field_spot_sigma_mm(
        z_mm=z_mm, wavelength_nm=wl_nm, waist_1e2_full_um=aperture_fwhm_um
    )


def fraunhofer_farfield_from_waves(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
) -> tuple[list[tuple], float, float, float, float]:
    sigma_x = farfield_sigma_mm(z_mm=z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=aperture_u_um)
    sigma_y = farfield_sigma_mm(z_mm=z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=aperture_v_um)
    patches: list[tuple] = []
    m0 = next((s for s in screens if s.m == 0), screens[0])
    for scr in screens:
        peak = order_peak_intensity(meta, waves, scr.m, z_mm=z_mm)
        patch = finite_spot_intensity_patch(
            x_center_mm=scr.x_center_mm,
            y_center_mm=scr.y_center_mm,
            peak_intensity=peak,
            sigma_x_mm=sigma_x,
            sigma_y_mm=sigma_y,
            order_half_width_mm=order_half_width_mm,
            y_half_span_mm=y_half_span_mm,
            nx=nx,
            ny=ny,
        )
        if patch is None:
            continue
        x_mm, y_mm, intensity = patch
        patches.append((x_mm, y_mm, intensity))
        print(
            f"  order m={scr.m:+d}  |E_m|²={peak:.6g}  σ_u={sigma_x:.4f} σ_v={sigma_y:.4f} mm  "
            f"center=({scr.x_center_mm:.2f}, {scr.y_center_mm:.2f}) mm",
            file=sys.stderr,
        )
    return patches, sigma_x, sigma_y, m0.x_center_mm, m0.y_center_mm


def resolve_asr_patch_grid(
    meta: SimMeta,
    *,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    local_half_mm: float,
    local_n: int,
) -> tuple[float, int, float, float]:
    """Spot-sized ASR patch half-width and sample count (avoids periodic midft artifacts)."""
    sigma_x = farfield_sigma_mm(z_mm=z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=aperture_u_um)
    sigma_y = farfield_sigma_mm(z_mm=z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=aperture_v_um)
    auto_half, auto_n = spot_grid_params(max(sigma_x, sigma_y))
    half = auto_half if local_half_mm <= 0 else min(local_half_mm, auto_half * 4.0)
    n = max(local_n, auto_n) if local_n > 0 else auto_n
    return half, n, sigma_x, sigma_y


def kdomain_asr_single_order_patch(
    meta: SimMeta,
    waves: list[WaveMode],
    w: WaveMode,
    scr: OrderScreen,
    *,
    z_mm: float,
    sigma_u_um: float,
    sigma_v_um: float,
    local_half_mm: float,
    local_n: int,
    asr_n_u: int,
    asr_n_v: int,
    theta2: float,
    phi2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One order: k-domain ellipse conv -> propagate -> ASR -> |E|² on local lab (u,v) patch."""
    wl_um = meta.wl_nm / 1000.0
    z_um = abs(z_mm) * 1000.0
    # Source-plane support for k-grid; far-field patch sized by z·σ_θ below.
    span_u_um = max(6.0 * sigma_u_um, 2.0 * sigma_u_um)
    span_v_um = max(6.0 * sigma_v_um, 2.0 * sigma_v_um)

    patch_half_mm, patch_n, _, _ = resolve_asr_patch_grid(
        meta,
        z_mm=z_mm,
        aperture_u_um=2.0 * sigma_u_um,
        aperture_v_um=2.0 * sigma_v_um,
        local_half_mm=local_half_mm,
        local_n=local_n,
    )

    fxy_m = vacuum_propagating_fxy_from_mode(w.kx, w.ky, w.kz.real, wavelength_um=wl_um)
    if fxy_m is None:
        return None
    fx_m, fy_m = fxy_m

    us_loc_mm = np.linspace(
        scr.x_center_mm - patch_half_mm, scr.x_center_mm + patch_half_mm, patch_n
    )
    vs_loc_mm = np.linspace(
        scr.y_center_mm - patch_half_mm, scr.y_center_mm + patch_half_mm, patch_n
    )
    # ASR midft must use coordinates relative to spot center (large lab v causes phase aliasing).
    us_rel_um = (us_loc_mm - scr.x_center_mm) * 1000.0
    vs_rel_um = (vs_loc_mm - scr.y_center_mm) * 1000.0

    fx, fy, fzz = build_order_propagation_k_grid(
        fx_m,
        fy_m,
        us_rel_um=us_rel_um,
        vs_rel_um=vs_rel_um,
        wavelength_um=wl_um,
        z_um=z_um,
        theta2=theta2,
        phi2=phi2,
        sigma_u_um=sigma_u_um,
        sigma_v_um=sigma_v_um,
        aperture_span_u_um=span_u_um,
        aperture_span_v_um=span_v_um,
        asr_n_u=asr_n_u,
        asr_n_v=asr_n_v,
    )
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")

    ex0, ey0, ez0 = mode_cartesian_amplitude(w)
    intensity = np.zeros((patch_n, patch_n), dtype=float)
    for amp in (ex0, ey0, ez0):
        if abs(amp) < 1e-30:
            continue
        f0 = order_spectrum_k_gaussian(
            fxx,
            fyy,
            sigma_u_um=sigma_u_um,
            sigma_v_um=sigma_v_um,
            fx_m=fx_m,
            fy_m=fy_m,
            amplitude=amp,
        )
        e_out = scalar_asr_from_spectrum(
            f0,
            fx,
            fy,
            fzz,
            us_rel_um,
            vs_rel_um,
            wavelength_um=wl_um,
            z_um=z_um,
            theta2=theta2,
            phi2=phi2,
            number_u=asr_n_u,
            number_v=asr_n_v,
            fx_m=fx_m,
            fy_m=fy_m,
        )
        intensity += np.abs(e_out) ** 2

    intensity_raw = intensity.copy()
    peak = float(np.max(intensity))
    ref = order_peak_intensity(meta, waves, scr.m, z_mm=z_mm)
    if peak > 0 and ref > 0:
        intensity *= ref / peak
    return us_loc_mm, vs_loc_mm, intensity, intensity_raw


def compute_asr_order_patches(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    local_half_mm: float,
    local_n: int,
    asr_n_u: int,
    asr_n_v: int,
    theta2_deg: float | None,
    phi2_deg: float | None,
) -> tuple[list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]], float, float, float, float]:
    """Per-order ASR |E|² patches on lab (u,v) grids (normalized to S4 peak)."""
    sigma_u = 0.5 * aperture_u_um
    sigma_v = 0.5 * aperture_v_um
    _, _, sigma_x_mm, sigma_y_mm = resolve_asr_patch_grid(
        meta,
        z_mm=z_mm,
        aperture_u_um=aperture_u_um,
        aperture_v_um=aperture_v_um,
        local_half_mm=local_half_mm,
        local_n=local_n,
    )

    if theta2_deg is not None and phi2_deg is not None:
        theta2_lab, phi2_lab = math.radians(theta2_deg), math.radians(phi2_deg)
    else:
        theta2_lab, phi2_lab = m0_observation_angles(
            meta, waves, theta2_deg=theta2_deg, phi2_deg=phi2_deg
        )
        print(
            f"ASR observation: θ₂=0 lab CCD (m=0 ref θ₂={math.degrees(theta2_lab):.2f}°, "
            f"φ₂={math.degrees(phi2_lab):.2f}° for plot labels only)",
            file=sys.stderr,
        )

    asr_n_base = max(asr_n_u, asr_n_v)
    asr_n_eff = adaptive_asr_grid_n(asr_n_base, meta.angle_deg)
    asr_n_u_eff = max(asr_n_u, asr_n_eff)
    asr_n_v_eff = max(asr_n_v, asr_n_eff)
    if asr_n_u_eff > asr_n_u or asr_n_v_eff > asr_n_v:
        print(
            f"ASR k-grid raised to {asr_n_u_eff}×{asr_n_v_eff} "
            f"(θ_inc={meta.angle_deg}°, base={asr_n_base})",
            file=sys.stderr,
        )

    div = beam_divergence_info(
        wavelength_nm=meta.wl_nm,
        aperture_u_um=aperture_u_um,
        aperture_v_um=aperture_v_um,
        z_mm=z_mm,
    )
    print(
        f"Beam divergence: σ_θ,u={div['sigma_theta_u_mrad']:.4f} mrad  "
        f"σ_θ,v={div['sigma_theta_v_mrad']:.4f} mrad  "
        f"far-field σ_u={div['spot_sigma_u_mm']:.4f} mm  σ_v={div['spot_sigma_v_mm']:.4f} mm  "
        f"z/z_R={div['z_over_zr']:.2f}",
        file=sys.stderr,
    )

    patches: list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]] = []
    for scr in screens:
        mw = waves_for_order(waves, scr.m)
        if not mw:
            continue
        w = mw[0]
        theta2, phi2 = observation_angles_for_mode(
            meta, w, theta2_deg=theta2_deg, phi2_deg=phi2_deg
        )
        patch_out = kdomain_asr_single_order_patch(
            meta,
            waves,
            w,
            scr,
            z_mm=z_mm,
            sigma_u_um=sigma_u,
            sigma_v_um=sigma_v,
            local_half_mm=local_half_mm,
            local_n=local_n,
            asr_n_u=asr_n_u_eff,
            asr_n_v=asr_n_v_eff,
            theta2=theta2,
            phi2=phi2,
        )
        if patch_out is None:
            print(f"  order m={scr.m:+d}  skip (evanescent)", file=sys.stderr)
            continue
        us_loc, vs_loc, intensity, _i_raw = patch_out
        patches.append((scr, us_loc, vs_loc, intensity))
        ref = order_peak_intensity(meta, waves, scr.m, z_mm=z_mm)
        print(
            f"  order m={scr.m:+d}  |E_m|²={ref:.6g}  σ_u={sigma_x_mm:.4f} σ_v={sigma_y_mm:.4f} mm  "
            f"center=({scr.x_center_mm:.2f}, {scr.y_center_mm:.2f}) mm",
            file=sys.stderr,
        )

    return patches, sigma_x_mm, sigma_y_mm, theta2_lab, phi2_lab


def kdomain_asr_farfield(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    axis_half_mm: float,
    obs_n: int,
    local_half_mm: float,
    local_n: int,
    asr_n_u: int,
    asr_n_v: int,
    theta2_deg: float | None,
    phi2_deg: float | None,
    patch_feather_frac: float = 0.0,
    paste_method: str = "native",
    stitch_m_max: int = 1,
    stitch_paste_method: str = "interp",
    order_patches: list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Per-order k-domain conv + ASR, stitched on horizontal lab CCD at each order landing.
    Incoherent sum: canvas += |E_m|² (non-overlapping spots).

    Only orders with |m| <= stitch_m_max define bounds and are pasted onto the stitched canvas.
    """
    spot_half_mm, _, sigma_x_mm, _ = resolve_asr_patch_grid(
        meta,
        z_mm=z_mm,
        aperture_u_um=aperture_u_um,
        aperture_v_um=aperture_v_um,
        local_half_mm=local_half_mm,
        local_n=local_n,
    )

    if order_patches is None:
        order_patches, _, _, theta2, phi2 = compute_asr_order_patches(
            meta,
            waves,
            screens,
            z_mm=z_mm,
            aperture_u_um=aperture_u_um,
            aperture_v_um=aperture_v_um,
            local_half_mm=local_half_mm,
            local_n=local_n,
            asr_n_u=asr_n_u,
            asr_n_v=asr_n_v,
            theta2_deg=theta2_deg,
            phi2_deg=phi2_deg,
        )
    elif theta2_deg is not None and phi2_deg is not None:
        theta2, phi2 = math.radians(theta2_deg), math.radians(phi2_deg)
    else:
        theta2, phi2 = m0_observation_angles(
            meta, waves, theta2_deg=theta2_deg, phi2_deg=phi2_deg
        )

    stitch_screens = filter_screens_for_stitch(screens, stitch_m_max)
    stitch_patch_m = {scr.m for scr in stitch_screens}
    stitch_patches = [
        p for p in order_patches if p[0].m in stitch_patch_m
    ]
    if stitch_m_max >= 0 and len(stitch_patches) < len(order_patches):
        print(
            f"Stitched canvas: |m|<={stitch_m_max} ({len(stitch_patches)}/{len(order_patches)} orders); "
            f"far orders still in per-order PNGs",
            file=sys.stderr,
        )

    if patch_feather_frac > 0.0:
        print(
            f"Patch stitch: soft edge feather={patch_feather_frac:.2f} "
            f"(raised-cosine taper on each order tile)",
            file=sys.stderr,
        )

    x_lo, x_hi, y_lo, y_hi = stitched_view_bounds(
        stitch_screens, half_w=spot_half_mm, axis_half_mm=axis_half_mm
    )
    span_u = x_hi - x_lo
    span_v = y_hi - y_lo
    nx_g = adaptive_stitch_obs_n(span_u, sigma_x_mm, obs_n)
    ny_g = max(128, int(nx_g * span_v / max(span_u, 1e-9)))
    ny_g = adaptive_stitch_obs_n(span_v, sigma_x_mm, ny_g)
    us_mm = np.linspace(x_lo, x_hi, nx_g)
    vs_mm = np.linspace(y_lo, y_hi, ny_g)
    canvas = np.zeros((ny_g, nx_g), dtype=float)

    for scr, us_loc, vs_loc, intensity in stitch_patches:
        if stitch_paste_method == "native" and patch_feather_frac <= 0.0:
            paste_patch_add_native(canvas, us_mm, vs_mm, us_loc, vs_loc, intensity)
        else:
            paste_patch_add(
                canvas,
                us_mm,
                vs_mm,
                us_loc,
                vs_loc,
                intensity,
                u_center=scr.x_center_mm,
                v_center=scr.y_center_mm,
                feather_frac=patch_feather_frac,
            )

    du_um = span_u / max(nx_g - 1, 1) * 1000.0
    dv_um = span_v / max(ny_g - 1, 1) * 1000.0
    print(
        f"Stitch grid: {nx_g}×{ny_g} px, du={du_um:.1f} μm dv={dv_um:.1f} μm, "
        f"paste={stitch_paste_method}",
        file=sys.stderr,
    )

    return canvas, us_mm, vs_mm, theta2, phi2


# --- legacy spatial-patch ASR (asr-legacy) ---


def elliptical_gaussian(x_um: np.ndarray, y_um: np.ndarray, sigma_u_um: float, sigma_v_um: float) -> np.ndarray:
    xg, yg = np.meshgrid(x_um, y_um, indexing="xy")
    return np.exp(-0.5 * (xg / sigma_u_um) ** 2 - 0.5 * (yg / sigma_v_um) ** 2)


def asr_order_patch_legacy(
    meta: SimMeta,
    mode_waves: list[WaveMode],
    scr: OrderScreen,
    *,
    z_mm: float,
    sigma_u_um: float,
    sigma_v_um: float,
    aperture_n: int,
    local_half_mm: float,
    local_n: int,
    theta2: float,
    phi2: float,
    asr_n_u: int,
    asr_n_v: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_u = max(3.0 * sigma_u_um, sigma_u_um)
    half_v = max(3.0 * sigma_v_um, sigma_v_um)
    x_um = np.linspace(-half_u, half_u, aperture_n)
    y_um = np.linspace(-half_v, half_v, aperture_n)
    win = elliptical_gaussian(x_um, y_um, sigma_u_um, sigma_v_um)
    xg, yg = np.meshgrid(x_um, y_um, indexing="xy")

    ex = ey = ez = 0.0
    for w in mode_waves:
        ex0, ey0, ez0 = mode_cartesian_amplitude(w)
        ex = ex + ex0
        ey = ey + ey0
        ez = ez + ez0
    ex *= win
    ey *= win
    ez *= win

    wl_um = meta.wl_nm / 1000.0
    z_um = abs(z_mm) * 1000.0
    us_loc_mm = np.linspace(scr.x_center_mm - local_half_mm, scr.x_center_mm + local_half_mm, local_n)
    vs_loc_mm = np.linspace(scr.y_center_mm - local_half_mm, scr.y_center_mm + local_half_mm, local_n)
    us_um = us_loc_mm * 1000.0
    vs_um = vs_loc_mm * 1000.0
    src_span_u = float(x_um[-1] - x_um[0])
    src_span_v = float(y_um[-1] - y_um[0])
    kw = dict(
        wavelength_um=wl_um,
        z_um=z_um,
        us=us_um,
        vs=vs_um,
        theta2=theta2,
        phi2=phi2,
        number_u=asr_n_u,
        number_v=asr_n_v,
        source_span_u_um=src_span_u,
        source_span_v_um=src_span_v,
        max_n_freq=min(384, max(asr_n_u, asr_n_v)),
    )
    ox = scalar_asr_propagate(ex, xg, yg, **kw)
    oy = scalar_asr_propagate(ey, xg, yg, **kw)
    oz = scalar_asr_propagate(ez, xg, yg, **kw)
    return us_loc_mm, vs_loc_mm, np.abs(ox) ** 2 + np.abs(oy) ** 2 + np.abs(oz) ** 2


def patch_feather_mask(
    us_loc: np.ndarray,
    vs_loc: np.ndarray,
    u_center: float,
    v_center: float,
    *,
    feather_frac: float,
) -> np.ndarray:
    """
    Raised-cosine taper on a per-order patch: 1 in the interior, smoothly → 0 at edges.

    Avoids a hard rectangular cutoff when the patch is pasted onto the global CCD grid
    (the outer feather_frac fraction of each half-width rolls off to zero).
    """
    if feather_frac <= 0.0:
        ug, _vg = np.meshgrid(us_loc, vs_loc, indexing="xy")
        return np.ones_like(ug, dtype=float)
    half_u = max(0.5 * (float(us_loc[-1]) - float(us_loc[0])), 1e-12)
    half_v = max(0.5 * (float(vs_loc[-1]) - float(vs_loc[0])), 1e-12)
    ug, vg = np.meshgrid(us_loc, vs_loc, indexing="xy")
    ru = np.abs(ug - u_center) / half_u
    rv = np.abs(vg - v_center) / half_v
    r = np.maximum(ru, rv)
    inner = max(1.0 - feather_frac, 0.0)
    t = np.clip((r - inner) / max(feather_frac, 1e-12), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def paste_patch_add(
    canvas: np.ndarray,
    us_g: np.ndarray,
    vs_g: np.ndarray,
    us_loc: np.ndarray,
    vs_loc: np.ndarray,
    patch: np.ndarray,
    *,
    u_center: float | None = None,
    v_center: float | None = None,
    feather_frac: float = 0.25,
) -> None:
    """
    Paste one order patch onto the lab CCD canvas (incoherent sum).

    With feather_frac > 0, multiply the patch by a soft edge window before
    interpolation so intensity fades to zero at the patch boundary instead of
    ending in a sharp square tile.
    """
    from scipy.interpolate import RegularGridInterpolator

    data = patch
    if feather_frac > 0.0:
        uc = float(us_loc[len(us_loc) // 2]) if u_center is None else u_center
        vc = float(vs_loc[len(vs_loc) // 2]) if v_center is None else v_center
        data = patch * patch_feather_mask(
            us_loc, vs_loc, uc, vc, feather_frac=feather_frac
        )

    interp = RegularGridInterpolator(
        (vs_loc, us_loc), data, bounds_error=False, fill_value=0.0
    )
    ug, vg = np.meshgrid(us_g, vs_g, indexing="xy")
    sampled = interp(np.column_stack([vg.ravel(), ug.ravel()])).reshape(vg.shape)
    canvas += sampled


def paste_patch_add_native(
    canvas: np.ndarray,
    us_g: np.ndarray,
    vs_g: np.ndarray,
    us_loc: np.ndarray,
    vs_loc: np.ndarray,
    patch: np.ndarray,
) -> None:
    """
    Deposit patch pixels onto global canvas without interpolation (physical |E|² values).

    Each patch sample maps to the nearest global grid node; no feather or fill_value cutoff.
    """
    if len(us_g) < 2 or len(vs_g) < 2:
        return
    du = (float(us_g[-1]) - float(us_g[0])) / (len(us_g) - 1)
    dv = (float(vs_g[-1]) - float(vs_g[0])) / (len(vs_g) - 1)
    if du <= 0 or dv <= 0:
        return
    u0, v0 = float(us_g[0]), float(vs_g[0])
    for j_loc, v in enumerate(vs_loc):
        jj = int(round((float(v) - v0) / dv))
        if jj < 0 or jj >= canvas.shape[0]:
            continue
        for i_loc, u in enumerate(us_loc):
            ii = int(round((float(u) - u0) / du))
            if ii < 0 or ii >= canvas.shape[1]:
                continue
            canvas[jj, ii] += patch[j_loc, i_loc]


def paste_patch_onto_canvas(
    canvas: np.ndarray,
    us_g: np.ndarray,
    vs_g: np.ndarray,
    us_loc: np.ndarray,
    vs_loc: np.ndarray,
    patch: np.ndarray,
) -> None:
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (vs_loc, us_loc), patch, bounds_error=False, fill_value=0.0
    )
    ug, vg = np.meshgrid(us_g, vs_g, indexing="xy")
    sampled = interp(np.column_stack([vg.ravel(), ug.ravel()])).reshape(vg.shape)
    np.maximum(canvas, sampled, out=canvas)


def asr_legacy_farfield(
    meta: SimMeta,
    waves: list[WaveMode],
    screens: list[OrderScreen],
    *,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    aperture_n: int,
    axis_half_mm: float,
    obs_n: int,
    local_half_mm: float,
    local_n: int,
    asr_n_u: int,
    asr_n_v: int,
    theta2_deg: float | None,
    phi2_deg: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    sigma_u = 0.5 * aperture_u_um
    sigma_v = 0.5 * aperture_v_um
    mw0 = waves_for_order(waves, 0)
    if not mw0:
        raise ValueError("no m=0 mode")
    if theta2_deg is not None and phi2_deg is not None:
        theta2, phi2 = math.radians(theta2_deg), math.radians(phi2_deg)
    else:
        theta2, phi2 = observation_angles_from_k(*k_to_um_components(mw0[0], meta.period_nm))

    m0 = next((s for s in screens if s.m == 0), screens[0])
    us_mm = np.linspace(m0.x_center_mm - axis_half_mm, m0.x_center_mm + axis_half_mm, obs_n)
    vs_mm = np.linspace(m0.y_center_mm - axis_half_mm, m0.y_center_mm + axis_half_mm, obs_n)
    canvas = np.zeros((obs_n, obs_n), dtype=float)
    for scr in screens:
        mw = waves_for_order(waves, scr.m)
        if not mw:
            continue
        us_loc, vs_loc, patch = asr_order_patch_legacy(
            meta,
            mw,
            scr,
            z_mm=z_mm,
            sigma_u_um=sigma_u,
            sigma_v_um=sigma_v,
            aperture_n=aperture_n,
            local_half_mm=local_half_mm,
            local_n=local_n,
            theta2=theta2,
            phi2=phi2,
            asr_n_u=asr_n_u,
            asr_n_v=asr_n_v,
        )
        paste_patch_onto_canvas(canvas, us_mm, vs_mm, us_loc, vs_loc, patch)
    return canvas, us_mm, vs_mm, theta2, phi2


def _order_tag(m: int) -> str:
    return f"m{m:+d}".replace("+", "p").replace("-", "m")


def _patch_extent_mm(
    us_loc_mm: np.ndarray,
    vs_loc_mm: np.ndarray,
    *,
    origin_u_mm: float,
    origin_v_mm: float,
) -> tuple[float, float, float, float]:
    """Lab patch extent relative to m=0 theory center, in mm."""
    u0 = (us_loc_mm - origin_u_mm).astype(float)
    v0 = (vs_loc_mm - origin_v_mm).astype(float)
    return float(u0[0]), float(u0[-1]), float(v0[0]), float(v0[-1])


def plot_asr_order_patch(
    scr: OrderScreen,
    us_loc: np.ndarray,
    vs_loc: np.ndarray,
    intensity: np.ndarray,
    *,
    meta: SimMeta,
    z_mm: float,
    out_path: Path,
    dpi: int,
    polarization: str,
    aperture_u_um: float,
    aperture_v_um: float,
    sigma_x_mm: float,
    sigma_y_mm: float,
    origin_u_mm: float,
    origin_v_mm: float,
    intensity_scale: str = "linear",
) -> None:
    """Single-order ASR spot (large figure); axes in mm relative to m=0 theory center."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pol = polarization.strip().upper() or "TE"
    peak = float(np.max(intensity)) if intensity.size else 1.0
    norm = _intensity_norm_for_scale(intensity_scale, peak, peaks=[peak])
    display = _prepare_canvas_for_display(intensity, norm, intensity_scale)
    cmap = _display_cmap("inferno")
    extent = _patch_extent_mm(
        us_loc, vs_loc, origin_u_mm=origin_u_mm, origin_v_mm=origin_v_mm
    )
    du_mm = scr.x_center_mm - origin_u_mm
    dv_mm = scr.y_center_mm - origin_v_mm

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5), facecolor="w")
    im = ax.imshow(
        display,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("u (mm)  [origin: m=0 theory center]", fontsize=12)
    ax.set_ylabel("v (mm)", fontsize=12)
    ax.set_title(
        f"ASR order m = {scr.m:+d}  —  λ = {meta.wl_nm:.1f} nm, {farfield_angle_title(meta)}, "
        f"{pol}, z = {z_mm:g} mm\n"
        f"landing Δu = {du_mm:.3f} mm, Δv = {dv_mm:.3f} mm  "
        f"σ_u = {sigma_x_mm:.4f} mm  σ_v = {sigma_y_mm:.4f} mm  "
        f"R_m = {scr.r_abs:.4g}  (aperture {aperture_u_um:g}×{aperture_v_um:g} μm)",
        fontsize=11,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(
        "|E|² (log)" if intensity_scale == "log" else "|E|²",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_asr_order_patches(
    patches: list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]],
    *,
    meta: SimMeta,
    z_mm: float,
    out_dir: Path,
    dpi: int,
    polarization: str,
    aperture_u_um: float,
    aperture_v_um: float,
    sigma_x_mm: float,
    sigma_y_mm: float,
    origin_u_mm: float,
    origin_v_mm: float,
    intensity_scale: str = "linear",
    also_panel: bool = True,
    name_prefix: str = "order",
) -> list[Path]:
    """Write one large PNG per diffraction order plus an optional overview panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scr, us_loc, vs_loc, intensity in patches:
        path = out_dir / f"{name_prefix}_{_order_tag(scr.m)}.png"
        plot_asr_order_patch(
            scr,
            us_loc,
            vs_loc,
            intensity,
            meta=meta,
            z_mm=z_mm,
            out_path=path,
            dpi=dpi,
            polarization=polarization,
            aperture_u_um=aperture_u_um,
            aperture_v_um=aperture_v_um,
            sigma_x_mm=sigma_x_mm,
            sigma_y_mm=sigma_y_mm,
            origin_u_mm=origin_u_mm,
            origin_v_mm=origin_v_mm,
            intensity_scale=intensity_scale,
        )
        written.append(path)

    if also_panel and patches:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pol = polarization.strip().upper() or "TE"
        n = len(patches)
        ncols = min(n, 3)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 5.0 * nrows),
            facecolor="w",
            squeeze=False,
        )
        for (scr, us_loc, vs_loc, intensity), ax in zip(patches, axes.ravel()):
            peak = float(np.max(intensity)) if intensity.size else 1.0
            norm = _intensity_norm_for_scale(intensity_scale, peak, peaks=[peak])
            display = _prepare_canvas_for_display(intensity, norm, intensity_scale)
            extent = _patch_extent_mm(
                us_loc, vs_loc, origin_u_mm=origin_u_mm, origin_v_mm=origin_v_mm
            )
            du_mm = scr.x_center_mm - origin_u_mm
            dv_mm = scr.y_center_mm - origin_v_mm
            im = ax.imshow(
                display,
                origin="lower",
                aspect="equal",
                extent=extent,
                cmap=_display_cmap("inferno"),
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xlabel("u (mm)", fontsize=10)
            ax.set_ylabel("v (mm)", fontsize=10)
            ax.set_title(
                f"m = {scr.m:+d}  Δu = {du_mm:.2f}  Δv = {dv_mm:.2f} mm\n"
                f"R_m = {scr.r_abs:.4g}",
                fontsize=10,
                fontweight="bold",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(
            f"ASR per-order spots (u,v relative to m=0 center, mm) — "
            f"λ = {meta.wl_nm:.1f} nm, {farfield_angle_title(meta)}, "
            f"{pol}, z = {z_mm:g} mm, aperture {aperture_u_um:g}×{aperture_v_um:g} μm",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        panel_path = out_dir / f"{name_prefix}s_panel.png"
        fig.savefig(panel_path, dpi=dpi, facecolor="w", bbox_inches="tight")
        plt.close(fig)
        written.append(panel_path)

    return written


def resolve_order_patch_outputs(
    out: Path,
    order_patches_dir: Path | None,
) -> tuple[Path, str]:
    """
    Default: per-order PNGs beside the stitched far-field image.

    Example: ..._farfield_asr.png -> ..._farfield_asr_order_mp0.png
    """
    if order_patches_dir is not None:
        return order_patches_dir, "order"
    return out.parent, f"{out.stem}_order"


def plot_asr_farfield(
    intensity: np.ndarray,
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    *,
    meta: SimMeta,
    z_mm: float,
    out_path: Path,
    dpi: int,
    polarization: str,
    aperture_u_um: float,
    aperture_v_um: float,
    theta2_rad: float,
    phi2_rad: float,
    method_note: str,
    intensity_scale: str = "log",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pol = polarization.strip().upper() or "TE"
    vmax = float(np.max(intensity)) if intensity.size else 1.0
    norm = _intensity_norm_for_scale(intensity_scale, vmax, peaks=[vmax])
    display = _prepare_canvas_for_display(intensity, norm, intensity_scale)
    cmap = _display_cmap("inferno")

    fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FARFIELD_FIGSIZE, facecolor="w")
    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=(us_mm[0], us_mm[-1], vs_mm[0], vs_mm[-1]),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("u (mm)", fontsize=12)
    ax.set_ylabel("v (mm)", fontsize=12)
    t2 = math.degrees(theta2_rad)
    p2 = math.degrees(phi2_rad)
    ax.set_title(
        f"Far-field ASR (aperture {aperture_u_um:g}×{aperture_v_um:g} μm) — "
        f"λ = {meta.wl_nm:.1f} nm, {farfield_angle_title(meta)}, {pol}, z = {z_mm:g} mm\n"
        f"{method_note} (θ₂={t2:.1f}°, φ₂={p2:.1f}°)",
        fontsize=11,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|E|² (log)" if intensity_scale == "log" else "|E|²", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="run4: S4 Floquet + elliptical aperture far-field")
    ap.add_argument("waves_tsv", help="au_grating_farfield.lua output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-mm", type=float, default=40.0)
    ap.add_argument("--aperture-u-um", type=float, default=15.0, help="1/e² full width u (μm)")
    ap.add_argument("--aperture-v-um", type=float, default=25.0, help="1/e² full width v (μm)")
    ap.add_argument(
        "--method",
        choices=("asr", "asr-legacy", "fraunhofer"),
        default="asr",
        help="asr: k-conv + ASR stitched on lab CCD (default); asr-legacy: spatial patch; fraunhofer: analytic",
    )
    ap.add_argument("--obs-span-mm", type=float, default=None, help="half-span override for m0 (u,v) grid")
    ap.add_argument("--order-half-width-mm", type=float, default=8.0)
    ap.add_argument("--y-half-span-mm", type=float, default=2.0)
    ap.add_argument("--spot-nx", type=int, default=128)
    ap.add_argument("--spot-ny", type=int, default=32)
    ap.add_argument("--aperture-n", type=int, default=128, help="asr-legacy source grid")
    ap.add_argument("--obs-n", type=int, default=2400, help="stitched lab CCD horizontal samples")
    ap.add_argument("--local-half-mm", type=float, default=0.0, help="per-order ASR patch half-width (0=auto from σ)")
    ap.add_argument("--local-n", type=int, default=128, help="samples per order ASR patch edge")
    ap.add_argument(
        "--patch-feather-frac",
        type=float,
        default=0.0,
        help="non-physical display taper on patch edges (0=off, use native paste)",
    )
    ap.add_argument(
        "--paste-method",
        choices=("native", "interp"),
        default="native",
        help="legacy alias; stitched canvas uses --stitch-paste-method",
    )
    ap.add_argument(
        "--stitch-m-max",
        type=int,
        default=1,
        help="include orders with |m|<=N on stitched lab CCD (default 1; far orders still in patch PNGs)",
    )
    ap.add_argument(
        "--stitch-paste-method",
        choices=("native", "interp"),
        default="interp",
        help="paste method for stitched total canvas (default interp)",
    )
    ap.add_argument("--save-canvas-npz", type=Path, default=None, help="save stitched canvas arrays")
    ap.add_argument("--asr-n-u", type=int, default=192)
    ap.add_argument("--asr-n-v", type=int, default=192)
    ap.add_argument("--axis-half-mm", type=float, default=DEFAULT_AXIS_HALF_MM)
    ap.add_argument("--order-r-thresh", type=float, default=1e-5)
    ap.add_argument("--obs-theta2-deg", type=float, default=None)
    ap.add_argument("--obs-phi2-deg", type=float, default=None)
    ap.add_argument("--polarization", type=str, default="TE")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--intensity-scale", choices=("linear", "sqrt", "log", "asinh"), default="log")
    ap.add_argument(
        "--plot-order-patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write large per-order ASR spot PNGs alongside stitched far-field",
    )
    ap.add_argument(
        "--order-patches-dir",
        type=Path,
        default=None,
        help="directory for per-order spot PNGs (default: same folder as --out)",
    )
    ap.add_argument(
        "--order-patch-scale",
        choices=("linear", "sqrt", "log", "asinh"),
        default="linear",
        help="intensity scale for per-order morphology figures",
    )
    args = ap.parse_args()

    meta, waves = load_waves_tsv(Path(args.waves_tsv))
    flux = load_flux_tsv(Path(args.waves_tsv))
    screens = compute_order_screens(
        waves,
        z_mm=args.z_mm,
        period_nm=meta.period_nm,
        flux=flux,
        r_threshold=args.order_r_thresh,
    )
    print(
        f"Far-field run4 ({args.method}): {len(reflected_propagating_modes(waves))} modes",
        file=sys.stderr,
    )
    print(f"Order screens ({len(screens)}):", file=sys.stderr)
    for scr in screens:
        print(
            f"  m={scr.m:+d}  lab u={scr.x_center_mm:.2f} mm  v={scr.y_center_mm:.2f} mm  R={scr.r_abs:.6g}",
            file=sys.stderr,
        )

    out = Path(args.out)
    pol = args.polarization.strip().upper() or "TE"
    m0_scr = next((s for s in screens if s.m == 0), screens[0])
    origin_u_mm = m0_scr.x_center_mm
    origin_v_mm = m0_scr.y_center_mm

    if args.method == "fraunhofer":
        patches, sx, sy, u0, v0 = fraunhofer_farfield_from_waves(
            meta,
            waves,
            screens,
            z_mm=args.z_mm,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            order_half_width_mm=args.order_half_width_mm,
            y_half_span_mm=args.y_half_span_mm,
            nx=args.spot_nx,
            ny=args.spot_ny,
        )
        print(f"Observation center (m=0 lab): u={u0:.2f} mm, v={v0:.2f} mm", file=sys.stderr)
        plot_stitched_order_spots(
            screens,
            patches,
            z_mm=args.z_mm,
            meta=meta,
            out_path=out,
            dpi=args.dpi,
            polarization=pol,
            title_suffix=f"(aperture {args.aperture_u_um:g}×{args.aperture_v_um:g} μm, Fraunhofer k-conv)",
            sigma_x_mm=sx,
            axis_half_mm=args.axis_half_mm,
            intensity_scale=args.intensity_scale,
        )
        if args.plot_order_patches:
            order_dir, name_prefix = resolve_order_patch_outputs(out, args.order_patches_dir)
            fraunhofer_patches = [
                (scr, x_mm, y_mm, inten)
                for scr, (x_mm, y_mm, inten) in zip(screens, patches)
            ]
            written = plot_asr_order_patches(
                fraunhofer_patches,
                meta=meta,
                z_mm=args.z_mm,
                out_dir=order_dir,
                dpi=args.dpi,
                polarization=pol,
                aperture_u_um=args.aperture_u_um,
                aperture_v_um=args.aperture_v_um,
                sigma_x_mm=sx,
                sigma_y_mm=sy,
                origin_u_mm=origin_u_mm,
                origin_v_mm=origin_v_mm,
                intensity_scale=args.order_patch_scale,
                name_prefix=name_prefix,
            )
            for p in written:
                print(p.resolve())
        print(out.resolve())
        return

    order_patches: list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]] | None = None
    sigma_x_mm = sigma_y_mm = 0.0
    if args.method == "asr" and args.plot_order_patches:
        order_patches, sigma_x_mm, sigma_y_mm, _, _ = compute_asr_order_patches(
            meta,
            waves,
            screens,
            z_mm=args.z_mm,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            local_half_mm=args.local_half_mm,
            local_n=args.local_n,
            asr_n_u=args.asr_n_u,
            asr_n_v=args.asr_n_v,
            theta2_deg=args.obs_theta2_deg,
            phi2_deg=args.obs_phi2_deg,
        )

    if args.method == "asr-legacy":
        intensity, us_mm, vs_mm, theta2, phi2 = asr_legacy_farfield(
            meta,
            waves,
            screens,
            z_mm=args.z_mm,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            aperture_n=args.aperture_n,
            axis_half_mm=args.axis_half_mm,
            obs_n=args.obs_n,
            local_half_mm=args.local_half_mm,
            local_n=args.local_n,
            asr_n_u=args.asr_n_u,
            asr_n_v=args.asr_n_v,
            theta2_deg=args.obs_theta2_deg,
            phi2_deg=args.obs_phi2_deg,
        )
        note = "obs: spatial patch ASR (legacy)"
    else:
        intensity, us_mm, vs_mm, theta2, phi2 = kdomain_asr_farfield(
            meta,
            waves,
            screens,
            z_mm=args.z_mm,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            axis_half_mm=args.axis_half_mm,
            obs_n=args.obs_n,
            local_half_mm=args.local_half_mm,
            local_n=args.local_n,
            asr_n_u=args.asr_n_u,
            asr_n_v=args.asr_n_v,
            theta2_deg=args.obs_theta2_deg,
            phi2_deg=args.obs_phi2_deg,
            patch_feather_frac=args.patch_feather_frac,
            paste_method=args.paste_method,
            stitch_m_max=args.stitch_m_max,
            stitch_paste_method=args.stitch_paste_method,
            order_patches=order_patches,
        )
        if sigma_x_mm <= 0:
            _, _, sigma_x_mm, sigma_y_mm = resolve_asr_patch_grid(
                meta,
                z_mm=args.z_mm,
                aperture_u_um=args.aperture_u_um,
                aperture_v_um=args.aperture_v_um,
                local_half_mm=args.local_half_mm,
                local_n=args.local_n,
            )
        note = (
            f"lab CCD, per-order k-conv + ASR stitched (|m|<={args.stitch_m_max}, "
            f"paste={args.stitch_paste_method})"
        )

    print(
        f"lab CCD grid: u∈[{us_mm[0]:.2f},{us_mm[-1]:.2f}] mm, v∈[{vs_mm[0]:.2f},{vs_mm[-1]:.2f}] mm; "
        f"θ₂={math.degrees(theta2):.2f}°, φ₂={math.degrees(phi2):.2f}°",
        file=sys.stderr,
    )
    plot_asr_farfield(
        intensity,
        us_mm,
        vs_mm,
        meta=meta,
        z_mm=args.z_mm,
        out_path=out,
        dpi=args.dpi,
        polarization=pol,
        aperture_u_um=args.aperture_u_um,
        aperture_v_um=args.aperture_v_um,
        theta2_rad=theta2,
        phi2_rad=phi2,
        method_note=note,
        intensity_scale=args.intensity_scale,
    )
    if args.plot_order_patches and order_patches:
        order_dir, name_prefix = resolve_order_patch_outputs(out, args.order_patches_dir)
        written = plot_asr_order_patches(
            order_patches,
            meta=meta,
            z_mm=args.z_mm,
            out_dir=order_dir,
            dpi=args.dpi,
            polarization=pol,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            sigma_x_mm=sigma_x_mm,
            sigma_y_mm=sigma_y_mm,
            origin_u_mm=origin_u_mm,
            origin_v_mm=origin_v_mm,
            intensity_scale=args.order_patch_scale,
            name_prefix=name_prefix,
        )
        for p in written:
            print(p.resolve())
    if args.save_canvas_npz is not None:
        args.save_canvas_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_canvas_npz,
            intensity=intensity,
            us_mm=us_mm,
            vs_mm=vs_mm,
        )
        print(f"Saved canvas NPZ: {args.save_canvas_npz.resolve()}", file=sys.stderr)
    print(out.resolve())


if __name__ == "__main__":
    main()
