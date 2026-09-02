#!/usr/bin/env python3
"""
run5: HHG harmonic overlay on one lab CCD.

For 800 nm fundamental, select harmonic orders q with λ=800/q in [wl_min, wl_max],
run S4+ASR per harmonic, paste all diffraction-order spots at their lab (u,v)
positions (weighted by S4 energy), and render one combined figure.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FAR = Path(__file__).resolve().parent
if str(_FAR) not in sys.path:
    sys.path.insert(0, str(_FAR))

from s4_farfield_asr import (  # noqa: E402
    compute_asr_order_patches,
    paste_patch_add_native,
    resolve_asr_patch_grid,
)
from s4_farfield_reconstruct import (  # noqa: E402
    DEFAULT_AXIS_HALF_MM,
    DEFAULT_FARFIELD_FIGSIZE,
    OrderScreen,
    SimMeta,
    _display_cmap,
    _intensity_norm_for_scale,
    _prepare_canvas_for_display,
    compute_order_screens,
    farfield_angle_title,
    load_flux_tsv,
    load_waves_tsv,
)


def tag_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def harmonic_wavelength_nm(fundamental_nm: float, q: int) -> float:
    return fundamental_nm / q


def harmonic_orders_in_wl_range(
    *,
    fundamental_nm: float,
    wl_min: float,
    wl_max: float,
    odd_only: bool,
    order_step: int,
    order_list: str,
) -> list[int]:
    if order_list.strip():
        orders = sorted({int(x.strip()) for x in order_list.split(",") if x.strip()})
        return [q for q in orders if wl_min <= fundamental_nm / q <= wl_max]
    if wl_min <= 0 or wl_max <= 0 or wl_min > wl_max:
        raise ValueError("require 0 < wl_min <= wl_max")
    q_lo = int(math.ceil(fundamental_nm / wl_max))
    q_hi = int(math.floor(fundamental_nm / wl_min))
    if q_lo > q_hi:
        raise ValueError(
            f"no harmonic orders: λ range [{wl_min}, {wl_max}] nm "
            f"↔ q∈[{q_lo}, {q_hi}] for fundamental={fundamental_nm} nm"
        )
    step = max(1, order_step)
    if odd_only:
        start = q_lo if q_lo % 2 == 1 else q_lo + 1
        return list(range(start, q_hi + 1, 2 * step if step > 1 else 2))
    return list(range(q_lo, q_hi + 1, step))


def waves_path_for_harmonic(
    far_dir: Path,
    q: int,
    angle_deg: float,
    azimuth_deg: float,
    period_nm: float,
) -> Path:
    return far_dir / (
        f"s4_farfield_hhg_q{q}_th{tag_num(angle_deg)}deg_"
        f"az{tag_num(azimuth_deg)}deg_L{tag_num(period_nm)}nm_waves_air.txt"
    )


def ensure_waves_tsv(
    *,
    s4_bin: Path,
    lua: Path,
    waves_path: Path,
    s4_arg: str,
    force_s4: bool,
) -> None:
    if waves_path.is_file() and not force_s4:
        return
    waves_path.parent.mkdir(parents=True, exist_ok=True)
    with waves_path.open("w", encoding="utf-8") as fout:
        subprocess.run([str(s4_bin), str(lua), "-a", s4_arg], check=True, stdout=fout)


@dataclass
class SpotRecord:
    h_order: int
    wl_nm: float
    m: int
    u_mm: float
    v_mm: float
    r_abs: float
    peak_intensity: float
    sigma_u_mm: float
    sigma_v_mm: float


@dataclass
class HarmonicRun:
    h_order: int
    wl_nm: float
    meta: SimMeta
    patches: list[tuple[OrderScreen, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    spot_half_mm: float = 0.0
    records: list[SpotRecord] = field(default_factory=list)


def filter_screens_by_m(screens: list[OrderScreen], diff_m_max: int) -> list[OrderScreen]:
    return [s for s in screens if abs(s.m) <= diff_m_max]


def compute_harmonic(
    waves_path: Path,
    *,
    h_order: int,
    wl_nm: float,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    local_half_mm: float,
    local_n: int,
    asr_n_u: int,
    asr_n_v: int,
    order_r_thresh: float,
    diff_m_max: int,
    obs_theta2_deg: float | None,
    obs_phi2_deg: float | None,
) -> HarmonicRun:
    meta, waves = load_waves_tsv(waves_path)
    meta = SimMeta(
        wl_nm=wl_nm,
        angle_deg=meta.angle_deg,
        azimuth_deg=meta.azimuth_deg,
        period_nm=meta.period_nm,
        depth_nm=meta.depth_nm,
        duty=meta.duty,
        ng=meta.ng,
    )
    flux = load_flux_tsv(waves_path)
    screens = compute_order_screens(
        waves,
        z_mm=z_mm,
        period_nm=meta.period_nm,
        flux=flux,
        r_threshold=order_r_thresh,
    )
    screens = filter_screens_by_m(screens, diff_m_max)
    order_patches, sigma_x_mm, sigma_y_mm, _, _ = compute_asr_order_patches(
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
        theta2_deg=obs_theta2_deg,
        phi2_deg=obs_phi2_deg,
    )
    spot_half_mm, _, _, _ = resolve_asr_patch_grid(
        meta,
        z_mm=z_mm,
        aperture_u_um=aperture_u_um,
        aperture_v_um=aperture_v_um,
        local_half_mm=local_half_mm,
        local_n=local_n,
    )
    records: list[SpotRecord] = []
    for scr, _us, _vs, intensity in order_patches:
        peak = float(np.max(intensity)) if intensity.size else 0.0
        records.append(
            SpotRecord(
                h_order=h_order,
                wl_nm=wl_nm,
                m=scr.m,
                u_mm=scr.x_center_mm,
                v_mm=scr.y_center_mm,
                r_abs=scr.r_abs,
                peak_intensity=peak,
                sigma_u_mm=sigma_x_mm,
                sigma_v_mm=sigma_y_mm,
            )
        )
    return HarmonicRun(
        h_order=h_order,
        wl_nm=wl_nm,
        meta=meta,
        patches=order_patches,
        spot_half_mm=spot_half_mm,
        records=records,
    )


def global_view_bounds(
    runs: list[HarmonicRun],
    *,
    axis_half_mm: float,
) -> tuple[float, float, float, float]:
    centers: list[tuple[float, float]] = []
    margin = 0.0
    for run in runs:
        margin = max(margin, run.spot_half_mm)
        for rec in run.records:
            centers.append((rec.u_mm, rec.v_mm))
    if not centers:
        half = max(axis_half_mm, 50.0)
        return -half, half, -half, half
    margin += 5.0
    us = [c[0] for c in centers]
    vs = [c[1] for c in centers]
    x_lo = min(us) - margin
    x_hi = max(us) + margin
    y_lo_raw = min(vs) - margin
    y_hi_raw = max(vs) + margin
    y_mid = 0.5 * (y_lo_raw + y_hi_raw)
    y_half = max(0.5 * (y_hi_raw - y_lo_raw), abs(axis_half_mm))
    return x_lo, x_hi, y_mid - y_half, y_mid + y_half


def stitch_hhg_canvas(
    runs: list[HarmonicRun],
    *,
    obs_n: int,
    axis_half_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_lo, x_hi, y_lo, y_hi = global_view_bounds(runs, axis_half_mm=axis_half_mm)
    nx_g = max(obs_n, 256)
    ny_g = max(128, int(nx_g * (y_hi - y_lo) / max(x_hi - x_lo, 1e-9)))
    us_mm = np.linspace(x_lo, x_hi, nx_g)
    vs_mm = np.linspace(y_lo, y_hi, ny_g)
    canvas = np.zeros((ny_g, nx_g), dtype=float)
    for run in runs:
        for _scr, us_loc, vs_loc, intensity in run.patches:
            paste_patch_add_native(canvas, us_mm, vs_mm, us_loc, vs_loc, intensity)
    return canvas, us_mm, vs_mm


def plot_hhg_overlay(
    canvas: np.ndarray,
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    records: list[SpotRecord],
    *,
    out_path: Path,
    fundamental_nm: float,
    wl_min: float,
    wl_max: float,
    meta: SimMeta,
    z_mm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    polarization: str,
    dpi: int,
    intensity_scale: str,
    diff_m_max: int,
    annotate: bool,
    annotate_min_r: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    pol = polarization.strip().upper() or "TE"
    vmax = float(np.max(canvas)) if canvas.size else 1.0
    norm = _intensity_norm_for_scale(intensity_scale, vmax, peaks=[vmax])
    display = _prepare_canvas_for_display(canvas, norm, intensity_scale)
    cmap = _display_cmap("inferno")

    q_vals = sorted({r.h_order for r in records})
    q_min = min(q_vals) if q_vals else 0
    q_max = max(q_vals) if q_vals else 0

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

    q_norm = plt.Normalize(vmin=min(q_vals), vmax=max(q_vals)) if q_vals else None
    q_cmap = colormaps["cool"]

    if annotate and records and q_norm is not None:
        for rec in records:
            if rec.r_abs < annotate_min_r:
                continue
            color = q_cmap(q_norm(rec.h_order))
            ax.plot(rec.u_mm, rec.v_mm, "+", color=color, markersize=6, markeredgewidth=1.2)
            ax.annotate(
                f"H{rec.h_order}\nm={rec.m:+d}\nR={rec.r_abs:.2g}",
                (rec.u_mm, rec.v_mm),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                alpha=0.95,
            )

    ax.set_title(
        f"HHG combined far-field — {fundamental_nm:g} nm fundamental, "
        f"H{q_min}–H{q_max} (λ={wl_min:g}–{wl_max:g} nm)\n"
        f"{farfield_angle_title(meta)}, {pol}, z={z_mm:g} mm, "
        f"aperture {aperture_u_um:g}×{aperture_v_um:g} μm\n"
        f"Diffraction orders m∈[−{diff_m_max}, +{diff_m_max}]  |  "
        f"incoherent sum, energy from S4 R_m",
        fontsize=11,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|E|² (log)" if intensity_scale == "log" else "|E|²", fontsize=10)
    if q_vals and q_norm is not None:
        sm = plt.cm.ScalarMappable(cmap=q_cmap, norm=q_norm)
        sm.set_array([])
        cax2 = fig.add_axes([0.92, 0.55, 0.015, 0.30])
        fig.colorbar(sm, cax=cax2, label="harmonic order q")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def save_spot_table(records: list[SpotRecord], path: Path) -> None:
    rows = [
        {
            "h_order": r.h_order,
            "wl_nm": r.wl_nm,
            "m": r.m,
            "u_mm": r.u_mm,
            "v_mm": r.v_mm,
            "R_m": r.r_abs,
            "peak_intensity": r.peak_intensity,
            "sigma_u_mm": r.sigma_u_mm,
            "sigma_v_mm": r.sigma_v_mm,
        }
        for r in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="run5: HHG harmonics (λ=800/q in range) on one lab CCD"
    )
    ap.add_argument("--far-dir", type=Path, default=_FAR)
    ap.add_argument("--s4-bin", type=Path, required=True)
    ap.add_argument("--lua", type=Path, default=_FAR / "au_grating_farfield.lua")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spots-json", type=Path, default=None)
    ap.add_argument("--fundamental-nm", type=float, default=800.0)
    ap.add_argument("--wl-min", type=float, default=10.0)
    ap.add_argument("--wl-max", type=float, default=30.0)
    ap.add_argument("--hhg-order-list", default="", help="comma q list, filtered by λ range")
    ap.add_argument("--hhg-order-step", type=int, default=1)
    ap.add_argument("--odd-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--angle-deg", type=float, default=60.0)
    ap.add_argument("--azimuth-deg", type=float, default=90.0)
    ap.add_argument("--period-nm", type=float, default=80.0)
    ap.add_argument("--depth-nm", type=float, default=40.0)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--ng", type=int, default=31)
    ap.add_argument("--z-mm", type=float, default=40.0)
    ap.add_argument("--aperture-u-um", type=float, default=15.0)
    ap.add_argument("--aperture-v-um", type=float, default=25.0)
    ap.add_argument("--obs-n", type=int, default=4800)
    ap.add_argument("--local-half-mm", type=float, default=0.0)
    ap.add_argument("--local-n", type=int, default=512)
    ap.add_argument("--asr-n-u", type=int, default=384)
    ap.add_argument("--asr-n-v", type=int, default=384)
    ap.add_argument("--axis-half-mm", type=float, default=DEFAULT_AXIS_HALF_MM)
    ap.add_argument("--order-r-thresh", type=float, default=1e-5)
    ap.add_argument(
        "--diff-m-max",
        type=int,
        default=2,
        help="include diffraction orders m with |m| <= this value (default: ±2)",
    )
    ap.add_argument("--obs-theta2-deg", type=float, default=None)
    ap.add_argument("--obs-phi2-deg", type=float, default=None)
    ap.add_argument("--polarization", default="TE")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--intensity-scale",
        choices=("linear", "sqrt", "log", "asinh"),
        default="log",
    )
    ap.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--annotate-min-r", type=float, default=1e-4)
    ap.add_argument("--force-s4", action="store_true")
    args = ap.parse_args()

    orders = harmonic_orders_in_wl_range(
        fundamental_nm=args.fundamental_nm,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        odd_only=args.odd_only,
        order_step=args.hhg_order_step,
        order_list=args.hhg_order_list,
    )
    if not orders:
        raise SystemExit("no harmonic orders in wavelength range")

    print(
        f"HHG overlay: {len(orders)} harmonic orders, "
        f"λ={args.wl_min:g}–{args.wl_max:g} nm, fundamental={args.fundamental_nm:g} nm, "
        f"diffraction m∈[−{args.diff_m_max}, +{args.diff_m_max}]",
        file=sys.stderr,
    )
    for q in orders:
        wl = harmonic_wavelength_nm(args.fundamental_nm, q)
        print(f"  H{q}: λ={wl:.4f} nm", file=sys.stderr)

    runs: list[HarmonicRun] = []
    all_records: list[SpotRecord] = []

    for q in orders:
        wl = harmonic_wavelength_nm(args.fundamental_nm, q)
        waves_path = waves_path_for_harmonic(
            args.far_dir, q, args.angle_deg, args.azimuth_deg, args.period_nm
        )
        s4_arg = (
            f"wl_nm={wl};angle_deg={args.angle_deg};azimuth_deg={args.azimuth_deg};"
            f"period_nm={args.period_nm};depth_nm={args.depth_nm};duty={args.duty};NG={args.ng}"
        )
        print(f"S4 + ASR: H{q} (λ={wl:.4f} nm) -> {waves_path.name}", file=sys.stderr)
        ensure_waves_tsv(
            s4_bin=args.s4_bin,
            lua=args.lua,
            waves_path=waves_path,
            s4_arg=s4_arg,
            force_s4=args.force_s4,
        )
        run = compute_harmonic(
            waves_path,
            h_order=q,
            wl_nm=wl,
            z_mm=args.z_mm,
            aperture_u_um=args.aperture_u_um,
            aperture_v_um=args.aperture_v_um,
            local_half_mm=args.local_half_mm,
            local_n=args.local_n,
            asr_n_u=args.asr_n_u,
            asr_n_v=args.asr_n_v,
            order_r_thresh=args.order_r_thresh,
            diff_m_max=args.diff_m_max,
            obs_theta2_deg=args.obs_theta2_deg,
            obs_phi2_deg=args.obs_phi2_deg,
        )
        runs.append(run)
        all_records.extend(run.records)
        for rec in run.records:
            print(
                f"    H{rec.h_order} m={rec.m:+d}  u={rec.u_mm:.3f} v={rec.v_mm:.3f} mm  "
                f"R_m={rec.r_abs:.6g}  peak={rec.peak_intensity:.4g}",
                file=sys.stderr,
            )

    canvas, us_mm, vs_mm = stitch_hhg_canvas(
        runs, obs_n=args.obs_n, axis_half_mm=args.axis_half_mm
    )
    meta0 = runs[0].meta
    plot_hhg_overlay(
        canvas,
        us_mm,
        vs_mm,
        all_records,
        out_path=args.out,
        fundamental_nm=args.fundamental_nm,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        meta=meta0,
        z_mm=args.z_mm,
        aperture_u_um=args.aperture_u_um,
        aperture_v_um=args.aperture_v_um,
        polarization=args.polarization,
        dpi=args.dpi,
        intensity_scale=args.intensity_scale,
        diff_m_max=args.diff_m_max,
        annotate=args.annotate,
        annotate_min_r=args.annotate_min_r,
    )

    spots_path = args.spots_json or args.out.with_suffix(".spots.json")
    save_spot_table(all_records, spots_path)
    print(f"Spot table: {spots_path.resolve()}", file=sys.stderr)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
