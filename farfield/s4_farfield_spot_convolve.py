#!/usr/bin/env python3
"""
Finite-spot far-field via single-angle RCWA + per-order receiving screens.

At grazing incidence, diffraction orders land at separate x positions; a unified
screen shows interference fringes. Default: stitched discrete order spots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FAR = Path(__file__).resolve().parent
if str(_FAR) not in sys.path:
    sys.path.insert(0, str(_FAR))

from s4_farfield_reconstruct import (  # noqa: E402
    DEFAULT_AXIS_HALF_MM,
    choose_farfield_screen_mode,
    compute_order_screens,
    finite_spot_intensity_patch,
    load_flux_tsv,
    load_waves_tsv,
    order_peak_intensity,
    plot_order_spot_panels,
    plot_stitched_order_spots,
    sigma_x_mm,
)


def collect_order_patches(
    meta,
    waves,
    screens,
    *,
    z_mm: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
    sigma_x_mm: float,
):
    patches = []
    for scr in screens:
        peak = order_peak_intensity(meta, waves, scr.m, z_mm=z_mm)
        patch = finite_spot_intensity_patch(
            x_center_mm=scr.x_center_mm,
            y_center_mm=scr.y_center_mm,
            peak_intensity=peak,
            sigma_x_mm=sigma_x_mm,
            order_half_width_mm=order_half_width_mm,
            y_half_span_mm=y_half_span_mm,
            nx=nx,
            ny=ny,
        )
        if patch is None:
            continue
        x_mm, y_mm, intensity = patch
        patches.append((scr, x_mm, y_mm, intensity))
    return patches


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Finite-spot far-field: single RCWA + per-order receiving screens"
    )
    ap.add_argument("waves_tsv", help="au_grating_farfield.lua output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-mm", type=float, default=40.0)
    ap.add_argument("--spot-w0-um", type=float, default=50.0)
    ap.add_argument("--conv-sigma-scale", type=float, default=1.0)
    ap.add_argument("--order-half-width-mm", type=float, default=8.0)
    ap.add_argument("--y-half-span-mm", type=float, default=2.0)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--order-r-thresh", type=float, default=1e-5)
    ap.add_argument(
        "--screen-mode",
        choices=("auto", "stitched", "orders", "unified"),
        default="auto",
    )
    ap.add_argument("--panels-out", default="", help="optional multi-panel PNG")
    ap.add_argument("--polarization", type=str, default="TE")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--intensity-scale",
        choices=("linear", "sqrt", "log", "asinh"),
        default="log",
        help="overview colormap scaling (default: log)",
    )
    ap.add_argument(
        "--axis-half-mm",
        type=float,
        default=DEFAULT_AXIS_HALF_MM,
        help="v-axis half-span (mm); default 50 mm (CCD 100 mm)",
    )
    args = ap.parse_args()

    meta, waves = load_waves_tsv(Path(args.waves_tsv))
    flux = load_flux_tsv(Path(args.waves_tsv))
    mode = choose_farfield_screen_mode(args.screen_mode, meta.angle_deg)

    sx = sigma_x_mm(
        z_mm=args.z_mm,
        wl_nm=meta.wl_nm,
        w0_um=args.spot_w0_um,
        scale=args.conv_sigma_scale,
    )
    print(
        f"screen_mode={mode}; spot blur σ_x={sx:.4f} mm (w0={args.spot_w0_um} μm)",
        file=sys.stderr,
    )

    screens = compute_order_screens(
        waves,
        z_mm=args.z_mm,
        period_nm=meta.period_nm,
        flux=flux,
        r_threshold=args.order_r_thresh,
    )
    if not screens:
        raise SystemExit("No propagating orders; lower --order-r-thresh")

    print(f"Order receiving screens ({len(screens)}):", file=sys.stderr)
    for scr in screens:
        print(
            f"  m={scr.m:+d}  u={scr.x_center_mm:.2f} mm  v={scr.y_center_mm:.2f} mm  "
            f"R_abs={scr.r_abs:.6g}",
            file=sys.stderr,
        )

    pol = args.polarization.strip().upper() or "TE"
    out = Path(args.out)
    suffix = f"(conv, w₀={args.spot_w0_um:g} μm)" if sx > 0 else ""

    if mode == "unified":
        raise SystemExit(
            "unified screen at grazing incidence shows interference fringes, not discrete spots; "
            "use --screen-mode stitched (default for |θ|≥60°)"
        )

    patches_raw = collect_order_patches(
        meta,
        waves,
        screens,
        z_mm=args.z_mm,
        order_half_width_mm=args.order_half_width_mm,
        y_half_span_mm=args.y_half_span_mm,
        nx=args.nx,
        ny=args.ny,
        sigma_x_mm=sx,
    )
    screens_ok = [p[0] for p in patches_raw]
    patch_tuples = [(p[1], p[2], p[3]) for p in patches_raw]

    if mode == "stitched":
        plot_stitched_order_spots(
            screens_ok,
            patch_tuples,
            z_mm=args.z_mm,
            meta=meta,
            out_path=out,
            dpi=args.dpi,
            polarization=pol,
            title_suffix=suffix,
            title_angle_deg=meta.angle_deg,
            sigma_x_mm=sx,
            axis_half_mm=args.axis_half_mm,
            intensity_scale=args.intensity_scale,
        )
    else:
        plot_order_spot_panels(
            meta,
            waves,
            screens_ok,
            z_mm=args.z_mm,
            order_half_width_mm=args.order_half_width_mm,
            y_half_span_mm=args.y_half_span_mm,
            nx=args.nx,
            ny=args.ny,
            out_path=out,
            dpi=args.dpi,
            polarization=pol,
            title_suffix=suffix,
        )

    if args.panels_out and mode == "stitched":
        plot_order_spot_panels(
            meta,
            waves,
            screens_ok,
            z_mm=args.z_mm,
            order_half_width_mm=args.order_half_width_mm,
            y_half_span_mm=args.y_half_span_mm,
            nx=args.nx,
            ny=args.ny,
            out_path=Path(args.panels_out),
            dpi=args.dpi,
            polarization=pol,
            title_suffix=suffix,
        )

    print(out.resolve())


if __name__ == "__main__":
    main()
