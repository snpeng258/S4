#!/usr/bin/env python3
"""
Finite-spot far-field via angular-spectrum multi-angle RCWA + per-order screens.

Coherent sum per diffraction order on its receiving screen (not unified plane).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_FAR = Path(__file__).resolve().parent
if str(_FAR) not in sys.path:
    sys.path.insert(0, str(_FAR))

from s4_farfield_reconstruct import (  # noqa: E402
    DEFAULT_AXIS_HALF_MM,
    SimMeta,
    choose_farfield_screen_mode,
    compute_order_screens,
    finite_spot_intensity_patch,
    load_flux_tsv,
    load_waves_tsv,
    order_field_at_origin,
    order_peak_intensity,
    plot_stitched_order_spots,
    sigma_x_mm,
    waves_for_order,
)


def load_manifest(path: Path) -> list[tuple[Path, float, float]]:
    rows: list[tuple[Path, float, float]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            rows.append((Path(parts[0]), float(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def incoherent_order_peak_intensity(
    manifest: list[tuple[Path, float, float]],
    scr_m: int,
    *,
    z_mm: float,
) -> float:
    """
    Angular-spectrum intensity: Σ w(θ) |E_m(θ)|².

    S4 GetWaves coefficients carry a θ-dependent global phase (TE: ey rotates
    with incidence). Coherent vector sum at one point cancels even for m=0;
    incoherent weighting matches the finite-beam power budget per order.
    """
    total = 0.0
    for waves_path, weight, _angle_deg in manifest:
        meta, waves = load_waves_tsv(waves_path)
        total += weight * order_peak_intensity(meta, waves, scr_m, z_mm=z_mm)
    return total


def coherent_order_peak_intensity(
    manifest: list[tuple[Path, float, float]],
    scr_m: int,
    *,
    z_mm: float,
) -> float:
    """Coherent |E_m|² at spot center from angular-spectrum sum (local phase)."""
    ex_sum = ey_sum = ez_sum = 0.0 + 0.0j
    got = False
    for waves_path, weight, _angle_deg in manifest:
        meta, waves = load_waves_tsv(waves_path)
        mw = waves_for_order(waves, scr_m)
        if not mw:
            continue
        ex, ey, ez = order_field_at_origin(meta, mw, z_mm=z_mm)
        ex_sum += weight * ex
        ey_sum += weight * ey
        ez_sum += weight * ez
        got = True
    if not got:
        return 0.0
    return float(abs(ex_sum) ** 2 + abs(ey_sum) ** 2 + abs(ez_sum) ** 2)


def coherent_order_patch_from_manifest(
    manifest: list[tuple[Path, float, float]],
    scr_m: int,
    x_center_mm: float,
    y_center_mm: float,
    *,
    meta_ref: SimMeta,
    z_mm: float,
    order_half_width_mm: float,
    y_half_span_mm: float,
    nx: int,
    ny: int,
    sigma_x_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    peak = incoherent_order_peak_intensity(manifest, scr_m, z_mm=z_mm)
    patch = finite_spot_intensity_patch(
        x_center_mm=x_center_mm,
        y_center_mm=y_center_mm,
        peak_intensity=peak,
        sigma_x_mm=sigma_x_mm,
        order_half_width_mm=order_half_width_mm,
        y_half_span_mm=y_half_span_mm,
        nx=nx,
        ny=ny,
    )
    if patch is None:
        return None
    return patch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Finite-spot far-field: multi-angle per-order receiving screens"
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-mm", type=float, default=40.0)
    ap.add_argument("--spot-w0-um", type=float, default=50.0)
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
    ap.add_argument("--panels-out", default="")
    ap.add_argument("--polarization", type=str, default="TE")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--theta0-deg", type=float, default=60.0)
    ap.add_argument("--wl-nm", type=float, default=13.5)
    ap.add_argument(
        "--intensity-scale",
        choices=("linear", "sqrt", "log", "asinh"),
        default="log",
    )
    ap.add_argument("--axis-half-mm", type=float, default=DEFAULT_AXIS_HALF_MM)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    center_path = min(manifest, key=lambda r: abs(r[2] - args.theta0_deg))[0]
    meta_c, waves_c = load_waves_tsv(center_path)
    meta_c.angle_deg = args.theta0_deg
    flux_c = load_flux_tsv(center_path)
    mode = choose_farfield_screen_mode(args.screen_mode, args.theta0_deg)

    sx = sigma_x_mm(z_mm=args.z_mm, wl_nm=args.wl_nm, w0_um=args.spot_w0_um)
    print(f"Angular spectrum: {len(manifest)} angles; screen_mode={mode}", file=sys.stderr)
    print(f"σ_θ blur σ_x ≈ {sx:.4f} mm", file=sys.stderr)

    screens = compute_order_screens(
        waves_c,
        z_mm=args.z_mm,
        period_nm=meta_c.period_nm,
        flux=flux_c,
        r_threshold=args.order_r_thresh,
    )
    if not screens:
        raise SystemExit("No orders above threshold")

    print(f"Order screens ({len(screens)}):", file=sys.stderr)
    for scr in screens:
        print(
            f"  m={scr.m:+d}  u={scr.x_center_mm:.2f} mm  v={scr.y_center_mm:.2f} mm  "
            f"R={scr.r_abs:.6g}",
            file=sys.stderr,
        )

    if mode == "unified":
        raise SystemExit(
            "unified screen shows interference at grazing incidence; use stitched/orders"
        )

    patches_raw = []
    for scr in screens:
        got = coherent_order_patch_from_manifest(
            manifest,
            scr.m,
            scr.x_center_mm,
            scr.y_center_mm,
            meta_ref=meta_c,
            z_mm=args.z_mm,
            order_half_width_mm=args.order_half_width_mm,
            y_half_span_mm=args.y_half_span_mm,
            nx=args.nx,
            ny=args.ny,
            sigma_x_mm=sx,
        )
        if got is None:
            continue
        x_mm, y_mm, intensity = got
        patches_raw.append((scr, x_mm, y_mm, intensity))

    screens_ok = [p[0] for p in patches_raw]
    patch_tuples = [(p[1], p[2], p[3]) for p in patches_raw]
    pol = args.polarization.strip().upper() or "TE"
    suffix = f"(angular spectrum, w₀={args.spot_w0_um:g} μm, Nθ={len(manifest)})"
    out = Path(args.out)

    if mode == "stitched":
        plot_stitched_order_spots(
            screens_ok,
            patch_tuples,
            z_mm=args.z_mm,
            meta=meta_c,
            out_path=out,
            dpi=args.dpi,
            polarization=pol,
            title_suffix=suffix,
            title_angle_deg=args.theta0_deg,
            sigma_x_mm=sx,
            axis_half_mm=args.axis_half_mm,
            intensity_scale=args.intensity_scale,
        )
    else:
        # panels: reuse single-angle helper per order with precomputed intensity
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import math

        n = len(patches_raw)
        ncols = min(4, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), facecolor="w", squeeze=False
        )
        for (scr, x_mm, y_mm, intensity), ax in zip(patches_raw, axes.ravel()):
            im = ax.imshow(
                intensity,
                origin="lower",
                aspect="auto",
                extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
                cmap="viridis",
            )
            ax.set_title(
                f"m={scr.m:+d}, u={scr.x_center_mm:.1f}, v={scr.y_center_mm:.1f} mm\n"
                f"R={scr.r_abs:.4g}",
                fontsize=10,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Per-order screens {suffix} — θ₀={args.theta0_deg}°, z={args.z_mm:g} mm",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, facecolor="w", bbox_inches="tight")
        plt.close(fig)

    print(out.resolve())


if __name__ == "__main__":
    main()
