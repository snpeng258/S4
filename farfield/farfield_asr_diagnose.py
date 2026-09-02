#!/usr/bin/env python3
"""
Export ASR far-field pipeline data for spot-shape diagnosis (no fake stamps).

Saves per-order I_raw / I_norm, stitched canvases (native vs interp), profiles,
and quantitative metrics vs theoretical Gaussian σ = zλ/(πw₀).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_FAR = Path(__file__).resolve().parent
if str(_FAR) not in sys.path:
    sys.path.insert(0, str(_FAR))

from scipy import ndimage

from s4_farfield_asr import (  # noqa: E402
    DEFAULT_AXIS_HALF_MM,
    compute_asr_order_patches,
    farfield_sigma_mm,
    kdomain_asr_farfield,
    observation_angles_for_mode,
    paste_patch_add_native,
)
from s4_farfield_reconstruct import (  # noqa: E402
    compute_order_screens,
    load_flux_tsv,
    load_waves_tsv,
    order_peak_intensity,
    waves_for_order,
)


def _theory_intensity(
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    u0: float,
    v0: float,
    *,
    peak: float,
    sigma_u_mm: float,
    sigma_v_mm: float,
) -> np.ndarray:
    ug, vg = np.meshgrid(us_mm, vs_mm, indexing="xy")
    du = (ug - u0) / max(sigma_u_mm, 1e-12)
    dv = (vg - v0) / max(sigma_v_mm, 1e-12)
    return peak * np.exp(-0.5 * du**2 - 0.5 * dv**2)


def _fit_sigma_1d(r_mm: np.ndarray, i: np.ndarray) -> float | None:
    mask = (r_mm > 0) & (i > 0) & np.isfinite(i)
    if mask.sum() < 4:
        return None
    r = r_mm[mask]
    y = np.log(i[mask])
    slope, _ = np.polyfit(r**2, y, 1)
    if slope >= 0:
        return None
    return float(np.sqrt(-0.5 / slope))


def _patch_metrics(
    i: np.ndarray,
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    u0: float,
    v0: float,
    *,
    sigma_u_mm: float,
    sigma_v_mm: float,
) -> dict:
    peak = float(np.max(i))
    if peak <= 0:
        return {"peak": 0.0}
    ug, vg = np.meshgrid(us_mm, vs_mm, indexing="xy")
    ru = (ug - u0) / max(sigma_u_mm, 1e-12)
    rv = (vg - v0) / max(sigma_v_mm, 1e-12)
    r = np.sqrt(ru**2 + rv**2)
    wings = i[(r > 2.0) & (i > 0)]
    wing_med = float(np.median(wings)) if wings.size else 1e-30
    plateau = float(np.mean(i > 0.5 * peak))
    sigma_r_fit = _fit_sigma_1d(r.ravel(), i.ravel())
    j_pk, i_pk = np.unravel_index(int(np.argmax(i)), i.shape)
    peak_offset_u_um = float((us_mm[i_pk] - u0) * 1000.0)
    peak_offset_v_um = float((vs_mm[j_pk] - v0) * 1000.0)
    mx = ndimage.maximum_filter(i, size=5) == i
    n_local_max = int((mx & (i > 0.1 * peak)).sum())
    return {
        "peak": peak,
        "min": float(np.min(i)),
        "std": float(np.std(i)),
        "dynamic_range": peak / max(wing_med, 1e-30),
        "plateau_frac": plateau,
        "sigma_radial_fit_mm": sigma_r_fit,
        "sigma_u_theory_mm": sigma_u_mm,
        "sigma_v_theory_mm": sigma_v_mm,
        "peak_offset_u_um": peak_offset_u_um,
        "peak_offset_v_um": peak_offset_v_um,
        "n_local_max": n_local_max,
    }


def _canvas_stitch_metrics(
    canvas: np.ndarray,
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    patches: list,
) -> dict:
    span_u = float(us_mm[-1] - us_mm[0]) if len(us_mm) > 1 else 0.0
    span_v = float(vs_mm[-1] - vs_mm[0]) if len(vs_mm) > 1 else 0.0
    du_um = span_u / max(len(us_mm) - 1, 1) * 1000.0
    dv_um = span_v / max(len(vs_mm) - 1, 1) * 1000.0
    nz = int((canvas > 0).sum())
    total = int(canvas.size)
    per_order: dict[str, dict] = {}
    for scr, us_loc, vs_loc, intensity in patches:
        c = np.zeros_like(canvas)
        paste_patch_add_native(c, us_mm, vs_mm, us_loc, vs_loc, intensity)
        pk = float(intensity.max()) if intensity.size else 0.0
        per_order[str(scr.m)] = {
            "deposited_px": int((c > 0).sum()),
            "canvas_peak": float(c.max()),
            "patch_peak": pk,
            "pileup_ratio": float(c.max() / pk) if pk > 0 else 0.0,
        }
    return {
        "du_um": du_um,
        "dv_um": dv_um,
        "span_u_mm": span_u,
        "span_v_mm": span_v,
        "nonzero_px": nz,
        "nonzero_frac": nz / max(total, 1),
        "canvas_peak": float(canvas.max()),
        "per_order": per_order,
    }


def _canvas_profile(
    canvas: np.ndarray,
    us_mm: np.ndarray,
    vs_mm: np.ndarray,
    u0: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iu = int(np.argmin(np.abs(us_mm - u0)))
    iv = int(np.argmin(np.abs(vs_mm - v0)))
    return us_mm, canvas[iv, :], vs_mm, canvas[:, iu]


def _save_profile_plot(
    path: Path,
    x_u: np.ndarray,
    prof_u: np.ndarray,
    x_v: np.ndarray,
    prof_v: np.ndarray,
    *,
    theory_u: np.ndarray | None = None,
    theory_v: np.ndarray | None = None,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="w")
    ax1.plot(x_u, prof_u, "b-", label="ASR")
    if theory_u is not None:
        ax1.plot(x_u, theory_u, "r--", label="theory")
    ax1.set_xlabel("u (mm)")
    ax1.set_ylabel("|E|²")
    ax1.set_title("u cut")
    ax1.legend()
    ax2.plot(x_v, prof_v, "b-", label="ASR")
    if theory_v is not None:
        ax2.plot(x_v, theory_v, "r--", label="theory")
    ax2.set_xlabel("v (mm)")
    ax2.set_ylabel("|E|²")
    ax2.set_title("v cut")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor="w")
    plt.close(fig)


def _save_patch_heatmap(path: Path, us: np.ndarray, vs: np.ndarray, i: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="w")
    im = ax.imshow(
        i,
        origin="lower",
        aspect="auto",
        extent=(us[0], us[-1], vs[0], vs[-1]),
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor="w")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="ASR spot diagnostics (real ASR data only)")
    ap.add_argument("waves_tsv")
    ap.add_argument("--dump-dir", type=Path, required=True)
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
    ap.add_argument("--stitch-m-max", type=int, default=1)
    args = ap.parse_args()

    dump = args.dump_dir
    dump.mkdir(parents=True, exist_ok=True)

    meta, waves = load_waves_tsv(Path(args.waves_tsv))
    flux = load_flux_tsv(Path(args.waves_tsv))
    screens = compute_order_screens(
        waves,
        z_mm=args.z_mm,
        period_nm=meta.period_nm,
        flux=flux,
        r_threshold=args.order_r_thresh,
    )
    sigma_u_mm = farfield_sigma_mm(
        z_mm=args.z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=args.aperture_u_um
    )
    sigma_v_mm = farfield_sigma_mm(
        z_mm=args.z_mm, wl_nm=meta.wl_nm, aperture_fwhm_um=args.aperture_v_um
    )
    sigma_u = 0.5 * args.aperture_u_um
    sigma_v = 0.5 * args.aperture_v_um

    summary: dict = {
        "meta": {
            "wl_nm": meta.wl_nm,
            "aperture_u_um": args.aperture_u_um,
            "aperture_v_um": args.aperture_v_um,
            "z_mm": args.z_mm,
            "sigma_u_theory_mm": sigma_u_mm,
            "sigma_v_theory_mm": sigma_v_mm,
        },
        "orders": {},
    }

    order_patches, _, _, _, _ = compute_asr_order_patches(
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
        theta2_deg=None,
        phi2_deg=None,
    )
    patch_by_m = {scr.m: (scr, us_loc, vs_loc, i_norm) for scr, us_loc, vs_loc, i_norm in order_patches}

    for scr in screens:
        mw = waves_for_order(waves, scr.m)
        if not mw:
            continue
        entry = patch_by_m.get(scr.m)
        if entry is None:
            continue
        scr, us_loc, vs_loc, i_norm = entry
        w = mw[0]
        theta2, phi2 = observation_angles_for_mode(
            meta, w, theta2_deg=None, phi2_deg=None
        )
        ref = order_peak_intensity(meta, waves, scr.m, z_mm=args.z_mm)
        i_raw = i_norm * (ref / max(float(i_norm.max()), 1e-30))
        i_theory = _theory_intensity(
            us_loc,
            vs_loc,
            scr.x_center_mm,
            scr.y_center_mm,
            peak=ref,
            sigma_u_mm=sigma_u_mm,
            sigma_v_mm=sigma_v_mm,
        )
        m_raw = _patch_metrics(
            i_raw,
            us_loc,
            vs_loc,
            scr.x_center_mm,
            scr.y_center_mm,
            sigma_u_mm=sigma_u_mm,
            sigma_v_mm=sigma_v_mm,
        )
        m_norm = _patch_metrics(
            i_norm,
            us_loc,
            vs_loc,
            scr.x_center_mm,
            scr.y_center_mm,
            sigma_u_mm=sigma_u_mm,
            sigma_v_mm=sigma_v_mm,
        )
        tag = f"m{scr.m:+d}".replace("+", "p").replace("-", "m")
        np.savez(
            dump / f"patch_{tag}.npz",
            us_loc_mm=us_loc,
            vs_loc_mm=vs_loc,
            I_raw=i_raw,
            I_norm=i_norm,
            I_theory=i_theory,
            u_center_mm=scr.x_center_mm,
            v_center_mm=scr.y_center_mm,
        )
        _save_patch_heatmap(
            dump / f"patch_{tag}_I_raw_linear.png",
            us_loc,
            vs_loc,
            i_raw,
            f"m={scr.m:+d} I_raw (linear)",
        )
        _save_patch_heatmap(
            dump / f"patch_{tag}_I_norm_linear.png",
            us_loc,
            vs_loc,
            i_norm,
            f"m={scr.m:+d} I_norm (linear)",
        )
        iu = int(np.argmin(np.abs(us_loc - scr.x_center_mm)))
        iv = int(np.argmin(np.abs(vs_loc - scr.y_center_mm)))
        _save_profile_plot(
            dump / f"patch_{tag}_profile.png",
            us_loc,
            i_norm[iv, :],
            vs_loc,
            i_norm[:, iu],
            theory_u=i_theory[iv, :],
            theory_v=i_theory[:, iu],
            title=f"m={scr.m:+d} patch profile vs theory",
        )
        summary["orders"][str(scr.m)] = {
            "center_mm": [scr.x_center_mm, scr.y_center_mm],
            "I_raw": m_raw,
            "I_norm": m_norm,
            "theta2_deg": math.degrees(theta2),
            "phi2_deg": math.degrees(phi2),
            "diagnosis": (
                "flat_in_patch"
                if m_raw.get("dynamic_range", 0) < 10
                else "gaussian_like_in_patch"
            ),
        }
        print(
            f"order m={scr.m:+d}  I_raw dynamic_range={m_raw.get('dynamic_range', 0):.2g}  "
            f"n_local_max={m_raw.get('n_local_max', 0)}  "
            f"peak_off=({m_raw.get('peak_offset_u_um', 0):.0f},"
            f"{m_raw.get('peak_offset_v_um', 0):.0f}) um",
            file=sys.stderr,
        )

    order_patches_for_stitch = order_patches
    canvas_native, us_mm, vs_mm, _, _ = kdomain_asr_farfield(
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
        theta2_deg=None,
        phi2_deg=None,
        stitch_m_max=args.stitch_m_max,
        stitch_paste_method="native",
        order_patches=order_patches_for_stitch,
    )
    canvas_interp, _, _, _, _ = kdomain_asr_farfield(
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
        theta2_deg=None,
        phi2_deg=None,
        stitch_m_max=args.stitch_m_max,
        stitch_paste_method="interp",
        order_patches=order_patches_for_stitch,
    )
    stitch_subset = [p for p in order_patches_for_stitch if abs(p[0].m) <= args.stitch_m_max]
    summary["stitch_native"] = _canvas_stitch_metrics(
        canvas_native, us_mm, vs_mm, stitch_subset
    )
    summary["stitch_interp"] = _canvas_stitch_metrics(
        canvas_interp, us_mm, vs_mm, stitch_subset
    )
    np.savez(
        dump / "canvas_stitched.npz",
        canvas_native=canvas_native,
        canvas_interp=canvas_interp,
        us_mm=us_mm,
        vs_mm=vs_mm,
    )

    m0 = next((s for s in screens if s.m == 0), screens[0])
    cu, pu_u, cv, pu_v = _canvas_profile(
        canvas_native, us_mm, vs_mm, m0.x_center_mm, m0.y_center_mm
    )
    _, pi_u, _, pi_v = _canvas_profile(
        canvas_interp, us_mm, vs_mm, m0.x_center_mm, m0.y_center_mm
    )
    ref0 = order_peak_intensity(meta, waves, m0.m, z_mm=args.z_mm)
    th_prof_u = ref0 * np.exp(-0.5 * ((cu - m0.x_center_mm) / max(sigma_u_mm, 1e-12)) ** 2)
    th_prof_v = ref0 * np.exp(-0.5 * ((cv - m0.y_center_mm) / max(sigma_v_mm, 1e-12)) ** 2)
    _save_profile_plot(
        dump / "canvas_m0_native_vs_interp.png",
        cu,
        pu_u,
        cv,
        pu_v,
        theory_u=th_prof_u,
        theory_v=th_prof_v,
        title="m=0 canvas profiles (native cut) vs theory",
    )
    fig_lines = dump / "canvas_m0_profiles.tsv"
    with fig_lines.open("w") as f:
        f.write("axis\tcoord_mm\tI_native\tI_interp\n")
        for u, a, b in zip(cu, pu_u, pi_u):
            f.write(f"u\t{u:.6g}\t{a:.6g}\t{b:.6g}\n")
        for v, a, b in zip(cv, pu_v, pi_v):
            f.write(f"v\t{v:.6g}\t{a:.6g}\t{b:.6g}\n")

    raw0 = summary["orders"].get(str(m0.m), {}).get("I_raw", {})
    if raw0.get("dynamic_range", 0) >= 10:
        summary["root_cause"] = (
            "patch_has_gradient; if canvas looks flat use native paste + higher OBS_N"
        )
    else:
        summary["root_cause"] = "ASR patch I_raw is flat — k-grid / midft resolution"

    with (dump / "diagnose_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "ASR spot diagnosis summary",
        f"aperture {args.aperture_u_um} x {args.aperture_v_um} um  z={args.z_mm} mm",
        f"theory sigma_u={sigma_u_mm:.4f} mm  sigma_v={sigma_v_mm:.4f} mm",
        f"stitch |m|<={args.stitch_m_max}  native du={summary['stitch_native']['du_um']:.1f} um",
        f"root_cause: {summary['root_cause']}",
        "",
    ]
    for m, info in summary["orders"].items():
        lines.append(f"order m={m}: {info['diagnosis']}")
        lines.append(f"  I_raw dynamic_range={info['I_raw'].get('dynamic_range', 0):.4g}")
        lines.append(f"  n_local_max={info['I_raw'].get('n_local_max', 0)}")
        lines.append(
            f"  peak_offset_um=({info['I_raw'].get('peak_offset_u_um', 0):.0f},"
            f"{info['I_raw'].get('peak_offset_v_um', 0):.0f})"
        )
    (dump / "diagnose_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote diagnostics: {dump.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
