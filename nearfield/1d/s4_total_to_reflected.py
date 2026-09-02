#!/usr/bin/env python3
"""
Convert S4 near-field TSV (total field) to reflected-only TSV by subtracting
the S4 planewave incident (MakeExcitationPlanewave hx/hy + vacuum k×H → E).

Aligns with S4 GetFieldAtPoint spatial phase exp(+i(k_x x + k_y y)) in period-normalized
coordinates (see S4/rcwa.cpp, S4/S4.cpp Simulation_MakeExcitationPlanewave).
"""
from __future__ import annotations

import argparse
import cmath
import math
import re
import sys
from pathlib import Path


def parse_header_meta(path: Path) -> tuple[float | None, float | None, float | None]:
    wl_nm = None
    angle_deg = None
    period_nm = None
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                m = re.search(
                    r"wl\s*=\s*([\d.]+)\s*nm.*?angle\s*=\s*([\d.]+).*?period\s*=\s*([\d.]+)",
                    line,
                    re.I,
                )
                if m:
                    wl_nm = float(m.group(1))
                    angle_deg = float(m.group(2))
                    period_nm = float(m.group(3))
                    break
                m2 = re.search(r"wl\s*=\s*([\d.]+)\s*nm.*?angle\s*=\s*([\d.]+)", line, re.I)
                if m2:
                    wl_nm = float(m2.group(1))
                    angle_deg = float(m2.group(2))
    except OSError:
        pass
    return wl_nm, angle_deg, period_nm


def iter_tsv_rows(path: Path):
    in_block = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if s == "#BEGIN_NEARFIELD_TSV":
                in_block = True
                continue
            if s == "#END_NEARFIELD_TSV":
                break
            if not in_block or not s or s.startswith("#") or s.startswith("x_norm"):
                continue
            parts = s.split("\t")
            if len(parts) < 11:
                continue
            try:
                yield [float(x) for x in parts[:11]]
            except ValueError:
                continue


def s4_make_excitation_hx_hy(
    angle_deg: float,
    phi_deg: float,
    *,
    pol_s_amp: float,
    pol_s_phase_deg: float,
    pol_p_amp: float,
    pol_p_phase_deg: float,
    n_inc_real: float,
) -> tuple[complex, complex, float, float]:
    """
    Port of Simulation_MakeExcitationPlanewave (S4/S4.cpp ~3322–3336).
    Returns hx, hy (complex), k_parallel x/y in period-normalized units (S->k[0], S->k[1]).
    """
    th = math.radians(angle_deg)
    ph = math.radians(phi_deg)
    c0, s0 = math.cos(th), math.sin(th)
    c1, s1 = math.cos(ph), math.sin(ph)
    ps = math.radians(pol_s_phase_deg)
    pp = math.radians(pol_p_phase_deg)
    cs, ss = math.cos(ps), math.sin(ps)
    cp, sp = math.cos(pp), math.sin(pp)
    root_eps = math.sqrt(max(n_inc_real, 1e-30))
    hx = complex(
        -c0 * c1 * pol_s_amp * cs - s1 * pol_p_amp * cp,
        -c0 * c1 * pol_s_amp * ss - s1 * pol_p_amp * sp,
    )
    hy = complex(
        -c0 * s1 * pol_s_amp * cs + c1 * pol_p_amp * cp,
        -c0 * s1 * pol_s_amp * ss + c1 * pol_p_amp * sp,
    )
    kx0_norm = c1 * s0 * root_eps
    ky0_norm = s1 * s0 * root_eps
    return hx, hy, kx0_norm, ky0_norm


def s4_planewave_incident_fields(
    x_norm: float,
    y_norm: float,
    x_nm: float,
    z_nm: float,
    *,
    wl_nm: float,
    period_nm: float,
    angle_deg: float,
    phi_deg: float,
    n_inc_real: float,
    phase_sign: str,
    z_term_sign: str,
    pol_s_amp: float,
    pol_s_phase_deg: float,
    pol_p_amp: float,
    pol_p_phase_deg: float,
    te_scale: float,
) -> tuple[complex, complex, complex]:
    """
    Incident Ex, Ey, Ez at (x,y,z): H from MakeExcitationPlanewave × exp(+i phase),
    E = (k × H) / omega0 (vacuum, forward wave; matches S4 +i spatial convention).
    """
    k0 = 2.0 * math.pi / wl_nm
    omega0 = 2.0 * math.pi * (period_nm / wl_nm)
    kz_phys = -k0 * n_inc_real * math.cos(math.radians(angle_deg))

    hx, hy, kx0_norm, ky0_norm = s4_make_excitation_hx_hy(
        angle_deg,
        phi_deg,
        pol_s_amp=pol_s_amp,
        pol_s_phase_deg=pol_s_phase_deg,
        pol_p_amp=pol_p_amp,
        pol_p_phase_deg=pol_p_phase_deg,
        n_inc_real=n_inc_real,
    )

    sign = 1.0 if phase_sign == "plus" else -1.0
    z_sign = 1.0 if z_term_sign == "plus" else -1.0
    phase = sign * (
        kx0_norm * x_norm
        + ky0_norm * y_norm
        + z_sign * kz_phys * z_nm
    )
    pf = cmath.exp(1j * phase)

    Hx = hx * pf
    Hy = hy * pf
    Hz = 0.0 + 0.0j

    kx_p = kx0_norm / period_nm
    ky_p = ky0_norm / period_nm
    kz_p = kz_phys

    # E = (k × H) / omega0
    ex_i = te_scale * (ky_p * Hz - kz_p * Hy) / omega0
    ey_i = te_scale * (kz_p * Hx - kx_p * Hz) / omega0
    ez_i = te_scale * (kx_p * Hy - ky_p * Hx) / omega0
    return ex_i, ey_i, ez_i


def write_reflected(
    src: Path,
    dst: Path,
    *,
    wl_nm: float,
    period_nm: float,
    angle_deg: float,
    phi_deg: float,
    n_inc_real: float,
    phase_sign: str,
    te_sign: int,
    carrier_gauge: str,
    z_term_sign: str,
    pol_s_amp: float,
    pol_s_phase_deg: float,
    pol_p_amp: float,
    pol_p_phase_deg: float,
) -> int:
    k0 = 2.0 * math.pi / wl_nm
    th = math.radians(angle_deg)
    omega0 = 2.0 * math.pi * (period_nm / wl_nm)
    root_eps = math.sqrt(max(n_inc_real, 1e-30))
    kx0_norm = math.cos(math.radians(phi_deg)) * math.sin(th) * root_eps
    ky0_norm = math.sin(math.radians(phi_deg)) * math.sin(th) * root_eps
    kx_inc = k0 * n_inc_real * math.sin(th) * math.cos(math.radians(phi_deg))
    ky_inc = k0 * n_inc_real * math.sin(th) * math.sin(math.radians(phi_deg))
    kz_inc = -k0 * n_inc_real * math.cos(th)
    te_scale = 1.0 if te_sign >= 0 else -1.0

    n_rows = 0
    with dst.open("w", encoding="utf-8") as out:
        out.write(
            (
                f"# S4 reflected near-field (total minus S4 planewave incident) | "
                f"wl={wl_nm:g} nm angle={angle_deg:g} deg period={period_nm:g} nm "
                f"n_inc_real={n_inc_real:g}\n"
            )
        )
        out.write(
            (
                f"# Convention: phase_sign={phase_sign} te_sign={te_sign:+d} "
                f"carrier_gauge={carrier_gauge} z_term_sign={z_term_sign} "
                f"pol_s=({pol_s_amp},{pol_s_phase_deg}deg) pol_p=({pol_p_amp},{pol_p_phase_deg}deg)\n"
            )
        )
        out.write(
            (
                f"# k_inc phys (rad/nm): kx={kx_inc:.10g} ky={ky_inc:.10g} kz={kz_inc:.10g}; "
                f"kx0_norm={kx0_norm:.10g} omega0={omega0:.10g} (S4 InitSolution G=0)\n"
            )
        )
        out.write("#BEGIN_NEARFIELD_TSV\n")
        out.write("x_norm\tz_norm\tx_nm\tz_nm\tExr\tExi\tEyr\tEyi\tEzr\tEzi\tE_mag\n")

        y_norm = 0.0
        y_nm = 0.0
        for row in iter_tsv_rows(src):
            x_norm, z_norm, x_nm, z_nm = row[0], row[1], row[2], row[3]
            ex = complex(row[4], row[5])
            ey = complex(row[6], row[7])
            ez = complex(row[8], row[9])

            ex_i, ey_i, ez_i = s4_planewave_incident_fields(
                x_norm,
                y_norm,
                x_nm,
                z_nm,
                wl_nm=wl_nm,
                period_nm=period_nm,
                angle_deg=angle_deg,
                phi_deg=phi_deg,
                n_inc_real=n_inc_real,
                phase_sign=phase_sign,
                z_term_sign=z_term_sign,
                pol_s_amp=pol_s_amp,
                pol_s_phase_deg=pol_s_phase_deg,
                pol_p_amp=pol_p_amp,
                pol_p_phase_deg=pol_p_phase_deg,
                te_scale=te_scale,
            )

            ex_r = ex - ex_i
            ey_r = ey - ey_i
            ez_r = ez - ez_i

            if carrier_gauge == "remove":
                cg = cmath.exp(-1j * kx_inc * x_nm)
                ex_r *= cg
                ey_r *= cg
                ez_r *= cg
            elif carrier_gauge == "add":
                cg = cmath.exp(1j * kx_inc * x_nm)
                ex_r *= cg
                ey_r *= cg
                ez_r *= cg

            e_mag = math.sqrt(abs(ex_r) ** 2 + abs(ey_r) ** 2 + abs(ez_r) ** 2)

            out.write(
                (
                    f"{x_norm:.10g}\t{z_norm:.10g}\t{x_nm:.10g}\t{z_nm:.10g}\t"
                    f"{ex_r.real:.10g}\t{ex_r.imag:.10g}\t"
                    f"{ey_r.real:.10g}\t{ey_r.imag:.10g}\t"
                    f"{ez_r.real:.10g}\t{ez_r.imag:.10g}\t"
                    f"{e_mag:.10g}\n"
                )
            )
            n_rows += 1

        out.write("#END_NEARFIELD_TSV\n")
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert S4 total near-field TSV into reflected-only TSV."
    )
    ap.add_argument("s4_total_tsv", help="Input S4 raw log with TSV block")
    ap.add_argument("--out", required=True, help="Output reflected TSV file")
    ap.add_argument("--wl", type=float, default=None, help="Override wavelength (nm)")
    ap.add_argument("--angle", type=float, default=None, help="Override incidence angle (deg)")
    ap.add_argument("--period-nm", type=float, default=None, help="Grating period (nm)")
    ap.add_argument("--phi-deg", type=float, default=0.0, help="Azimuth angle (deg), default 0")
    ap.add_argument("--n-inc-real", type=float, default=1.0, help="Incident medium real n")
    ap.add_argument(
        "--phase-sign",
        choices=("plus", "minus"),
        default="plus",
        help="Spatial phase: exp(+i*phase) matches S4 GetFieldAtPoint",
    )
    ap.add_argument(
        "--te-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Global scale on incident E",
    )
    ap.add_argument(
        "--carrier-gauge",
        choices=("none", "remove", "add"),
        default="none",
        help="Post-subtract gauge: multiply reflected field by exp(±i*kx_inc*x_nm)",
    )
    ap.add_argument(
        "--z-term-sign",
        choices=("plus", "minus"),
        default="plus",
        help="Sign of kz*z in incident spatial phase",
    )
    ap.add_argument("--pol-s-amp", type=float, default=1.0)
    ap.add_argument("--pol-s-phase-deg", type=float, default=0.0)
    ap.add_argument("--pol-p-amp", type=float, default=0.0)
    ap.add_argument("--pol-p-phase-deg", type=float, default=0.0)
    args = ap.parse_args()

    src = Path(args.s4_total_tsv)
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(2)

    wl_h, ang_h, per_h = parse_header_meta(src)
    wl_nm = args.wl if args.wl is not None else (wl_h if wl_h is not None else 13.5)
    angle_deg = args.angle if args.angle is not None else (ang_h if ang_h is not None else 80.0)
    period_nm = args.period_nm if args.period_nm is not None else (per_h if per_h is not None else 80.0)

    dst = Path(args.out)
    n = write_reflected(
        src,
        dst,
        wl_nm=wl_nm,
        period_nm=period_nm,
        angle_deg=angle_deg,
        phi_deg=args.phi_deg,
        n_inc_real=args.n_inc_real,
        phase_sign=args.phase_sign,
        te_sign=args.te_sign,
        carrier_gauge=args.carrier_gauge,
        z_term_sign=args.z_term_sign,
        pol_s_amp=args.pol_s_amp,
        pol_s_phase_deg=args.pol_s_phase_deg,
        pol_p_amp=args.pol_p_amp,
        pol_p_phase_deg=args.pol_p_phase_deg,
    )
    print(dst.resolve(), file=sys.stdout)
    print(f"# rows={n}", file=sys.stdout)


if __name__ == "__main__":
    main()
