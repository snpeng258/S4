#!/usr/bin/env python3
"""
S4 near-field raw output -> spatial 4×2（|Eα| 与相位随 x 的曲线，或 x–z 伪彩）+ k-space。
默认：固定 z（单 z 或多 z 时取最接近 --line-at-z-nm 的一层）下对 x 画线，避免 1D 几何在 y 向重复成条纹。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from kspace_plot_common import mirror_kspace_for_display, reflect_kx_about_zero


def extract_block(path: str) -> list[list[float]]:
    rows: list[list[float]] = []
    p = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "#BEGIN_NEARFIELD_TSV":
                p = True
                continue
            if line == "#END_NEARFIELD_TSV":
                break
            if not p or line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue
            try:
                rows.append([float(x) for x in parts[:11]])
            except ValueError:
                continue
    return rows


def parse_header_meta(path: str) -> tuple[float | None, float | None]:
    """Parse `# S4 近场 | wl=... nm angle=... deg` if present."""
    wl, ang = None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                m = re.search(r"wl\s*=\s*([\d.]+)\s*nm.*?angle\s*=\s*([\d.]+)", line, re.I)
                if m:
                    wl = float(m.group(1))
                    ang = float(m.group(2))
                    break
    except OSError:
        pass
    return wl, ang


def rows_to_grids(rows: list[list[float]]):
    """Return x_nm_1d, z_nm_1d, Ex, Ey, Ez complex (nz, nx)."""
    import numpy as np

    if not rows:
        return None
    xs = sorted({r[0] for r in rows})
    zs = sorted({r[1] for r in rows})
    nx, nz = len(xs), len(zs)
    xi = {v: i for i, v in enumerate(xs)}
    zi = {v: i for i, v in enumerate(zs)}
    Ex = np.zeros((nz, nx), dtype=complex)
    Ey = np.zeros((nz, nx), dtype=complex)
    Ez = np.zeros((nz, nx), dtype=complex)
    xnm_map: dict[tuple[int, int], float] = {}
    znm_map: dict[tuple[int, int], float] = {}
    for r in rows:
        ix, iz = xi[r[0]], zi[r[1]]
        Ex[iz, ix] = complex(r[4], r[5])
        Ey[iz, ix] = complex(r[6], r[7])
        Ez[iz, ix] = complex(r[8], r[9])
        xnm_map[(iz, ix)] = r[2]
        znm_map[(iz, ix)] = r[3]
    x1d = np.array([xnm_map[(0, i)] for i in range(nx)], dtype=float)
    z1d = np.array([znm_map[(j, 0)] for j in range(nz)], dtype=float)
    return x1d, z1d, Ex, Ey, Ez


def _spectrum_demod_fft_2d(
    field: "np.ndarray",
    demod_phase: "np.ndarray",
    n_rep: int,
    pad_size: int,
) -> "np.ndarray":
    """旧版 plot_kspace（kx–kz 伪彩）用：二维解调 → 周期复制 → 高斯窗 → 补零 → fft2。当前默认未调用。"""
    import numpy as np

    e_env = field * demod_phase
    e_large = np.tile(e_env, (n_rep, n_rep))
    h, w = e_large.shape
    yg, xg = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    sigma = min(h, w) / 4.5
    win = np.exp(-(((xg - w / 2) ** 2 + (yg - h / 2) ** 2) / (2 * sigma**2)))
    e_win = e_large * win
    e_pad = np.pad(
        e_win,
        ((pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
        constant_values=0,
    )
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(e_pad)))


def _spectrum_demod_fft_1d(
    field_1d: "np.ndarray",
    demod_phase_1d: "np.ndarray",
    n_rep: int,
    pad_size: int,
) -> "np.ndarray":
    """沿 x 线：解调 → 周期拼接 → 窗 → 补零 → FFT。"""
    import numpy as np

    e_env = field_1d * demod_phase_1d
    e_large = np.tile(e_env, n_rep)
    n = int(e_large.size)
    xg = np.arange(n, dtype=float)
    sigma = max(n / 4.5, 1e-9)
    win = np.exp(-((xg - n / 2) ** 2) / (2 * sigma**2))
    e_win = e_large * win
    e_pad = np.pad(e_win, (pad_size, pad_size), mode="constant", constant_values=0.0)
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(e_pad)))


def _spectrum_raw_fft_1d(field_1d: "np.ndarray") -> "np.ndarray":
    """裸 FFT：无解调、周期拼接、高斯窗、补零。"""
    import numpy as np

    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field_1d)))


def plot_spatial(
    x1d,
    z1d,
    Ex,
    Ey,
    Ez,
    *,
    out_path: Path,
    dpi: int,
    wl_nm: float,
    angle_deg: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    extent = (float(x1d.min()), float(x1d.max()), float(z1d.min()), float(z1d.max()))
    fields = [Ex, Ey, Ez]
    labels = ("x", "y", "z")
    cmap_mag = "viridis"
    cmap_phase = "twilight"

    fig, axes = plt.subplots(4, 2, figsize=(14.0, 9.0), facecolor="w")
    for i, (f, lab) in enumerate(zip(fields, labels)):
        ax_mag = axes[i, 0]
        ax_ph = axes[i, 1]
        mag = np.abs(f)
        ph = np.angle(f)
        im0 = ax_mag.imshow(
            mag,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap_mag,
        )
        ax_mag.set_xlabel("x (nm)", fontsize=12)
        ax_mag.set_ylabel("z (nm)", fontsize=12)
        ax_mag.set_title(f"|E_{lab}|", fontsize=14)
        cb0 = fig.colorbar(im0, ax=ax_mag, fraction=0.046, pad=0.04)
        cb0.set_label("|E|", fontsize=10)

        im1 = ax_ph.imshow(
            ph,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap_phase,
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax_ph.set_xlabel("x (nm)", fontsize=12)
        ax_ph.set_ylabel("z (nm)", fontsize=12)
        cb1 = fig.colorbar(im1, ax=ax_ph, fraction=0.046, pad=0.04)
        cb1.set_label("phase (rad)", fontsize=10)

    etot = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
    e_sum = Ex + Ey + Ez
    phase_tot = np.angle(e_sum)

    ax7 = axes[3, 0]
    im7 = ax7.imshow(etot, extent=extent, origin="lower", aspect="auto", cmap=cmap_mag)
    ax7.set_xlabel("x (nm)", fontsize=12)
    ax7.set_ylabel("z (nm)", fontsize=12)
    ax7.set_title(r"$|E_{total}|$", fontsize=14, fontweight="bold")
    cb7 = fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cb7.set_label("|E|", fontsize=10)

    ax8 = axes[3, 1]
    im8 = ax8.imshow(
        phase_tot,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax8.set_xlabel("x (nm)", fontsize=12)
    ax8.set_ylabel("z (nm)", fontsize=12)
    ax8.set_title(r"$\arg(E_x+E_y+E_z)$", fontsize=14)
    cb8 = fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    cb8.set_label("phase (rad)", fontsize=10)

    if len(z1d) == 1:
        z0 = float(z1d[0])
        supt = (
            f"S4 near-field (x at fixed z = {z0:.2f} nm) — λ = {wl_nm:.1f} nm, θ_i = {angle_deg:.2f}°"
        )
    else:
        supt = f"S4 near-field (x–z plane) — λ = {wl_nm:.1f} nm, θ_i = {angle_deg:.2f}°"
    fig.suptitle(supt, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def _pick_z_index(z1d: "np.ndarray", line_at_z_nm: float) -> tuple[int, float]:
    import numpy as np

    z1d = np.asarray(z1d, dtype=float)
    iz = int(np.argmin(np.abs(z1d - line_at_z_nm)))
    return iz, float(z1d[iz])


def plot_spatial_lines(
    x1d_nm,
    z1d_nm,
    Ex,
    Ey,
    Ez,
    *,
    out_path: Path,
    dpi: int,
    wl_nm: float,
    angle_deg: float,
    line_at_z_nm: float,
    phys_tag: str = "",
) -> None:
    """固定 z（或最接近 line_at_z_nm 的采样层）下，各分量 |E|、phase 随 x 的曲线（4×2）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x1d_nm = np.asarray(x1d_nm, dtype=float)
    z1d_nm = np.asarray(z1d_nm, dtype=float)
    iz, z_used = _pick_z_index(z1d_nm, line_at_z_nm)
    ex = np.asarray(Ex[iz, :], dtype=complex)
    ey = np.asarray(Ey[iz, :], dtype=complex)
    ez = np.asarray(Ez[iz, :], dtype=complex)
    fields = [ex, ey, ez]
    labels = ("x", "y", "z")

    fig, axes = plt.subplots(4, 2, figsize=(14.0, 9.0), facecolor="w")
    for i, (f, lab) in enumerate(zip(fields, labels)):
        ax_mag = axes[i, 0]
        ax_ph = axes[i, 1]
        ax_mag.plot(x1d_nm, np.abs(f), color="C0", lw=1.2)
        ax_mag.set_xlabel("x (nm)", fontsize=12)
        ax_mag.set_ylabel(f"$|E_{lab}|$", fontsize=12)
        ax_mag.set_title(f"$|E_{lab}|$", fontsize=14)
        ax_mag.grid(True, alpha=0.3)

        ax_ph.plot(x1d_nm, np.angle(f), color="C1", lw=1.2)
        ax_ph.set_xlabel("x (nm)", fontsize=12)
        ax_ph.set_ylabel("phase (rad)", fontsize=12)
        ax_ph.set_ylim(-np.pi - 0.2, np.pi + 0.2)
        ax_ph.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax_ph.grid(True, alpha=0.3)

    etot = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    e_sum = ex + ey + ez
    phase_tot = np.angle(e_sum)

    axes[3, 0].plot(x1d_nm, etot, color="C0", lw=1.4)
    axes[3, 0].set_xlabel("x (nm)", fontsize=12)
    axes[3, 0].set_ylabel(r"$|E|$", fontsize=12)
    axes[3, 0].set_title(r"$|E_{total}|$", fontsize=14, fontweight="bold")
    axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].plot(x1d_nm, phase_tot, color="C1", lw=1.4)
    axes[3, 1].set_xlabel("x (nm)", fontsize=12)
    axes[3, 1].set_ylabel("phase (rad)", fontsize=12)
    axes[3, 1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[3, 1].axhline(0.0, color="k", lw=0.5, alpha=0.4)
    axes[3, 1].set_title(r"$\arg(E_x+E_y+E_z)$", fontsize=14)
    axes[3, 1].grid(True, alpha=0.3)

    ttl = f"S4 near-field vs x at z = {z_used:.2f} nm — λ = {wl_nm:.1f} nm, θ_i = {angle_deg:.2f}°"
    if phys_tag:
        ttl += f" | {phys_tag}"
    fig.suptitle(ttl, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def _kspace_demod_factor(
    x1d,
    kx_inc: float,
    sign: str,
):
    import numpy as np

    if sign == "none":
        return np.ones_like(x1d, dtype=complex), "no demod"
    if sign == "minus":
        return np.exp(-1j * kx_inc * x1d), "demod exp(-i kx x)"
    if sign == "plus":
        return np.exp(1j * kx_inc * x1d), "demod exp(+i kx x)"
    raise ValueError(f"unknown kspace_demod_sign: {sign!r}")


def plot_kspace(
    x1d,
    z1d,
    Ex,
    Ey,
    Ez,
    *,
    wavelength_nm: float,
    angle_deg: float,
    out_path: Path,
    dpi: int,
    n_rep: int,
    pad_size: int,
    nth_root: float = 3.0,
    line_at_z_nm: float = -5.0,
    n_inc_real: float = 1.0,
    phys_tag: str = "",
    kspace_demod_sign: str = "minus",
    kspace_mode: str = "dsp",
    kspace_s4_mirror: bool = True,
) -> None:
    """固定 z（与空域线切割一致）下沿 x 的一维 k_x 谱。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x1d = np.asarray(x1d, dtype=float)
    z1d = np.asarray(z1d, dtype=float)
    iz, z_used = _pick_z_index(z1d, line_at_z_nm)
    ex = np.asarray(Ex[iz, :], dtype=complex)
    ey = np.asarray(Ey[iz, :], dtype=complex)
    ez = np.asarray(Ez[iz, :], dtype=complex)

    th = math.radians(angle_deg)
    k0 = 2.0 * math.pi / wavelength_nm
    kx_inc = k0 * n_inc_real * math.sin(th)
    raw = kspace_mode == "raw"
    if raw:
        demod_label = "raw FFT, no demod/tile/window/pad"
    else:
        demod_1d, demod_label = _kspace_demod_factor(x1d, kx_inc, kspace_demod_sign)
    # --- 旧版整图回退：x–z 二维 k 谱（imshow kx,kz）；需用本文件中的 _spectrum_demod_fft_2d ---
    # kz_inc = -k0 * math.cos(th)
    # XX, ZZ = np.meshgrid(x1d, z1d, indexing="xy")
    # demod_2d = np.exp(1j * (kx_inc * XX + kz_inc * ZZ))
    # ny_u, nx_u = Ex.shape
    # dz = float((z1d.max() - z1d.min()) / max(ny_u - 1, 1))
    # if dz <= 0.0 or not math.isfinite(dz):
    #     dz = dx if dx > 0.0 else 1.0
    # for fld in (Ex, Ey, Ez):
    #     spec = _spectrum_demod_fft_2d(fld, demod_2d, n_rep, pad_size)
    # ... 再用 kx_axis/kz_axis 与 ax.imshow 绘图（见 git 历史或自行补全）
    nx_u = ex.size
    dx = float((x1d.max() - x1d.min()) / max(nx_u - 1, 1))

    i_sum = None
    for fld in (ex, ey, ez):
        if raw:
            spec = _spectrum_raw_fft_1d(fld)
        else:
            spec = _spectrum_demod_fft_1d(fld, demod_1d, n_rep, pad_size)
        t = np.abs(spec) ** 2
        i_sum = t if i_sum is None else i_sum + t

    assert i_sum is not None
    if np.max(i_sum) > 0:
        i_sum = i_sum / np.max(i_sum.ravel())

    disp = np.maximum(i_sum, 0.0) ** (1.0 / nth_root)
    nx_fft = int(disp.size)
    kx_axis = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx_fft, d=dx))
    kx_mirror_note = ""
    if kspace_s4_mirror:
        kx_axis, disp = mirror_kspace_for_display(kx_axis, disp)
        kx_axis, disp = reflect_kx_about_zero(kx_axis, disp)
        kx_mirror_note = "; kx mirrored"

    nr = int(round(nth_root))
    line_color = "#1565c0"
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.5), facecolor="w")
    ax.fill_between(kx_axis, 0.0, disp, color=line_color, alpha=0.25)
    ax.plot(kx_axis, disp, color=line_color, lw=1.4)
    ax.set_xlabel(r"$k_x$ (rad/nm)", fontsize=12)
    ax.set_ylabel(f"I^(1/{nr}) (norm.)", fontsize=11)
    if raw:
        subtitle = (
            f"λ = {wavelength_nm:.1f} nm, θ = {angle_deg:.2f}°, n_inc = {n_inc_real:.4g}"
            + (f", {phys_tag}" if phys_tag else "")
        )
    else:
        subtitle = (
            f"λ = {wavelength_nm:.1f} nm, θ = {angle_deg:.2f}°, n_inc = {n_inc_real:.4g}, "
            f"n_rep = {n_rep}, pad = {pad_size}"
            + (f", {phys_tag}" if phys_tag else "")
        )
    ax.set_title(
        f"1D k-spectrum along x (fixed z = {z_used:.2f} nm): {demod_label}{kx_mirror_note}; "
        f"FFT, sum |FFT(E_α)|², norm.; I^(1/{nr})\n"
        + subtitle,
        fontsize=11,
    )
    ax.grid(True, alpha=0.35)
    ax.set_xlim(float(kx_axis[0]), float(kx_axis[-1]))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="S4 near-field TSV -> spatial 4×2（默认 |E|/phase 随 x 曲线）+ 可选 k-space PNG。"
    )
    ap.add_argument("s4_raw", help="S4 stdout log file containing #BEGIN_NEARFIELD_TSV block")
    ap.add_argument("--out", type=str, required=True, help="spatial 4x2 PNG path")
    ap.add_argument("--out-kspace", type=str, default=None, help="k-space PNG; default <out_stem>_kspace.png")
    ap.add_argument("--no-kspace", action="store_true")
    ap.add_argument("--wl", type=float, default=None, help="wavelength (nm); else parse from S4 header")
    ap.add_argument("--angle", type=float, default=None, help="incidence angle (deg); else parse from header")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--kspace-n-rep", type=int, default=2)
    ap.add_argument("--kspace-pad", type=int, default=256)
    ap.add_argument("--n-inc-real", type=float, default=1.0, help="incident medium real refractive index for kx demod")
    ap.add_argument("--phys-tag", type=str, default="", help="label appended in spatial/k-space title")
    ap.add_argument(
        "--kspace-demod-sign",
        choices=("minus", "plus", "none"),
        default="minus",
        help="k-space carrier: minus=exp(-i kx x) default (S4 +i spatial); plus=legacy; none=identity",
    )
    ap.add_argument(
        "--skip-kspace-demod",
        action="store_true",
        help="alias for --kspace-demod-sign none",
    )
    ap.add_argument(
        "--kspace-mode",
        choices=("dsp", "raw"),
        default="dsp",
        help="dsp: demod+tile+window+pad+FFT; raw: direct fft(ifftshift E_line)",
    )
    ap.add_argument(
        "--kspace-s4-mirror",
        action="store_true",
        help="optional kx display flip (off by default)",
    )
    ap.add_argument(
        "--no-kspace-s4-mirror",
        action="store_true",
        help="deprecated: kx mirror is already off by default",
    )
    ap.add_argument(
        "--spatial-style",
        choices=("line", "imshow"),
        default="line",
        help="line: |E|/phase vs x at fixed z (see --line-at-z-nm); imshow: x–z pseudocolor",
    )
    ap.add_argument(
        "--line-at-z-nm",
        type=float,
        default=-5.0,
        help="when spatial-style=line and nz>1, pick z-layer closest to this (nm); ignored if nz==1",
    )
    args = ap.parse_args()
    kspace_demod = "none" if args.skip_kspace_demod else args.kspace_demod_sign

    path = Path(args.s4_raw)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(2)

    rows = extract_block(str(path))
    if not rows:
        print("No TSV block in file.", file=sys.stderr)
        sys.exit(2)

    g = rows_to_grids(rows)
    if g is None:
        sys.exit(2)
    x1d, z1d, Ex, Ey, Ez = g

    wl = args.wl
    ang = args.angle
    if wl is None or ang is None:
        hw, ha = parse_header_meta(str(path))
        if wl is None:
            wl = hw if hw is not None else 13.5
        if ang is None:
            ang = ha if ha is not None else 80.0

    out_sp = Path(args.out)
    if args.spatial_style == "line":
        plot_spatial_lines(
            x1d,
            z1d,
            Ex,
            Ey,
            Ez,
            out_path=out_sp,
            dpi=args.dpi,
            wl_nm=wl,
            angle_deg=ang,
            line_at_z_nm=args.line_at_z_nm,
            phys_tag=args.phys_tag,
        )
    else:
        plot_spatial(x1d, z1d, Ex, Ey, Ez, out_path=out_sp, dpi=args.dpi, wl_nm=wl, angle_deg=ang)
    print(out_sp.resolve(), file=sys.stdout)

    if not args.no_kspace:
        out_k = args.out_kspace
        if out_k is None:
            out_k = str(out_sp.with_name(out_sp.stem + "_kspace" + out_sp.suffix))
        plot_kspace(
            x1d,
            z1d,
            Ex,
            Ey,
            Ez,
            wavelength_nm=wl,
            angle_deg=ang,
            out_path=Path(out_k),
            dpi=args.dpi,
            n_rep=args.kspace_n_rep,
            pad_size=args.kspace_pad,
            line_at_z_nm=args.line_at_z_nm,
            n_inc_real=args.n_inc_real,
            phys_tag=args.phys_tag,
            kspace_demod_sign=kspace_demod,
            kspace_mode=args.kspace_mode,
            kspace_s4_mirror=args.kspace_s4_mirror and not args.no_kspace_s4_mirror,
        )
        print(Path(out_k).resolve(), file=sys.stdout)


if __name__ == "__main__":
    main()
