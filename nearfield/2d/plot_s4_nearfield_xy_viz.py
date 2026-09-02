#!/usr/bin/env python3
"""
S4 2D 近场 TSV（x–y 平面, 固定 z）-> 空间 4x2 图 + k-space（kx–ky）。
几何：与 au_square_grating_nearfield_xy.lua 一致 — 正方格周期 Lx=Ly=period_nm，
每原胞中心一个 Au 正方形突起，边长 = duty×period_nm。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


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
    """Return x_nm_1d, y_nm_1d, Ex, Ey, Ez complex (ny, nx)."""
    import numpy as np

    if not rows:
        return None
    xs = sorted({r[0] for r in rows})
    ys = sorted({r[1] for r in rows})
    nx, ny = len(xs), len(ys)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    Ex = np.zeros((ny, nx), dtype=complex)
    Ey = np.zeros((ny, nx), dtype=complex)
    Ez = np.zeros((ny, nx), dtype=complex)
    xnm_map: dict[tuple[int, int], float] = {}
    ynm_map: dict[tuple[int, int], float] = {}
    for r in rows:
        ix, iy = xi[r[0]], yi[r[1]]
        Ex[iy, ix] = complex(r[4], r[5])
        Ey[iy, ix] = complex(r[6], r[7])
        Ez[iy, ix] = complex(r[8], r[9])
        xnm_map[(iy, ix)] = r[2]
        ynm_map[(iy, ix)] = r[3]
    x1d = np.array([xnm_map[(0, i)] for i in range(nx)], dtype=float)
    y1d = np.array([ynm_map[(j, 0)] for j in range(ny)], dtype=float)
    return x1d, y1d, Ex, Ey, Ez


def _spectrum_demod_fft_2d(
    field: "np.ndarray",
    demod_phase: "np.ndarray",
    n_rep: int,
    pad_size: int,
) -> "np.ndarray":
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


def plot_spatial(
    x1d,
    y1d,
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

    extent = (float(x1d.min()), float(x1d.max()), float(y1d.min()), float(y1d.max()))
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
        ax_mag.set_ylabel("y (nm)", fontsize=12)
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
        ax_ph.set_ylabel("y (nm)", fontsize=12)
        cb1 = fig.colorbar(im1, ax=ax_ph, fraction=0.046, pad=0.04)
        cb1.set_label("phase (rad)", fontsize=10)

    etot = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
    e_sum = Ex + Ey + Ez
    phase_tot = np.angle(e_sum)

    ax7 = axes[3, 0]
    im7 = ax7.imshow(etot, extent=extent, origin="lower", aspect="auto", cmap=cmap_mag)
    ax7.set_xlabel("x (nm)", fontsize=12)
    ax7.set_ylabel("y (nm)", fontsize=12)
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
    ax8.set_ylabel("y (nm)", fontsize=12)
    ax8.set_title(r"$\arg(E_x+E_y+E_z)$", fontsize=14)
    cb8 = fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    cb8.set_label("phase (rad)", fontsize=10)

    fig.suptitle(
        f"S4 near-field (x–y plane) — λ = {wl_nm:.1f} nm, θ_i = {angle_deg:.2f}°",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_kspace(
    x1d,
    y1d,
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
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    th = math.radians(angle_deg)
    k0 = 2.0 * math.pi / wavelength_nm
    # 与 au_square_grating_nearfield_xy.lua 中 SetExcitationPlanewave({angle_deg, 0}, ...) 一致：xz 面内入射
    kx_inc = k0 * math.sin(th)
    ky_inc = 0.0

    ny_u, nx_u = Ex.shape
    demod = np.exp(
        1j * (kx_inc * x1d[np.newaxis, :] + ky_inc * y1d[:, np.newaxis])
    )
    assert demod.shape == (ny_u, nx_u)

    dx = float((x1d.max() - x1d.min()) / max(nx_u - 1, 1))
    dy = float((y1d.max() - y1d.min()) / max(ny_u - 1, 1))

    i_sum = None
    for fld in (Ex, Ey, Ez):
        spec = _spectrum_demod_fft_2d(fld, demod, n_rep, pad_size)
        t = np.abs(spec) ** 2
        i_sum = t if i_sum is None else i_sum + t

    assert i_sum is not None
    if np.max(i_sum) > 0:
        i_sum = i_sum / np.max(i_sum.ravel())

    disp = np.maximum(i_sum, 0.0) ** (1.0 / nth_root)
    ny_fft, nx_fft = i_sum.shape
    kx_axis = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx_fft, d=dx))
    ky_axis = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny_fft, d=dy))

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 7.0), facecolor="w")
    extent = (
        float(kx_axis[0]),
        float(kx_axis[-1]),
        float(ky_axis[0]),
        float(ky_axis[-1]),
    )
    im = ax.imshow(disp, extent=extent, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel(r"$k_x$ (rad/nm)", fontsize=12)
    ax.set_ylabel(r"$k_y$ (rad/nm)", fontsize=12)
    nr = int(round(nth_root))
    ax.set_title(
        f"k-space (x–y slice): demod. FFT, sum |FFT(E_α)|², norm.; I^(1/{nr})\n"
        f"λ = {wavelength_nm:.1f} nm, θ = {angle_deg:.2f}°, n_rep = {n_rep}, pad = {pad_size}",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"I^(1/{nr}) (norm.)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="S4 2D 近场 TSV -> 空间 4x2 + k-space PNG（x–y 平面）。"
    )
    ap.add_argument("s4_raw", help="含 #BEGIN_NEARFIELD_TSV 的 S4 输出文件")
    ap.add_argument("--out", type=str, required=True, help="空间 4x2 PNG 路径")
    ap.add_argument("--out-kspace", type=str, default=None)
    ap.add_argument("--no-kspace", action="store_true")
    ap.add_argument("--wl", type=float, default=None)
    ap.add_argument("--angle", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--kspace-n-rep", type=int, default=2)
    ap.add_argument("--kspace-pad", type=int, default=256)
    args = ap.parse_args()

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
    x1d, y1d, Ex, Ey, Ez = g

    wl = args.wl
    ang = args.angle
    if wl is None or ang is None:
        hw, ha = parse_header_meta(str(path))
        if wl is None:
            wl = hw if hw is not None else 13.5
        if ang is None:
            ang = ha if ha is not None else 80.0

    out_sp = Path(args.out)
    plot_spatial(x1d, y1d, Ex, Ey, Ez, out_path=out_sp, dpi=args.dpi, wl_nm=wl, angle_deg=ang)
    print(out_sp.resolve(), file=sys.stdout)

    if not args.no_kspace:
        out_k = args.out_kspace
        if out_k is None:
            out_k = str(out_sp.with_name(out_sp.stem + "_kspace" + out_sp.suffix))
        plot_kspace(
            x1d,
            y1d,
            Ex,
            Ey,
            Ez,
            wavelength_nm=wl,
            angle_deg=ang,
            out_path=Path(out_k),
            dpi=args.dpi,
            n_rep=args.kspace_n_rep,
            pad_size=args.kspace_pad,
        )
        print(Path(out_k).resolve(), file=sys.stdout)


if __name__ == "__main__":
    main()
