#!/usr/bin/env python3
"""
Diagnose reflected near-field carrier: phase slope dφ/dx and FFT peak kx (with/without demod).
Usage: python3 diag_reflected_carrier.py <*_reflected.txt> [--wl NM] [--angle DEG] [--period NM]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


def parse_meta(path: Path) -> dict[str, float]:
    meta: dict[str, float] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for _ in range(30):
            line = f.readline()
            if not line:
                break
            m = re.search(
                r"wl\s*=\s*([\d.]+).*?angle\s*=\s*([\d.]+).*?period\s*=\s*([\d.]+)",
                line,
                re.I,
            )
            if m:
                meta["wl"] = float(m.group(1))
                meta["angle"] = float(m.group(2))
                meta["period"] = float(m.group(3))
            m2 = re.search(r"kx0_norm\s*=\s*([\d.eE+-]+)", line)
            if m2:
                meta["kx0_norm"] = float(m2.group(1))
            m3 = re.search(r"kx\s*=\s*([\d.eE+-]+)", line)
            if m3 and "kx_inc" not in meta:
                meta["kx_inc"] = float(m3.group(1))
    return meta


def load_tsv(path: Path):
    xs: list[float] = []
    ex: list[complex] = []
    ey: list[complex] = []
    ez: list[complex] = []
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
            p = s.split("\t")
            if len(p) < 10:
                continue
            try:
                xs.append(float(p[2]))
                ex.append(complex(float(p[4]), float(p[5])))
                ey.append(complex(float(p[6]), float(p[7])))
                ez.append(complex(float(p[8]), float(p[9])))
            except ValueError:
                continue
    return xs, ex, ey, ez


def phase_slope(x_nm: list[float], e: list[complex]) -> float:
    import numpy as np

    x = np.asarray(x_nm, dtype=float)
    ph = np.unwrap(np.angle(np.asarray(e, dtype=complex)))
    if x.size < 2:
        return float("nan")
    return float(np.polyfit(x, ph, 1)[0])


def fft_peak_kx(x_nm: list[float], e: list[complex], demod_sign: str, kx_inc: float) -> float:
    import numpy as np

    x = np.asarray(x_nm, dtype=float)
    f = np.asarray(e, dtype=complex)
    if demod_sign == "minus":
        f = f * np.exp(-1j * kx_inc * x)
    elif demod_sign == "plus":
        f = f * np.exp(1j * kx_inc * x)
    nx = f.size
    dx = float((x.max() - x.min()) / max(nx - 1, 1))
    spec = np.fft.fftshift(np.fft.fft(f))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    power = np.abs(spec) ** 2
    return float(kx[int(np.argmax(power))])


def main() -> None:
    ap = argparse.ArgumentParser(description="Reflected near-field carrier diagnostic")
    ap.add_argument("reflected_tsv", type=Path)
    ap.add_argument("--wl", type=float, default=None)
    ap.add_argument("--angle", type=float, default=None)
    ap.add_argument("--period", type=float, default=80.0)
    ap.add_argument("--n-inc-real", type=float, default=1.0)
    args = ap.parse_args()

    path = args.reflected_tsv
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(2)

    meta = parse_meta(path)
    wl = args.wl if args.wl is not None else meta.get("wl", 13.5)
    ang = args.angle if args.angle is not None else meta.get("angle", 0.0)
    period = meta.get("period", args.period)
    k0 = 2.0 * math.pi / wl
    kx_inc = meta.get("kx_inc", k0 * args.n_inc_real * math.sin(math.radians(ang)))
    k_grid = 2.0 * math.pi / period

    xs, ex, ey, ez = load_tsv(path)
    if not xs:
        print("No TSV rows.", file=sys.stderr)
        sys.exit(2)

    print(f"file: {path}")
    print(f"wl={wl:g} nm  angle={ang:g} deg  period={period:g} nm")
    print(f"kx_inc (demod) = {kx_inc:.6g} rad/nm   2π/Λ = {k_grid:.6g} rad/nm")
    if "kx0_norm" in meta:
        print(f"kx0_norm (header) = {meta['kx0_norm']:.6g}")

    for name, fld in (("Ex", ex), ("Ey", ey), ("Ez", ez)):
        slope = phase_slope(xs, fld)
        print(f"  dφ/dx ({name}) = {slope:.6g} rad/nm")

    for comp_name, fld in (("Ey", ey),):
        for dem in ("none", "minus", "plus"):
            pk = fft_peak_kx(xs, fld, dem, kx_inc)
            print(f"  FFT peak kx ({comp_name}, demod={dem:5s}) = {pk:.6g} rad/nm")


if __name__ == "__main__":
    main()
