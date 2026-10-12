#!/usr/bin/env python3
"""Zeroth-order R(λ) vs CD / depth / SWA at φ=0° and 90° (Corazza Fig. 1 analog)."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from config import GridAxisConfig, ScatterometryConfig, load_config
from forward_model import reset_runner, simulate_reflectivity
from recipe import expand_harmonic_orders

HERE = Path(__file__).resolve().parent
KIND_DEPTH, KIND_CD, KIND_SWA = 0, 1, 2
KIND_NAME = {KIND_DEPTH: "depth", KIND_CD: "cd", KIND_SWA: "swa"}

_worker_cfg: ScatterometryConfig | None = None


def _jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def axis_from_spec(raw: dict | None, default: GridAxisConfig) -> GridAxisConfig:
    if not raw:
        return default
    return GridAxisConfig(float(raw["min"]), float(raw["max"]), float(raw["step"]))


def reflection_height_period_nm(wl_nm: float, angle_deg: float) -> float:
    c = math.cos(math.radians(float(angle_deg)))
    if abs(c) < 1e-12:
        return float("inf")
    return float(wl_nm) / (2.0 * abs(c))


@dataclass(frozen=True)
class Phi90Spec:
    azimuths: np.ndarray
    wavelengths: np.ndarray
    cd: np.ndarray
    depth: np.ndarray
    swa: np.ndarray
    workers: int
    checkpoint_every: int
    two_beam_top: str
    two_beam_floor: str
    truth_cd: float
    truth_depth: float
    truth_swa: float
    pitch: float
    angle_deg: float
    out_dir: Path
    dpi: int


def parse_phi90(cfg: ScatterometryConfig, raw: dict) -> Phi90Spec:
    block = dict(raw.get("phi90_cd") or {})
    wl = axis_from_spec(block.get("wl_nm"), GridAxisConfig(12.0, 16.0, 0.2))
    cd = axis_from_spec(block.get("cd_nm"), GridAxisConfig(32.0, 48.0, 2.0))
    depth = axis_from_spec(block.get("depth_nm"), GridAxisConfig(24.0, 56.0, 2.0))
    swa = axis_from_spec(block.get("swa_deg"), GridAxisConfig(85.0, 94.0, 0.5))
    az = np.asarray(block.get("azimuths_deg") or [0.0, 90.0], dtype=float)
    two = dict(block.get("two_beam") or {})
    out = Path(cfg.paths.output_dir)
    if not out.is_absolute():
        base = (cfg.config_path or HERE / "config.yaml").parent
        out = (base / out).resolve()
    return Phi90Spec(
        azimuths=az,
        wavelengths=wl.points(),
        cd=cd.points(),
        depth=depth.points(),
        swa=swa.points(),
        workers=int(block.get("workers") or 8),
        checkpoint_every=int(block.get("checkpoint_every") or 80),
        two_beam_top=str(two.get("top") or cfg.structure.grating_material),
        two_beam_floor=str(two.get("floor") or cfg.structure.substrate_material),
        truth_cd=float(cfg.structure.cd_nm),
        truth_depth=float(cfg.structure.depth_nm),
        truth_swa=float(cfg.structure.swa_deg),
        pitch=float(cfg.structure.pitch_nm),
        angle_deg=float(cfg.optical.angle_deg),
        out_dir=out,
        dpi=int(getattr(cfg.eval, "plot_dpi", 150) or 150),
    )


def n_s4_calls(spec: Phi90Spec) -> int:
    n_y = int(spec.cd.size + spec.depth.size + spec.swa.size)
    return int(spec.azimuths.size * spec.wavelengths.size * n_y)


def harmonic_markers(cfg: ScatterometryConfig) -> list[tuple[float, str]]:
    rec = cfg.optical.recipe
    if rec is None:
        return []
    fund = float(rec.fundamental_nm)
    out: list[tuple[float, str]] = []
    for q in expand_harmonic_orders(rec):
        out.append((fund / float(q), f"H{q}"))
    return out


def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(axis, dtype=float) - float(value))))


def fresnel_rs(eps: complex, angle_deg: float, n1: float = 1.0) -> complex:
    """Vacuum → medium s-pol amplitude reflectance."""
    n2 = np.sqrt(complex(eps))
    if n2.real < 0:
        n2 = -n2
    theta = math.radians(float(angle_deg))
    sin_t = (n1 / n2) * math.sin(theta)
    cos_i = math.cos(theta)
    cos_t = np.sqrt(1.0 - sin_t * sin_t)
    if (n2 * cos_t).real < 0:
        cos_t = -cos_t
    num = n1 * cos_i - n2 * cos_t
    den = n1 * cos_i + n2 * cos_t
    if abs(den) < 1e-18:
        return complex(1.0, 0.0)
    return complex(num / den)


def two_beam_R0(
    wl_nm: float,
    depth_nm: float,
    fill: float,
    angle_deg: float,
    r_top: complex,
    r_floor: complex,
) -> float:
    """Mesa + trench-floor interference; phase independent of fill in (0, 1)."""
    f = float(fill)
    psi = 4.0 * math.pi * float(depth_nm) * math.cos(math.radians(float(angle_deg))) / float(wl_nm)
    amp = f * complex(r_top) + (1.0 - f) * complex(r_floor) * np.exp(1j * psi)
    return float(abs(amp) ** 2)


_WL_ROW = re.compile(
    r"wl\s*=\s*([0-9.eE+-]+)\s*,\s*eps_r\s*=\s*([0-9.eE+-]+)\s*,\s*eps_i\s*=\s*([0-9.eE+-]+)"
)
_ASSIGN = re.compile(
    r'materials\["([^"]+)"\]\s*=\s*materials\["\1"\]\s*or\s*\{'
)


def _material_block(text: str, name: str) -> str:
    matches = list(_ASSIGN.finditer(text))
    for i, m in enumerate(matches):
        if m.group(1) != name:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end() : end]
        extra = re.findall(
            rf'table\.insert\(materials\["{re.escape(name)}"\],\s*\{{([^}}]+)\}}',
            text,
        )
        if extra:
            block += "\n" + "\n".join("{" + e + "}" for e in extra)
        return block
    raise KeyError(f"material {name!r} not in table")


def load_material_eps(lua_path: Path, name: str, wl_nm: float) -> complex:
    text = Path(lua_path).read_text(encoding="utf-8")
    block = _material_block(text, name)
    rows = [(float(a), float(b), float(c)) for a, b, c in _WL_ROW.findall(block)]
    if not rows:
        raise KeyError(f"no ε rows for {name!r} in {lua_path}")
    wls = np.array([r[0] for r in rows], dtype=float)
    i = int(np.argmin(np.abs(wls - float(wl_nm))))
    return complex(rows[i][1], rows[i][2])


def materials_lua(cfg: ScatterometryConfig) -> Path:
    db = Path(cfg.paths.materials_db).expanduser()
    if not db.is_absolute():
        base = (cfg.config_path or HERE / "config.yaml").parent
        db = (base / db).resolve()
    return db / "materials.lua"


def rms_partial(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """rms(∂R/∂y) over (y, λ) for each azimuth. R shape (n_az, n_y, n_wl)."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return np.full(R.shape[0], np.nan)
    dR = np.gradient(np.asarray(R, dtype=float), y, axis=1)
    return np.sqrt(np.nanmean(dR**2, axis=(1, 2)))


def sensitivity_table(spec: Phi90Spec, R_depth: np.ndarray, R_cd: np.ndarray, R_swa: np.ndarray) -> dict:
    d_depth = rms_partial(R_depth, spec.depth)
    d_cd = rms_partial(R_cd, spec.cd)
    d_swa = rms_partial(R_swa, spec.swa)
    out: dict[str, dict] = {}
    for i, az in enumerate(spec.azimuths):
        depth = float(d_depth[i])
        cd = float(d_cd[i])
        swa = float(d_swa[i])
        out[f"{az:g}"] = {
            "rms_dR_ddepth": depth,
            "rms_dR_dCD": cd,
            "rms_dR_dSWA": swa,
            "cd_over_depth": (cd / depth) if depth > 0 else None,
        }
    return out


def _init_pool_worker(config_path: str) -> None:
    global _worker_cfg
    _worker_cfg = load_config(config_path)


def _apply_trial(cfg: ScatterometryConfig, cd: float, depth: float, swa: float, wl: float, az: float):
    trial = cfg.copy()
    trial.structure.cd_nm = float(cd)
    trial.structure.depth_nm = float(depth)
    trial.structure.swa_deg = float(swa)
    trial.optical.recipe = None
    trial.optical.wl_nm = float(wl)
    trial.optical.azimuth_deg = float(az)
    return trial


def _compute_point(task: tuple) -> tuple[int, int, int, int, float]:
    kind, i_az, i_y, i_wl, cd, depth, swa, wl, az = task
    reset_runner()
    assert _worker_cfg is not None
    trial = _apply_trial(_worker_cfg, cd, depth, swa, wl, az)
    _, r_map = simulate_reflectivity(trial)
    return int(kind), int(i_az), int(i_y), int(i_wl), float(r_map.get(0, 0.0))


def _empty_arrays(spec: Phi90Spec):
    naz, nwl = spec.azimuths.size, spec.wavelengths.size
    R_depth = np.full((naz, spec.depth.size, nwl), np.nan)
    R_cd = np.full((naz, spec.cd.size, nwl), np.nan)
    R_swa = np.full((naz, spec.swa.size, nwl), np.nan)
    filled_depth = np.zeros(R_depth.shape, dtype=bool)
    filled_cd = np.zeros(R_cd.shape, dtype=bool)
    filled_swa = np.zeros(R_swa.shape, dtype=bool)
    return R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa


def _npz_path(spec: Phi90Spec) -> Path:
    return spec.out_dir / "phi90_cd_sensitivity.npz"


def _json_path(spec: Phi90Spec) -> Path:
    return spec.out_dir / "phi90_cd_sensitivity.json"


def save_npz(
    path: Path,
    spec: Phi90Spec,
    R_depth: np.ndarray,
    R_cd: np.ndarray,
    R_swa: np.ndarray,
    filled_depth: np.ndarray,
    filled_cd: np.ndarray,
    filled_swa: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        azimuths=spec.azimuths,
        wavelengths=spec.wavelengths,
        cd=spec.cd,
        depth=spec.depth,
        swa=spec.swa,
        R_depth=R_depth,
        R_cd=R_cd,
        R_swa=R_swa,
        filled_depth=filled_depth,
        filled_cd=filled_cd,
        filled_swa=filled_swa,
        truth_cd=spec.truth_cd,
        truth_depth=spec.truth_depth,
        truth_swa=spec.truth_swa,
        pitch=spec.pitch,
        angle_deg=spec.angle_deg,
    )


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _arrays_for_kind(kind: int, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa):
    if kind == KIND_DEPTH:
        return R_depth, filled_depth
    if kind == KIND_CD:
        return R_cd, filled_cd
    return R_swa, filled_swa


def build_tasks(spec: Phi90Spec, filled_depth, filled_cd, filled_swa) -> list[tuple]:
    tasks: list[tuple] = []
    truth_cd, truth_depth, truth_swa = spec.truth_cd, spec.truth_depth, spec.truth_swa
    for i_az, az in enumerate(spec.azimuths):
        for i_wl, wl in enumerate(spec.wavelengths):
            for i_y, depth in enumerate(spec.depth):
                if not filled_depth[i_az, i_y, i_wl]:
                    tasks.append(
                        (KIND_DEPTH, i_az, i_y, i_wl, truth_cd, float(depth), truth_swa, float(wl), float(az))
                    )
            for i_y, cd in enumerate(spec.cd):
                if not filled_cd[i_az, i_y, i_wl]:
                    tasks.append(
                        (KIND_CD, i_az, i_y, i_wl, float(cd), truth_depth, truth_swa, float(wl), float(az))
                    )
            for i_y, swa in enumerate(spec.swa):
                if not filled_swa[i_az, i_y, i_wl]:
                    tasks.append(
                        (KIND_SWA, i_az, i_y, i_wl, truth_cd, truth_depth, float(swa), float(wl), float(az))
                    )
    return tasks


def _store(kind, i_az, i_y, i_wl, R0, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa) -> None:
    arr, filled = _arrays_for_kind(kind, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
    arr[i_az, i_y, i_wl] = R0
    filled[i_az, i_y, i_wl] = True


def _harmonic_vlines(ax, harmonics: list[tuple[float, str]], x0: float, x1: float) -> None:
    labeled = False
    for wl, name in harmonics:
        if wl < x0 or wl > x1:
            continue
        ax.axvline(wl, color="0.65", ls=":", lw=0.8, zorder=3)
        if not labeled:
            ax.axvline(wl, color="0.65", ls=":", lw=0.8, label="H55–61" if name.startswith("H") else name)
            labeled = True


def plot_corazza_analog(
    path: Path,
    spec: Phi90Spec,
    R_depth: np.ndarray,
    R_cd: np.ndarray,
    R_swa: np.ndarray,
    harmonics: list[tuple[float, str]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 11.4), sharex=True, constrained_layout=True)
    rows = [
        ("grating height (nm)", spec.depth, R_depth, spec.truth_depth),
        ("CD (nm)", spec.cd, R_cd, spec.truth_cd),
        ("SWA (deg)", spec.swa, R_swa, spec.truth_swa),
    ]
    wl = spec.wavelengths
    for i, (ylabel, y, R, truth) in enumerate(rows):
        finite = np.isfinite(R)
        vmin = float(np.nanmin(R)) if finite.any() else 0.0
        vmax = float(np.nanmax(R)) if finite.any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12
        for j, az in enumerate(spec.azimuths):
            ax = axes[i, j]
            im = ax.imshow(
                R[j],
                origin="lower",
                aspect="auto",
                extent=(float(wl[0]), float(wl[-1]), float(y[0]), float(y[-1])),
                cmap="afmhot",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.axhline(truth, color="white", ls="--", lw=0.7, alpha=0.85)
            _harmonic_vlines(ax, harmonics, float(wl[0]), float(wl[-1]))
            if i == 0:
                ax.set_title(rf"azimuth ${az:g}^\circ$")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 2:
                ax.set_xlabel("wavelength (nm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(r"zeroth-order $R_0(\lambda)$  (truth CD / GH / SWA dashed)", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=spec.dpi)
    plt.close(fig)


def plot_two_beam_depth90(
    path: Path,
    spec: Phi90Spec,
    R_depth: np.ndarray,
    harmonics: list[tuple[float, str]],
    lua_path: Path,
) -> None:
    i90 = nearest_index(spec.azimuths, 90.0)
    marks = [(wl, name) for wl, name in harmonics if spec.wavelengths[0] <= wl <= spec.wavelengths[-1]]
    if not marks:
        mid = spec.wavelengths[spec.wavelengths.size // 2]
        marks = [(float(mid), f"{mid:.2f} nm")]
    marks = marks[:4]
    fill = spec.truth_cd / spec.pitch
    n = len(marks)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    h_dense = np.linspace(float(spec.depth[0]), float(spec.depth[-1]), 400)
    for ax, (wl, name) in zip(axes, marks):
        i_wl = nearest_index(spec.wavelengths, wl)
        wl_use = float(spec.wavelengths[i_wl])
        r_top = fresnel_rs(load_material_eps(lua_path, spec.two_beam_top, wl_use), spec.angle_deg)
        r_floor = fresnel_rs(load_material_eps(lua_path, spec.two_beam_floor, wl_use), spec.angle_deg)
        tb = [
            two_beam_R0(wl_use, h, fill, spec.angle_deg, r_top, r_floor) for h in h_dense
        ]
        ax.plot(spec.depth, R_depth[i90, :, i_wl], "o", ms=3.5, color="C0", label="S4 m=0")
        ax.plot(h_dense, tb, color="C1", lw=1.2, label="two-beam")
        ax.axvline(spec.truth_depth, color="0.4", ls="--", lw=0.8)
        ax.set_title(f"{name}  ({wl_use:.3g} nm)")
        ax.set_xlabel("grating height (nm)")
        ax.set_xlim(float(spec.depth[0]), float(spec.depth[-1]))
    axes[0].set_ylabel(r"$R_0$  ($\varphi=90^\circ$)")
    axes[-1].legend(fontsize=8, loc="best")
    period = reflection_height_period_nm(float(np.mean(spec.wavelengths)), spec.angle_deg)
    fig.suptitle(
        rf"two-beam overlay at $\varphi=90^\circ$, $f$=CD/pitch={fill:.2f}; "
        rf"$\Delta h\sim\lambda/(2\cos\theta)\approx{period:.1f}\,\mathrm{{nm}}$",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=spec.dpi)
    plt.close(fig)


def summarize(
    cfg: ScatterometryConfig,
    spec: Phi90Spec,
    R_depth: np.ndarray,
    R_cd: np.ndarray,
    R_swa: np.ndarray,
    filled_depth: np.ndarray,
    filled_cd: np.ndarray,
    filled_swa: np.ndarray,
) -> dict:
    sens = sensitivity_table(spec, R_depth, R_cd, R_swa)
    harmonics = harmonic_markers(cfg)
    n_done = int(filled_depth.sum() + filled_cd.sum() + filled_swa.sum())
    report = {
        "n_s4": n_s4_calls(spec),
        "n_done": n_done,
        "azimuths_deg": spec.azimuths.tolist(),
        "wavelengths_nm": spec.wavelengths.tolist(),
        "cd_nm": spec.cd.tolist(),
        "depth_nm": spec.depth.tolist(),
        "swa_deg": spec.swa.tolist(),
        "truth": {"cd_nm": spec.truth_cd, "depth_nm": spec.truth_depth, "swa_deg": spec.truth_swa},
        "pitch_nm": spec.pitch,
        "angle_deg": spec.angle_deg,
        "fill_factor": spec.truth_cd / spec.pitch,
        "harmonics": [{"wl_nm": w, "label": n} for w, n in harmonics],
        "height_fringe_nm": {
            "at_min_wl": reflection_height_period_nm(float(spec.wavelengths[0]), spec.angle_deg),
            "at_max_wl": reflection_height_period_nm(float(spec.wavelengths[-1]), spec.angle_deg),
        },
        "sensitivity": sens,
        "note": (
            "φ=90° CD row should be flatter than φ=0° (fringe phase set by height, "
            "fill only sets contrast). SWA row is extra vs Corazza Fig. 1."
        ),
    }
    path = _json_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    analog = spec.out_dir / "fig_phi90_cd_sensitivity.png"
    plot_corazza_analog(analog, spec, R_depth, R_cd, R_swa, harmonics)
    lua = materials_lua(cfg)
    two_path = spec.out_dir / "fig_two_beam_depth90.png"
    if lua.is_file() and 90.0 in [float(a) for a in spec.azimuths]:
        plot_two_beam_depth90(two_path, spec, R_depth, harmonics, lua)
    print(f"Wrote {path}")
    print(f"Wrote {analog}")
    if two_path.is_file():
        print(f"Wrote {two_path}")
    print("sensitivity rms(∂R₀/∂p):")
    for az, row in sens.items():
        ratio = row["cd_over_depth"]
        ratio_s = f"{ratio:.3g}" if ratio is not None else "n/a"
        print(
            f"  φ={az}°  CD={row['rms_dR_dCD']:.3g}  "
            f"depth={row['rms_dR_ddepth']:.3g}  SWA={row['rms_dR_dSWA']:.3g}  "
            f"CD/depth={ratio_s}"
        )
    return report


def spec_from_npz(cfg: ScatterometryConfig, raw: dict, data: dict, out_dir: Path) -> Phi90Spec:
    spec = parse_phi90(cfg, raw)
    return Phi90Spec(
        azimuths=np.asarray(data["azimuths"], dtype=float),
        wavelengths=np.asarray(data["wavelengths"], dtype=float),
        cd=np.asarray(data["cd"], dtype=float),
        depth=np.asarray(data["depth"], dtype=float),
        swa=np.asarray(data["swa"], dtype=float),
        workers=spec.workers,
        checkpoint_every=spec.checkpoint_every,
        two_beam_top=spec.two_beam_top,
        two_beam_floor=spec.two_beam_floor,
        truth_cd=float(data.get("truth_cd", spec.truth_cd)),
        truth_depth=float(data.get("truth_depth", spec.truth_depth)),
        truth_swa=float(data.get("truth_swa", spec.truth_swa)),
        pitch=float(data.get("pitch", spec.pitch)),
        angle_deg=float(data.get("angle_deg", spec.angle_deg)),
        out_dir=out_dir,
        dpi=spec.dpi,
    )


def replot(npz_path: Path, config_path: Path | None = None) -> dict:
    npz_path = Path(npz_path).expanduser().resolve()
    if npz_path.is_dir():
        npz_path = npz_path / "phi90_cd_sensitivity.npz"
    data = load_npz(npz_path)
    cfg_path = config_path or HERE / "config_phi90_cd_sensitivity_p80.yaml"
    cfg = load_config(cfg_path)
    raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    spec = spec_from_npz(cfg, raw, data, npz_path.parent)
    return summarize(
        cfg,
        spec,
        data["R_depth"],
        data["R_cd"],
        data["R_swa"],
        data["filled_depth"],
        data["filled_cd"],
        data["filled_swa"],
    )


def run_scan(
    cfg: ScatterometryConfig,
    spec: Phi90Spec,
    config_path: Path,
    workers: int,
    resume: bool,
) -> dict:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = _npz_path(spec)
    R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa = _empty_arrays(spec)
    if resume and npz_path.is_file():
        prev = load_npz(npz_path)
        same = (
            np.allclose(prev["azimuths"], spec.azimuths)
            and np.allclose(prev["wavelengths"], spec.wavelengths)
            and np.allclose(prev["cd"], spec.cd)
            and np.allclose(prev["depth"], spec.depth)
            and np.allclose(prev["swa"], spec.swa)
        )
        if same:
            R_depth, R_cd, R_swa = prev["R_depth"], prev["R_cd"], prev["R_swa"]
            filled_depth, filled_cd, filled_swa = (
                prev["filled_depth"],
                prev["filled_cd"],
                prev["filled_swa"],
            )
            n_fill = int(filled_depth.sum() + filled_cd.sum() + filled_swa.sum())
            print(f"resume: {n_fill}/{n_s4_calls(spec)} already filled")
        else:
            print("resume file does not match this grid; starting over")

    tasks = build_tasks(spec, filled_depth, filled_cd, filled_swa)
    t0 = time.perf_counter()
    if tasks and workers <= 1:
        _init_pool_worker(str(config_path.resolve()))
        for k, task in enumerate(tasks, start=1):
            kind, i_az, i_y, i_wl, R0 = _compute_point(task)
            _store(kind, i_az, i_y, i_wl, R0, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
            if spec.checkpoint_every and (k % spec.checkpoint_every == 0 or k == len(tasks)):
                save_npz(npz_path, spec, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
                rate = k / max(time.perf_counter() - t0, 1e-9)
                n_fill = int(filled_depth.sum() + filled_cd.sum() + filled_swa.sum())
                print(f"[{n_fill}/{n_s4_calls(spec)}] {rate:.3f} pts/s", flush=True)
    elif tasks:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_pool_worker,
            initargs=(str(config_path.resolve()),),
        ) as pool:
            futures = [pool.submit(_compute_point, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                kind, i_az, i_y, i_wl, R0 = fut.result()
                _store(kind, i_az, i_y, i_wl, R0, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
                done += 1
                if spec.checkpoint_every and (done % spec.checkpoint_every == 0 or done == len(tasks)):
                    save_npz(npz_path, spec, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
                    rate = done / max(time.perf_counter() - t0, 1e-9)
                    n_fill = int(filled_depth.sum() + filled_cd.sum() + filled_swa.sum())
                    print(f"[{n_fill}/{n_s4_calls(spec)}] {rate:.3f} pts/s", flush=True)

    save_npz(npz_path, spec, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)
    print(f"Saved {npz_path}")
    return summarize(cfg, spec, R_depth, R_cd, R_swa, filled_depth, filled_cd, filled_swa)


def dry_run_text(cfg: ScatterometryConfig, spec: Phi90Spec, workers: int) -> str:
    n = n_s4_calls(spec)
    harmonics = harmonic_markers(cfg)
    hlab = ", ".join(f"{n}@{w:.3g}nm" for w, n in harmonics) or "(none)"
    pmin = reflection_height_period_nm(float(spec.wavelengths[0]), spec.angle_deg)
    pmax = reflection_height_period_nm(float(spec.wavelengths[-1]), spec.angle_deg)
    return (
        f"dry-run φ={[float(a) for a in spec.azimuths]}  λ {spec.wavelengths[0]:g}…{spec.wavelengths[-1]:g} "
        f"({spec.wavelengths.size})\n"
        f"  CD {spec.cd[0]:g}…{spec.cd[-1]:g} n={spec.cd.size}  "
        f"depth {spec.depth[0]:g}…{spec.depth[-1]:g} n={spec.depth.size}  "
        f"SWA {spec.swa[0]:g}…{spec.swa[-1]:g} n={spec.swa.size}\n"
        f"  S4={n}  workers={workers}  out={spec.out_dir}\n"
        f"  harmonics {hlab}\n"
        f"  height fringe λ/(2 cos θ) = {pmin:.3g}–{pmax:.3g} nm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zeroth-order R(λ) vs CD / depth / SWA at φ=0° and 90°"
    )
    parser.add_argument(
        "--config",
        default=str(HERE / "config_phi90_cd_sensitivity_p80.yaml"),
        help="config_phi90_cd_sensitivity_p80.yaml",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="rebuild PNGs from saved phi90_cd_sensitivity.npz (no S4)",
    )
    args = parser.parse_args()
    if args.replot is not None:
        replot(args.replot, Path(args.config))
        return
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    spec = parse_phi90(cfg, raw)
    workers = int(args.workers or spec.workers)
    if args.dry_run:
        print(dry_run_text(cfg, spec, workers))
        return
    run_scan(cfg, spec, config_path=cfg_path, workers=workers, resume=not args.no_resume)


if __name__ == "__main__":
    main()
