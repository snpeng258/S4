"""Inverse-loss landscape on a 2-D structure slice (Gross 2009 Fig. 4 analog).

Z is the same data term the solver minimizes (GA ``_loss`` / LM data residual):
``sum((wsqrt * (R - R_meas))**2)``, with ``wsqrt`` from ``snr_weights`` or
``static_role_weights``. Forwards stay the full stored R_m vector; each FIM
mask is applied afterwards so prop / decoupling / m0_all / only90 share S4.

    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --dry-run
    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --workers 8
    python3 run_chi2_landscape.py --config config_chi2_p300.yaml --workers 4
    python3 run_chi2_landscape.py --config config_chi2_p300_d150.yaml --workers 4
    python3 run_chi2_landscape.py --config config_chi2_p80_wide_depth.yaml --workers 8
    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --slice cd_swa
    python3 run_chi2_landscape.py --replot ../runs/inverse/chi2_landscape
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from config import (
    GridAxisConfig,
    ScatterometryConfig,
    load_config,
    parse_config_arg,
    role_weights_enabled,
)
from decoupling import static_role_weights
from forward_model import generate_synthetic_measurement, reset_runner, simulate_reflectivity_multi
from order_collection import FIM_MASK_MODES, apply_collection, block_sizes
from recipe import expand_measurement_conditions
from residual_weights import snr_weights
from run_crlb_mc import apply_mask

HERE = Path(__file__).resolve().parent
SLICES = ("cd_depth", "cd_swa")
DEFAULT_MASKS = ("prop", "decoupling", "m0_all", "only90")

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


@dataclass(frozen=True)
class LandscapeSpec:
    slice: str
    x_name: str
    y_name: str
    x: GridAxisConfig
    y: GridAxisConfig
    fixed: dict[str, float]
    noiseless: bool
    masks: tuple[str, ...]
    workers: int
    checkpoint_every: int


def _axis(raw: dict | None, name: str, default: GridAxisConfig) -> GridAxisConfig:
    if not raw or name not in raw:
        return default
    item = raw[name]
    if isinstance(item, GridAxisConfig):
        return item
    return GridAxisConfig(float(item["min"]), float(item["max"]), float(item["step"]))


def parse_landscape(cfg: ScatterometryConfig, raw: dict, slice_name: str | None) -> LandscapeSpec:
    land = dict(raw.get("landscape") or {})
    slice_name = (slice_name or land.get("slice") or "cd_depth").replace("-", "_")
    if slice_name not in SLICES:
        raise ValueError(f"slice must be one of {SLICES}, got {slice_name!r}")
    truth = {
        "cd_nm": float(cfg.structure.cd_nm),
        "depth_nm": float(cfg.structure.depth_nm),
        "swa_deg": float(cfg.structure.swa_deg),
    }
    if slice_name == "cd_depth":
        x_name, y_name, fix_name = "cd_nm", "depth_nm", "swa_deg"
        x = _axis(land, "cd_nm", GridAxisConfig(32.0, 48.0, 1.0))
        y = _axis(land, "depth_nm", GridAxisConfig(32.0, 48.0, 1.0))
    else:
        x_name, y_name, fix_name = "cd_nm", "swa_deg", "depth_nm"
        x = _axis(land, "cd_nm", GridAxisConfig(32.0, 48.0, 1.0))
        y = _axis(land, "swa_deg", GridAxisConfig(85.0, 94.0, 0.5))
    masks = tuple(str(m) for m in (land.get("masks") or DEFAULT_MASKS))
    unknown = [m for m in masks if m not in FIM_MASK_MODES]
    if unknown:
        raise ValueError(f"unknown landscape masks {unknown}; choose from {FIM_MASK_MODES}")
    return LandscapeSpec(
        slice=slice_name,
        x_name=x_name,
        y_name=y_name,
        x=x,
        y=y,
        fixed={fix_name: truth[fix_name]},
        noiseless=bool(land.get("noiseless", True)),
        masks=masks,
        workers=max(1, int(land.get("workers", cfg.library.build_workers or 1))),
        checkpoint_every=max(0, int(land.get("checkpoint_every", 50))),
    )


def load_landscape_config(path: Path) -> tuple[ScatterometryConfig, dict]:
    cfg = load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg, raw


def structure_grid(spec: LandscapeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = spec.x.points()
    ys = spec.y.points()
    params = []
    for y in ys:
        for x in xs:
            p = {spec.x_name: float(x), spec.y_name: float(y), **spec.fixed}
            params.append([p["cd_nm"], p["depth_nm"], p["swa_deg"]])
    return xs, ys, np.asarray(params, dtype=float)


def reflection_height_period_nm(wl_nm: float, angle_deg: float) -> float:
    """Path-length period for a 2π phase in specular reflection: λ / (2 cos θ)."""
    c = math.cos(math.radians(float(angle_deg)))
    if abs(c) < 1e-12:
        return float("inf")
    return float(wl_nm) / (2.0 * abs(c))


def height_fringe_report(cfg: ScatterometryConfig) -> dict:
    angle = float(cfg.optical.angle_deg)
    wls = sorted({round(float(c.wl_nm), 6) for c in expand_measurement_conditions(cfg)})
    periods = [reflection_height_period_nm(w, angle) for w in wls]
    truth_h = float(cfg.structure.depth_nm)
    expected: list[float] = []
    seen: set[float] = set()
    for p in periods:
        for sign in (-1.0, 1.0):
            h = truth_h + sign * p
            key = round(h, 3)
            if key not in seen:
                seen.add(key)
                expected.append(float(h))
    expected.sort()
    return {
        "formula": "lambda / (2 * cos(theta))",
        "angle_deg": angle,
        "wavelengths_nm": wls,
        "period_nm": periods,
        "period_nm_min": min(periods) if periods else None,
        "period_nm_max": max(periods) if periods else None,
        "truth_depth_nm": truth_h,
        "expected_second_depth_nm": expected,
    }


def local_minima_1d(z: np.ndarray) -> list[int]:
    """Interior 1-D local minima; plateaus with a strictly lower neighbor are kept."""
    z = np.asarray(z, dtype=float)
    out: list[int] = []
    for i in range(1, z.size - 1):
        if not np.isfinite(z[i]):
            continue
        lesser_or_eq = True
        strictly_less = False
        for nb in (z[i - 1], z[i + 1]):
            if not np.isfinite(nb):
                continue
            if z[i] > nb:
                lesser_or_eq = False
                break
            if z[i] < nb:
                strictly_less = True
        if lesser_or_eq and strictly_less:
            out.append(i)
    return out


def local_minima_2d(grid: np.ndarray) -> list[tuple[int, int]]:
    """8-connected strict local minima as (iy, ix). Flat plateaus are skipped."""
    z = np.asarray(grid, dtype=float)
    ny, nx = z.shape
    out: list[tuple[int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            v = z[iy, ix]
            if not np.isfinite(v):
                continue
            strictly_less = False
            is_min = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    jy, jx = iy + dy, ix + dx
                    if jy < 0 or jy >= ny or jx < 0 or jx >= nx:
                        continue
                    nb = z[jy, jx]
                    if not np.isfinite(nb):
                        continue
                    if v > nb:
                        is_min = False
                        break
                    if v < nb:
                        strictly_less = True
                if not is_min:
                    break
            if is_min and strictly_less:
                out.append((iy, ix))
    return out


def describe_local_minima(
    xs: np.ndarray,
    ys: np.ndarray,
    grid: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    truth: dict[str, float],
    fixed: dict[str, float] | None = None,
) -> list[dict]:
    finite = np.asarray(grid, dtype=float)
    finite = finite[np.isfinite(finite)]
    gmin = float(np.min(finite)) if finite.size else None
    recs: list[dict] = []
    extra_fixed = dict(fixed or {})
    for iy, ix in local_minima_2d(grid):
        loss = float(grid[iy, ix])
        rec: dict = {
            "loss": loss,
            x_name: float(xs[ix]),
            y_name: float(ys[iy]),
            **extra_fixed,
            "delta_to_truth": {
                x_name: float(xs[ix] - truth[x_name]),
                y_name: float(ys[iy] - truth[y_name]),
            },
        }
        if gmin is not None and gmin > 0:
            rec["loss_ratio"] = loss / gmin
        elif gmin == 0.0:
            rec["loss_ratio"] = 0.0 if loss == 0.0 else None
        recs.append(rec)
    recs.sort(key=lambda r: (float(r["loss"]), abs(float(r["delta_to_truth"][y_name]))))
    return recs


def apply_structure(cfg: ScatterometryConfig, cd: float, depth: float, swa: float) -> ScatterometryConfig:
    out = cfg.copy()
    out.structure.set_param("cd_nm", cd)
    out.structure.set_param("depth_nm", depth)
    out.structure.set_param("swa_deg", swa)
    return out


def inverse_objective_wsqrt(cfg: ScatterometryConfig, r_meas_full: np.ndarray) -> np.ndarray:
    """Same collectible sqrt-weights as InverseProblem (fixed on R_meas)."""
    r_m = apply_collection(r_meas_full, cfg)
    blocks = block_sizes(cfg)
    use_roles = role_weights_enabled(cfg) and cfg.inverse.order_collection.lower() == "decoupling"
    if use_roles:
        return static_role_weights(cfg, r_m, blocks)
    return snr_weights(r_m, cfg.inverse, cfg=cfg, block_sizes=blocks)


def inverse_objective_loss(
    r_full: np.ndarray,
    r_meas_full: np.ndarray,
    wsqrt: np.ndarray,
    cfg: ScatterometryConfig,
    *,
    params: np.ndarray | None = None,
    p_ref: np.ndarray | None = None,
) -> float:
    """GA ``_loss`` data term (+ optional ``reg_weight``). Not the unweighted resnorm."""
    r_c = apply_collection(np.asarray(r_full, dtype=float), cfg)
    r_m = apply_collection(np.asarray(r_meas_full, dtype=float), cfg)
    loss = float(np.sum((np.asarray(wsqrt, dtype=float) * (r_c - r_m)) ** 2))
    if (
        params is not None
        and p_ref is not None
        and cfg.inverse.reg_weight > 0
    ):
        loss += float(cfg.inverse.reg_weight * np.sum((np.asarray(params) / np.asarray(p_ref) - 1.0) ** 2))
    return loss


def mask_objectives(
    cfg: ScatterometryConfig,
    masks: tuple[str, ...],
    r_meas_full: np.ndarray,
) -> dict[str, tuple[ScatterometryConfig, np.ndarray]]:
    out: dict[str, tuple[ScatterometryConfig, np.ndarray]] = {}
    for name in masks:
        work = apply_mask(cfg, name)
        out[name] = (work, inverse_objective_wsqrt(work, r_meas_full))
    return out


def _init_pool_worker(config_path: str) -> None:
    global _worker_cfg
    _worker_cfg = load_config(config_path)
    reset_runner()


def _compute_point(task: tuple[int, float, float, float]) -> tuple[int, np.ndarray]:
    if _worker_cfg is None:
        raise RuntimeError("pool worker not initialized")
    idx, cd, depth, swa = task
    reset_runner()
    trial = apply_structure(_worker_cfg, cd, depth, swa)
    r_vec, _ = simulate_reflectivity_multi(trial, condition_workers=1)
    return idx, np.asarray(r_vec, dtype=float)


def landscape_out_dir(cfg: ScatterometryConfig, noiseless: bool) -> Path:
    """Noisy runs go to ``<dir>_noisy`` so they do not overwrite the clean surface."""
    root = (HERE / cfg.paths.output_dir).resolve()
    if noiseless or root.name.endswith("_noisy"):
        return root
    return root.with_name(root.name + "_noisy")


def _npz_path(out_dir: Path, spec: LandscapeSpec) -> Path:
    return out_dir / f"chi2_{spec.slice}.npz"


def _save_npz(
    path: Path,
    *,
    spec: LandscapeSpec,
    xs: np.ndarray,
    ys: np.ndarray,
    params: np.ndarray,
    R: np.ndarray,
    r_meas: np.ndarray,
    filled: np.ndarray,
) -> None:
    tmp = path.with_suffix(".npz.tmp")
    np.savez(
        tmp,
        x=xs,
        y=ys,
        params=params,
        R=R,
        r_meas=r_meas,
        filled=filled.astype(np.uint8),
        x_name=np.array(spec.x_name),
        y_name=np.array(spec.y_name),
        slice=np.array(spec.slice),
        param_names=np.array(["cd_nm", "depth_nm", "swa_deg"]),
    )
    tmp.replace(path)


def log_contour_levels(chi2: np.ndarray, n: int = 12) -> np.ndarray | None:
    """Log-spaced iso-loss levels. Zeros (noiseless truth) are skipped; they sit inside the first ring."""
    finite = np.asarray(chi2, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size < 2:
        return None
    vmax = float(np.percentile(finite, 99))
    zmin = float(np.min(finite))
    lo = max(zmin, vmax / 1e8)
    if not np.isfinite(vmax) or vmax <= 0 or lo >= vmax:
        return None
    levels = np.logspace(np.log10(lo), np.log10(vmax), n)
    levels = np.unique(np.round(levels, decimals=16))
    return levels if levels.size >= 2 else None


def log_png_path(out_path: Path) -> Path:
    if out_path.stem.endswith("_log"):
        return out_path
    return out_path.with_name(out_path.stem + "_log" + out_path.suffix)


def plot_landscape(
    xs: np.ndarray,
    ys: np.ndarray,
    chi2: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    truth_xy: tuple[float, float],
    title: str,
    out_path: Path,
    dpi: int,
    scale: str = "linear",
    extra_minima: list[tuple[float, float]] | None = None,
    expected_y: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    z = np.ma.masked_invalid(chi2)
    fig = plt.figure(figsize=(11.2, 4.6))
    ax0 = fig.add_subplot(1, 2, 1)
    if scale == "log":
        levels = log_contour_levels(chi2)
        if levels is None:
            plt.close(fig)
            return
        norm = LogNorm(vmin=float(levels[0]), vmax=float(levels[-1]))
        cs = ax0.contour(xs, ys, z, levels=levels, cmap="viridis", norm=norm)
        ax0.clabel(cs, inline=True, fontsize=7, fmt="%1.0e")
        ax0.set_title("iso-loss (log)")
    else:
        cs = ax0.contour(xs, ys, z, levels=24, cmap="viridis")
        ax0.clabel(cs, inline=True, fontsize=7, fmt="%1.2g")
        ax0.set_title("iso-loss")
    ax0.plot(truth_xy[0], truth_xy[1], "kx", ms=8, mew=1.5, label="truth")
    if np.any(np.isfinite(chi2)):
        iy, ix = np.unravel_index(int(np.nanargmin(chi2)), chi2.shape)
        ax0.plot(xs[ix], ys[iy], "o", ms=6, mfc="none", mec="#00bfbf", mew=1.5, label="min loss")
    extras = extra_minima or []
    drawn_extra = False
    for x_m, y_m in extras:
        if abs(x_m - truth_xy[0]) < 1e-9 and abs(y_m - truth_xy[1]) < 1e-9:
            continue
        ax0.plot(x_m, y_m, "s", ms=5, mfc="none", mec="#c51b8a", mew=1.2)
        drawn_extra = True
    if drawn_extra:
        ax0.plot([], [], "s", ms=5, mfc="none", mec="#c51b8a", label="local min")
    if y_name == "depth_nm":
        for yv in expected_y or []:
            ax0.axhline(yv, color="0.75", ls=":", lw=0.8, zorder=0)
    ax0.set_xlabel(x_name)
    ax0.set_ylabel(y_name)
    ax0.legend(fontsize=8, loc="best")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    xx, yy = np.meshgrid(xs, ys)
    finite = np.asarray(chi2, dtype=float)
    finite = finite[np.isfinite(finite)]
    if scale == "log":
        levels = log_contour_levels(chi2)
        lo = float(levels[0]) if levels is not None else 1e-16
        hi = float(levels[-1]) if levels is not None else 1.0
        z3 = np.log10(np.clip(np.where(np.isfinite(chi2), chi2, np.nan), lo, hi))
        surf = ax1.plot_surface(xx, yy, z3, cmap=cm.jet, linewidth=0, antialiased=True)
        ax1.set_zlabel(r"$\log_{10}$ inverse loss")
    else:
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmin = float(np.min(finite)) if finite.size else 0.0
        surf = ax1.plot_surface(
            xx, yy, np.clip(np.where(np.isfinite(chi2), chi2, np.nan), vmin, vmax),
            cmap=cm.jet, linewidth=0, antialiased=True, vmin=vmin, vmax=vmax,
        )
        ax1.set_zlabel("inverse loss")
    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.08)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_landscape_pair(
    xs: np.ndarray,
    ys: np.ndarray,
    chi2: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    truth_xy: tuple[float, float],
    title: str,
    out_path: Path,
    dpi: int,
    extra_minima: list[tuple[float, float]] | None = None,
    expected_y: list[float] | None = None,
) -> None:
    """Write the linear PNG and a sibling ``*_log.png``; do not overwrite names."""
    plot_landscape(
        xs, ys, chi2, x_name=x_name, y_name=y_name, truth_xy=truth_xy,
        title=title, out_path=out_path, dpi=dpi, scale="linear",
        extra_minima=extra_minima, expected_y=expected_y,
    )
    plot_landscape(
        xs, ys, chi2, x_name=x_name, y_name=y_name, truth_xy=truth_xy,
        title=f"{title}  (log)", out_path=log_png_path(out_path), dpi=dpi, scale="log",
        extra_minima=extra_minima, expected_y=expected_y,
    )


def plot_depth_cut(
    ys: np.ndarray,
    z: np.ndarray,
    *,
    truth_depth: float,
    expected_y: list[float] | None,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Loss vs depth at the truth CD. Zeros (noiseless truth) are omitted on the log axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    z_plot = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    ax.semilogy(ys, z_plot, color="#1d4ed8", lw=1.4)
    ax.axvline(truth_depth, color="k", ls="--", lw=1.0, label="truth")
    for yv in expected_y or []:
        ax.axvline(yv, color="0.6", ls=":", lw=1.0)
    if expected_y:
        ax.axvline(np.nan, color="0.6", ls=":", lw=1.0, label=r"$\lambda/(2\cos\theta)$")
    for i in local_minima_1d(z):
        if np.isfinite(z[i]) and z[i] > 0:
            ax.plot(ys[i], z[i], "s", ms=6, mfc="none", mec="#c51b8a", mew=1.3)
    ax.set_xlabel("depth_nm")
    ax.set_ylabel("inverse loss")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def replot_directory(out_dir: Path, *, dpi: int = 150) -> list[Path]:
    """Rebuild linear + log PNGs from saved ``chi2_*.json`` / ``.npy`` / ``.npz``."""
    written: list[Path] = []
    for meta_path in sorted(out_dir.glob("chi2_*.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        slice_name = str(payload.get("slice") or "")
        x_name = str(payload.get("x_name") or "cd_nm")
        y_name = str(payload.get("y_name") or "depth_nm")
        truth = payload.get("truth") or {}
        if slice_name not in SLICES or x_name not in truth or y_name not in truth:
            continue
        npz_path = out_dir / f"chi2_{slice_name}.npz"
        if not npz_path.is_file():
            continue
        npz = np.load(npz_path)
        xs = np.asarray(npz["x"], dtype=float)
        ys = np.asarray(npz["y"], dtype=float)
        truth_xy = (float(truth[x_name]), float(truth[y_name]))
        n_filled = int(payload.get("n_filled") or 0)
        fringe = payload.get("height_fringe") or {}
        expected_y = list(fringe.get("expected_second_depth_nm") or []) if y_name == "depth_nm" else []
        for mask in payload.get("masks") or {}:
            grid_path = out_dir / f"chi2_{slice_name}_{mask}.npy"
            if not grid_path.is_file():
                continue
            grid = np.asarray(np.load(grid_path), dtype=float)
            extra = [
                (float(m[x_name]), float(m[y_name]))
                for m in describe_local_minima(
                    xs, ys, grid,
                    x_name=x_name, y_name=y_name, truth=truth,
                    fixed=dict(payload.get("fixed") or {}),
                )
            ]
            out_png = out_dir / f"chi2_{slice_name}_{mask}.png"
            plot_landscape_pair(
                xs, ys, grid,
                x_name=x_name, y_name=y_name, truth_xy=truth_xy,
                title=f"{slice_name}  {mask}  inverse loss  n={n_filled}",
                out_path=out_png, dpi=dpi,
                extra_minima=extra, expected_y=expected_y,
            )
            written.append(out_png)
            written.append(log_png_path(out_png))
            if y_name == "depth_nm" and x_name == "cd_nm":
                ix_cd = int(np.argmin(np.abs(xs - float(truth[x_name]))))
                cut_png = out_dir / f"chi2_{slice_name}_{mask}_depth_cut.png"
                plot_depth_cut(
                    ys, grid[:, ix_cd],
                    truth_depth=float(truth[y_name]),
                    expected_y=expected_y,
                    title=f"{slice_name}  {mask}  depth cut @ CD={xs[ix_cd]:g} nm",
                    out_path=cut_png, dpi=dpi,
                )
                written.append(cut_png)
    return written


def replot_tree(root: Path, *, dpi: int = 150) -> list[Path]:
    root = root.expanduser().resolve()
    dirs: list[Path] = []
    if any(root.glob("chi2_*.json")):
        dirs.append(root)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and any(child.glob("chi2_*.json")):
                dirs.append(child)
    written: list[Path] = []
    for d in dirs:
        paths = replot_directory(d, dpi=dpi)
        print(f"replot {d}: {len(paths)} pngs")
        written.extend(paths)
    if not written:
        raise FileNotFoundError(f"no chi2_*.json landscapes under {root}")
    return written


def summarize(
    cfg: ScatterometryConfig,
    spec: LandscapeSpec,
    xs: np.ndarray,
    ys: np.ndarray,
    params: np.ndarray,
    R: np.ndarray,
    r_meas: np.ndarray,
    filled: np.ndarray,
    objectives: dict[str, tuple[ScatterometryConfig, np.ndarray]],
    out_dir: Path,
) -> dict:
    ny, nx = len(ys), len(xs)
    truth = {
        "cd_nm": float(cfg.structure.cd_nm),
        "depth_nm": float(cfg.structure.depth_nm),
        "swa_deg": float(cfg.structure.swa_deg),
    }
    p_ref = np.array([cfg.structure.get_param(n) for n in cfg.inverse.param_names], dtype=float)
    fringe = height_fringe_report(cfg)
    expected_y = list(fringe["expected_second_depth_nm"]) if spec.y_name == "depth_nm" else []
    payload = {
        "slice": spec.slice,
        "x_name": spec.x_name,
        "y_name": spec.y_name,
        "fixed": spec.fixed,
        "truth": truth,
        "noiseless": spec.noiseless,
        "z_axis": "inverse_loss",
        "height_fringe": fringe,
        "n_conditions": len(expand_measurement_conditions(cfg)),
        "n_points": int(params.shape[0]),
        "n_filled": int(np.count_nonzero(filled)),
        "masks": {},
    }
    do_plot = bool(cfg.eval.plot)
    dpi = int(cfg.eval.plot_dpi)
    ix_cd = int(np.argmin(np.abs(xs - truth["cd_nm"]))) if spec.x_name == "cd_nm" else None
    for name, (work, wsqrt) in objectives.items():
        grid = np.full((ny, nx), np.nan)
        for i, ok in enumerate(filled):
            if not ok:
                continue
            iy, ix = divmod(i, nx)
            grid[iy, ix] = inverse_objective_loss(
                R[i], r_meas, wsqrt, work, params=params[i], p_ref=p_ref
            )
        rec: dict = {
            "n_rows": int(wsqrt.size),
            "loss_min": None,
            "at": None,
            "sum_w": float(np.sum(np.asarray(wsqrt) ** 2)),
            "n_local_minima": 0,
            "local_minima": [],
        }
        extra_xy: list[tuple[float, float]] = []
        if np.any(np.isfinite(grid)):
            iy, ix = np.unravel_index(int(np.nanargmin(grid)), grid.shape)
            rec["loss_min"] = float(grid[iy, ix])
            rec["at"] = {spec.x_name: float(xs[ix]), spec.y_name: float(ys[iy]), **spec.fixed}
            rec["delta_to_truth"] = {
                spec.x_name: float(xs[ix] - truth[spec.x_name]),
                spec.y_name: float(ys[iy] - truth[spec.y_name]),
            }
            mins = describe_local_minima(
                xs, ys, grid,
                x_name=spec.x_name, y_name=spec.y_name, truth=truth, fixed=spec.fixed,
            )
            rec["local_minima"] = mins
            rec["n_local_minima"] = len(mins)
            extra_xy = [(float(m[spec.x_name]), float(m[spec.y_name])) for m in mins]
            if spec.y_name == "depth_nm" and ix_cd is not None:
                cut = grid[:, ix_cd]
                cut_recs = []
                for j in local_minima_1d(cut):
                    cut_recs.append(
                        {
                            "depth_nm": float(ys[j]),
                            "loss": float(cut[j]),
                            "delta_depth_nm": float(ys[j] - truth["depth_nm"]),
                        }
                    )
                rec["depth_cut"] = {
                    "cd_nm": float(xs[ix_cd]),
                    "n_minima": len(cut_recs),
                    "minima": cut_recs,
                }
        payload["masks"][name] = rec
        print(
            f"  {name:<12} n={rec['n_rows']:4d}  "
            f"min_loss={rec['loss_min'] if rec['loss_min'] is None else format(rec['loss_min'], '.4g')}  "
            f"at={rec['at']}  local_minima={rec['n_local_minima']}"
        )
        if rec["n_local_minima"] > 1:
            print(f"    second={rec['local_minima'][1]}")
        if do_plot and np.any(np.isfinite(grid)):
            plot_landscape_pair(
                xs,
                ys,
                grid,
                x_name=spec.x_name,
                y_name=spec.y_name,
                truth_xy=(truth[spec.x_name], truth[spec.y_name]),
                title=f"{spec.slice}  {name}  inverse loss  n={int(np.count_nonzero(filled))}",
                out_path=out_dir / f"chi2_{spec.slice}_{name}.png",
                dpi=dpi,
                extra_minima=extra_xy,
                expected_y=expected_y,
            )
            if spec.y_name == "depth_nm" and ix_cd is not None:
                plot_depth_cut(
                    ys,
                    grid[:, ix_cd],
                    truth_depth=truth["depth_nm"],
                    expected_y=expected_y,
                    title=f"{spec.slice}  {name}  depth cut @ CD={xs[ix_cd]:g} nm",
                    out_path=out_dir / f"chi2_{spec.slice}_{name}_depth_cut.png",
                    dpi=dpi,
                )
        np.save(out_dir / f"chi2_{spec.slice}_{name}.npy", grid)
    (out_dir / f"chi2_{spec.slice}.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    return payload


def run_landscape(
    cfg: ScatterometryConfig,
    spec: LandscapeSpec,
    *,
    config_path: Path,
    workers: int,
    resume: bool,
    seed: int,
    noiseless: bool,
) -> dict:
    out_dir = landscape_out_dir(cfg, noiseless)
    out_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, params = structure_grid(spec)
    n_pts = params.shape[0]
    n_cond = len(expand_measurement_conditions(cfg))
    print(
        f"landscape slice={spec.slice}  {spec.x_name}×{spec.y_name} = "
        f"{len(xs)}×{len(ys)} = {n_pts}  conditions={n_cond}  "
        f"S4≈{n_pts * n_cond}  workers={workers}  out={out_dir}"
    )
    print(f"  {spec.x_name}: {spec.x.min:g} → {spec.x.max:g} step {spec.x.step:g}")
    print(f"  {spec.y_name}: {spec.y.min:g} → {spec.y.max:g} step {spec.y.step:g}")
    print(f"  fixed: {spec.fixed}")
    fringe = height_fringe_report(cfg)
    print(
        f"  height fringe λ/(2 cos θ) = {fringe['period_nm_min']:.3g}–{fringe['period_nm_max']:.3g} nm; "
        f"expected second depths {fringe['expected_second_depth_nm']}"
    )

    reset_runner()
    rng = np.random.default_rng(seed)
    r_meas = generate_synthetic_measurement(cfg, rng=rng, noiseless=noiseless)

    n_full = int(r_meas.size)
    R = np.full((n_pts, n_full), np.nan)
    filled = np.zeros(n_pts, dtype=bool)
    npz_path = _npz_path(out_dir, spec)
    if resume and npz_path.is_file():
        prev = np.load(npz_path, allow_pickle=True)
        if prev["R"].shape == R.shape and np.array_equal(prev["params"], params):
            R = np.asarray(prev["R"], dtype=float)
            filled = np.asarray(prev["filled"], dtype=bool)
            r_meas = np.asarray(prev["r_meas"], dtype=float)
            print(f"resume: {int(np.count_nonzero(filled))}/{n_pts} already filled")
        else:
            print("resume file does not match this grid; starting over")

    todo = [i for i in range(n_pts) if not filled[i]]
    t0 = time.perf_counter()
    if todo and workers <= 1:
        for k, i in enumerate(todo, start=1):
            cd, depth, swa = (float(v) for v in params[i])
            reset_runner()
            trial = apply_structure(cfg, cd, depth, swa)
            r_vec, _ = simulate_reflectivity_multi(trial, condition_workers=1)
            R[i] = r_vec
            filled[i] = True
            if spec.checkpoint_every and (k % spec.checkpoint_every == 0 or k == len(todo)):
                _save_npz(
                    npz_path, spec=spec, xs=xs, ys=ys, params=params, R=R, r_meas=r_meas, filled=filled
                )
                elapsed = time.perf_counter() - t0
                rate = k / max(elapsed, 1e-9)
                print(f"[{int(np.count_nonzero(filled))}/{n_pts}] {rate:.3f} pts/s", flush=True)
    elif todo:
        tasks = [(i, float(params[i, 0]), float(params[i, 1]), float(params[i, 2])) for i in todo]
        done_batch = 0
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_pool_worker,
            initargs=(str(config_path.resolve()),),
        ) as pool:
            futures = [pool.submit(_compute_point, task) for task in tasks]
            for fut in as_completed(futures):
                idx, r_vec = fut.result()
                R[idx] = r_vec
                filled[idx] = True
                done_batch += 1
                if spec.checkpoint_every and (
                    done_batch % spec.checkpoint_every == 0 or done_batch == len(todo)
                ):
                    _save_npz(
                        npz_path, spec=spec, xs=xs, ys=ys, params=params, R=R, r_meas=r_meas, filled=filled
                    )
                    elapsed = time.perf_counter() - t0
                    rate = done_batch / max(elapsed, 1e-9)
                    print(f"[{int(np.count_nonzero(filled))}/{n_pts}] {rate:.3f} pts/s", flush=True)

    _save_npz(npz_path, spec=spec, xs=xs, ys=ys, params=params, R=R, r_meas=r_meas, filled=filled)
    print(f"Saved {npz_path}")
    objectives = mask_objectives(cfg, spec.masks, r_meas)
    return summarize(cfg, spec, xs, ys, params, R, r_meas, filled, objectives, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse-loss landscape on a CD–depth or CD–SWA slice")
    parser.add_argument(
        "--config",
        default=None,
        help="config_chi2_p80.yaml / config_chi2_p300.yaml / config_chi2_p300_d150.yaml / config_chi2_p80_wide_depth.yaml",
    )
    parser.add_argument("--slice", choices=SLICES, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="add detector noise to R_meas")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="rebuild linear + log PNGs from saved chi2_*.json/npy (no S4); file a run dir or chi2_landscape/",
    )
    args = parser.parse_args()

    if args.replot is not None:
        replot_tree(args.replot)
        return

    cfg_path = Path(args.config).expanduser() if args.config else parse_config_arg()
    cfg, raw = load_landscape_config(cfg_path)
    spec = parse_landscape(cfg, raw, args.slice)
    workers = int(args.workers or spec.workers)
    noiseless = spec.noiseless and not args.noisy
    xs, ys, params = structure_grid(spec)
    n_cond = len(expand_measurement_conditions(cfg))
    if args.dry_run:
        print(
            f"dry-run {spec.slice}: {len(xs)}×{len(ys)}={params.shape[0]} points, "
            f"{n_cond} conditions, S4≈{params.shape[0] * n_cond}, workers={workers}"
        )
        print(f"  x {spec.x_name} {xs[0]:g}…{xs[-1]:g}")
        print(f"  y {spec.y_name} {ys[0]:g}…{ys[-1]:g}")
        print(f"  fixed {spec.fixed}  noiseless={noiseless}  masks={list(spec.masks)}")
        print(f"  out={landscape_out_dir(cfg, noiseless)}")
        fringe = height_fringe_report(cfg)
        print(
            f"  height fringe λ/(2 cos θ) = {fringe['period_nm_min']:.3g}–{fringe['period_nm_max']:.3g} nm; "
            f"expected second depths {fringe['expected_second_depth_nm']}"
        )
        return
    run_landscape(
        cfg,
        spec,
        config_path=cfg_path,
        workers=workers,
        resume=not args.no_resume,
        seed=args.seed,
        noiseless=noiseless,
    )


if __name__ == "__main__":
    main()
