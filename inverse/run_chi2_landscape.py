"""χ² landscape on a 2-D structure slice (Gross 2009 Fig. 4 analog).

Fixes one of {cd, depth, swa}, sweeps the other two against a single
measurement. Forwards are the full stored R_m vector; FIM masks are applied
afterwards so prop / decoupling / m0_all / only90 share the same S4 calls.

    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --dry-run
    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --workers 8
    python3 run_chi2_landscape.py --config config_chi2_p300.yaml --workers 4
    python3 run_chi2_landscape.py --config config_chi2_p80.yaml --slice cd_swa
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

from config import GridAxisConfig, ScatterometryConfig, load_config, parse_config_arg
from fim_study import full_observable_layout, mask_rows
from forward_model import generate_synthetic_measurement, reset_runner, simulate_reflectivity_multi
from noise_model import observation_variance
from order_collection import FIM_MASK_MODES, stored_order_m_vector
from recipe import expand_measurement_conditions

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


def apply_structure(cfg: ScatterometryConfig, cd: float, depth: float, swa: float) -> ScatterometryConfig:
    out = cfg.copy()
    out.structure.set_param("cd_nm", cd)
    out.structure.set_param("depth_nm", depth)
    out.structure.set_param("swa_deg", swa)
    return out


def _fim_windows(cfg: ScatterometryConfig) -> dict[str, list[float]]:
    return {k: list(v) for k, v in cfg.fim.azimuth_windows.items()}


def mask_selectors(cfg: ScatterometryConfig, masks: tuple[str, ...]) -> dict[str, np.ndarray]:
    rows = full_observable_layout(cfg)
    windows = _fim_windows(cfg)
    tol = float(cfg.fim.azimuth_tol_deg)
    keep = cfg.fim.keep_orders
    return {
        name: mask_rows(rows, name, windows=windows, tol=tol, keep_orders=keep)
        for name in masks
    }


def chi2_from_residual(r: np.ndarray, r_meas: np.ndarray, sigma: np.ndarray, sel: np.ndarray) -> float:
    d = (np.asarray(r, dtype=float)[sel] - np.asarray(r_meas, dtype=float)[sel]) / np.maximum(
        np.asarray(sigma, dtype=float)[sel], 1e-30
    )
    return float(np.dot(d, d))


def evaluate_masks(
    r: np.ndarray,
    r_meas: np.ndarray,
    sigma: np.ndarray,
    selectors: dict[str, np.ndarray],
) -> dict[str, float]:
    return {name: chi2_from_residual(r, r_meas, sigma, sel) for name, sel in selectors.items()}


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
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    z = np.ma.masked_invalid(chi2)
    fig = plt.figure(figsize=(11.2, 4.6))
    ax0 = fig.add_subplot(1, 2, 1)
    levels = 24
    cs = ax0.contour(xs, ys, z, levels=levels, cmap="viridis")
    ax0.clabel(cs, inline=True, fontsize=7, fmt="%1.2g")
    ax0.plot(truth_xy[0], truth_xy[1], "kx", ms=8, mew=1.5, label="truth")
    if np.any(np.isfinite(chi2)):
        iy, ix = np.unravel_index(int(np.nanargmin(chi2)), chi2.shape)
        ax0.plot(xs[ix], ys[iy], "o", ms=6, mfc="none", mec="#00bfbf", mew=1.5, label="min χ²")
    ax0.set_xlabel(x_name)
    ax0.set_ylabel(y_name)
    ax0.set_title("iso-χ²")
    ax0.legend(fontsize=8, loc="best")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    xx, yy = np.meshgrid(xs, ys)
    finite = np.asarray(chi2, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    vmin = float(np.min(finite)) if finite.size else 0.0
    surf = ax1.plot_surface(
        xx, yy, np.clip(np.where(np.isfinite(chi2), chi2, np.nan), vmin, vmax),
        cmap=cm.jet, linewidth=0, antialiased=True, vmin=vmin, vmax=vmax,
    )
    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    ax1.set_zlabel(r"$\chi^2$")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.08)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def summarize(
    cfg: ScatterometryConfig,
    spec: LandscapeSpec,
    xs: np.ndarray,
    ys: np.ndarray,
    params: np.ndarray,
    R: np.ndarray,
    r_meas: np.ndarray,
    filled: np.ndarray,
    selectors: dict[str, np.ndarray],
    sigma: np.ndarray,
    out_dir: Path,
) -> dict:
    ny, nx = len(ys), len(xs)
    truth = {
        "cd_nm": float(cfg.structure.cd_nm),
        "depth_nm": float(cfg.structure.depth_nm),
        "swa_deg": float(cfg.structure.swa_deg),
    }
    payload = {
        "slice": spec.slice,
        "x_name": spec.x_name,
        "y_name": spec.y_name,
        "fixed": spec.fixed,
        "truth": truth,
        "noiseless": spec.noiseless,
        "n_conditions": len(expand_measurement_conditions(cfg)),
        "n_points": int(params.shape[0]),
        "n_filled": int(np.count_nonzero(filled)),
        "masks": {},
    }
    do_plot = bool(cfg.eval.plot)
    dpi = int(cfg.eval.plot_dpi)
    for name, sel in selectors.items():
        grid = np.full((ny, nx), np.nan)
        for i, ok in enumerate(filled):
            if not ok:
                continue
            iy, ix = divmod(i, nx)
            grid[iy, ix] = chi2_from_residual(R[i], r_meas, sigma, sel)
        rec: dict = {"n_rows": int(np.count_nonzero(sel)), "chi2_min": None, "at": None}
        if np.any(np.isfinite(grid)):
            iy, ix = np.unravel_index(int(np.nanargmin(grid)), grid.shape)
            rec["chi2_min"] = float(grid[iy, ix])
            rec["at"] = {spec.x_name: float(xs[ix]), spec.y_name: float(ys[iy]), **spec.fixed}
            rec["delta_to_truth"] = {
                spec.x_name: float(xs[ix] - truth[spec.x_name]),
                spec.y_name: float(ys[iy] - truth[spec.y_name]),
            }
        payload["masks"][name] = rec
        print(
            f"  {name:<12} n={rec['n_rows']:4d}  "
            f"minχ²={rec['chi2_min'] if rec['chi2_min'] is None else format(rec['chi2_min'], '.4g')}  "
            f"at={rec['at']}"
        )
        if do_plot and np.any(np.isfinite(grid)):
            plot_landscape(
                xs,
                ys,
                grid,
                x_name=spec.x_name,
                y_name=spec.y_name,
                truth_xy=(truth[spec.x_name], truth[spec.y_name]),
                title=f"{spec.slice}  {name}  n={int(np.count_nonzero(filled))}",
                out_path=out_dir / f"chi2_{spec.slice}_{name}.png",
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
    out_dir = (HERE / cfg.paths.output_dir).resolve()
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

    reset_runner()
    rng = np.random.default_rng(seed)
    r_meas = generate_synthetic_measurement(cfg, rng=rng, noiseless=noiseless)
    order_m = stored_order_m_vector(cfg)
    sigma = np.sqrt(observation_variance(r_meas, cfg.inverse.noise, order_m))
    selectors = mask_selectors(cfg, spec.masks)

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
            sigma = np.sqrt(observation_variance(r_meas, cfg.inverse.noise, order_m))
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
    return summarize(cfg, spec, xs, ys, params, R, r_meas, filled, selectors, sigma, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="χ² landscape on a CD–depth or CD–SWA slice")
    parser.add_argument("--config", default=None, help="config_chi2_p80.yaml / config_chi2_p300.yaml")
    parser.add_argument("--slice", choices=SLICES, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="add detector noise to R_meas")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
