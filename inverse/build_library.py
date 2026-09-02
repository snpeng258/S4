"""Build spectral library: grid sweep + S4 forward model -> NPZ."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import ScatterometryConfig, load_config, parse_config_arg
from forward_model import reset_runner, simulate_reflectivity_multi
from recipe import expand_measurement_conditions
from spectrum_library import (
    SpectralLibrary,
    build_meta,
    count_grid_points,
    iter_grid_params,
    library_path,
    load_library,
    oracle_nearest_error,
    save_library,
    validate_library,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_EVERY = 500

_worker_cfg: ScatterometryConfig | None = None
_worker_names: list[str] | None = None
_worker_condition_workers: int = 1


def _apply_grid_point(cfg: ScatterometryConfig, names: list[str], params: np.ndarray) -> ScatterometryConfig:
    out = cfg.copy()
    for n, v in zip(names, params):
        out.structure.set_param(n, float(v))
    return out


def _init_pool_worker(config_path: str, condition_workers: int = 1) -> None:
    global _worker_cfg, _worker_names, _worker_condition_workers
    _worker_cfg = load_config(config_path)
    _worker_names = list(_worker_cfg.inverse.param_names)
    _worker_condition_workers = max(1, int(condition_workers))
    reset_runner()


def _compute_grid_point(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    if _worker_cfg is None or _worker_names is None:
        raise RuntimeError("pool worker not initialized")
    idx, params = task
    reset_runner()
    trial_cfg = _apply_grid_point(_worker_cfg, _worker_names, params)
    r_vec, _ = simulate_reflectivity_multi(
        trial_cfg,
        condition_workers=_worker_condition_workers,
    )
    return idx, np.asarray(params, dtype=float), np.asarray(r_vec, dtype=float)


def _resolve_workers(cfg: ScatterometryConfig, workers: int | None) -> int:
    n = workers if workers is not None else cfg.library.build_workers
    return max(1, int(n))


def _print_progress(done: int, total: int, start_idx: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    rate = (done - start_idx) / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print(f"[{done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s rate={rate:.3f} pts/s", flush=True)


def _build_serial(
    cfg: ScatterometryConfig,
    names: list[str],
    grid: dict,
    meta: dict,
    out_path: Path,
    *,
    total: int,
    start_idx: int,
    params_list: list[np.ndarray],
    r_list: list[np.ndarray],
    checkpoint_every: int,
    condition_workers: int,
) -> None:
    t0 = time.perf_counter()
    for i, p in enumerate(iter_grid_params(names, grid)):
        if i < start_idx:
            continue
        if i >= total:
            break
        reset_runner()
        trial_cfg = _apply_grid_point(cfg, names, p)
        r_vec, _ = simulate_reflectivity_multi(
            trial_cfg,
            condition_workers=condition_workers,
        )
        params_list.append(np.asarray(p, dtype=float))
        r_list.append(np.asarray(r_vec, dtype=float))
        done = i + 1
        if done % 50 == 0 or done == total:
            _print_progress(done, total, start_idx, t0)
        if checkpoint_every > 0 and done % checkpoint_every == 0:
            _save_partial(out_path, params_list, r_list, names, grid, meta)
            print(f"Checkpoint saved at {done} points", flush=True)


def _build_parallel(
    cfg: ScatterometryConfig,
    config_path: Path,
    names: list[str],
    grid: dict,
    meta: dict,
    out_path: Path,
    *,
    total: int,
    start_idx: int,
    workers: int,
    condition_workers: int,
    params_list: list[np.ndarray],
    r_list: list[np.ndarray],
    checkpoint_every: int,
) -> None:
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    next_idx = start_idx
    tasks = [
        (i, np.asarray(p, dtype=float))
        for i, p in enumerate(iter_grid_params(names, grid))
        if start_idx <= i < total
    ]
    if not tasks:
        return

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_pool_worker,
        initargs=(str(config_path.resolve()), condition_workers),
    ) as pool:
        futures = [pool.submit(_compute_grid_point, task) for task in tasks]
        for fut in as_completed(futures):
            idx, params, r_vec = fut.result()
            pending[idx] = (params, r_vec)
            while next_idx in pending:
                p, r = pending.pop(next_idx)
                params_list.append(p)
                r_list.append(r)
                next_idx += 1
                done = next_idx
                if done % 50 == 0 or done == total:
                    _print_progress(done, total, start_idx, t0)
                if checkpoint_every > 0 and done % checkpoint_every == 0:
                    _save_partial(out_path, params_list, r_list, names, grid, meta)
                    print(f"Checkpoint saved at {done} points", flush=True)


def build_library(
    cfg: ScatterometryConfig,
    *,
    max_points: int | None = None,
    resume: bool = False,
    checkpoint_every: int | None = None,
    workers: int | None = None,
    condition_workers: int | None = None,
    config_path: Path | None = None,
) -> SpectralLibrary:
    names = list(cfg.inverse.param_names)
    grid = {n: cfg.library.grid[n] for n in names}
    meta = build_meta(cfg)
    out_path = library_path(cfg)
    total = count_grid_points(grid)
    if max_points is not None:
        total = min(total, max_points)

    n_workers = _resolve_workers(cfg, workers)
    n_cond_workers = (
        int(condition_workers)
        if condition_workers is not None
        else max(1, int(cfg.library.condition_workers))
    )
    ckpt = (
        int(checkpoint_every)
        if checkpoint_every is not None
        else max(1, int(cfg.library.checkpoint_every))
    )
    cfg_path = config_path or cfg.config_path or (ROOT / "config.yaml")
    if cfg.config_path is None:
        cfg = cfg.copy()
        cfg.config_path = Path(cfg_path)

    params_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    start_idx = 0

    if resume and out_path.is_file():
        existing = load_library(out_path)
        validate_library(existing, cfg)
        params_list = [existing.params[i] for i in range(existing.size)]
        r_list = [existing.R[i] for i in range(existing.size)]
        start_idx = existing.size
        print(f"Resuming from {start_idx} / {total} points")

    n_cond = len(expand_measurement_conditions(cfg))
    print(
        f"Build workers: {n_workers}, condition_workers: {n_cond_workers}, "
        f"optical conditions: {n_cond}, checkpoint_every: {ckpt}"
    )

    if start_idx >= total:
        print("Already complete.")
    elif n_workers == 1:
        _build_serial(
            cfg, names, grid, meta, out_path,
            total=total, start_idx=start_idx,
            params_list=params_list, r_list=r_list,
            checkpoint_every=ckpt,
            condition_workers=n_cond_workers,
        )
    else:
        _build_parallel(
            cfg, cfg_path, names, grid, meta, out_path,
            total=total, start_idx=start_idx, workers=n_workers,
            condition_workers=n_cond_workers,
            params_list=params_list, r_list=r_list,
            checkpoint_every=ckpt,
        )

    lib = SpectralLibrary(
        params=np.asarray(params_list, dtype=float),
        R=np.asarray(r_list, dtype=float),
        param_names=names,
        grid=grid,
        meta=meta,
    )
    save_library(out_path, lib)
    print(f"Library saved: {out_path} ({lib.size} points)")
    return lib


def _save_partial(
    path: Path,
    params_list: list[np.ndarray],
    r_list: list[np.ndarray],
    names: list[str],
    grid,
    meta: dict,
) -> None:
    lib = SpectralLibrary(
        params=np.asarray(params_list, dtype=float),
        R=np.asarray(r_list, dtype=float),
        param_names=names,
        grid=grid,
        meta=meta,
    )
    save_library(path, lib)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build spectral library (grid -> S4 -> NPZ)")
    parser.add_argument("--config", type=Path, default=None, help="config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="print grid size only")
    parser.add_argument("--max-points", type=int, default=None, help="limit points (for testing)")
    parser.add_argument("--resume", action="store_true", help="resume from existing NPZ")
    parser.add_argument("--workers", type=int, default=None, help="override library.build_workers")
    parser.add_argument(
        "--condition-workers",
        type=int,
        default=None,
        help="override library.condition_workers (parallel S4 per grid point)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="override library.checkpoint_every",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve() if args.config else parse_config_arg([])
    cfg = load_config(cfg_path)
    names = list(cfg.inverse.param_names)
    grid = {n: cfg.library.grid[n] for n in names}
    n_grid = count_grid_points(grid)
    n_cond = len(expand_measurement_conditions(cfg))

    print(f"Config: {cfg_path}")
    print(f"Grid axes: { {n: len(grid[n].points()) for n in names} }")
    print(f"Total grid points: {n_grid}")
    print(f"Optical conditions: {n_cond}")
    print(f"S4 calls (est.): {n_grid * n_cond:,}")
    workers = _resolve_workers(cfg, args.workers)
    cond_w = args.condition_workers if args.condition_workers is not None else cfg.library.condition_workers
    print(f"Workers: {workers} (config build_workers={cfg.library.build_workers})")
    print(f"Condition workers: {cond_w} (config condition_workers={cfg.library.condition_workers})")
    if args.checkpoint_every is not None:
        print(f"Checkpoint every: {args.checkpoint_every}")
    if args.max_points:
        print(f"Limited to: {args.max_points}")

    if args.dry_run:
        return

    lib = build_library(
        cfg,
        max_points=args.max_points,
        resume=args.resume,
        workers=args.workers,
        condition_workers=args.condition_workers,
        checkpoint_every=args.checkpoint_every,
        config_path=cfg_path,
    )
    oracle = oracle_nearest_error(lib, cfg)
    print("Oracle nearest-grid error (true structure vs library):")
    for k, v in oracle.items():
        if k != "nearest_index":
            print(f"  {k}: {v:.4g}")
        else:
            print(f"  nearest_index: {int(v)}")


if __name__ == "__main__":
    main()
