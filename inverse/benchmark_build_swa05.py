"""Benchmark SWA 0.5° library build: sweep build_workers × condition_workers."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from build_library import build_library
from config import load_config
from recipe import expand_measurement_conditions
from spectrum_library import count_grid_points, library_path

ROOT = Path(__file__).resolve().parent
TARGET_PTS_PER_SEC = 0.317
POINTS_PER_RUN = 80
BUILD_WORKERS = [8, 16, 24, 32]
CONDITION_WORKERS = [4, 8, 16]


def _estimate_days(remaining: int, pts_per_sec: float) -> float:
    if pts_per_sec <= 0:
        return math.inf
    return remaining / (pts_per_sec * 86400.0)


def run_benchmark(
    cfg_path: Path,
    *,
    max_points: int = POINTS_PER_RUN,
) -> dict:
    cfg = load_config(cfg_path)
    bench_cfg = cfg.copy()
    bench_cfg.library.file = "../data/inverse/_benchmark_swa05.npz"
    bench_path = library_path(bench_cfg)
    if bench_path.is_file():
        bench_path.unlink()

    n_cond = len(expand_measurement_conditions(bench_cfg))
    names = list(bench_cfg.inverse.param_names)
    grid = {n: bench_cfg.library.grid[n] for n in names}
    n_grid = count_grid_points(grid)
    remaining = n_grid - 4000  # partial checkpoint from legacy run

    records: list[dict] = []
    print(
        f"SWA05 benchmark: {n_cond} conditions, {n_grid} grid points, "
        f"{max_points} pts/run, target {TARGET_PTS_PER_SEC} pts/s"
    )
    print(f"CPU count: {os.cpu_count()}")

    for bw in BUILD_WORKERS:
        for cw in CONDITION_WORKERS:
            if bench_path.is_file():
                bench_path.unlink()
            t0 = time.perf_counter()
            build_library(
                bench_cfg,
                max_points=max_points,
                resume=False,
                checkpoint_every=0,
                workers=bw,
                condition_workers=cw,
                config_path=cfg_path,
            )
            elapsed = time.perf_counter() - t0
            pts_per_sec = max_points / max(elapsed, 1e-9)
            s4_concurrency = bw * cw
            rec = {
                "build_workers": bw,
                "condition_workers": cw,
                "s4_concurrency": s4_concurrency,
                "elapsed_s": elapsed,
                "points_per_sec": pts_per_sec,
                "s4_per_sec": pts_per_sec * n_cond,
                "eta_days_full_grid": _estimate_days(remaining, pts_per_sec),
                "meets_10d_target": pts_per_sec >= TARGET_PTS_PER_SEC,
            }
            records.append(rec)
            mark = "OK" if rec["meets_10d_target"] else "--"
            print(
                f"  bw={bw:2d} cw={cw:2d}  conc={s4_concurrency:3d}  "
                f"elapsed={elapsed:6.1f}s  pts/s={pts_per_sec:6.3f}  "
                f"eta={rec['eta_days_full_grid']:5.1f}d  [{mark}]"
            )

    best = max(records, key=lambda r: r["points_per_sec"])
    meets_target = best["points_per_sec"] >= TARGET_PTS_PER_SEC

    if bench_path.is_file():
        bench_path.unlink()

    return {
        "config": str(cfg_path.name),
        "n_conditions": n_cond,
        "n_grid": n_grid,
        "remaining_after_checkpoint": remaining,
        "points_per_run": max_points,
        "target_pts_per_sec": TARGET_PTS_PER_SEC,
        "cpu_count": os.cpu_count(),
        "build_workers_sweep": BUILD_WORKERS,
        "condition_workers_sweep": CONDITION_WORKERS,
        "records": records,
        "best": best,
        "meets_10d_target": meets_target,
        "recommendation": {
            "build_workers": best["build_workers"],
            "condition_workers": best["condition_workers"],
            "points_per_sec": best["points_per_sec"],
            "eta_days": best["eta_days_full_grid"],
            "action": "resume_full_grid" if meets_target else "fallback_coarse_grid",
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SWA05 library build throughput")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config_build_swa05.yaml",
        help="build config (default: config_build_swa05.yaml)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=POINTS_PER_RUN,
        help=f"grid points per worker combo (default: {POINTS_PER_RUN})",
    )
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run_benchmark(cfg_path, max_points=args.max_points)
    out_json = out_dir / "build_benchmark_swa05.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rec = payload["recommendation"]
    print("\n===== SWA05 BENCHMARK =====")
    print(
        f"Best: build_workers={rec['build_workers']}  "
        f"condition_workers={rec['condition_workers']}  "
        f"pts/s={rec['points_per_sec']:.3f}  eta≈{rec['eta_days']:.1f}d"
    )
    print(f"10d target ({TARGET_PTS_PER_SEC} pts/s): {'MET' if payload['meets_10d_target'] else 'NOT MET'}")
    print(f"Action: {rec['action']}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
