"""Benchmark parallel library build throughput and estimate 48h recipe feasibility."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from build_library import build_library
from config import ScatterometryConfig, load_config, parse_config_arg
from recipe import expand_measurement_conditions
from spectrum_library import count_grid_points, library_path

ROOT = Path(__file__).resolve().parent
BUDGET_HOURS = 48.0
OVERHEAD = 1.1
POINTS_PER_RUN = 40

SCHEMES = {
    "E": {
        "angles_deg": [60, 65, 69, 70, 71, 75, 80, 85],
        "azimuths_deg": [0, 30, 45, 60, 90],
    },
    "C": {
        "angles_deg": [60, 69, 70, 71, 85],
        "azimuths_deg": [0, 30, 45, 60, 90],
    },
    "A": {
        "angles_deg": [60, 69, 70, 71, 85],
        "azimuths_deg": [0, 45, 90],
    },
}


def _workers_sweep() -> list[int]:
    cpu = os.cpu_count() or 4
    cap = min(cpu, 32)
    sweep = [1]
    w = 2
    while w <= cap:
        sweep.append(w)
        w *= 2
    if cap not in sweep:
        sweep.append(cap)
    return sorted(set(sweep))


def _apply_scheme(cfg: ScatterometryConfig, scheme: str) -> ScatterometryConfig:
    if cfg.optical.recipe is None:
        raise ValueError("config must have optical.recipe for HHG benchmark")
    out = cfg.copy()
    spec = SCHEMES[scheme]
    out.optical.recipe.angles_deg = list(spec["angles_deg"])
    out.optical.recipe.azimuths_deg = list(spec["azimuths_deg"])
    return out


def _estimate_hours(
    n_grid: int,
    n_conditions: int,
    points_per_sec: float,
) -> float:
    if points_per_sec <= 0:
        return math.inf
    return n_grid * OVERHEAD / (points_per_sec * 3600.0)


def _pick_scheme(etas: dict[str, float]) -> str:
    for name in ("E", "C", "A"):
        if etas[name] <= BUDGET_HOURS:
            return name
    return "A"


def run_benchmark(
    cfg: ScatterometryConfig,
    *,
    config_path: Path,
    points_per_run: int = POINTS_PER_RUN,
    scheme_for_timing: str = "E",
) -> dict:
    bench_cfg = _apply_scheme(cfg, scheme_for_timing)
    bench_cfg.library.file = "../data/inverse/_benchmark_build.npz"
    bench_path = library_path(bench_cfg)
    if bench_path.is_file():
        bench_path.unlink()

    n_cond = len(expand_measurement_conditions(bench_cfg))
    names = list(bench_cfg.inverse.param_names)
    grid = {n: bench_cfg.library.grid[n] for n in names}
    n_grid = count_grid_points(grid)

    records: list[dict] = []
    baseline_pts_per_sec: float | None = None

    print(f"Benchmark scheme {scheme_for_timing}: {n_cond} conditions, {n_grid} grid points")
    print(f"Points per worker sweep: {points_per_run}")

    for workers in _workers_sweep():
        if bench_path.is_file():
            bench_path.unlink()
        t0 = time.perf_counter()
        build_library(
            bench_cfg,
            max_points=points_per_run,
            resume=False,
            checkpoint_every=0,
            workers=workers,
            config_path=config_path,
        )
        elapsed = time.perf_counter() - t0
        pts_per_sec = points_per_run / max(elapsed, 1e-9)
        s4_per_sec = pts_per_sec * n_cond
        if baseline_pts_per_sec is None:
            baseline_pts_per_sec = pts_per_sec
        efficiency = pts_per_sec / max(baseline_pts_per_sec, 1e-9) / workers
        rec = {
            "workers": workers,
            "elapsed_s": elapsed,
            "points_per_sec": pts_per_sec,
            "s4_per_sec": s4_per_sec,
            "parallel_efficiency": efficiency,
        }
        records.append(rec)
        print(
            f"  workers={workers:2d}  elapsed={elapsed:6.1f}s  "
            f"pts/s={pts_per_sec:5.3f}  s4/s={s4_per_sec:7.1f}  eff={efficiency:.2f}"
        )

    best = max(records, key=lambda r: r["points_per_sec"])
    workers_best = best["workers"]
    pts_best = best["points_per_sec"]

    scheme_etas: dict[str, float] = {}
    scheme_details: dict[str, dict] = {}
    for name, spec in SCHEMES.items():
        trial = _apply_scheme(cfg, name)
        nc = len(expand_measurement_conditions(trial))
        eta_h = _estimate_hours(n_grid, nc, pts_best)
        scheme_etas[name] = eta_h
        scheme_details[name] = {
            "angles_deg": spec["angles_deg"],
            "azimuths_deg": spec["azimuths_deg"],
            "n_conditions": nc,
            "s4_calls": n_grid * nc,
            "eta_hours": eta_h,
            "fits_48h": eta_h <= BUDGET_HOURS,
        }

    chosen = _pick_scheme(scheme_etas)

    if bench_path.is_file():
        bench_path.unlink()

    return {
        "benchmark_scheme": scheme_for_timing,
        "n_grid": n_grid,
        "points_per_run": points_per_run,
        "budget_hours": BUDGET_HOURS,
        "overhead_factor": OVERHEAD,
        "cpu_count": os.cpu_count(),
        "workers_sweep": _workers_sweep(),
        "records": records,
        "workers_best": workers_best,
        "points_per_sec_best": pts_best,
        "schemes": scheme_details,
        "recommended_scheme": chosen,
        "recommended": {
            "scheme": chosen,
            "workers": workers_best,
            "eta_hours": scheme_etas[chosen],
            **SCHEMES[chosen],
        },
    }


def main() -> None:
    cfg_path = parse_config_arg()
    cfg = load_config(cfg_path)
    out_dir = ROOT / cfg.paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run_benchmark(cfg, config_path=cfg_path)
    out_json = out_dir / "build_benchmark.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rec = payload["recommended"]
    print("\n===== 48h RECOMMENDATION =====")
    print(f"workers_best: {payload['workers_best']}  ({payload['points_per_sec_best']:.3f} pts/s)")
    for name in ("E", "C", "A"):
        s = payload["schemes"][name]
        mark = "OK" if s["fits_48h"] else "NO"
        print(
            f"  scheme {name}: eta={s['eta_hours']:.1f}h  "
            f"conditions={s['n_conditions']}  s4={s['s4_calls']:,}  [{mark}]"
        )
    print(f"\nRecommended: scheme {rec['scheme']}  workers={rec['workers']}  eta≈{rec['eta_hours']:.1f}h")
    print(f"  angles_deg: {rec['angles_deg']}")
    print(f"  azimuths_deg: {rec['azimuths_deg']}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
