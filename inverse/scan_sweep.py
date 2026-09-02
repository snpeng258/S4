"""Multi-parameter forward scan (wavelength / angle / azimuth grid)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ScatterometryConfig
from forward_model import reset_runner, simulate_reflectivity
from sensitivity import param_corr_from_jacobian
from recipe import (
    OpticalCondition,
    cfg_with_condition,
    expand_scan_grid,
    iter_structure_sweep,
    n_orders,
    scan_recipe_source,
    scan_recipe_summary,
)


def _finite_diff_jacobian_single(
    cfg: ScatterometryConfig,
    cond: OpticalCondition,
    param_names: list[str],
    r0: np.ndarray,
    eps_frac: float = 0.01,
) -> np.ndarray:
    """Single-condition Jacobian (full order block); scan_sweep only."""
    n_p = len(param_names)
    jac = np.zeros((len(r0), n_p), dtype=float)
    base = cfg.copy()
    base = cfg_with_condition(base, cond)
    p0 = np.array([base.structure.get_param(n) for n in param_names], dtype=float)
    for j, name in enumerate(param_names):
        step = max(abs(p0[j]) * eps_frac, 1e-6)
        trial = base.copy()
        trial.structure.set_param(name, p0[j] + step)
        rp, _ = simulate_reflectivity(trial)
        jac[:, j] = (rp - r0) / step
    return jac


def plot_scan_sweep(records: list[dict], param_names: list[str], out_dir: Path, dpi: int) -> None:
    if not records:
        return
    labels = [r["label"] for r in records]
    x = np.arange(len(labels))

    if any("jacobian_col_norm" in r for r in records):
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
        width = 0.8 / max(len(param_names), 1)
        for j, pname in enumerate(param_names):
            vals = [r.get("jacobian_col_norm", {}).get(pname, 0.0) for r in records]
            offset = (j - len(param_names) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=pname)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("|dR/dp| (norm)")
        ax.set_title("Jacobian column norms by scan point")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "scan_sweep_jacobian.png", dpi=dpi, facecolor="w")
        plt.close(fig)

    if records and "param_corr" in records[0]:
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        im = ax2.imshow(records[0]["param_corr"], vmin=-1, vmax=1, cmap="RdBu_r")
        ax2.set_xticks(range(len(param_names)))
        ax2.set_yticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha="right")
        ax2.set_yticklabels(param_names)
        ax2.set_title("Param correlation (1st grid point)")
        fig2.colorbar(im, ax=ax2, fraction=0.046)
        fig2.tight_layout()
        fig2.savefig(out_dir / "scan_sweep_corr.png", dpi=dpi, facecolor="w")
        plt.close(fig2)


def run_scan_sweep(cfg: ScatterometryConfig, out_dir: Path | None = None) -> list[dict]:
    """Grid scan over scan.axes; optional structure_sweep and Jacobian diagnostics."""
    reset_runner()
    out_dir = out_dir or (Path(__file__).resolve().parent / cfg.paths.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = expand_scan_grid(cfg)
    param_names = list(cfg.inverse.param_names)
    n_ord = n_orders(cfg)
    records: list[dict] = []

    print(
        f"scan_sweep: source={scan_recipe_source(cfg)}, axes={cfg.scan.axes}, "
        f"grid_points={len(grid)}, structure_sweep={bool(cfg.scan.structure_sweep)}"
    )

    for struct_delta, struct_cfg in iter_structure_sweep(cfg):
        for cond in grid:
            t0 = time.perf_counter()
            trial = cfg_with_condition(struct_cfg, cond)
            r_vec, r_map = simulate_reflectivity(trial)
            elapsed = time.perf_counter() - t0
            peak_m = max(r_map, key=lambda m: r_map[m]) if r_map else 0
            rec: dict = {
                "label": cond.label(),
                "harmonic_order": cond.harmonic_order,
                "wl_nm": cond.wl_nm,
                "angle_deg": cond.angle_deg,
                "azimuth_deg": cond.azimuth_deg,
                "structure": struct_delta or None,
                "R_sum": float(r_vec.sum()),
                "R_peak": float(r_vec.max()),
                "peak_order_m": int(peak_m),
                "n_orders": n_ord,
                "timing_s": elapsed,
            }
            if cfg.scan.compute_jacobian:
                jac = _finite_diff_jacobian_single(trial, cond, param_names, r_vec)
                rec["jacobian_col_norm"] = {
                    n: float(np.linalg.norm(jac[:, j])) for j, n in enumerate(param_names)
                }
                if not records:
                    rec["param_corr"] = param_corr_from_jacobian(jac).tolist()
            records.append(rec)
            print(
                f"  {rec['label']} struct={struct_delta or 'ref'} "
                f"R_sum={rec['R_sum']:.4g} peak_m={rec['peak_order_m']} t={elapsed:.2f}s"
            )

    payload = {
        "scan_axes": cfg.scan.axes,
        "scan_recipe": scan_recipe_summary(cfg),
        "n_records": len(records),
        "records": records,
    }
    (out_dir / "scan_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'scan_sweep.json'}")

    if cfg.scan.plot:
        plot_scan_sweep(records, param_names, out_dir, cfg.eval.plot_dpi)

    return records
