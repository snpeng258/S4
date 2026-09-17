"""Performance evaluation: noise robustness, timing, plots."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import EVAL_TASKS, INVERSE_METHODS, ScatterometryConfig, load_config, parse_config_arg
from forward_model import (
    generate_synthetic_measurement,
    noise_db_to_fraction,
    reset_runner,
    simulate_reflectivity_multi,
)
from inverse_solver import InverseResult, run_inverse
from scan_sweep import run_scan_sweep


def _param_unit(name: str) -> str:
    if name.endswith("_nm"):
        return "nm"
    if name.endswith("_deg"):
        return "deg"
    return ""


def _abs_error(record: dict, pname: str) -> float:
    abs_e = record.get("abs_errors") or {}
    if pname in abs_e:
        return abs(float(abs_e[pname]))
    rel = record.get("rel_errors") or {}
    if pname in rel:
        return abs(float(rel[pname]))
    return float("nan")


def _abs_ylabel(pname: str) -> str:
    unit = _param_unit(pname)
    return f"|abs error| ({unit})" if unit else "|abs error|"


def _format_elapsed(seconds: float) -> str:
    if seconds >= 60.0:
        m, s = divmod(seconds, 60.0)
        return f"{int(m)}m {s:.1f}s"
    return f"{seconds:.1f}s"


def _timing_caption(timing: dict[str, float], key: str) -> str:
    t = timing.get(key, 0.0)
    return f"t = {_format_elapsed(t)}" if t > 0 else ""


def plot_convergence(result: InverseResult, cfg: ScatterometryConfig, out_path: Path) -> None:
    timing = result.timing
    t_total = timing.get("t_total", 0.0)
    suptitle = "Inverse convergence"
    if t_total > 0:
        suptitle += f"  (total {_format_elapsed(t_total)})"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if result.ga_history:
        axes[0].plot(result.ga_history, "b-o", ms=3)
        axes[0].set_title("GA loss")
        axes[0].set_xlabel("eval")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.3)
        ga_cap = _timing_caption(timing, "t_ga")
        if ga_cap:
            axes[0].text(
                0.98,
                0.98,
                ga_cap,
                transform=axes[0].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
            )
    else:
        axes[0].text(0.5, 0.5, "GA skipped", ha="center", va="center")
        axes[0].set_axis_off()

    if result.lm_history:
        axes[1].plot(result.lm_history, "r-o", ms=3)
        axes[1].set_title("LM data residual^2")
        axes[1].set_xlabel("iter")
        axes[1].grid(True, alpha=0.3)
        lm_cap = _timing_caption(timing, "t_lm")
        if lm_cap:
            axes[1].text(
                0.98,
                0.98,
                lm_cap,
                transform=axes[1].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
            )
    else:
        axes[1].text(0.5, 0.5, "LM skipped", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle(suptitle)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.eval.plot_dpi, facecolor="w")
    plt.close(fig)


def _noise_db_label(db: float) -> str:
    return f"{db:g} dB"


def _noise_pct_label(db: float) -> str:
    pct = noise_db_to_fraction(db) * 100.0
    if pct >= 1.0:
        return f"{pct:.3g}%"
    return f"{pct:.4g}%"


def _apply_dual_noise_xaxis(ax: plt.Axes, noise_dbs: list[float]) -> None:
    """Bottom x-axis: dB; top x-axis: sigma/R_peak as percentage."""
    ax.set_xticks(noise_dbs)
    ax.set_xticklabels([f"{db:g}" for db in noise_dbs])
    ax.set_xlabel("noise (dB)")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(noise_dbs)
    ax_top.set_xticklabels([_noise_pct_label(db) for db in noise_dbs])
    ax_top.set_xlabel("noise (% of peak R)")


def _apply_n0_xaxis(ax: plt.Axes, n0_vals: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(n0_vals)
    ax.set_xticklabels([f"{n0:.3g}" for n0 in n0_vals])
    ax.set_xlabel(r"$N_0$ (photoelectrons at $R=1$)")


def plot_noise_sweep(df_records: list[dict], param_names: list[str], out_dir: Path, dpi: int) -> None:
    if not df_records:
        return
    if "n0_electrons" in df_records[0]:
        keys = sorted({float(r["n0_electrons"]) for r in df_records})
        match = lambda rec, k: float(rec["n0_electrons"]) == k
        xlabel_fn = _apply_n0_xaxis
    else:
        keys = sorted({r["noise_db"] for r in df_records})
        match = lambda rec, k: rec["noise_db"] == k
        xlabel_fn = _apply_dual_noise_xaxis

    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4.5), squeeze=False)
    for j, pname in enumerate(param_names):
        ax = axes[0, j]
        means, stds = [], []
        for k in keys:
            errs = [_abs_error(r, pname) for r in df_records if match(r, k)]
            means.append(np.mean(errs))
            stds.append(np.std(errs))
        ax.errorbar(keys, means, yerr=stds, fmt="-o", capsize=3)
        ax.set_ylabel(_abs_ylabel(pname))
        ax.set_title(pname)
        ax.grid(True, alpha=0.3)
        xlabel_fn(ax, keys)
    fig.suptitle("Parameter error vs noise")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_noise_curve.png", dpi=dpi, facecolor="w")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    data = []
    for k in keys:
        vals = []
        for pname in param_names:
            vals.extend([_abs_error(r, pname) for r in df_records if match(r, k)])
        data.append(vals)
    if "n0_electrons" in df_records[0]:
        ax2.boxplot(data, positions=list(range(len(keys))), widths=0.6)
        ax2.set_xticks(list(range(len(keys))))
        ax2.set_xticklabels([f"{k:.3g}" for k in keys])
        ax2.set_xlabel(r"$N_0$ (photoelectrons at $R=1$)")
    else:
        ax2.boxplot(data, positions=keys, widths=1.2)
        xlabel_fn(ax2, keys)
    ax2.set_ylabel("|abs error|")
    ax2.set_title("Absolute error distribution")
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(out_dir / "eval_noise_boxplot.png", dpi=dpi, facecolor="w")
    plt.close(fig2)


def plot_timing(timing: dict[str, float], forward_evals: int, out_dir: Path, dpi: int) -> None:
    labels = [k for k in ("t_ga", "t_lm", "t_s4_forward", "t_total") if k in timing and timing[k] > 0]
    if not labels:
        labels = list(timing.keys())
    vals = [timing[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#8172B2"][: len(labels)])
    ax.set_ylabel("seconds")
    ax.set_title(f"Timing (forward_evals={forward_evals})")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_timing.png", dpi=dpi, facecolor="w")
    plt.close(fig)


def evaluate_noise(cfg: ScatterometryConfig) -> list[dict]:
    records = []
    n0_list = list(cfg.eval.noise_n0_electrons or [])
    if n0_list and cfg.inverse.noise.apply:
        for n0 in n0_list:
            for trial in range(cfg.eval.n_trials):
                reset_runner()
                trial_cfg = cfg.copy()
                trial_cfg.inverse.noise.set_n0_electrons(float(n0))
                rng = np.random.default_rng(
                    cfg.inverse.ga_seed + trial if cfg.inverse.ga_seed else trial
                )
                r_meas = generate_synthetic_measurement(
                    trial_cfg, rng=rng, n0_electrons=float(n0)
                )
                inv_cfg = cfg.copy()
                inv_cfg.inverse.noise.set_n0_electrons(float(n0))
                result = run_inverse(inv_cfg, r_meas=r_meas)
                records.append({
                    "n0_electrons": float(n0),
                    "trial": trial,
                    "rel_errors": result.relative_errors_pct,
                    "abs_errors": result.absolute_errors,
                    "resnorm": result.resnorm,
                    "timing": result.timing,
                    "forward_evals": result.forward_eval_count,
                })
                print(
                    f"N0={n0:.3g} trial={trial} abs={result.absolute_errors}"
                )
        return records

    for noise_db in cfg.eval.noise_levels_db:
        noise_frac = noise_db_to_fraction(noise_db)
        for trial in range(cfg.eval.n_trials):
            reset_runner()
            trial_cfg = cfg.copy()
            trial_cfg.inverse.noise.apply = False
            rng = np.random.default_rng(cfg.inverse.ga_seed + trial if cfg.inverse.ga_seed else trial)
            r_meas = generate_synthetic_measurement(trial_cfg, noise_db=noise_db, rng=rng)
            inv_cfg = cfg.copy()
            inv_cfg.inverse.noise.apply = False
            inv_cfg.inverse.noise_level = noise_frac
            result = run_inverse(inv_cfg, r_meas=r_meas)
            records.append({
                "noise_db": noise_db,
                "noise_level": noise_frac,
                "trial": trial,
                "rel_errors": result.relative_errors_pct,
                "abs_errors": result.absolute_errors,
                "resnorm": result.resnorm,
                "timing": result.timing,
                "forward_evals": result.forward_eval_count,
            })
            print(
                f"noise={_noise_db_label(noise_db)} (sigma/R_peak={noise_frac:.4g}) "
                f"trial={trial} abs={result.absolute_errors}"
            )
    return records


def evaluate_timing(cfg: ScatterometryConfig) -> dict:
    reset_runner()
    t0 = time.perf_counter()
    _, _ = simulate_reflectivity_multi(cfg)
    t_forward = time.perf_counter() - t0
    t1 = time.perf_counter()
    r_meas = generate_synthetic_measurement(cfg, noiseless=True)
    result = run_inverse(cfg, r_meas=r_meas)
    t_inv = time.perf_counter() - t1
    timing = dict(result.timing)
    timing["t_s4_forward_single"] = t_forward
    timing["t_inverse_total"] = t_inv
    timing["forward_evals"] = result.forward_eval_count
    return timing


def _summarize_ga_workers(records: list[dict]) -> dict:
    workers = sorted({r["ga_workers"] for r in records})
    per_worker: dict[int, dict] = {}
    for w in workers:
        rows = [r for r in records if r["ga_workers"] == w]
        stats: dict[str, dict[str, float]] = {}
        for key in ("t_ga", "t_lm", "t_total"):
            vals = [r["timing"][key] for r in rows if key in r["timing"]]
            stats[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        per_worker[w] = {
            "n_trials": len(rows),
            "timing": stats,
            "forward_evals_mean": float(np.mean([r["forward_evals"] for r in rows])),
            "resnorm_mean": float(np.mean([r["resnorm"] for r in rows])),
        }

    best_w = min(workers, key=lambda w: per_worker[w]["timing"]["t_total"]["mean"])
    best = per_worker[best_w]["timing"]["t_total"]
    return {
        "per_worker": {str(w): per_worker[w] for w in workers},
        "best_by_total": {
            "ga_workers": best_w,
            "t_total_mean": best["mean"],
            "t_total_std": best["std"],
        },
    }


def _trial_ga_seed(base: int | None, trial: int, stride: int) -> int | None:
    if base is None:
        return trial + 1
    return int(base + trial * stride)


def evaluate_ga_workers(cfg: ScatterometryConfig, out_dir: Path | None = None) -> dict:
    reset_runner()
    r_meas = generate_synthetic_measurement(cfg, noiseless=True)
    records: list[dict] = []
    sweep = cfg.eval.ga_workers_sweep
    n_trials = cfg.eval.ga_workers_trials
    n_runs = len(sweep) * n_trials

    base_seed = cfg.inverse.ga_seed
    seed_stride = cfg.eval.ga_workers_seed_stride

    print(f"GA workers sweep: {sweep}, trials={n_trials}, fixed noiseless r_meas")
    print(f"ga_seed per trial: base={base_seed}, stride={seed_stride}")
    print(f"Total runs: {n_runs} (run via: python3 evaluate.py)")

    def _save_checkpoint(complete: bool) -> None:
        if out_dir is None:
            return
        payload = {
            "records": records,
            "summary": _summarize_ga_workers(records) if records else {},
            "complete": complete,
            "expected_runs": n_runs,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_ga_workers.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    for workers in sweep:
        for trial in range(n_trials):
            reset_runner()
            trial_cfg = cfg.copy()
            trial_cfg.inverse.ga_workers = int(workers)
            trial_cfg.inverse.lm_verbose = 0
            trial_seed = _trial_ga_seed(base_seed, trial, seed_stride)
            trial_cfg.inverse.ga_seed = trial_seed
            show_bounds = workers == sweep[0] and trial == 0
            result = run_inverse(trial_cfg, r_meas=r_meas, print_bounds=show_bounds)
            record = {
                "ga_workers": int(workers),
                "trial": trial,
                "ga_seed": trial_seed,
                "timing": result.timing,
                "forward_evals": result.forward_eval_count,
                "resnorm": result.resnorm,
            }
            records.append(record)
            t = result.timing
            print(
                f"[{len(records)}/{n_runs}] ga_workers={workers} trial={trial} seed={trial_seed} "
                f"t_ga={t.get('t_ga', 0):.1f}s t_lm={t.get('t_lm', 0):.1f}s "
                f"t_total={t.get('t_total', 0):.1f}s forward_evals={result.forward_eval_count}"
            )
        _save_checkpoint(complete=False)
        if out_dir is not None:
            print(f"Checkpoint saved ({len(records)}/{n_runs} runs)")

    summary = _summarize_ga_workers(records)
    payload = {"records": records, "summary": summary, "complete": True, "expected_runs": n_runs}
    if out_dir is not None:
        (out_dir / "eval_ga_workers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def plot_ga_workers_sweep(payload: dict, out_dir: Path, dpi: int) -> None:
    records = payload["records"]
    summary = payload["summary"]
    if not records:
        return

    workers = sorted({r["ga_workers"] for r in records})
    per_worker = summary["per_worker"]
    best = summary["best_by_total"]

    fig, ax = plt.subplots(figsize=(10, 5))
    series = [
        ("t_ga", "#4C72B0", "GA"),
        ("t_lm", "#DD8452", "LM"),
        ("t_total", "#55A868", "total"),
    ]
    for key, color, label in series:
        means = [per_worker[str(w)]["timing"][key]["mean"] for w in workers]
        stds = [per_worker[str(w)]["timing"][key]["std"] for w in workers]
        ax.errorbar(workers, means, yerr=stds, fmt="-o", capsize=3, color=color, label=label, ms=4)

    best_w = best["ga_workers"]
    ax.axvline(best_w, color="0.4", linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        0.98,
        f"best ga_workers={best_w}\n(t_total={best['t_total_mean']:.1f}s)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    ax.set_xlabel("ga_workers")
    ax.set_ylabel("seconds")
    ax.set_title("Inverse timing vs ga_workers")
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_ga_workers.png", dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_methods_comparison(records: list[dict], param_names: list[str], out_dir: Path, dpi: int) -> None:
    if not records:
        return
    methods = [r["method"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(methods))
    width = 0.6
    for j, pname in enumerate(param_names):
        errs = [_abs_error(r, pname) for r in records]
        offset = (j - len(param_names) / 2 + 0.5) * width / len(param_names)
        axes[0].bar(x + offset, errs, width / len(param_names), label=pname)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha="right")
    axes[0].set_ylabel("|abs error| (nm / deg)")
    axes[0].set_title("Parameter error by method")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    t_ga = [r["timing"].get("t_ga", 0.0) for r in records]
    t_lm = [r["timing"].get("t_lm", 0.0) for r in records]
    t_lib = [r["timing"].get("t_lib_match", 0.0) for r in records]
    t_tot = [r["timing"].get("t_total", 0.0) for r in records]
    axes[1].bar(x, t_ga, width, label="t_ga")
    axes[1].bar(x, t_lm, width, bottom=t_ga, label="t_lm")
    axes[1].bar(x, t_lib, width, bottom=np.array(t_ga) + np.array(t_lm), label="t_lib_match")
    axes[1].plot(x, t_tot, "ko", ms=6, label="t_total")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha="right")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Timing by method")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Inverse methods comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_methods.png", dpi=dpi, facecolor="w")
    plt.close(fig)


def evaluate_methods(cfg: ScatterometryConfig) -> list[dict]:
    """Compare ga_lm, lib_pop_ga_lm, lib_pop_rand_ga_lm on the same r_meas."""
    reset_runner()
    r_meas = generate_synthetic_measurement(cfg, noiseless=True)
    records: list[dict] = []
    for i, method in enumerate(INVERSE_METHODS):
        reset_runner()
        trial_cfg = cfg.copy()
        trial_cfg.inverse.method = method
        trial_cfg.inverse.lm_verbose = 0
        show_bounds = i == 0
        result = run_inverse(trial_cfg, r_meas=r_meas, print_bounds=show_bounds)
        record = {
            "method": method,
            "rel_errors": result.relative_errors_pct,
            "abs_errors": result.absolute_errors,
            "resnorm": result.resnorm,
            "timing": result.timing,
            "forward_evals": result.forward_eval_count,
            "p_est": result.p_est,
            "lib_match": result.lib_match,
        }
        records.append(record)
        t = result.timing
        print(
            f"method={method} abs={result.absolute_errors} "
            f"resnorm={result.resnorm:.4g} "
            f"t_lib={t.get('t_lib_match', 0):.3f}s "
            f"t_total={t.get('t_total', 0):.1f}s "
            f"s4_evals={result.forward_eval_count}"
        )
    return records


def main() -> None:
    cfg_path = parse_config_arg()
    cfg = load_config(cfg_path)
    out_dir = (Path(__file__).resolve().parent / cfg.paths.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task = cfg.eval.task.lower()
    if task not in EVAL_TASKS:
        raise ValueError(f"Unknown eval.task={task!r}; choose from {EVAL_TASKS}")

    print(f"Config: {cfg_path}, eval.task={task}")

    if task == "scan_sweep":
        run_scan_sweep(cfg, out_dir)
        print(f"Results in {out_dir}")
        return

    if task == "fim_study":
        from fim_study import run_fim_study

        run_fim_study(cfg, out_dir)
        print(f"Results in {out_dir}")
        return

    mode = cfg.eval.mode.lower()
    known = ("noise", "timing", "ga_workers", "methods", "all")
    print(f"eval.mode={mode}")
    if mode not in known:
        raise ValueError(f"Unknown eval.mode={mode!r}; choose from {known}")

    if mode in ("noise", "all"):
        records = evaluate_noise(cfg)
        (out_dir / "eval_noise.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if cfg.eval.plot:
            plot_noise_sweep(records, cfg.inverse.param_names, out_dir, cfg.eval.plot_dpi)

    if mode in ("timing", "all"):
        timing = evaluate_timing(cfg)
        (out_dir / "eval_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        print(f"Timing: {timing}")
        if cfg.eval.plot:
            plot_timing(timing, int(timing.get("forward_evals", 0)), out_dir, cfg.eval.plot_dpi)

    if mode in ("ga_workers",):
        payload = evaluate_ga_workers(cfg, out_dir=out_dir)
        if not payload.get("complete"):
            print("Warning: sweep incomplete")
        best = payload["summary"]["best_by_total"]
        print(
            f"Best ga_workers={best['ga_workers']} by t_total "
            f"(mean={best['t_total_mean']:.2f}s, std={best['t_total_std']:.2f}s)"
        )
        if cfg.eval.plot:
            plot_ga_workers_sweep(payload, out_dir, cfg.eval.plot_dpi)

    if mode in ("methods",):
        records = evaluate_methods(cfg)
        (out_dir / "eval_methods.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if cfg.eval.plot:
            plot_methods_comparison(records, cfg.inverse.param_names, out_dir, cfg.eval.plot_dpi)

    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
