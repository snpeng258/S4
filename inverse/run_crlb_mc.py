"""Compact-recipe FIM vs Monte-Carlo inverse (CRLB validation).

Recomputes FIM on the inverse observation vector, then runs:

  A  local LM from the true parameters, raw 1/σ weights (MLE, matches FIM)
  B  the production solver (library + GA + LM when available)

Each (mask, mode) writes to its own subdirectory so several terminals can run
in parallel. Jacobian is computed once and reused via a lock file.

    python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode layout
    python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode fim
    python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode probe
    python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask decoupling --mode A
    python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mask m0_all --mode A
    python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode summarize
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from config import ScatterometryConfig, load_config, parse_config_arg
from fim_study import fisher_and_crlb, full_observable_layout, mask_rows
from forward_model import generate_synthetic_measurement, reset_runner, simulate_reflectivity_multi
from inverse_solver import run_inverse
from noise_model import observation_variance
from order_collection import FIM_MASK_MODES, active_orders_per_condition, format_collection_summary, n_collectible
from recipe import expand_measurement_conditions
from sensitivity import full_stored_jacobian

HERE = Path(__file__).resolve().parent
JACOBIAN_NAME = "jacobian_compact.npz"
FIM_NAME = "fim_compact.json"
LOCK_NAME = "jacobian_compact.lock"
DEFAULT_TRIALS = {"A": 40, "B": 20}


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


def _out_root(cfg: ScatterometryConfig) -> Path:
    return (HERE / cfg.paths.output_dir).resolve()


def _cell_dir(cfg: ScatterometryConfig, mask: str, mode: str) -> Path:
    return _out_root(cfg) / f"{mask}_{mode}"


def apply_mask(cfg: ScatterometryConfig, mask: str) -> ScatterometryConfig:
    if mask not in FIM_MASK_MODES:
        raise ValueError(f"mask must be one of {FIM_MASK_MODES}, got {mask!r}")
    out = cfg.copy()
    out.inverse.order_collection = mask
    if mask == "decoupling":
        out.inverse.decoupling.enabled = True
    return out


def apply_experiment(cfg: ScatterometryConfig, mask: str, mode: str) -> ScatterometryConfig:
    out = apply_mask(cfg, mask)
    if mode == "A":
        out.inverse.use_role_weights = False
        out.inverse.use_ga = False
        out.inverse.use_lm = True
        out.inverse.method = "ga_lm"
        out.inverse.decoupling.dynamic_lm.enabled = False
        return out
    if mode == "B":
        out.inverse.use_ga = True
        out.inverse.use_lm = True
        if mask == "decoupling":
            out.inverse.use_role_weights = True
            out.inverse.decoupling.enabled = True
            out.inverse.decoupling.dynamic_lm.enabled = True
        else:
            out.inverse.use_role_weights = False
            out.inverse.decoupling.dynamic_lm.enabled = False
        lib = Path(out.library.file)
        if not lib.is_absolute():
            lib = (HERE / lib).resolve()
        if mask == "decoupling" and lib.is_file():
            out.inverse.method = "lib_pop_rand_ga_lm"
        else:
            out.inverse.method = "ga_lm"
            if mask == "decoupling" and not lib.is_file():
                print(f"Library not found ({lib}); mode B uses ga_lm")
        return out
    raise ValueError(f"experiment mode must be A or B, got {mode!r}")


def _fim_windows(cfg: ScatterometryConfig) -> dict[str, list[float]]:
    return {k: list(v) for k, v in cfg.fim.azimuth_windows.items()}


def _scales(cfg: ScatterometryConfig) -> np.ndarray:
    return np.array(
        [float(cfg.fim.param_scales.get(n, 1.0)) for n in cfg.inverse.param_names],
        dtype=float,
    )


def layout_pairs(cfg: ScatterometryConfig, mask: str) -> set[tuple[float, int]]:
    masked = apply_mask(cfg, mask)
    pairs: set[tuple[float, int]] = set()
    for cond, orders in zip(expand_measurement_conditions(masked), active_orders_per_condition(masked)):
        for m in orders:
            pairs.add((float(cond.azimuth_deg), int(m)))
    return pairs


def fim_mask_pairs(cfg: ScatterometryConfig, mask: str) -> set[tuple[float, int]]:
    rows = full_observable_layout(cfg)
    sel = mask_rows(
        rows,
        mask,
        windows=_fim_windows(cfg),
        tol=cfg.fim.azimuth_tol_deg,
        keep_orders=cfg.fim.keep_orders,
    )
    return {
        (float(r.azimuth_deg), int(r.order_m))
        for r, keep in zip(rows, sel)
        if keep
    }


def _wait_for_file(path: Path, timeout_s: float) -> None:
    t0 = time.time()
    while not path.is_file():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(2.0)


def _acquire_jacobian(cfg: ScatterometryConfig, workers: int) -> tuple[np.ndarray, np.ndarray]:
    root = _out_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / JACOBIAN_NAME
    lock_path = root / LOCK_NAME
    if npz_path.is_file():
        data = np.load(npz_path)
        return data["J"], data["r0"]

    created = False
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        created = True
    except FileExistsError:
        print(f"Waiting for {npz_path.name} (another process holds {lock_path.name}) ...")
        _wait_for_file(npz_path, timeout_s=8 * 3600)
        data = np.load(npz_path)
        return data["J"], data["r0"]

    try:
        if npz_path.is_file():
            data = np.load(npz_path)
            return data["J"], data["r0"]
        names = list(cfg.inverse.param_names)
        p = np.array([cfg.structure.get_param(n) for n in names], dtype=float)
        print(
            f"Computing compact J: {len(expand_measurement_conditions(cfg))} conditions × "
            f"{1 + len(names)} forwards, condition_workers={workers}"
        )
        t0 = time.perf_counter()
        J, r0, steps = full_stored_jacobian(
            cfg, p, names, eps_frac=float(cfg.fim.eps_frac), condition_workers=workers
        )
        elapsed = time.perf_counter() - t0
        tmp = npz_path.with_suffix(".npz.tmp")
        np.savez(tmp, J=J, r0=r0, steps=steps, param_names=np.array(names))
        tmp.replace(npz_path)
        print(f"Saved {npz_path}  ({elapsed:.1f}s)")
        return J, r0
    finally:
        if created:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def metrics_for_mask(
    cfg: ScatterometryConfig,
    mask: str,
    J: np.ndarray,
    r0: np.ndarray,
) -> dict:
    rows = full_observable_layout(cfg)
    sel = mask_rows(
        rows,
        mask,
        windows=_fim_windows(cfg),
        tol=cfg.fim.azimuth_tol_deg,
        keep_orders=cfg.fim.keep_orders,
    )
    order_m = np.array([r.order_m for r in rows], dtype=int)
    J_m = np.asarray(J, dtype=float)[sel]
    r_m = np.asarray(r0, dtype=float)[sel]
    sigma = np.sqrt(observation_variance(r_m, cfg.inverse.noise, order_m[sel]))
    raw = fisher_and_crlb(J_m, sigma, _scales(cfg), rcond=float(cfg.fim.rcond))
    names = list(cfg.inverse.param_names)
    crlb = {
        n: (None if not math.isfinite(float(v)) else float(v))
        for n, v in zip(names, raw["crlb"])
    }
    return {
        "mask": mask,
        "n_rows": int(raw["n_rows"]),
        "rank": int(raw["rank"]),
        "cond_scaled": float(raw["cond_scaled"]),
        "crlb": crlb,
        "corr": np.asarray(raw["corr"], dtype=float).tolist(),
        "singular": [names[i] for i, flag in enumerate(raw["singular"]) if flag],
        "rho_depth_cd": float(raw["corr"][0][1]) if raw["corr"].shape[0] >= 2 else None,
        "rho_lswa_rswa": float(raw["corr"][2][3]) if raw["corr"].shape[0] >= 4 else None,
    }


def run_fim(cfg: ScatterometryConfig, workers: int, masks: tuple[str, ...] = FIM_MASK_MODES) -> dict:
    J, r0 = _acquire_jacobian(cfg, workers)
    payload = {
        "structure": {n: cfg.structure.get_param(n) for n in ("pitch_nm", *cfg.inverse.param_names)},
        "n_conditions": len(expand_measurement_conditions(cfg)),
        "n_s4_forwards_for_J": (1 + len(cfg.inverse.param_names)) * len(expand_measurement_conditions(cfg)),
        "masks": {m: metrics_for_mask(cfg, m, J, r0) for m in masks},
    }
    path = _out_root(cfg) / FIM_NAME
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved {path}")
    for m, rec in payload["masks"].items():
        crlb = rec["crlb"]
        print(
            f"  {m:<12} n={rec['n_rows']:4d}  rank={rec['rank']}  "
            f"CD={crlb['cd_nm']:.3g}  H={crlb['depth_nm']:.3g}  "
            f"L={crlb['lswa_deg']:.3g}  R={crlb['rswa_deg']:.3g}  "
            f"ρ(d,CD)={rec['rho_depth_cd']:+.2f}  ρ(L,R)={rec['rho_lswa_rswa']:+.2f}"
        )
    return payload


def load_or_compute_fim_mask(cfg: ScatterometryConfig, mask: str, workers: int) -> dict:
    path = _out_root(cfg) / FIM_NAME
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rec = (payload.get("masks") or {}).get(mask)
        if rec:
            return rec
    payload = run_fim(cfg, workers)
    return payload["masks"][mask]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def summarize_trials(records: list[dict], names: list[str], fim: dict | None) -> dict:
    if not records:
        return {"n": 0}
    arr = {n: np.array([r["p_est"][n] for r in records], dtype=float) for n in names}
    ref = records[0]["p_ref"]
    bias = {n: float(arr[n].mean() - ref[n]) for n in names}
    std = {n: float(arr[n].std(ddof=1)) if len(records) > 1 else 0.0 for n in names}
    rmse = {n: float(np.sqrt(np.mean((arr[n] - ref[n]) ** 2))) for n in names}
    stacked = np.column_stack([arr[n] for n in names])
    emp_corr = np.corrcoef(stacked, rowvar=False) if len(records) > 2 else None
    crlb = (fim or {}).get("crlb") or {}
    efficiency = {}
    for n in names:
        c = crlb.get(n)
        s = std[n]
        if c is None or not math.isfinite(float(c)) or s <= 0:
            efficiency[n] = None
        else:
            efficiency[n] = float(c) / s
    lswa = arr.get("lswa_deg")
    rswa = arr.get("rswa_deg")
    swap_rate = None
    if lswa is not None and rswa is not None:
        true_l, true_r = ref["lswa_deg"], ref["rswa_deg"]
        swap = (np.abs(lswa - true_r) + np.abs(rswa - true_l)) < (
            np.abs(lswa - true_l) + np.abs(rswa - true_r)
        )
        swap_rate = float(np.mean(swap))
    return {
        "n": len(records),
        "bias": bias,
        "std": std,
        "rmse": rmse,
        "efficiency_crlb_over_std": efficiency,
        "empirical_corr": None if emp_corr is None else emp_corr.tolist(),
        "rho_lswa_rswa": None if emp_corr is None else float(emp_corr[2, 3]),
        "swap_rate": swap_rate,
        "mean_forward_evals": float(np.mean([r["forward_eval_count"] for r in records])),
        "mean_t_total": float(np.mean([r["timing"].get("t_total", 0.0) for r in records])),
        "fim": fim,
    }


def plot_scatter(records: list[dict], names: list[str], fim: dict | None, out_path: Path) -> None:
    if len(records) < 2:
        return
    import matplotlib.pyplot as plt

    arr = {n: np.array([r["p_est"][n] for r in records], dtype=float) for n in names}
    ref = records[0]["p_ref"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    pairs = (("cd_nm", "depth_nm"), ("lswa_deg", "rswa_deg"))
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(arr[x], arr[y], s=18, alpha=0.75, label="MC")
        ax.scatter([ref[x]], [ref[y]], c="k", marker="x", s=60, label="truth")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"n={len(records)}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="w")
    plt.close(fig)


def run_mc(
    cfg: ScatterometryConfig,
    mask: str,
    mode: str,
    n_trials: int,
    workers: int,
    seed0: int,
) -> dict:
    work = apply_experiment(cfg, mask, mode)
    work.library.condition_workers = workers
    out_dir = _cell_dir(cfg, mask, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    done = {int(r["trial"]) for r in _read_jsonl(trials_path)}
    fim = load_or_compute_fim_mask(cfg, mask, workers)
    print(format_collection_summary(work))
    print(f"mode={mode} mask={mask} trials={n_trials} out={out_dir}")
    print(f"already finished: {sorted(done)}")

    for trial in range(n_trials):
        if trial in done:
            continue
        seed = seed0 + trial
        rng = np.random.default_rng(seed)
        reset_runner()
        r_meas = generate_synthetic_measurement(work, rng=rng)
        result = run_inverse(
            work,
            r_meas=r_meas,
            print_bounds=(trial == 0 and not done),
            start_at_truth=(mode == "A"),
        )
        row = {
            "trial": trial,
            "seed": seed,
            "mask": mask,
            "mode": mode,
            "method": result.method,
            "p_est": result.p_est,
            "p_ref": result.p_ref,
            "relative_errors_pct": result.relative_errors_pct,
            "resnorm": result.resnorm,
            "forward_eval_count": result.forward_eval_count,
            "timing": result.timing,
        }
        _append_jsonl(trials_path, row)
        print(
            f"[{trial + 1}/{n_trials}] {mask} {mode}  "
            f"CD={result.p_est['cd_nm']:.4g}  H={result.p_est['depth_nm']:.4g}  "
            f"L={result.p_est['lswa_deg']:.3f}  R={result.p_est['rswa_deg']:.3f}  "
            f"evals={result.forward_eval_count}  t={result.timing.get('t_total', 0):.1f}s"
        )

    records = _read_jsonl(trials_path)
    summary = summarize_trials(records, list(cfg.inverse.param_names), fim)
    summary.update({"mask": mask, "mode": mode, "n_trials_requested": n_trials})
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if cfg.eval.plot:
        plot_scatter(records, list(cfg.inverse.param_names), fim, out_dir / "scatter.png")
    print(f"Saved {out_dir / 'summary.json'}")
    return summary


def run_layout(cfg: ScatterometryConfig) -> None:
    n_cond = len(expand_measurement_conditions(cfg))
    print(f"conditions={n_cond}  pitch={cfg.structure.pitch_nm}  NG={cfg.optical.NG}")
    print(f"{'mask':<12} {'inv':>5} {'fim':>5} match")
    for mask in FIM_MASK_MODES:
        inv = layout_pairs(cfg, mask)
        fim = fim_mask_pairs(cfg, mask)
        ok = inv == fim
        print(f"{mask:<12} {len(inv):5d} {len(fim):5d}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  only inverse: {sorted(inv - fim)}")
            print(f"  only FIM:     {sorted(fim - inv)}")
        work = apply_mask(cfg, mask)
        print(f"  n_collectible={n_collectible(work)}")


def run_probe(cfg: ScatterometryConfig, workers: int) -> None:
    work = apply_experiment(cfg, "decoupling", "A")
    work.library.condition_workers = workers
    reset_runner()
    t0 = time.perf_counter()
    simulate_reflectivity_multi(work, condition_workers=workers)
    t_fwd = time.perf_counter() - t0
    r_meas = generate_synthetic_measurement(work, noiseless=True)
    reset_runner()
    t1 = time.perf_counter()
    result = run_inverse(work, r_meas=r_meas, print_bounds=True, start_at_truth=True)
    t_inv = time.perf_counter() - t1
    print(
        f"probe: t_forward={t_fwd:.2f}s  t_A={t_inv:.2f}s  "
        f"evals={result.forward_eval_count}"
    )
    print(
        f"  40 × A ≈ {40 * t_inv / 3600:.2f} h   "
        f"4 masks × 40 × A ≈ {160 * t_inv / 3600:.2f} h  (this process only)"
    )


def run_summarize(cfg: ScatterometryConfig) -> None:
    root = _out_root(cfg)
    names = list(cfg.inverse.param_names)
    print(f"{'cell':<22} {'n':>3}  CD_std  H_std  L_std  R_std  eff_CD  ρ_LR  swap")
    for mode in ("A", "B"):
        for mask in FIM_MASK_MODES:
            path = root / f"{mask}_{mode}" / "summary.json"
            if not path.is_file():
                continue
            s = json.loads(path.read_text(encoding="utf-8"))
            std = s.get("std") or {}
            eff = s.get("efficiency_crlb_over_std") or {}

            def fmt(d, key, spec=".3g"):
                v = d.get(key)
                return "n/a" if v is None else format(v, spec)

            rho = s.get("rho_lswa_rswa")
            swap = s.get("swap_rate")
            print(
                f"{mask}_{mode:<19} {s.get('n', 0):3d}  "
                f"{fmt(std, 'cd_nm'):>6} {fmt(std, 'depth_nm'):>6} "
                f"{fmt(std, 'lswa_deg'):>6} {fmt(std, 'rswa_deg'):>6} "
                f"{fmt(eff, 'cd_nm'):>6} "
                f"{'n/a' if rho is None else format(rho, '+.2f'):>6} "
                f"{'n/a' if swap is None else format(swap, '.2f')}"
            )
    print(f"root: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact-recipe FIM vs Monte-Carlo inverse")
    parser.add_argument("--config", default=None, help="config_crlb_mc_p80.yaml / p300.yaml")
    parser.add_argument("--mask", choices=FIM_MASK_MODES, default=None)
    parser.add_argument(
        "--mode",
        choices=("layout", "fim", "probe", "A", "B", "summarize"),
        required=True,
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--condition-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser() if args.config else parse_config_arg()
    cfg = load_config(cfg_path)
    workers = int(args.condition_workers or cfg.library.condition_workers or 1)
    cfg.library.condition_workers = workers

    if args.mode == "layout":
        run_layout(cfg)
        return
    if args.mode == "fim":
        run_fim(cfg, workers)
        return
    if args.mode == "probe":
        run_probe(cfg, workers)
        return
    if args.mode == "summarize":
        run_summarize(cfg)
        return
    if args.mask is None:
        raise SystemExit("--mask is required for mode A / B")
    n_trials = int(args.n_trials or DEFAULT_TRIALS[args.mode])
    run_mc(cfg, args.mask, args.mode, n_trials, workers, args.seed)


if __name__ == "__main__":
    main()
