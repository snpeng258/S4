"""Inverse scatterometry: GA (DE) + LM using reflectivity R_m."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from config import (
    EVAL_TASKS,
    INVERSE_METHODS,
    PARAM_NAMES,
    ScatterometryConfig,
    decoupling_is_active,
    load_config,
    parse_config_arg,
    role_weights_enabled,
)
from decoupling import (
    cd_anchor_value,
    dynamic_weights_from_jacobian,
    static_role_weights,
    warn_missing_roles,
)
from forward_model import generate_synthetic_measurement, reset_runner, simulate_reflectivity_multi
from library_match import build_de_init_population, de_popsize_multiplier, match_library
from library_slice import format_library_slice_summary
from order_collection import (
    apply_collection,
    block_sizes,
    collection_summary,
    format_collection_summary,
    n_collectible,
    n_observables_full,
)
from residual_weights import snr_weights
from sensitivity import collectible_jacobian
from spectrum_library import describe_library_slice, library_path, load_library


@dataclass
class InverseResult:
    p_est: dict[str, float]
    p_ref: dict[str, float]
    relative_errors_pct: dict[str, float]
    method: str = "ga_lm"
    ga_history: list[float] = field(default_factory=list)
    lm_history: list[float] = field(default_factory=list)
    param_history: list[list[float]] = field(default_factory=list)
    forward_eval_count: int = 0
    timing: dict[str, float] = field(default_factory=dict)
    resnorm: float = 0.0
    lib_match: dict | None = None
    lm_weight_history: list[dict] = field(default_factory=list)


def _has_key(d: dict[str, float] | None, key: str) -> bool:
    return d is not None and key in d


def _build_bounds(
    cfg: ScatterometryConfig,
    *,
    use_yaml_init: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build p_ref, init_guess, lb, ub per param_names entry.

    Manual init_guess / lb / ub in config take precedence per parameter.
    Unset entries among param_names fall back to perturb_frac / bound_frac
    relative to structure reference values.

    When use_yaml_init=False (library methods), yaml init_guess is ignored.
    """
    inv = cfg.inverse
    names = inv.param_names
    p_ref = np.array([cfg.structure.get_param(n) for n in names], dtype=float)

    rng = np.random.default_rng(inv.ga_seed)
    p0 = np.empty(len(names), dtype=float)
    lb = np.empty(len(names), dtype=float)
    ub = np.empty(len(names), dtype=float)

    for i, n in enumerate(names):
        ref = p_ref[i]
        if use_yaml_init and _has_key(inv.init_guess, n):
            p0[i] = float(inv.init_guess[n])  # type: ignore[index]
        else:
            p0[i] = ref * (1.0 + inv.perturb_frac * (2.0 * rng.random() - 1.0))

        if _has_key(inv.lb, n):
            lb[i] = float(inv.lb[n])  # type: ignore[index]
        else:
            lb[i] = ref * (1.0 - inv.bound_frac)

        if _has_key(inv.ub, n):
            ub[i] = float(inv.ub[n])  # type: ignore[index]
        else:
            ub[i] = ref * (1.0 + inv.bound_frac)

        if n.endswith("_deg"):
            if not _has_key(inv.lb, n):
                lb[i] = max(lb[i], 1.0)
            if not _has_key(inv.ub, n):
                ub[i] = min(ub[i], 179.0)

        if lb[i] >= ub[i]:
            raise ValueError(f"invalid bounds for {n}: lb={lb[i]} >= ub={ub[i]}")

        p0[i] = float(np.clip(p0[i], lb[i], ub[i]))

    return p_ref, p0, lb, ub


def _print_bounds(problem: InverseProblem) -> None:
    inv = problem.base_cfg.inverse
    use_yaml = inv.method == "ga_lm"
    print(f"Inverse method: {inv.method}")
    print("Init / bounds (* = manual override in config):")
    for i, n in enumerate(problem.names):
        ig_m = "*" if use_yaml and _has_key(inv.init_guess, n) else " "
        lb_m = "*" if _has_key(inv.lb, n) else " "
        ub_m = "*" if _has_key(inv.ub, n) else " "
        print(
            f"  {n}: ref={problem.p_ref[i]:.6g} init={problem.p0[i]:.6g}{ig_m}  "
            f"lb={problem.lb[i]:.6g}{lb_m}  ub={problem.ub[i]:.6g}{ub_m}"
        )


def _apply_params(cfg: ScatterometryConfig, names: list[str], x: np.ndarray) -> ScatterometryConfig:
    out = cfg.copy()
    for n, v in zip(names, x):
        out.structure.set_param(n, float(v))
    return out


def _map_from_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return lb + (ub - lb) * (0.5 * (np.tanh(x) + 1.0))


def _map_to_unbounded(p: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    t = 2.0 * (p - lb) / (ub - lb + 1e-30) - 1.0
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return np.arctanh(t)


class _GaObjective:
    """Pickleable GA objective (required when differential_evolution workers > 1)."""

    __slots__ = ("_problem", "_record_history")

    def __init__(self, problem: InverseProblem, record_history: bool) -> None:
        self._problem = problem
        self._record_history = record_history

    def __call__(self, p) -> float:
        loss = self._problem._loss(np.asarray(p, dtype=float))
        if self._record_history:
            self._problem.ga_history.append(loss)
        return loss


class _GaGenerationCallback:
    """Record generation-best loss when GA runs with workers > 1."""

    __slots__ = ("_objective", "_history")

    def __init__(self, objective: _GaObjective, history: list[float]) -> None:
        self._objective = objective
        self._history = history

    def __call__(self, xk, convergence=0) -> bool:
        self._history.append(float(self._objective(np.asarray(xk, dtype=float))))
        return False


class InverseProblem:
    def __init__(
        self,
        cfg: ScatterometryConfig,
        r_meas: np.ndarray,
        *,
        use_yaml_init: bool = True,
    ):
        self.base_cfg = cfg
        r = np.asarray(r_meas, dtype=float)
        if r.ndim != 1:
            raise ValueError("r_meas must be 1-D")
        full_n = n_observables_full(cfg)
        coll_n = n_collectible(cfg)
        if r.size == full_n:
            self.r_meas = apply_collection(r, cfg)
        elif r.size == coll_n:
            self.r_meas = r.copy()
        else:
            raise ValueError(
                f"r_meas length {r.size} != full {full_n} or collectible {coll_n}"
            )
        self.names = list(cfg.inverse.param_names)
        self.p_ref, self.p0, self.lb, self.ub = _build_bounds(cfg, use_yaml_init=use_yaml_init)
        self._block_sizes = block_sizes(cfg)
        if role_weights_enabled(cfg):
            if not decoupling_is_active(cfg):
                raise ValueError("use_role_weights=True requires decoupling order collection")
            self.wsqrt = static_role_weights(cfg, self.r_meas, self._block_sizes)
            warn_missing_roles(cfg)
        else:
            self.wsqrt = snr_weights(
                self.r_meas, cfg.inverse, cfg=cfg, block_sizes=self._block_sizes
            )
        self._wsqrt_lm: np.ndarray | None = None
        self._cd_reg_sqrt: float = 0.0
        self._cd_anchor_val: float | None = None
        self.lm_weight_history: list[dict] = []
        self.eval_count = 0
        self.ga_history: list[float] = []
        self.lm_history: list[float] = []
        self.param_history: list[list[float]] = []

    def _sim_collectible(self, cfg: ScatterometryConfig) -> np.ndarray:
        r_sim, _ = simulate_reflectivity_multi(cfg)
        return apply_collection(r_sim, cfg)

    def _loss(self, p: np.ndarray) -> float:
        cfg = _apply_params(self.base_cfg, self.names, p)
        r_sim = self._sim_collectible(cfg)
        r_data = self.wsqrt * (r_sim - self.r_meas)
        reg = 0.0
        if self.base_cfg.inverse.reg_weight > 0:
            reg = self.base_cfg.inverse.reg_weight * np.sum(((p / self.p_ref) - 1.0) ** 2)
        loss = float(np.sum(r_data ** 2) + reg)
        self.param_history.append(p.tolist())
        return loss

    def _residuals_unbounded(self, x: np.ndarray) -> np.ndarray:
        p = _map_from_bounds(x, self.lb, self.ub)
        cfg = _apply_params(self.base_cfg, self.names, p)
        r_sim = self._sim_collectible(cfg)
        self.eval_count += 1
        wsqrt = self._wsqrt_lm if self._wsqrt_lm is not None else self.wsqrt
        r_data = wsqrt * (r_sim - self.r_meas)
        extra: list[np.ndarray] = []
        if self.base_cfg.inverse.reg_weight > 0:
            extra.append(np.sqrt(self.base_cfg.inverse.reg_weight) * ((p / self.p_ref) - 1.0))
        if self._cd_reg_sqrt > 0 and self._cd_anchor_val is not None and "cd_nm" in self.names:
            cd_i = self.names.index("cd_nm")
            extra.append(np.array([self._cd_reg_sqrt * (p[cd_i] - self._cd_anchor_val)]))
        if extra:
            out = np.concatenate([r_data, *extra])
        else:
            out = r_data
        self.lm_history.append(float(np.sum(r_data ** 2)))
        self.param_history.append(p.tolist())
        return out

    def run_ga(self, init_pop: np.ndarray | None = None) -> np.ndarray:
        inv = self.base_cfg.inverse
        parallel = inv.ga_workers > 1
        objective = _GaObjective(self, record_history=not parallel)
        gen_history: list[float] = []
        callback = _GaGenerationCallback(objective, gen_history) if parallel else None

        ndim = len(self.lb)
        popsize_mult = de_popsize_multiplier(inv.ga_popsize, ndim)
        de_kwargs: dict = dict(
            func=objective,
            bounds=list(zip(self.lb.tolist(), self.ub.tolist())),
            maxiter=inv.ga_maxiter,
            popsize=popsize_mult,
            mutation=inv.ga_mutation,
            recombination=inv.ga_recombination,
            seed=inv.ga_seed,
            workers=inv.ga_workers,
            polish=False,
            callback=callback,
        )
        if init_pop is not None:
            expected = (inv.ga_popsize, ndim)
            if init_pop.shape != expected:
                raise ValueError(f"init_pop shape {init_pop.shape} != expected {expected}")
            de_kwargs["init"] = init_pop

        result = differential_evolution(**de_kwargs)
        self.eval_count += int(result.nfev)
        if parallel:
            self.ga_history.extend(gen_history)
        return np.asarray(result.x, dtype=float)

    def run_lm(self, p_init: np.ndarray, *, p_ga: np.ndarray | None = None) -> np.ndarray:
        inv = self.base_cfg.inverse
        if (
            role_weights_enabled(self.base_cfg)
            and decoupling_is_active(self.base_cfg)
            and inv.decoupling.dynamic_lm.enabled
        ):
            return self.run_lm_decoupled(p_init, p_ga=p_ga)
        x0 = _map_to_unbounded(p_init, self.lb, self.ub)
        res = least_squares(
            self._residuals_unbounded,
            x0,
            method="lm",
            max_nfev=inv.lm_max_nfev,
            ftol=inv.lm_ftol,
            xtol=inv.lm_xtol,
            verbose=inv.lm_verbose,
        )
        return _map_from_bounds(res.x, self.lb, self.ub)

    def run_lm_decoupled(self, p_init: np.ndarray, *, p_ga: np.ndarray | None = None) -> np.ndarray:
        inv = self.base_cfg.inverse
        dyn = inv.decoupling.dynamic_lm
        p = np.asarray(p_init, dtype=float).copy()
        x = _map_to_unbounded(p, self.lb, self.ub)
        self._cd_anchor_val = cd_anchor_value(
            self.base_cfg, p_ga=p_ga, p_init=p_init, names=self.names
        )
        self._wsqrt_lm = self.wsqrt.copy()
        self._cd_reg_sqrt = 0.0
        nfev_since_refresh = dyn.refresh_every_nfev
        total_nfev = 0

        for outer in range(dyn.outer_max_rounds):
            if nfev_since_refresh >= dyn.refresh_every_nfev:
                J, layout = collectible_jacobian(self.base_cfg, p, self.names)
                self._wsqrt_lm, lambda_cd, diag = dynamic_weights_from_jacobian(
                    J,
                    layout,
                    self.base_cfg,
                    self.r_meas,
                    self._block_sizes,
                    base_wsqrt=self.wsqrt,
                )
                self._cd_reg_sqrt = float(np.sqrt(max(lambda_cd, 0.0)))
                diag["outer_round"] = outer
                diag["p_snapshot"] = {n: float(p[i]) for i, n in enumerate(self.names)}
                self.lm_weight_history.append(diag)
                nfev_since_refresh = 0

            max_inner = min(
                dyn.inner_max_nfev,
                max(1, inv.lm_max_nfev - total_nfev),
            )
            if max_inner <= 0:
                break
            res = least_squares(
                self._residuals_unbounded,
                x,
                method="lm",
                max_nfev=max_inner,
                ftol=inv.lm_ftol,
                xtol=inv.lm_xtol,
                verbose=inv.lm_verbose,
            )
            x = res.x
            p = _map_from_bounds(x, self.lb, self.ub)
            nfev_since_refresh += int(res.nfev)
            total_nfev += int(res.nfev)
            if res.success or total_nfev >= inv.lm_max_nfev:
                break

        return p


def _uses_library_init(method: str) -> bool:
    return method in ("lib_pop_ga_lm", "lib_pop_rand_ga_lm")


def run_inverse(
    cfg: ScatterometryConfig,
    r_meas: np.ndarray | None = None,
    *,
    print_bounds: bool = True,
    start_at_truth: bool = False,
) -> InverseResult:
    reset_runner()
    method = cfg.inverse.method
    if method not in INVERSE_METHODS:
        raise ValueError(f"unknown inverse.method={method!r}; choose from {INVERSE_METHODS}")

    if r_meas is None:
        r_meas = generate_synthetic_measurement(cfg)

    use_yaml_init = method == "ga_lm"
    problem = InverseProblem(cfg, r_meas, use_yaml_init=use_yaml_init)
    if print_bounds:
        _print_bounds(problem)
        print(format_collection_summary(cfg))
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    init_pop: np.ndarray | None = None
    lib_match: dict | None = None
    lm_weight_history: list[dict] = field(default_factory=list)

    if _uses_library_init(method):
        lib = load_library(library_path(cfg), cfg)
        if print_bounds:
            print(format_library_slice_summary(describe_library_slice(lib, cfg)))
        order, scores, t_match = match_library(
            lib, problem.r_meas, problem.wsqrt, cfg.library.prior, cfg
        )
        timing["t_lib_match"] = t_match
        pop_size = cfg.inverse.ga_popsize
        top_k = order[:pop_size]
        top_params = lib.params[top_k]
        rng = np.random.default_rng(cfg.inverse.ga_seed)
        init_pop = build_de_init_population(
            method,  # type: ignore[arg-type]
            top_params,
            problem.lb,
            problem.ub,
            pop_size,
            rng,
        )
        lib_match = {
            "top_k": top_k.tolist(),
            "top_scores": scores[top_k].tolist(),
            "library_size": lib.size,
            "n_collectible": int(problem.r_meas.size),
        }

    if start_at_truth:
        problem.p0 = problem.p_ref.copy()

    p_init = problem.p0.copy()
    if cfg.inverse.use_ga:
        tg = time.perf_counter()
        p_init = problem.run_ga(init_pop=init_pop)
        timing["t_ga"] = time.perf_counter() - tg

    p_est = p_init.copy()
    if cfg.inverse.use_lm:
        tl = time.perf_counter()
        p_est = problem.run_lm(p_init, p_ga=p_init if cfg.inverse.use_ga else None)
        timing["t_lm"] = time.perf_counter() - tl

    timing["t_total"] = time.perf_counter() - t0
    runner_evals = problem.eval_count

    p_ref_d = {n: float(problem.p_ref[i]) for i, n in enumerate(problem.names)}
    p_est_d = {n: float(p_est[i]) for i, n in enumerate(problem.names)}
    rel_err = {n: 100.0 * (p_est_d[n] - p_ref_d[n]) / (p_ref_d[n] + 1e-30) for n in problem.names}

    cfg_est = _apply_params(cfg, problem.names, p_est)
    r_fit = problem._sim_collectible(cfg_est)
    resnorm = float(np.sum((r_fit - problem.r_meas) ** 2))

    return InverseResult(
        p_est=p_est_d,
        p_ref=p_ref_d,
        relative_errors_pct=rel_err,
        method=method,
        ga_history=problem.ga_history,
        lm_history=problem.lm_history,
        param_history=problem.param_history,
        forward_eval_count=runner_evals,
        timing=timing,
        resnorm=resnorm,
        lib_match=lib_match,
        lm_weight_history=problem.lm_weight_history,
    )


def _eval_mode_hint(cfg: ScatterometryConfig, cfg_path: Path) -> None:
    task = cfg.eval.task.lower()
    if task == "scan_sweep":
        return
    mode = cfg.eval.mode.lower()
    if mode in ("noise", "timing", "ga_workers", "methods", "all"):
        print(
            f"Note: eval.mode={mode!r} is for evaluate.py only "
            f"(this script runs one inverse when eval.task=inverse). "
            f"Use: python3 evaluate.py --config {cfg_path}"
        )


def main() -> None:
    cfg_path = parse_config_arg()
    cfg = load_config(cfg_path)
    out_dir = (Path(__file__).resolve().parent / cfg.paths.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task = cfg.eval.task.lower()
    if task not in EVAL_TASKS:
        raise ValueError(f"unknown eval.task={task!r}; choose from {EVAL_TASKS}")

    print(f"Config: {cfg_path}, eval.task={task}")

    if task == "scan_sweep":
        from scan_sweep import run_scan_sweep

        run_scan_sweep(cfg, out_dir)
        return

    if task == "fim_study":
        from fim_study import run_fim_study

        print("eval.task=fim_study: running Jacobian/FIM study (no GA+LM).")
        run_fim_study(cfg, out_dir)
        return

    s, inv = cfg.structure, cfg.inverse
    _eval_mode_hint(cfg, cfg_path)
    print("Reference (true) structure:")
    for n in PARAM_NAMES:
        print(f"  {n} = {s.get_param(n)}")
    print(f"Inverse method: {inv.method}, param_names: {inv.param_names}")
    print(f"GA: maxiter={inv.ga_maxiter} popsize={inv.ga_popsize} | LM: max_nfev={inv.lm_max_nfev}")

    result = run_inverse(cfg)
    print("\n===== INVERSE RESULTS =====")
    for n in inv.param_names:
        print(
            f"{n} = {result.p_est[n]:.6g} (ref={result.p_ref[n]:.6g}, "
            f"err={result.relative_errors_pct[n]:+.2f}%)"
        )
    print(f"resnorm={result.resnorm:.6g}, forward_evals={result.forward_eval_count}")
    print(f"timing: {result.timing}")

    out_json = out_dir / "p_est.json"
    payload = {
        "method": result.method,
        "order_collection": collection_summary(cfg),
        "p_est": result.p_est,
        "p_ref": result.p_ref,
        "relative_errors_pct": result.relative_errors_pct,
        "resnorm": result.resnorm,
        "forward_eval_count": result.forward_eval_count,
        "timing": result.timing,
        "lib_match": result.lib_match,
        "ga_history": result.ga_history,
        "lm_history": result.lm_history,
        "param_names": inv.param_names,
        "lm_weight_history": result.lm_weight_history,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")

    if cfg.eval.plot:
        from evaluate import plot_convergence

        plot_convergence(result, cfg, out_dir / "inverse_convergence.png")


if __name__ == "__main__":
    main()
