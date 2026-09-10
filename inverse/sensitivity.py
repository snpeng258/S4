"""Finite-difference Jacobian on collectible observables."""
from __future__ import annotations

import numpy as np

from config import ScatterometryConfig
from forward_model import simulate_reflectivity_multi
from order_collection import CollectibleSlot, collectible_layout


def _apply_params(cfg: ScatterometryConfig, names: list[str], p: np.ndarray) -> ScatterometryConfig:
    out = cfg.copy()
    for n, v in zip(names, p):
        out.structure.set_param(n, float(v))
    return out


def full_stored_jacobian(
    cfg: ScatterometryConfig,
    p: np.ndarray,
    param_names: list[str] | None = None,
    *,
    eps_frac: float = 0.01,
    condition_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference J on the full concatenated R_m vector (no collection mask).

    Returns (J, r0, steps) with J shape (n_conditions * n_orders, n_params).
    """
    names = list(param_names or cfg.inverse.param_names)
    base = _apply_params(cfg, names, np.asarray(p, dtype=float))
    workers = max(int(condition_workers), 1)
    r0, _ = simulate_reflectivity_multi(base, condition_workers=workers)
    n_p = len(names)
    jac = np.zeros((r0.size, n_p), dtype=float)
    p0 = np.array([base.structure.get_param(n) for n in names], dtype=float)
    steps = np.zeros(n_p, dtype=float)
    for j, name in enumerate(names):
        step = max(abs(p0[j]) * eps_frac, 1e-6)
        steps[j] = step
        trial = base.copy()
        trial.structure.set_param(name, p0[j] + step)
        rp, _ = simulate_reflectivity_multi(trial, condition_workers=workers)
        if rp.shape != r0.shape:
            raise RuntimeError(f"perturbed R shape {rp.shape} != base {r0.shape}")
        jac[:, j] = (rp - r0) / step
    return jac, r0, steps


def collectible_jacobian(
    cfg: ScatterometryConfig,
    p: np.ndarray,
    param_names: list[str] | None = None,
    *,
    eps_frac: float = 0.01,
) -> tuple[np.ndarray, list[CollectibleSlot]]:
    """Return (J, layout) with J shape (n_collectible, n_params)."""
    names = list(param_names or cfg.inverse.param_names)
    base = _apply_params(cfg, names, np.asarray(p, dtype=float))
    jac, _r0, _steps = full_stored_jacobian(base, p, names, eps_frac=eps_frac)
    from order_collection import active_indices

    idx = active_indices(base)
    jac_c = jac[idx]
    layout = collectible_layout(base)
    if len(layout) != jac_c.shape[0]:
        raise RuntimeError(f"layout size {len(layout)} != J rows {jac_c.shape[0]}")
    return jac_c, layout


def param_corr_from_jacobian(jac: np.ndarray) -> np.ndarray:
    if jac.shape[1] < 2:
        return np.eye(jac.shape[1])
    j = jac.copy()
    norms = np.linalg.norm(j, axis=0) + 1e-30
    j /= norms
    c = j.T @ j
    d = np.sqrt(np.diag(c)) + 1e-30
    return c / np.outer(d, d)
