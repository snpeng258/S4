"""Finite-difference Jacobian on collectible observables."""
from __future__ import annotations

import numpy as np

from config import ScatterometryConfig
from forward_model import simulate_reflectivity_multi
from order_collection import CollectibleSlot, apply_collection, collectible_layout, n_collectible


def _apply_params(cfg: ScatterometryConfig, names: list[str], p: np.ndarray) -> ScatterometryConfig:
    out = cfg.copy()
    for n, v in zip(names, p):
        out.structure.set_param(n, float(v))
    return out


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
    r0, _ = simulate_reflectivity_multi(base)
    r0c = apply_collection(r0, base)
    n_p = len(names)
    jac = np.zeros((n_collectible(base), n_p), dtype=float)
    p0 = np.array([base.structure.get_param(n) for n in names], dtype=float)
    for j, name in enumerate(names):
        step = max(abs(p0[j]) * eps_frac, 1e-6)
        trial = base.copy()
        trial.structure.set_param(name, p0[j] + step)
        rp, _ = simulate_reflectivity_multi(trial)
        rpc = apply_collection(rp, trial)
        jac[:, j] = (rpc - r0c) / step
    layout = collectible_layout(base)
    if len(layout) != jac.shape[0]:
        raise RuntimeError(f"layout size {len(layout)} != J rows {jac.shape[0]}")
    return jac, layout


def param_corr_from_jacobian(jac: np.ndarray) -> np.ndarray:
    if jac.shape[1] < 2:
        return np.eye(jac.shape[1])
    j = jac.copy()
    norms = np.linalg.norm(j, axis=0) + 1e-30
    j /= norms
    c = j.T @ j
    d = np.sqrt(np.diag(c)) + 1e-30
    return c / np.outer(d, d)
