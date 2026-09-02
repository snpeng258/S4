"""Library matching: score spectra, optional prior, build DE initial population."""
from __future__ import annotations

import time
from typing import Literal

import numpy as np

from config import LibraryPriorConfig, ScatterometryConfig
from order_collection import slice_library_matrix
from spectrum_library import SpectralLibrary


def _prior_penalty(params: np.ndarray, names: list[str], prior: LibraryPriorConfig) -> np.ndarray:
    n = params.shape[0]
    if not prior.enabled:
        return np.zeros(n, dtype=float)
    pen = np.zeros(n, dtype=float)
    swa_lo, swa_hi = prior.swa_deg_range[0], prior.swa_deg_range[1]
    for j, name in enumerate(names):
        if name.endswith("_deg"):
            below = np.maximum(swa_lo - params[:, j], 0.0)
            above = np.maximum(params[:, j] - swa_hi, 0.0)
            pen += (below + above) ** 2
    if prior.penalize_asymmetry:
        idx = {n: i for i, n in enumerate(names)}
        if "lswa_deg" in idx and "rswa_deg" in idx:
            d = params[:, idx["lswa_deg"]] - params[:, idx["rswa_deg"]]
            pen += d * d
    return prior.lambda_prior * pen


def _prior_reject_mask(params: np.ndarray, names: list[str], prior: LibraryPriorConfig) -> np.ndarray:
    if not prior.enabled or not prior.hard_reject_out_of_range:
        return np.ones(params.shape[0], dtype=bool)
    swa_lo, swa_hi = prior.swa_deg_range[0], prior.swa_deg_range[1]
    ok = np.ones(params.shape[0], dtype=bool)
    for j, name in enumerate(names):
        if name.endswith("_deg"):
            ok &= (params[:, j] >= swa_lo) & (params[:, j] <= swa_hi)
    return ok


def match_library(
    lib: SpectralLibrary,
    r_meas: np.ndarray,
    wsqrt: np.ndarray,
    prior: LibraryPriorConfig,
    cfg: ScatterometryConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (sorted_indices, scores, elapsed_seconds). Lower score is better.

    When cfg is given, lib.R is sliced to inverse collectible orders (library file
    still stores the full order range).
    """
    t0 = time.perf_counter()
    R = (
        slice_library_matrix(lib.R, cfg, lib.meta.get("measurement"))
        if cfg is not None
        else lib.R
    )
    resid = (R - r_meas) * wsqrt
    scores = np.sum(resid * resid, axis=1)
    scores += _prior_penalty(lib.params, lib.param_names, prior)
    ok = _prior_reject_mask(lib.params, lib.param_names, prior)
    scores = np.where(ok, scores, np.inf)
    order = np.argsort(scores)
    return order, scores, time.perf_counter() - t0


def random_in_bounds(lb: np.ndarray, ub: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return lb + (ub - lb) * rng.random((n, len(lb)))


def build_de_init_population(
    method: Literal["lib_pop_ga_lm", "lib_pop_rand_ga_lm"],
    top_params: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build (pop_size, ndim) initial population within bounds."""
    ndim = len(lb)
    pop = np.empty((pop_size, ndim), dtype=float)
    if method == "lib_pop_ga_lm":
        n_lib = min(pop_size, top_params.shape[0])
        pop[:n_lib] = top_params[:n_lib]
        if n_lib < pop_size:
            pop[n_lib:] = random_in_bounds(lb, ub, pop_size - n_lib, rng)
    elif method == "lib_pop_rand_ga_lm":
        n_lib = min(int(0.8 * pop_size), top_params.shape[0])
        n_rand = pop_size - n_lib
        pop[:n_lib] = top_params[:n_lib]
        if n_rand > 0:
            pop[n_lib:] = random_in_bounds(lb, ub, n_rand, rng)
    else:
        raise ValueError(f"not a library init method: {method}")
    return np.clip(pop, lb, ub)


def de_popsize_multiplier(pop_size: int, ndim: int) -> int:
    """SciPy popsize multiplier so popsize * ndim == pop_size."""
    if pop_size % ndim != 0:
        raise ValueError(
            f"ga_popsize={pop_size} must be divisible by n_params={ndim} "
            f"(DE population = popsize × n_params)"
        )
    mult = pop_size // ndim
    return max(mult, 5)
