"""Azimuth-order role weights for decoupled inverse (static GA + dynamic LM)."""
from __future__ import annotations

import logging
import math

import numpy as np

from config import DecouplingConfig, ScatterometryConfig, decoupling_is_active
from order_collection import CollectibleSlot, collectible_layout
from recipe import expand_measurement_conditions
from residual_weights import snr_weights as _snr_weights

logger = logging.getLogger(__name__)

ROLES = ("depth_anchor", "swa", "lateral", "aux")


def _deg2rad(deg: float) -> float:
    return math.radians(float(deg))


def _is_near_90(azimuth_deg: float, tol: float) -> bool:
    d = abs((float(azimuth_deg) % 360.0) - 90.0)
    return min(d, 180.0 - d) < tol


def _lateral_azimuth_set(cfg: ScatterometryConfig) -> set[float] | None:
    """None => all non-90 recipe azimuths; else explicit set."""
    spec = cfg.inverse.decoupling.roles.lateral
    if spec.azimuths_deg is None:
        return None
    if len(spec.azimuths_deg) == 0:
        return None
    return {float(a) for a in spec.azimuths_deg}


def _lateral_azimuth_match(azimuth_deg: float, cfg: ScatterometryConfig, tol: float) -> bool:
    if _is_near_90(azimuth_deg, tol):
        return False
    want = _lateral_azimuth_set(cfg)
    if want is None:
        conds = expand_measurement_conditions(cfg)
        recipe_az = {float(c.azimuth_deg) for c in conds}
        return any(not _is_near_90(a, tol) and abs(a - azimuth_deg) < 1e-6 for a in recipe_az)
    return any(abs(azimuth_deg - a) < max(tol, 1e-6) for a in want)


def classify_role(azimuth_deg: float, order_m: int, cfg: ScatterometryConfig) -> str:
    dec = cfg.inverse.decoupling
    tol = dec.azimuth_tol_deg
    m = int(order_m)
    if _is_near_90(azimuth_deg, tol):
        if m in dec.roles.depth_anchor.orders:
            return "depth_anchor"
        if m in dec.roles.swa.orders:
            return "swa"
        return "aux"
    if _lateral_azimuth_match(azimuth_deg, cfg, tol) and m in dec.roles.lateral.orders:
        return "lateral"
    return "aux"


def _static_role_multiplier(role: str, dec: DecouplingConfig) -> float:
    sw = dec.static_weights
    return float(getattr(sw, role, sw.aux))


def _kx0_shadow_factor(slot: CollectibleSlot) -> float:
    kx0 = abs(math.sin(_deg2rad(slot.angle_deg)) * math.cos(_deg2rad(slot.azimuth_deg)))
    return max(kx0, 1e-3)


def static_role_weights(
    cfg: ScatterometryConfig,
    r_meas: np.ndarray,
    block_sizes: list[int],
) -> np.ndarray:
    if not decoupling_is_active(cfg):
        raise ValueError("static_role_weights requires decoupling mode")
    layout = collectible_layout(cfg)
    wsqrt = _snr_weights(np.asarray(r_meas, dtype=float), cfg.inverse, block_sizes=block_sizes)
    dec = cfg.inverse.decoupling
    mult = np.array([_static_role_multiplier(s.role, dec) for s in layout], dtype=float)
    if dec.dynamic_lm.use_kx0_shadow_prior:
        for i, slot in enumerate(layout):
            if slot.role == "lateral":
                mult[i] *= _kx0_shadow_factor(slot)
    w = (wsqrt * mult) ** 2
    w /= w.sum() + 1e-30
    return np.sqrt(w)


def _param_index(names: list[str], key: str) -> int | None:
    if key not in names:
        return None
    return names.index(key)


def dynamic_weights_from_jacobian(
    J: np.ndarray,
    layout: list[CollectibleSlot],
    cfg: ScatterometryConfig,
    r_meas: np.ndarray,
    block_sizes: list[int],
    *,
    base_wsqrt: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Return (wsqrt, cd_reg_lambda, diagnostics)."""
    dec = cfg.inverse.decoupling
    dyn = dec.dynamic_lm
    names = list(cfg.inverse.param_names)
    wsqrt = base_wsqrt.copy() if base_wsqrt is not None else static_role_weights(cfg, r_meas, block_sizes)
    w = wsqrt ** 2

    depth_i = _param_index(names, "depth_nm")
    cd_i = _param_index(names, "cd_nm")
    anchor_rows = [i for i, s in enumerate(layout) if s.role == "depth_anchor"]
    lateral_rows = [i for i, s in enumerate(layout) if s.role == "lateral"]

    j_depth = 0.0
    j_cd = 0.0
    if depth_i is not None and anchor_rows:
        sub = J[np.asarray(anchor_rows, dtype=int), depth_i]
        j_depth = float(np.linalg.norm(sub))
    if cd_i is not None and anchor_rows:
        sub = J[np.asarray(anchor_rows, dtype=int), cd_i]
        j_cd = float(np.linalg.norm(sub))

    eps = dyn.coupling_eps
    r_ratio = j_depth / (j_cd + eps)
    lambda_cd = dyn.cd_reg_lambda0 / (r_ratio + eps)

    if r_ratio < 1.0 and lateral_rows and cd_i is not None:
        boost = dyn.boost_lateral_when_coupled
        cd_sens = np.abs(J[np.asarray(lateral_rows, dtype=int), cd_i]).astype(float).ravel()
        cd_sens = cd_sens / (cd_sens.sum() + 1e-30)
        for k, row in enumerate(lateral_rows):
            w[row] *= 1.0 + boost * cd_sens[k]

    if r_ratio >= 1.0 and anchor_rows and depth_i is not None:
        boost_d = min(r_ratio, 5.0)
        for row in anchor_rows:
            w[row] *= boost_d

    swa_rows = [i for i, s in enumerate(layout) if s.role == "swa"]
    for pname in ("lswa_deg", "rswa_deg"):
        pi = _param_index(names, pname)
        if pi is None:
            continue
        for row in swa_rows:
            sens = abs(float(J[row, pi]))
            w[row] *= 1.0 + min(sens, 10.0)

    w /= w.sum() + 1e-30
    diag = {
        "coupling_ratio_r": r_ratio,
        "j_depth_norm_anchor": j_depth,
        "j_cd_norm_anchor": j_cd,
        "cd_reg_lambda": lambda_cd,
        "n_depth_anchor": len(anchor_rows),
        "n_swa": len(swa_rows),
        "n_lateral": len(lateral_rows),
    }
    return np.sqrt(w), float(lambda_cd), diag


def cd_anchor_value(
    cfg: ScatterometryConfig,
    *,
    p_ga: np.ndarray | None,
    p_init: np.ndarray,
    names: list[str],
) -> float:
    dec = cfg.inverse.decoupling
    if dec.cd_anchor == "ga" and p_ga is not None and "cd_nm" in names:
        return float(p_ga[names.index("cd_nm")])
    if "cd_nm" in names:
        return float(p_init[names.index("cd_nm")])
    return float(cfg.structure.cd_nm)


def warn_missing_roles(cfg: ScatterometryConfig) -> None:
    if not decoupling_is_active(cfg):
        return
    layout = collectible_layout(cfg)
    roles_present = {s.role for s in layout}
    if "depth_anchor" not in roles_present:
        logger.warning("decoupling: no depth_anchor rows (need phi~90, m=0 propagating)")
    if "swa" not in roles_present and cfg.inverse.decoupling.roles.swa.orders:
        logger.warning("decoupling: no swa rows (phi~90, m=±1 may be evanescent)")
    if "lateral" not in roles_present:
        logger.warning("decoupling: no lateral rows (add non-90 azimuths to recipe)")
