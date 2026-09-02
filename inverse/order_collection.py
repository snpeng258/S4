"""Collectible diffraction orders for inverse / library matching (library stores all orders)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import ScatterometryConfig
from library_slice import map_conditions_to_library_blocks, stored_conditions_from_measurement
from recipe import OpticalCondition, expand_measurement_conditions, n_orders

OrderCollectionMode = Literal["all", "propagating", "list", "decoupling"]

COLLECTION_MODES: tuple[str, ...] = ("all", "propagating", "list", "decoupling")


@dataclass(frozen=True)
class CollectibleSlot:
    flat_index: int
    condition_index: int
    order_m: int
    role: str
    label: str
    wl_nm: float
    angle_deg: float
    azimuth_deg: float


def _decoupling_orders_for_condition(cond: OpticalCondition, cfg: ScatterometryConfig) -> list[int]:
    from decoupling import classify_role

    prop = propagating_orders(
        pitch_nm=cfg.structure.pitch_nm,
        wl_nm=cond.wl_nm,
        angle_deg=cond.angle_deg,
        azimuth_deg=cond.azimuth_deg,
        order_min=cfg.optical.order_min,
        order_max=cfg.optical.order_max,
    )
    return [m for m in prop if classify_role(cond.azimuth_deg, m, cfg) != "aux"]


def collectible_layout(cfg: ScatterometryConfig) -> list[CollectibleSlot]:
    from config import decoupling_is_active
    from decoupling import classify_role

    if not decoupling_is_active(cfg):
        raise ValueError("collectible_layout requires decoupling mode")
    conditions = expand_measurement_conditions(cfg)
    per_cond = active_orders_per_condition(cfg)
    layout: list[CollectibleSlot] = []
    flat = 0
    for c_idx, (cond, orders) in enumerate(zip(conditions, per_cond)):
        for m in orders:
            layout.append(
                CollectibleSlot(
                    flat_index=flat,
                    condition_index=c_idx,
                    order_m=int(m),
                    role=classify_role(cond.azimuth_deg, m, cfg),
                    label=cond.label(),
                    wl_nm=float(cond.wl_nm),
                    angle_deg=float(cond.angle_deg),
                    azimuth_deg=float(cond.azimuth_deg),
                )
            )
            flat += 1
    return layout


def _deg2rad(deg: float) -> float:
    return math.radians(float(deg))


def order_m_values(cfg: ScatterometryConfig) -> list[int]:
    o = cfg.optical
    return list(range(o.order_min, o.order_max + 1))


def propagating_orders(
    *,
    pitch_nm: float,
    wl_nm: float,
    angle_deg: float,
    azimuth_deg: float,
    order_min: int,
    order_max: int,
) -> list[int]:
    """Air-side reflected orders with real in-plane k (|k_parallel/k0| <= 1).

    1D grating periodicity along x; groove along y. Incident unit k_parallel:
      kx0 = sin(theta)*cos(phi), ky0 = sin(theta)*sin(phi)
    Order m adds kx_m = kx0 + m * (wl/pitch) (ky unchanged).
    """
    kx0 = math.sin(_deg2rad(angle_deg)) * math.cos(_deg2rad(azimuth_deg))
    ky0 = math.sin(_deg2rad(angle_deg)) * math.sin(_deg2rad(azimuth_deg))
    g = wl_nm / pitch_nm
    out: list[int] = []
    for m in range(order_min, order_max + 1):
        kx_m = kx0 + m * g
        if kx_m * kx_m + ky0 * ky0 <= 1.0 + 1e-12:
            out.append(m)
    return out


def collectible_orders_for_condition(
    cond: OpticalCondition,
    cfg: ScatterometryConfig,
) -> list[int]:
    o = cfg.optical
    inv = cfg.inverse
    mode = inv.order_collection.lower()
    stored = order_m_values(cfg)

    if mode == "all":
        return stored

    if mode == "list":
        if not inv.accepted_orders:
            raise ValueError("inverse.order_collection=list requires inverse.accepted_orders")
        want = {int(m) for m in inv.accepted_orders}
        return [m for m in stored if m in want]

    if mode == "propagating":
        return propagating_orders(
            pitch_nm=cfg.structure.pitch_nm,
            wl_nm=cond.wl_nm,
            angle_deg=cond.angle_deg,
            azimuth_deg=cond.azimuth_deg,
            order_min=o.order_min,
            order_max=o.order_max,
        )

    if mode == "decoupling":
        return _decoupling_orders_for_condition(cond, cfg)

    raise ValueError(
        f"unknown inverse.order_collection={mode!r}; choose from {COLLECTION_MODES}"
    )


def active_orders_per_condition(cfg: ScatterometryConfig) -> list[list[int]]:
    conditions = expand_measurement_conditions(cfg)
    return [collectible_orders_for_condition(c, cfg) for c in conditions]


def active_indices(cfg: ScatterometryConfig) -> np.ndarray:
    """Column indices into the full concatenated R_m vector (n_conditions * n_orders)."""
    o = cfg.optical
    ms = order_m_values(cfg)
    m_to_local = {m: i for i, m in enumerate(ms)}
    n_block = n_orders(cfg)
    idx: list[int] = []
    for c_idx, orders in enumerate(active_orders_per_condition(cfg)):
        base = c_idx * n_block
        for m in orders:
            idx.append(base + m_to_local[m])
    return np.asarray(idx, dtype=int)


def block_sizes(cfg: ScatterometryConfig) -> list[int]:
    return [len(orders) for orders in active_orders_per_condition(cfg)]


def n_collectible(cfg: ScatterometryConfig) -> int:
    return int(active_indices(cfg).size)


def n_observables_full(cfg: ScatterometryConfig) -> int:
    return len(expand_measurement_conditions(cfg)) * n_orders(cfg)


def apply_collection(vec: np.ndarray, cfg: ScatterometryConfig) -> np.ndarray:
    full_n = n_observables_full(cfg)
    if vec.size != full_n:
        raise ValueError(f"expected full vector length {full_n}, got {vec.size}")
    idx = active_indices(cfg)
    return np.asarray(vec, dtype=float)[idx]


def library_active_column_indices(
    cfg: ScatterometryConfig,
    stored_measurement: dict,
) -> np.ndarray:
    """Column indices into lib.R: condition subset blocks + order_collection mask."""
    stored_conds = stored_conditions_from_measurement(stored_measurement)
    current_conds = expand_measurement_conditions(cfg)
    n_block = int(stored_measurement["n_orders"])
    if n_block != n_orders(cfg):
        raise ValueError(
            f"library n_orders={n_block} != config {n_orders(cfg)}"
        )
    block_map = map_conditions_to_library_blocks(current_conds, stored_conds)
    ms = order_m_values(cfg)
    m_to_local = {m: i for i, m in enumerate(ms)}
    idx: list[int] = []
    for cond, s_block in zip(current_conds, block_map):
        base = s_block * n_block
        for m in collectible_orders_for_condition(cond, cfg):
            idx.append(base + m_to_local[m])
    return np.asarray(idx, dtype=int)


def slice_library_matrix(
    R: np.ndarray,
    cfg: ScatterometryConfig,
    stored_measurement: dict | None = None,
) -> np.ndarray:
    if R.ndim != 2:
        raise ValueError("library R must be 2-D")

    if stored_measurement is None:
        idx = active_indices(cfg)
        expected_cols = n_observables_full(cfg)
        if R.shape[1] != expected_cols:
            raise ValueError(
                f"library R columns {R.shape[1]} != full observables {expected_cols}"
            )
        return R[:, idx]

    n_stored_full = int(stored_measurement["n_conditions"]) * int(stored_measurement["n_orders"])
    if R.shape[1] != n_stored_full:
        raise ValueError(
            f"library R columns {R.shape[1]} != stored full width {n_stored_full}"
        )
    idx = library_active_column_indices(cfg, stored_measurement)
    return R[:, idx]


def collection_summary(cfg: ScatterometryConfig) -> dict:
    conditions = expand_measurement_conditions(cfg)
    per_cond = active_orders_per_condition(cfg)
    return {
        "mode": cfg.inverse.order_collection,
        "accepted_orders": cfg.inverse.accepted_orders,
        "n_conditions": len(conditions),
        "n_orders_stored": n_orders(cfg),
        "n_collectible": n_collectible(cfg),
        "per_condition": [
            {
                "label": cond.label(),
                "wl_nm": cond.wl_nm,
                "angle_deg": cond.angle_deg,
                "azimuth_deg": cond.azimuth_deg,
                "orders_m": orders,
            }
            for cond, orders in zip(conditions, per_cond)
        ],
    }


def format_collection_summary(cfg: ScatterometryConfig) -> str:
    s = collection_summary(cfg)
    lines = [
        f"order_collection={s['mode']!r}: "
        f"{s['n_collectible']} / {s['n_conditions'] * s['n_orders_stored']} observables"
    ]
    if s["mode"] == "list" and s["accepted_orders"]:
        lines.append(f"  accepted_orders={s['accepted_orders']}")
    for row in s["per_condition"][:8]:
        lines.append(
            f"  {row['label']}: m={row['orders_m']}"
        )
    if len(s["per_condition"]) > 8:
        lines.append(f"  ... ({len(s['per_condition']) - 8} more conditions)")
    return "\n".join(lines)
