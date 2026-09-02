"""Map inverse recipe conditions to column blocks in a stored spectral library."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from config import ScatterometryConfig
from recipe import OpticalCondition, expand_measurement_conditions, n_orders, recipe_meta_dict


def optical_condition_from_dict(d: dict) -> OpticalCondition:
    q = d.get("harmonic_order")
    return OpticalCondition(
        wl_nm=float(d["wl_nm"]),
        angle_deg=float(d["angle_deg"]),
        azimuth_deg=float(d["azimuth_deg"]),
        harmonic_order=int(q) if q is not None else None,
    )


def conditions_match(
    a: OpticalCondition,
    b: OpticalCondition,
    *,
    wl_rel_tol: float = 1e-9,
    wl_abs_tol: float = 1e-12,
    angle_abs_tol: float = 1e-6,
) -> bool:
    if a.harmonic_order != b.harmonic_order:
        return False
    if not np.isclose(a.wl_nm, b.wl_nm, rtol=wl_rel_tol, atol=wl_abs_tol):
        return False
    if not np.isclose(a.angle_deg, b.angle_deg, rtol=0.0, atol=angle_abs_tol):
        return False
    if not np.isclose(a.azimuth_deg, b.azimuth_deg, rtol=0.0, atol=angle_abs_tol):
        return False
    return True


def find_stored_condition_index(
    stored: list[OpticalCondition],
    target: OpticalCondition,
) -> int:
    for i, cand in enumerate(stored):
        if conditions_match(cand, target):
            return i
    raise ValueError(
        f"condition not found in library: {target.label()!r}; "
        f"library has {len(stored)} stored conditions"
    )


def map_conditions_to_library_blocks(
    current: list[OpticalCondition],
    stored: list[OpticalCondition],
) -> list[int]:
    """For each current condition, return its block index in the stored library."""
    return [find_stored_condition_index(stored, cond) for cond in current]


def stored_conditions_from_measurement(stored_measurement: dict) -> list[OpticalCondition]:
    return [optical_condition_from_dict(c) for c in stored_measurement["conditions"]]


def measurement_recipe_equal(stored: dict, current: dict) -> bool:
    return stored == current


def optical_common_compatible(stored: dict, current: dict) -> None:
    key = "optical_common"
    if stored.get(key) != current.get(key):
        raise ValueError(
            f"library optical settings mismatch: stored {stored.get(key)!r} vs config {current.get(key)!r}"
        )


def validate_condition_subset(stored_measurement: dict, cfg: ScatterometryConfig) -> list[int]:
    """Ensure every current condition exists in the library; return block indices."""
    stored_conds = stored_conditions_from_measurement(stored_measurement)
    current_conds = expand_measurement_conditions(cfg)
    if not current_conds:
        raise ValueError("no measurement conditions in config")
    if stored_measurement.get("n_orders") != n_orders(cfg):
        raise ValueError(
            f"library n_orders={stored_measurement.get('n_orders')} != config {n_orders(cfg)}"
        )
    return map_conditions_to_library_blocks(current_conds, stored_conds)


def library_slice_summary(stored_measurement: dict, cfg: ScatterometryConfig) -> dict:
    stored_conds = stored_conditions_from_measurement(stored_measurement)
    current_conds = expand_measurement_conditions(cfg)
    cur_meta = recipe_meta_dict(cfg)
    block_indices = map_conditions_to_library_blocks(current_conds, stored_conds)
    exact = measurement_recipe_equal(stored_measurement, cur_meta)
    return {
        "exact_recipe_match": exact,
        "n_stored_conditions": len(stored_conds),
        "n_current_conditions": len(current_conds),
        "condition_map": [
            {
                "current": asdict(cur),
                "stored_index": int(s_idx),
                "stored_label": stored_conds[s_idx].label(),
            }
            for cur, s_idx in zip(current_conds, block_indices)
        ],
    }


def format_library_slice_summary(summary: dict) -> str:
    if summary["exact_recipe_match"]:
        return (
            f"library recipe: exact match ({summary['n_current_conditions']} conditions)"
        )
    lines = [
        "library recipe: subset slice "
        f"({summary['n_current_conditions']} / {summary['n_stored_conditions']} conditions)"
    ]
    for row in summary["condition_map"][:8]:
        cur = row["current"]
        q = cur.get("harmonic_order")
        q_part = f"q{q}_" if q is not None else ""
        cur_label = (
            f"{q_part}{cur['wl_nm']:.4g}nm_th{cur['angle_deg']:g}_az{cur['azimuth_deg']:g}"
        )
        lines.append(
            f"  {cur_label} -> stored[{row['stored_index']}] ({row['stored_label']})"
        )
    extra = len(summary["condition_map"]) - 8
    if extra > 0:
        lines.append(f"  ... ({extra} more)")
    return "\n".join(lines)
