"""Measurement recipe: HHG harmonics, condition expansion, config helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np

from config import GridAxisConfig, MeasurementRecipe, OpticalConfig, ScatterometryConfig

EVAL_TASKS = ("inverse", "scan_sweep", "fim_study")


@dataclass(frozen=True)
class OpticalCondition:
    wl_nm: float
    angle_deg: float
    azimuth_deg: float
    harmonic_order: int | None = None

    def label(self) -> str:
        q = f"q{self.harmonic_order}" if self.harmonic_order is not None else "wl"
        return f"{q}_{self.wl_nm:.4g}nm_th{self.angle_deg:g}_az{self.azimuth_deg:g}"


def _as_float_list(val: list[float] | float | None, fallback: float) -> list[float]:
    if val is None:
        return [float(fallback)]
    if isinstance(val, (int, float)):
        return [float(val)]
    if not val:
        return [float(fallback)]
    return [float(x) for x in val]


def expand_harmonic_orders(recipe: MeasurementRecipe) -> list[int]:
    if recipe.harmonic_orders:
        orders = [int(q) for q in recipe.harmonic_orders]
    elif recipe.harmonic_order_ranges:
        orders: list[int] = []
        for band in recipe.harmonic_order_ranges:
            for q in range(int(band.min), int(band.max) + 1):
                if band.odd_only and q % 2 == 0:
                    continue
                orders.append(q)
        orders = sorted(set(orders))
    else:
        return []
    return [q for q in orders if q > 0]


def recipe_wavelength_pairs(
    recipe: MeasurementRecipe, optical: OpticalConfig
) -> list[tuple[float, int | None]]:
    if recipe.wavelengths_nm:
        return [(float(w), None) for w in recipe.wavelengths_nm]
    orders = expand_harmonic_orders(recipe)
    if orders:
        f0 = recipe.fundamental_nm
        return [(f0 / q, q) for q in orders]
    return [(optical.wl_nm, None)]


def expand_measurement_conditions(cfg: ScatterometryConfig) -> list[OpticalCondition]:
    """Full Cartesian product for inverse / library."""
    o = cfg.optical
    if o.recipe is None:
        return [OpticalCondition(o.wl_nm, o.angle_deg, o.azimuth_deg)]
    r = o.recipe
    pairs = recipe_wavelength_pairs(r, o)
    angles = _as_float_list(r.angles_deg, o.angle_deg)
    azimuths = _as_float_list(r.azimuths_deg, o.azimuth_deg)
    out: list[OpticalCondition] = []
    for wl, q in pairs:
        for ang in angles:
            for az in azimuths:
                out.append(OpticalCondition(wl, ang, az, q))
    return out


def scan_measurement_recipe(cfg: ScatterometryConfig) -> MeasurementRecipe | None:
    """Recipe for scan_sweep: scan.recipe if set, else optical.recipe."""
    if cfg.scan.recipe is not None:
        return cfg.scan.recipe
    return cfg.optical.recipe


def scan_recipe_source(cfg: ScatterometryConfig) -> str:
    return "scan.recipe" if cfg.scan.recipe is not None else "optical.recipe"


def scan_recipe_summary(cfg: ScatterometryConfig) -> dict:
    o = cfg.optical
    recipe = scan_measurement_recipe(cfg)
    grid = expand_scan_grid(cfg)
    out: dict = {
        "source": scan_recipe_source(cfg),
        "scan_axes": list(cfg.scan.axes or ["wavelength"]),
        "n_grid_points": len(grid),
    }
    if recipe is None:
        out["legacy_optical"] = {
            "wl_nm": o.wl_nm,
            "angle_deg": o.angle_deg,
            "azimuth_deg": o.azimuth_deg,
        }
        return out
    out["recipe"] = {
        "fundamental_nm": recipe.fundamental_nm,
        "harmonic_orders": expand_harmonic_orders(recipe) or recipe.harmonic_orders,
        "wavelengths_nm": recipe.wavelengths_nm,
        "angles_deg": _as_float_list(recipe.angles_deg, o.angle_deg),
        "azimuths_deg": _as_float_list(recipe.azimuths_deg, o.azimuth_deg),
    }
    return out


def expand_scan_grid(cfg: ScatterometryConfig) -> list[OpticalCondition]:
    """Scan grid: scan.recipe (or optical.recipe); only scan.axes vary."""
    o = cfg.optical
    scan = cfg.scan
    axes = {a.lower() for a in (scan.axes or ["wavelength"])}
    recipe = scan_measurement_recipe(cfg)
    if recipe is None:
        return [OpticalCondition(o.wl_nm, o.angle_deg, o.azimuth_deg)]
    pairs = recipe_wavelength_pairs(recipe, o)
    angles = _as_float_list(recipe.angles_deg, o.angle_deg)
    azimuths = _as_float_list(recipe.azimuths_deg, o.azimuth_deg)
    wl_pairs = pairs if "wavelength" in axes else [pairs[0]]
    ang_vals = angles if "angle" in axes else [angles[0]]
    az_vals = azimuths if "azimuth" in axes else [azimuths[0]]
    out: list[OpticalCondition] = []
    for wl, q in wl_pairs:
        for ang in ang_vals:
            for az in az_vals:
                out.append(OpticalCondition(wl, ang, az, q))
    return out


def cfg_with_condition(cfg: ScatterometryConfig, cond: OpticalCondition) -> ScatterometryConfig:
    out = cfg.copy()
    out.optical.wl_nm = cond.wl_nm
    out.optical.angle_deg = cond.angle_deg
    out.optical.azimuth_deg = cond.azimuth_deg
    return out


def n_orders(cfg: ScatterometryConfig) -> int:
    o = cfg.optical
    return o.order_max - o.order_min + 1


def n_observables(cfg: ScatterometryConfig) -> int:
    return len(expand_measurement_conditions(cfg)) * n_orders(cfg)


def recipe_meta_dict(cfg: ScatterometryConfig) -> dict:
    o = cfg.optical
    conditions = expand_measurement_conditions(cfg)
    meta: dict = {
        "n_conditions": len(conditions),
        "n_orders": n_orders(cfg),
        "conditions": [asdict(c) for c in conditions],
    }
    if o.recipe is not None:
        r = o.recipe
        meta["recipe"] = {
            "fundamental_nm": r.fundamental_nm,
            "harmonic_orders": expand_harmonic_orders(r) or r.harmonic_orders,
            "wavelengths_nm": r.wavelengths_nm,
            "angles_deg": _as_float_list(r.angles_deg, o.angle_deg),
            "azimuths_deg": _as_float_list(r.azimuths_deg, o.azimuth_deg),
        }
    else:
        meta["recipe"] = None
        meta["legacy_optical"] = {
            "wl_nm": o.wl_nm,
            "angle_deg": o.angle_deg,
            "azimuth_deg": o.azimuth_deg,
        }
    meta["optical_common"] = {
        "polarization": o.polarization,
        "pol_s_amp": o.pol_s_amp,
        "pol_s_phase_deg": o.pol_s_phase_deg,
        "pol_p_amp": o.pol_p_amp,
        "pol_p_phase_deg": o.pol_p_phase_deg,
        "NG": o.NG,
        "order_min": o.order_min,
        "order_max": o.order_max,
    }
    return meta


def iter_structure_sweep(cfg: ScatterometryConfig) -> Iterator[tuple[dict[str, float], ScatterometryConfig]]:
    sweep = cfg.scan.structure_sweep
    if not sweep:
        yield {}, cfg.copy()
        return
    names = sorted(sweep.keys())
    axes = [sweep[n] for n in names]
    for combo in np.ndindex(*[len(ax.points()) for ax in axes]):
        delta = {names[i]: float(axes[i].points()[combo[i]]) for i in range(len(names))}
        trial = cfg.copy()
        for n, v in delta.items():
            trial.structure.set_param(n, v)
        yield delta, trial
