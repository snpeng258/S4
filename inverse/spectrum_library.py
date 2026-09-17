"""Spectral library: grid generation, NPZ I/O, meta validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from config import GridAxisConfig, ScatterometryConfig
from library_slice import (
    library_slice_summary,
    measurement_recipe_equal,
    optical_common_compatible,
    validate_condition_subset,
)
from recipe import n_observables, n_orders, recipe_meta_dict

ROOT = Path(__file__).resolve().parent


@dataclass
class SpectralLibrary:
    params: np.ndarray
    R: np.ndarray
    param_names: list[str]
    grid: dict[str, GridAxisConfig]
    meta: dict

    @property
    def size(self) -> int:
        return int(self.params.shape[0])


def library_path(cfg: ScatterometryConfig) -> Path:
    p = Path(cfg.library.file)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def count_grid_points(grid: dict[str, GridAxisConfig]) -> int:
    n = 1
    for ax in grid.values():
        n *= len(ax.points())
    return n


def iter_grid_params(
    param_names: list[str],
    grid: dict[str, GridAxisConfig],
) -> Iterator[np.ndarray]:
    axes = [grid[name].points() for name in param_names]
    for combo in np.ndindex(*[len(a) for a in axes]):
        yield np.array([axes[i][combo[i]] for i in range(len(param_names))], dtype=float)


def build_meta(cfg: ScatterometryConfig) -> dict:
    s = cfg.structure
    meta = {
        "pitch_nm": s.pitch_nm,
        "n_slices": s.n_slices,
        "grating_material": s.grating_material,
        "substrate_material": s.substrate_material,
        "param_names": list(cfg.inverse.param_names),
        "measurement": recipe_meta_dict(cfg),
    }
    return meta


def _meta_compatible(stored: dict, cfg: ScatterometryConfig) -> None:
    cur = build_meta(cfg)
    for key in ("pitch_nm", "n_slices", "grating_material", "substrate_material", "param_names"):
        if stored.get(key) != cur.get(key):
            raise ValueError(f"library meta mismatch on {key}: {stored.get(key)!r} vs {cur.get(key)!r}")

    stored_m = stored.get("measurement")
    cur_m = cur.get("measurement")
    if stored_m is None or cur_m is None:
        raise ValueError("library or config missing measurement meta")

    optical_common_compatible(stored_m, cur_m)

    if measurement_recipe_equal(stored_m, cur_m):
        return

    validate_condition_subset(stored_m, cfg)


def save_library(path: Path, lib: SpectralLibrary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_steps = {k: np.array([v.min, v.max, v.step]) for k, v in lib.grid.items()}
    np.savez_compressed(
        path,
        params=lib.params,
        R=lib.R,
        param_names=np.array(lib.param_names, dtype=object),
        meta_json=json.dumps(lib.meta),
        **{f"grid_{k}": grid_steps[k] for k in lib.grid},
    )


def load_library(path: Path, cfg: ScatterometryConfig | None = None) -> SpectralLibrary:
    if not path.is_file():
        raise FileNotFoundError(f"spectral library not found: {path}")
    data = np.load(path, allow_pickle=True)
    param_names = [str(x) for x in data["param_names"].tolist()]
    meta = json.loads(str(data["meta_json"]))
    grid: dict[str, GridAxisConfig] = {}
    for name in param_names:
        key = f"grid_{name}"
        if key in data:
            mn, mx, st = data[key].tolist()
            grid[name] = GridAxisConfig(min=float(mn), max=float(mx), step=float(st))
    lib = SpectralLibrary(
        params=np.asarray(data["params"], dtype=float),
        R=np.asarray(data["R"], dtype=float),
        param_names=param_names,
        grid=grid,
        meta=meta,
    )
    if cfg is not None:
        validate_library(lib, cfg)
    return lib


def validate_library(lib: SpectralLibrary, cfg: ScatterometryConfig) -> None:
    _meta_compatible(lib.meta, cfg)
    if list(cfg.inverse.param_names) != lib.param_names:
        raise ValueError(
            f"library param_names {lib.param_names} != config {cfg.inverse.param_names}; "
            "rebuild the library after unifying SWA (one swa_deg axis, left=right)"
        )

    stored_m = lib.meta["measurement"]
    cur_m = recipe_meta_dict(cfg)
    n_block = n_orders(cfg)

    if measurement_recipe_equal(stored_m, cur_m):
        expected = n_observables(cfg)
        if lib.R.shape[1] != expected:
            raise ValueError(
                f"library R width {lib.R.shape[1]} != expected {expected} "
                f"({len(cur_m['conditions'])} conditions × {n_block} orders)"
            )
        return

    n_stored_full = int(stored_m["n_conditions"]) * int(stored_m["n_orders"])
    if lib.R.shape[1] != n_stored_full:
        raise ValueError(
            f"library R width {lib.R.shape[1]} != stored full width {n_stored_full}"
        )
    validate_condition_subset(stored_m, cfg)


def library_match_measurement(lib: SpectralLibrary) -> dict:
    """Measurement meta used for column slicing (always the build-time recipe)."""
    return lib.meta["measurement"]


def describe_library_slice(lib: SpectralLibrary, cfg: ScatterometryConfig) -> dict:
    return library_slice_summary(lib.meta["measurement"], cfg)


def oracle_nearest_error(lib: SpectralLibrary, cfg: ScatterometryConfig) -> dict[str, float]:
    """Distance from true structure params to nearest library grid point."""
    names = lib.param_names
    true = np.array([cfg.structure.get_param(n) for n in names], dtype=float)
    diff = lib.params - true
    dist2 = np.sum(diff * diff, axis=1)
    i = int(np.argmin(dist2))
    nearest = lib.params[i]
    out: dict[str, float] = {"nearest_index": float(i)}
    for j, n in enumerate(names):
        out[f"delta_{n}"] = float(nearest[j] - true[j])
        out[f"abs_delta_{n}"] = float(abs(nearest[j] - true[j]))
    return out
