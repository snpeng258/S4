"""Scatterometry configuration: dataclasses + YAML load/save."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

import numpy as np

from noise_model import DetectorNoiseConfig, parse_detector_noise

PARAM_NAMES = ("pitch_nm", "cd_nm", "depth_nm", "lswa_deg", "rswa_deg")
INVERSE_METHODS = ("ga_lm", "lib_pop_ga_lm", "lib_pop_rand_ga_lm")
EVAL_TASKS = ("inverse", "scan_sweep", "fim_study")
DEFAULT_FIM_MASKS = (
    "prop",
    "decoupling",
    "m0_all",
    "only90",
    "no90",
    "near90",
    "far",
    "mid",
    "two_cam",
    "drop_phi45",
)
DEFAULT_FIM_WINDOWS = {
    "only90": [90.0],
    "near90": [60.0, 90.0],
    "far": [0.0, 30.0],
    "mid": [30.0, 45.0],
    "two_cam": [0.0, 30.0, 60.0, 90.0],
}


@dataclass
class StructureConfig:
    pitch_nm: float = 80.0
    cd_nm: float = 44.0
    depth_nm: float = 44.0
    lswa_deg: float = 94.0
    rswa_deg: float = 94.0
    n_slices: int = 10
    grating_material: str = "Au"
    substrate_material: str = "SiO2"

    def get_param(self, name: str) -> float:
        if name not in PARAM_NAMES:
            raise KeyError(f"unknown structure param: {name}")
        return float(getattr(self, name))

    def set_param(self, name: str, value: float) -> None:
        if name not in PARAM_NAMES:
            raise KeyError(f"unknown structure param: {name}")
        setattr(self, name, float(value))

    def copy_with(self, **kwargs: Any) -> "StructureConfig":
        data = asdict(self)
        data.update(kwargs)
        return StructureConfig(**data)


@dataclass
class HarmonicOrderRange:
    min: int
    max: int
    odd_only: bool = True


def _parse_harmonic_order_ranges(bands: Any) -> list[HarmonicOrderRange]:
    """YAML harmonic_order_ranges: list of {min,max,odd_only}; single band may omit '-'."""
    if bands is None:
        return []
    if isinstance(bands, dict):
        if "min" in bands and "max" in bands:
            return [HarmonicOrderRange(**bands)]
        raise ValueError(
            "harmonic_order_ranges dict must be one band {min, max, odd_only}; "
            "for multiple bands use a YAML list with '-' per item"
        )
    if isinstance(bands, list):
        out: list[HarmonicOrderRange] = []
        for i, b in enumerate(bands):
            if not isinstance(b, dict):
                raise ValueError(
                    f"harmonic_order_ranges[{i}] must be a mapping {{min, max, odd_only}}, got {type(b).__name__!r}; "
                    "did you forget the '-' list prefix in config.yaml?"
                )
            out.append(HarmonicOrderRange(**b))
        return out
    raise ValueError(f"harmonic_order_ranges must be a list or dict, got {type(bands).__name__!r}")


@dataclass
class MeasurementRecipe:
    fundamental_nm: float = 800.0
    harmonic_order_ranges: list[HarmonicOrderRange] | None = None
    harmonic_orders: list[int] | None = None
    wavelengths_nm: list[float] | None = None
    angles_deg: list[float] | None = None
    azimuths_deg: list[float] | None = None


@dataclass
class ScanConfig:
    axes: list[str] = field(default_factory=lambda: ["wavelength"])
    compute_jacobian: bool = True
    plot: bool = True
    structure_sweep: dict[str, GridAxisConfig] | None = None
    # 若设置，scan_sweep 仅用此 recipe，与 optical.recipe（逆问题/建库）独立
    recipe: MeasurementRecipe | None = None


@dataclass
class OpticalConfig:
    wl_nm: float = 13.5
    angle_deg: float = 70.0
    azimuth_deg: float = 90.0
    recipe: MeasurementRecipe | None = None
    polarization: str = "TE"
    pol_s_amp: float = 1.0
    pol_s_phase_deg: float = 0.0
    pol_p_amp: float = 0.0
    pol_p_phase_deg: float = 0.0
    NG: int = 31
    order_min: int = -15
    order_max: int = 15

    def apply_polarization_preset(self) -> None:
        pol = self.polarization.upper()
        if pol == "TE":
            self.pol_s_amp, self.pol_s_phase_deg = 1.0, 0.0
            self.pol_p_amp, self.pol_p_phase_deg = 0.0, 0.0
        elif pol == "TM":
            self.pol_s_amp, self.pol_s_phase_deg = 0.0, 0.0
            self.pol_p_amp, self.pol_p_phase_deg = 1.0, 0.0


@dataclass
class PathsConfig:
    s4_bin: str | None = None
    lua_script: str = "lua/grating_slice_reflection.lua"
    materials_db: str = "../shared"
    output_dir: str = "../runs/inverse"
    cache_size: int = 128


@dataclass
class GridAxisConfig:
    min: float
    max: float
    step: float

    def points(self) -> np.ndarray:
        n = int(round((self.max - self.min) / self.step)) + 1
        return np.linspace(self.min, self.max, n, dtype=float)


@dataclass
class LibraryPriorConfig:
    enabled: bool = False
    lambda_prior: float = 0.01
    swa_deg_range: list[float] = field(default_factory=lambda: [85.0, 90.0])
    penalize_asymmetry: bool = False
    hard_reject_out_of_range: bool = True


def _default_library_grid() -> dict[str, GridAxisConfig]:
    return {
        "cd_nm": GridAxisConfig(32.0, 48.0, 1.0),
        "depth_nm": GridAxisConfig(32.0, 48.0, 1.0),
        "lswa_deg": GridAxisConfig(85.0, 95.0, 1.0),
        "rswa_deg": GridAxisConfig(85.0, 95.0, 1.0),
    }


@dataclass
class LibraryConfig:
    file: str = "../data/inverse/spectra.npz"
    build_workers: int = 1
    condition_workers: int = 1
    checkpoint_every: int = 500
    grid: dict[str, GridAxisConfig] = field(default_factory=_default_library_grid)
    prior: LibraryPriorConfig = field(default_factory=LibraryPriorConfig)


@dataclass
class DecouplingRoleSpec:
    azimuth_deg: float = 90.0
    azimuths_deg: list[float] | None = None
    orders: list[int] = field(default_factory=lambda: [0])


@dataclass
class DecouplingRolesConfig:
    depth_anchor: DecouplingRoleSpec = field(
        default_factory=lambda: DecouplingRoleSpec(azimuth_deg=90.0, orders=[0])
    )
    swa: DecouplingRoleSpec = field(
        default_factory=lambda: DecouplingRoleSpec(azimuth_deg=90.0, orders=[-1, 1])
    )
    lateral: DecouplingRoleSpec = field(
        default_factory=lambda: DecouplingRoleSpec(azimuths_deg=None, orders=[0])
    )


@dataclass
class DecouplingStaticWeights:
    depth_anchor: float = 1.0
    swa: float = 0.8
    lateral: float = 1.0
    aux: float = 0.2


@dataclass
class DecouplingDynamicLmConfig:
    enabled: bool = True
    refresh_every_nfev: int = 20
    inner_max_nfev: int = 15
    outer_max_rounds: int = 10
    coupling_eps: float = 1e-3
    cd_reg_lambda0: float = 0.01
    boost_lateral_when_coupled: float = 2.0
    use_kx0_shadow_prior: bool = True


@dataclass
class DecouplingConfig:
    enabled: bool = False
    azimuth_tol_deg: float = 0.5
    roles: DecouplingRolesConfig = field(default_factory=DecouplingRolesConfig)
    static_weights: DecouplingStaticWeights = field(default_factory=DecouplingStaticWeights)
    dynamic_lm: DecouplingDynamicLmConfig = field(default_factory=DecouplingDynamicLmConfig)
    cd_anchor: str = "ga"


def _parse_decoupling_role(raw: dict | None, *, default_azimuth: float | None = 90.0) -> DecouplingRoleSpec:
    if not raw:
        az = default_azimuth if default_azimuth is not None else 90.0
        return DecouplingRoleSpec(azimuth_deg=az)
    orders = [int(m) for m in raw.get("orders", [0])]
    if "azimuths_deg" in raw:
        az_list = raw["azimuths_deg"]
        if az_list is None:
            return DecouplingRoleSpec(azimuths_deg=None, orders=orders)
        if isinstance(az_list, (int, float)):
            return DecouplingRoleSpec(azimuths_deg=[float(az_list)], orders=orders)
        return DecouplingRoleSpec(azimuths_deg=[float(a) for a in az_list], orders=orders)
    az = float(raw.get("azimuth_deg", default_azimuth if default_azimuth is not None else 90.0))
    return DecouplingRoleSpec(azimuth_deg=az, orders=orders)


def _parse_decoupling(raw: dict | None) -> DecouplingConfig:
    if not raw:
        return DecouplingConfig()
    roles_raw = raw.get("roles") or {}
    static_raw = raw.get("static_weights") or {}
    dyn_raw = raw.get("dynamic_lm") or {}
    return DecouplingConfig(
        enabled=bool(raw.get("enabled", False)),
        azimuth_tol_deg=float(raw.get("azimuth_tol_deg", 0.5)),
        roles=DecouplingRolesConfig(
            depth_anchor=_parse_decoupling_role(roles_raw.get("depth_anchor"), default_azimuth=90.0),
            swa=_parse_decoupling_role(roles_raw.get("swa"), default_azimuth=90.0),
            lateral=_parse_decoupling_role(roles_raw.get("lateral"), default_azimuth=None),
        ),
        static_weights=DecouplingStaticWeights(
            depth_anchor=float(static_raw.get("depth_anchor", 1.0)),
            swa=float(static_raw.get("swa", 0.8)),
            lateral=float(static_raw.get("lateral", 1.0)),
            aux=float(static_raw.get("aux", 0.2)),
        ),
        dynamic_lm=DecouplingDynamicLmConfig(
            enabled=bool(dyn_raw.get("enabled", True)),
            refresh_every_nfev=int(dyn_raw.get("refresh_every_nfev", 20)),
            inner_max_nfev=int(dyn_raw.get("inner_max_nfev", 15)),
            outer_max_rounds=int(dyn_raw.get("outer_max_rounds", 10)),
            coupling_eps=float(dyn_raw.get("coupling_eps", 1e-3)),
            cd_reg_lambda0=float(dyn_raw.get("cd_reg_lambda0", 0.01)),
            boost_lateral_when_coupled=float(dyn_raw.get("boost_lateral_when_coupled", 2.0)),
            use_kx0_shadow_prior=bool(dyn_raw.get("use_kx0_shadow_prior", True)),
        ),
        cd_anchor=str(raw.get("cd_anchor", "ga")).lower(),
    )


def decoupling_is_active(cfg: "ScatterometryConfig") -> bool:
    inv = cfg.inverse
    if inv.order_collection.lower() == "decoupling":
        return True
    return bool(inv.decoupling.enabled)


def role_weights_enabled(cfg: "ScatterometryConfig") -> bool:
    """Role / kx0 multipliers and dynamic LM. None => on only when decoupling is active."""
    flag = cfg.inverse.use_role_weights
    if flag is None:
        return decoupling_is_active(cfg)
    return bool(flag)


@dataclass
class InverseConfig:
    method: str = "ga_lm"
    # all | propagating | list | decoupling | prop | m0_all | only90
    # prop / m0_all / only90 match FIM masks (propagating ∩ m∈{-1,0,1}).
    order_collection: str = "propagating"
    accepted_orders: list[int] | None = None
    # None = auto (on iff decoupling is active). False = raw 1/σ, matches FIM.
    use_role_weights: bool | None = None
    param_names: list[str] = field(
        default_factory=lambda: ["cd_nm", "depth_nm", "lswa_deg", "rswa_deg"]
    )
    perturb_frac: float = 0.05
    bound_frac: float = 0.2
    reg_weight: float = 0.0
    noise_level: float = 0.0
    noise_threshold: float = 1.0
    noise: DetectorNoiseConfig = field(default_factory=DetectorNoiseConfig)
    use_ga: bool = True
    use_lm: bool = True
    ga_maxiter: int = 50
    ga_popsize: int = 15
    ga_mutation: tuple[float, float] = (0.5, 1.0)
    ga_recombination: float = 0.7
    ga_seed: int | None = None
    ga_workers: int = 1
    lm_max_nfev: int = 500
    lm_ftol: float = 1e-8
    lm_xtol: float = 1e-8
    lm_verbose: int = 2
    init_guess: dict[str, float] | None = None
    lb: dict[str, float] | None = None
    ub: dict[str, float] | None = None
    decoupling: DecouplingConfig = field(default_factory=DecouplingConfig)


def _default_fim_param_scales() -> dict[str, float]:
    return {
        "cd_nm": 1.0,
        "depth_nm": 1.0,
        "lswa_deg": 1.0,
        "rswa_deg": 1.0,
    }


def _default_fim_windows() -> dict[str, list[float]]:
    return {k: list(v) for k, v in DEFAULT_FIM_WINDOWS.items()}


def _default_fim_keep_orders() -> list[int]:
    return [-1, 0, 1]


@dataclass
class FimStudyConfig:
    """Layer-1 Jacobian / FIM study (row masks on one full-order J)."""

    eps_frac: float = 0.01
    keep_orders: list[int] = field(default_factory=_default_fim_keep_orders)
    masks: list[str] = field(default_factory=lambda: list(DEFAULT_FIM_MASKS))
    n0_electrons: list[float] | None = None
    flicker_a: list[float | None] = field(default_factory=lambda: [None, 0.0])
    param_scales: dict[str, float] = field(default_factory=_default_fim_param_scales)
    azimuth_windows: dict[str, list[float]] = field(default_factory=_default_fim_windows)
    azimuth_tol_deg: float = 0.5
    rcond: float = 1e-12
    n0_sweep_masks: list[str] = field(
        default_factory=lambda: ["prop", "decoupling", "only90"]
    )
    corr_masks: list[str] = field(
        default_factory=lambda: ["prop", "decoupling", "m0_all", "only90"]
    )


def _parse_fim(raw: dict | None) -> FimStudyConfig:
    if not raw:
        return FimStudyConfig()
    data = dict(raw)
    windows = _default_fim_windows()
    raw_windows = data.pop("azimuth_windows", None) or {}
    for name, vals in raw_windows.items():
        if vals is None:
            continue
        if isinstance(vals, (int, float)):
            windows[str(name)] = [float(vals)]
        else:
            windows[str(name)] = [float(v) for v in vals]
    scales = _default_fim_param_scales()
    raw_scales = data.pop("param_scales", None) or {}
    for name, val in raw_scales.items():
        scales[str(name)] = float(val)
    flicker_raw = data.pop("flicker_a", [None, 0.0])
    flicker: list[float | None] = []
    if flicker_raw is None:
        flicker = [None]
    else:
        for item in flicker_raw:
            flicker.append(None if item is None else float(item))
    n0_raw = data.pop("n0_electrons", None)
    n0_vals: list[float] | None
    if n0_raw is None:
        n0_vals = None
    else:
        n0_vals = [float(v) for v in n0_raw]
    keep_raw = data.pop("keep_orders", None)
    keep_orders = (
        [int(m) for m in keep_raw] if keep_raw is not None else _default_fim_keep_orders()
    )
    return FimStudyConfig(
        eps_frac=float(data.get("eps_frac", 0.01)),
        keep_orders=keep_orders,
        masks=[str(m) for m in data.get("masks", list(DEFAULT_FIM_MASKS))],
        n0_electrons=n0_vals,
        flicker_a=flicker,
        param_scales=scales,
        azimuth_windows=windows,
        azimuth_tol_deg=float(data.get("azimuth_tol_deg", 0.5)),
        rcond=float(data.get("rcond", 1e-12)),
        n0_sweep_masks=[str(m) for m in data.get("n0_sweep_masks", ["prop", "decoupling", "only90"])],
        corr_masks=[str(m) for m in data.get("corr_masks", ["prop", "decoupling", "m0_all", "only90"])],
    )


@dataclass
class EvalConfig:
    # task: inverse | scan_sweep | fim_study  (inverse_solver.py 与 evaluate.py 均读)
    task: str = "inverse"
    # mode 仅 evaluate.py 且 task=inverse: noise | timing | ga_workers | methods | all
    mode: str = "all"
    # Legacy peak-relative sweep: sigma/R_peak = 10^(dB/20). Ignored if noise_n0_electrons is set.
    noise_levels_db: list[float] = field(
        default_factory=lambda: [-40.0, -30.0, -25.0, -20.0, -15.0]
    )
    # Detector-noise sweep: photoelectrons at R=1 (shot term f/N0). Preferred over noise_levels_db.
    noise_n0_electrons: list[float] = field(
        default_factory=lambda: [1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7]
    )
    n_trials: int = 10
    ga_workers_sweep: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 8, 16, 24, 28, 32, 36, 40, 48, 56]
    )
    ga_workers_trials: int = 3
    ga_workers_seed_stride: int = 1
    plot: bool = True
    plot_dpi: int = 150


@dataclass
class ScatterometryConfig:
    structure: StructureConfig = field(default_factory=StructureConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    inverse: InverseConfig = field(default_factory=InverseConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    fim: FimStudyConfig = field(default_factory=FimStudyConfig)
    config_path: Path | None = None

    def copy(self) -> "ScatterometryConfig":
        return copy.deepcopy(self)


def load_config(path: str | Path) -> ScatterometryConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ScatterometryConfig()
    if "structure" in raw:
        cfg.structure = StructureConfig(**raw["structure"])
    if "optical" in raw:
        opt_raw = dict(raw["optical"])
        recipe_raw = opt_raw.pop("recipe", None)
        cfg.optical = OpticalConfig(**opt_raw)
        if recipe_raw:
            r_copy = dict(recipe_raw)
            bands = r_copy.pop("harmonic_order_ranges", None)
            if bands:
                r_copy["harmonic_order_ranges"] = _parse_harmonic_order_ranges(bands)
            cfg.optical.recipe = MeasurementRecipe(**r_copy)
    cfg.optical.apply_polarization_preset()
    if "paths" in raw:
        cfg.paths = PathsConfig(**raw["paths"])
    if "inverse" in raw:
        inv = dict(raw["inverse"])
        if "ga_mutation" in inv and isinstance(inv["ga_mutation"], list):
            inv["ga_mutation"] = tuple(inv["ga_mutation"])
        dec_raw = inv.pop("decoupling", None)
        dec = _parse_decoupling(dec_raw)
        noise_raw = inv.pop("noise", None)
        noise = parse_detector_noise(noise_raw)
        cfg.inverse = InverseConfig(**inv, decoupling=dec, noise=noise)
        if cfg.inverse.order_collection.lower() == "decoupling":
            cfg.inverse.decoupling.enabled = True
    if "eval" in raw:
        ev = dict(raw["eval"])
        if "noise_levels_db" not in ev and "noise_levels" in ev:
            import math

            ev["noise_levels_db"] = [
                20.0 * math.log10(f) if f > 0 else -120.0 for f in ev.pop("noise_levels")
            ]
        cfg.eval = EvalConfig(**ev)
    if "library" in raw:
        lib_raw = dict(raw["library"])
        grid_raw = lib_raw.pop("grid", {})
        prior_raw = lib_raw.pop("prior", {})
        prior = LibraryPriorConfig(**prior_raw) if prior_raw else LibraryPriorConfig()
        grid = (
            {k: GridAxisConfig(**v) for k, v in grid_raw.items()}
            if grid_raw
            else _default_library_grid()
        )
        cfg.library = LibraryConfig(
            file=lib_raw.get("file", "../data/inverse/spectra.npz"),
            build_workers=int(lib_raw.get("build_workers", 1)),
            condition_workers=int(lib_raw.get("condition_workers", 1)),
            checkpoint_every=int(lib_raw.get("checkpoint_every", 500)),
            grid=grid,
            prior=prior,
        )
    if "scan" in raw:
        sc = dict(raw["scan"])
        ss = sc.pop("structure_sweep", None)
        recipe_raw = sc.pop("recipe", None)
        if ss:
            sc["structure_sweep"] = {k: GridAxisConfig(**v) for k, v in ss.items()}
        cfg.scan = ScanConfig(**sc)
        if recipe_raw is not None:
            r_copy = dict(recipe_raw)
            bands = r_copy.pop("harmonic_order_ranges", None)
            if bands:
                r_copy["harmonic_order_ranges"] = _parse_harmonic_order_ranges(bands)
            cfg.scan.recipe = MeasurementRecipe(**r_copy)
    if "fim" in raw:
        cfg.fim = _parse_fim(raw["fim"])
    cfg.config_path = path
    return cfg


def _recipe_to_dict(recipe: MeasurementRecipe) -> dict:
    return {
        "fundamental_nm": recipe.fundamental_nm,
        "harmonic_orders": recipe.harmonic_orders,
        "wavelengths_nm": recipe.wavelengths_nm,
        "angles_deg": recipe.angles_deg,
        "azimuths_deg": recipe.azimuths_deg,
        "harmonic_order_ranges": (
            [asdict(b) for b in recipe.harmonic_order_ranges] if recipe.harmonic_order_ranges else None
        ),
    }


def _scan_to_dict(scan: ScanConfig) -> dict:
    d = asdict(scan)
    if scan.structure_sweep:
        d["structure_sweep"] = {k: asdict(v) for k, v in scan.structure_sweep.items()}
    if scan.recipe is not None:
        d["recipe"] = _recipe_to_dict(scan.recipe)
    else:
        d.pop("recipe", None)
    return d


def _optical_to_dict(optical: OpticalConfig) -> dict:
    d = asdict(optical)
    if optical.recipe is not None:
        d["recipe"] = _recipe_to_dict(optical.recipe)
    return d


def save_config(cfg: ScatterometryConfig, path: str | Path) -> None:
    path = Path(path)
    data = {
        "structure": asdict(cfg.structure),
        "optical": _optical_to_dict(cfg.optical),
        "paths": asdict(cfg.paths),
        "inverse": asdict(cfg.inverse),
        "library": {
            "file": cfg.library.file,
            "build_workers": cfg.library.build_workers,
            "condition_workers": cfg.library.condition_workers,
            "checkpoint_every": cfg.library.checkpoint_every,
            "grid": {k: asdict(v) for k, v in cfg.library.grid.items()},
            "prior": asdict(cfg.library.prior),
        },
        "scan": _scan_to_dict(cfg.scan),
        "eval": asdict(cfg.eval),
        "fim": asdict(cfg.fim),
    }
    inv = data["inverse"]
    if isinstance(inv.get("ga_mutation"), tuple):
        inv["ga_mutation"] = list(inv["ga_mutation"])
    d = cfg.inverse.decoupling
    inv["decoupling"] = {
        "enabled": d.enabled,
        "azimuth_tol_deg": d.azimuth_tol_deg,
        "roles": {
            "depth_anchor": asdict(d.roles.depth_anchor),
            "swa": asdict(d.roles.swa),
            "lateral": asdict(d.roles.lateral),
        },
        "static_weights": asdict(d.static_weights),
        "dynamic_lm": asdict(d.dynamic_lm),
        "cd_anchor": d.cd_anchor,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def parse_config_arg(argv: list[str] | None = None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="配置文件路径 (default: inverse/config.yaml)",
    )
    args, _ = parser.parse_known_args(argv)
    return args.config.resolve()
