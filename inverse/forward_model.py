"""Forward model: structure + optical -> S4 -> R_m vector."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import ScatterometryConfig, load_config, parse_config_arg
from order_collection import collection_summary, stored_order_m_vector
from recipe import cfg_with_condition, expand_measurement_conditions, n_orders
from s4_runner import S4Runner, reflectivity_vector
from noise_model import add_detector_noise


_runner: S4Runner | None = None


def _get_runner(cfg: ScatterometryConfig) -> S4Runner:
    global _runner
    if _runner is None:
        _runner = S4Runner(cfg)
    return _runner


def reset_runner() -> None:
    global _runner
    _runner = None


def _simulate_condition_at_index(
    cfg: ScatterometryConfig,
    cond_idx: int,
) -> tuple[int, np.ndarray]:
    conditions = expand_measurement_conditions(cfg)
    trial = cfg_with_condition(cfg, conditions[cond_idx])
    runner = S4Runner(trial)
    r_map = runner.run_reflection(trial)
    vec = reflectivity_vector(r_map, trial.optical)
    return cond_idx, vec


def simulate_reflectivity(
    cfg: ScatterometryConfig,
) -> tuple[np.ndarray, dict[int, float]]:
    """Single optical condition (cfg.optical wl/angle/azimuth)."""
    runner = _get_runner(cfg)
    r_map = runner.run_reflection(cfg)
    vec = reflectivity_vector(r_map, cfg.optical)
    return vec, r_map


def simulate_reflectivity_multi(
    cfg: ScatterometryConfig,
    *,
    condition_workers: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    """All recipe conditions concatenated into one R_m vector."""
    conditions = expand_measurement_conditions(cfg)
    if len(conditions) == 1 and cfg.optical.recipe is None:
        vec, r_map = simulate_reflectivity(cfg)
        return vec, [{"condition": conditions[0].label(), "r_map": r_map}]

    n_cond = len(conditions)
    if condition_workers <= 1 or n_cond <= 1:
        parts: list[np.ndarray] = []
        details: list[dict] = []
        for cond in conditions:
            trial = cfg_with_condition(cfg, cond)
            vec, r_map = simulate_reflectivity(trial)
            parts.append(vec)
            details.append({
                "condition": cond.label(),
                "harmonic_order": cond.harmonic_order,
                "wl_nm": cond.wl_nm,
                "angle_deg": cond.angle_deg,
                "azimuth_deg": cond.azimuth_deg,
                "r_map": r_map,
            })
        return np.concatenate(parts), details

    cfg_path = cfg.config_path
    if cfg_path is None:
        raise ValueError("condition_workers>1 requires cfg.config_path for grid pool workers")
    parts: list[np.ndarray | None] = [None] * n_cond
    with ThreadPoolExecutor(max_workers=int(condition_workers)) as pool:
        futures = [
            pool.submit(_simulate_condition_at_index, cfg, i)
            for i in range(n_cond)
        ]
        for fut in as_completed(futures):
            idx, vec = fut.result()
            parts[idx] = vec
    assert all(p is not None for p in parts)
    return np.concatenate(parts), []


def noise_db_to_fraction(db: float) -> float:
    """Amplitude noise floor relative to peak R: sigma/R_peak = 10^(db/20)."""
    return float(10.0 ** (db / 20.0))


def noise_fraction_to_db(fraction: float) -> float:
    if fraction <= 0.0:
        raise ValueError("noise fraction must be positive")
    return float(20.0 * np.log10(fraction))


def generate_synthetic_measurement(
    cfg: ScatterometryConfig,
    noise_level: float | None = None,
    rng: np.random.Generator | None = None,
    noise_db: float | None = None,
    *,
    noiseless: bool = False,
    n0_electrons: float | None = None,
) -> np.ndarray:
    """Full stored-order vector; inverse applies collectible mask.

    Default: per-order detector model (Poisson shot + camera floor + optional
    flicker). m=0 and |m|>=1 use inverse.noise.zero / .first. Legacy additive
    peak-relative noise is used only if noise_db / noise_level is passed, or if
    cfg.inverse.noise.apply is false and noise_level>0.
    """
    vec, _ = simulate_reflectivity_multi(cfg)
    if noiseless:
        return vec.copy()
    rng = rng or np.random.default_rng()
    noise = cfg.inverse.noise
    use_legacy = noise_db is not None or (
        noise_level is not None and noise_level > 0 and not noise.apply
    )
    if noise_db is not None:
        noise_level = noise_db_to_fraction(noise_db)
        use_legacy = True
    if use_legacy:
        level = float(noise_level or 0.0)
        if level <= 0:
            return vec.copy()
        scale = level * max(float(vec.max()), 1e-12)
        noisy = vec + rng.normal(0.0, scale, size=vec.shape)
        return np.clip(noisy, 0.0, None)
    if not noise.apply:
        return vec.copy()
    return add_detector_noise(
        vec,
        noise,
        rng,
        stored_order_m_vector(cfg),
        n0_electrons=n0_electrons,
    )


def save_reflectivity_tsv(path: Path, r_map: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["m\tR_abs"]
    for m in sorted(r_map):
        lines.append(f"{m}\t{r_map[m]:.10g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg_path = parse_config_arg()
    cfg = load_config(cfg_path)
    out_dir = (Path(__file__).resolve().parent / cfg.paths.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s = cfg.structure
    o = cfg.optical
    print(f"Config: {cfg_path}")
    print(
        f"Structure: pitch={s.pitch_nm} cd={s.cd_nm} depth={s.depth_nm} "
        f"LSWA={s.lswa_deg} RSWA={s.rswa_deg} n_slices={s.n_slices}"
    )
    n_cond = len(expand_measurement_conditions(cfg))
    print(
        f"Optical: conditions={n_cond} pol={o.polarization} NG={o.NG} "
        f"orders=[{o.order_min},{o.order_max}]"
    )
    coll = collection_summary(cfg)
    print(
        f"Collectible (inverse): mode={coll['mode']!r} "
        f"{coll['n_collectible']} / {n_cond * n_orders(cfg)} observables"
    )

    vec, details = simulate_reflectivity_multi(cfg)
    runner = _get_runner(cfg)
    print(f"S4 calls: {runner.call_count}")
    print(f"R_m vector length: {len(vec)}, sum(R_m)={vec.sum():.6g}")

    if details and "r_map" in details[0]:
        save_reflectivity_tsv(out_dir / "forward_Rm.tsv", details[0]["r_map"])
    meta = {
        "config": str(cfg_path),
        "structure": s.__dict__,
        "n_conditions": n_cond,
        "R_total": float(vec.sum()),
    }
    (out_dir / "forward_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'forward_meta.json'}")


if __name__ == "__main__":
    main()
