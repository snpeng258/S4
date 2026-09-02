"""Run S4 RCWA via subprocess and parse reflectivity TSV."""
from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

from config import OpticalConfig, PathsConfig, ScatterometryConfig, StructureConfig
from structure_model import slice_grating


def find_s4_binary(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"S4 binary not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    for base in [here, *here.parents]:
        for cand in (base / "upstream" / "build" / "S4", base / "build" / "S4"):
            if cand.is_file():
                return cand
    raise FileNotFoundError("upstream/build/S4 not found; set paths.s4_bin in config.yaml")


def _fmt_csv(values: list[float]) -> str:
    return ",".join(f"{v:.12g}" for v in values)


def build_s4_arg(cfg: ScatterometryConfig) -> str:
    s, o = cfg.structure, cfg.optical
    layers = slice_grating(s)
    parts = [
        f"wl_nm={o.wl_nm}",
        f"angle_deg={o.angle_deg}",
        f"azimuth_deg={o.azimuth_deg}",
        f"period_nm={s.pitch_nm}",
        f"NG={o.NG}",
        f"n_slices={len(layers)}",
        f'duties="{_fmt_csv([x.duty for x in layers])}"',
        f'offsets="{_fmt_csv([x.offset_norm for x in layers])}"',
        f'slice_thicknesses="{_fmt_csv([x.thickness_norm for x in layers])}"',
        f'grating_mat="{s.grating_material}"',
        f'substrate_mat="{s.substrate_material}"',
        f"pol_s_amp={o.pol_s_amp}",
        f"pol_s_phase={o.pol_s_phase_deg}",
        f"pol_p_amp={o.pol_p_amp}",
        f"pol_p_phase={o.pol_p_phase_deg}",
    ]
    return ";".join(parts)


def parse_compare_tsv(text: str) -> dict[int, float]:
    block = re.search(r"#BEGIN_COMPARE_TSV\s*(.*?)\s*#END_COMPARE_TSV", text, re.S)
    if not block:
        raise ValueError("TSV block #BEGIN_COMPARE_TSV not found in S4 output")
    out: dict[int, float] = {}
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("m"):
            continue
        m_s, r_s = line.split()[:2]
        out[int(m_s)] = float(r_s)
    return out


def _cache_key(cfg: ScatterometryConfig) -> str:
    s, o = cfg.structure, cfg.optical
    layers = slice_grating(s)
    layer_sig = tuple((x.duty, x.offset_norm, x.thickness_norm) for x in layers)
    return (
        f"{s.pitch_nm}|{s.cd_nm}|{s.depth_nm}|{s.lswa_deg}|{s.rswa_deg}|{s.n_slices}|"
        f"{o.wl_nm}|{o.angle_deg}|{o.azimuth_deg}|{o.NG}|{o.pol_s_amp}|{o.pol_p_amp}|{layer_sig}"
    )


class S4Runner:
    def __init__(self, cfg: ScatterometryConfig):
        self.cfg = cfg
        self.root = Path(__file__).resolve().parent
        self.s4_bin = find_s4_binary(cfg.paths.s4_bin)
        lua_rel = cfg.paths.lua_script
        self.lua = (self.root / lua_rel).resolve()
        db = Path(cfg.paths.materials_db).expanduser()
        if not db.is_absolute():
            db = (self.root / db).resolve()
        if not (db / "load_materials.lua").is_file():
            for base in [self.root, *self.root.parents]:
                shared = (base / "shared").resolve()
                if (shared / "load_materials.lua").is_file():
                    db = shared
                    break
        self.materials_db = db
        self._cache: dict[str, dict[int, float]] = {}
        self.cache_size = max(0, int(cfg.paths.cache_size))
        self.call_count = 0

    def run_reflection(self, cfg: ScatterometryConfig | None = None) -> dict[int, float]:
        cfg = cfg or self.cfg
        key = _cache_key(cfg)
        if key in self._cache:
            return self._cache[key]

        env = os.environ.copy()
        env["S4_MATERIALS_DB"] = str(self.materials_db)
        s4_arg = build_s4_arg(cfg)
        try:
            proc = subprocess.run(
                [str(self.s4_bin), str(self.lua), "-a", s4_arg],
                capture_output=True,
                text=True,
                cwd=str(self.root),
                env=env,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"S4 timed out after 180s\narg={s4_arg[:200]}...") from exc
        self.call_count += 1
        if proc.returncode != 0:
            raise RuntimeError(
                f"S4 failed (code {proc.returncode})\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
            )
        result = parse_compare_tsv(proc.stdout)
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
        return result


def orders_to_vector(r_map: dict[int, float], order_min: int, order_max: int) -> list[int]:
    return list(range(order_min, order_max + 1))


def reflectivity_vector(r_map: dict[int, float], optical: OpticalConfig) -> "numpy.ndarray":
    import numpy as np

    ms = range(optical.order_min, optical.order_max + 1)
    return np.array([r_map.get(m, 0.0) for m in ms], dtype=float)
