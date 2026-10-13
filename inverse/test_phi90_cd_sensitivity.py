"""90°/0° zeroth-order CD sensitivity helpers (no S4)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from config import load_config
from run_phi90_cd_sensitivity import (
    Phi90Spec,
    dark_wavelengths_nm,
    dry_run_text,
    fresnel_rs,
    harmonic_markers,
    load_material_eps,
    n_s4_calls,
    parse_phi90,
    reflection_height_period_nm,
    replot,
    rms_partial,
    save_npz,
    sensitivity_table,
    summarize,
    two_beam_R0,
)

HERE = Path(__file__).resolve().parent
CFG = HERE / "config_phi90_cd_sensitivity_p80.yaml"
CFG_P700 = HERE / "config_phi90_cd_sensitivity_p700.yaml"
LUA = HERE.parent / "shared" / "materials.lua"


def _raw() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}


def test_p80_grid_and_s4_count():
    cfg = load_config(CFG)
    spec = parse_phi90(cfg, _raw())
    assert spec.wavelengths.size == 21
    assert spec.wavelengths[0] == pytest.approx(12.0)
    assert spec.wavelengths[-1] == pytest.approx(16.0)
    assert spec.cd.size == 9
    assert spec.depth.size == 17
    assert spec.swa.size == 19
    assert list(spec.azimuths) == [0.0, 90.0]
    assert n_s4_calls(spec) == 2 * 21 * (9 + 17 + 19)
    assert n_s4_calls(spec) == 1890
    assert spec.truth_cd == pytest.approx(40.0)
    assert spec.truth_swa == pytest.approx(89.45)
    text = dry_run_text(cfg, spec, 8)
    assert "S4=1890" in text
    assert "H55" in text


def test_p700_paper_grid_and_s4_count():
    cfg = load_config(CFG_P700)
    spec = parse_phi90(cfg, yaml.safe_load(CFG_P700.read_text(encoding="utf-8")) or {})
    assert cfg.structure.pitch_nm == 700.0
    assert cfg.structure.grating_material == "Si"
    assert cfg.structure.substrate_material == "Si"
    assert cfg.structure.swa_deg == pytest.approx(90.0)
    assert cfg.structure.n_slices == 10
    assert cfg.optical.angle_deg == pytest.approx(78.0)
    assert cfg.optical.NG == 201
    assert spec.include_swa is True
    assert spec.wavelengths.size == 25
    assert spec.wavelengths[0] == pytest.approx(11.0)
    assert spec.wavelengths[-1] == pytest.approx(35.0)
    assert spec.cd.size == 16
    assert spec.depth.size == 21
    assert spec.swa.size == 19
    assert spec.truth_cd == pytest.approx(350.0)
    assert spec.truth_depth == pytest.approx(100.0)
    assert n_s4_calls(spec) == 2 * 25 * (16 + 21 + 19)
    assert n_s4_calls(spec) == 2800
    text = dry_run_text(cfg, spec, 4)
    assert "S4=2800" in text
    assert "SWA 85" in text
    assert "pitch 700" in text


def test_harmonic_markers_h55_61():
    cfg = load_config(CFG)
    marks = harmonic_markers(cfg)
    assert [n for _, n in marks] == ["H55", "H57", "H59", "H61"]
    by_name = {n: w for w, n in marks}
    assert by_name["H55"] == pytest.approx(800.0 / 55.0)


def test_two_beam_fringe_period():
    wl, ang = 14.0, 70.0
    r_t, r_f = 0.35 + 0.20j, 0.12 + 0.05j
    h = np.linspace(10.0, 90.0, 8001)
    R = np.array([two_beam_R0(wl, hi, 0.5, ang, r_t, r_f) for hi in h])
    interior = (R[1:-1] < R[:-2]) & (R[1:-1] < R[2:])
    hs = h[1:-1][interior]
    assert hs.size >= 3
    period = reflection_height_period_nm(wl, ang)
    assert abs(float(np.median(np.diff(hs))) - period) / period < 0.02


def test_dark_wavelength_is_spectral_min_at_fixed_height():
    h, ang = 40.0, 70.0
    predicted = dark_wavelengths_nm(h, ang)
    assert predicted[0] == pytest.approx(4.0 * h * abs(math.cos(math.radians(ang))))
    target = next(w for w in predicted if 8.0 < w < 25.0)
    wls = np.linspace(target - 3.0, target + 3.0, 3001)
    R = np.array([two_beam_R0(w, h, 0.5, ang, 0.40, 0.10) for w in wls])
    assert abs(wls[int(np.argmin(R))] - target) < 0.05
    R_other_cd = np.array([two_beam_R0(w, h, 0.3, ang, 0.40, 0.10) for w in wls])
    assert abs(wls[int(np.argmin(R_other_cd))] - target) < 0.05


def test_two_beam_phase_independent_of_fill():
    wl, ang = 14.55, 70.0
    r_t, r_f = 0.40 + 0.15j, 0.10 + 0.04j
    h = np.linspace(20.0, 70.0, 5001)
    r03 = np.array([two_beam_R0(wl, hi, 0.3, ang, r_t, r_f) for hi in h])
    r07 = np.array([two_beam_R0(wl, hi, 0.7, ang, r_t, r_f) for hi in h])
    dh = h[1] - h[0]
    assert abs(h[int(np.argmin(r03))] - h[int(np.argmin(r07))]) < 2.0 * dh
    # f and 1-f share the same contrast 2f(1-f)|r_t r_f|; the offset |α|²+|β|² still moves.
    assert abs(float(np.mean(r03)) - float(np.mean(r07))) > 1e-3


def test_fresnel_and_au_eps():
    n = 0.89 - 0.07j
    r = fresnel_rs(n * n, 70.0)
    assert abs(r) < 1.01
    assert LUA.is_file()
    eps = load_material_eps(LUA, "Au", 14.55)
    assert eps.real > 0
    assert eps.imag < 0
    r_au = fresnel_rs(eps, 70.0)
    assert abs(r_au) < 1.01


def test_sensitivity_cd_null_at_90():
    depth = np.linspace(24.0, 56.0, 9)
    cd = np.linspace(32.0, 48.0, 9)
    swa = np.linspace(85.0, 94.0, 7)
    wl = np.linspace(12.0, 16.0, 6)
    rng = np.random.default_rng(0)
    R_depth = 0.2 + 0.04 * rng.normal(size=(2, depth.size, wl.size))
    R_swa = 0.2 + 0.02 * rng.normal(size=(2, swa.size, wl.size))
    base = 0.18 + 0.03 * rng.normal(size=(wl.size,))
    R_cd = np.empty((2, cd.size, wl.size))
    R_cd[0] = base + np.linspace(-0.06, 0.06, cd.size)[:, None]
    R_cd[1] = np.broadcast_to(base, (cd.size, wl.size))
    d_cd = rms_partial(R_cd, cd)
    d_depth = rms_partial(R_depth, depth)
    assert d_cd[1] < 1e-12
    assert d_cd[0] > 10.0 * max(d_cd[1], 1e-16)
    spec = Phi90Spec(
        azimuths=np.array([0.0, 90.0]),
        wavelengths=wl,
        cd=cd,
        depth=depth,
        swa=swa,
        workers=1,
        checkpoint_every=10,
        two_beam_top="Au",
        two_beam_floor="SiO2",
        truth_cd=40.0,
        truth_depth=40.0,
        truth_swa=89.45,
        pitch=80.0,
        angle_deg=70.0,
        include_swa=True,
        out_dir=HERE,
        dpi=80,
    )
    table = sensitivity_table(spec, R_depth, R_cd, R_swa)
    assert table["90"]["cd_over_depth"] < 1e-6
    assert table["0"]["cd_over_depth"] > table["90"]["cd_over_depth"]
    assert d_depth[0] > 0


def test_replot_from_npz(tmp_path: Path):
    cfg = load_config(CFG)
    spec = Phi90Spec(
        azimuths=np.array([0.0, 90.0]),
        wavelengths=np.linspace(12.0, 16.0, 5),
        cd=np.linspace(32.0, 48.0, 4),
        depth=np.linspace(24.0, 56.0, 5),
        swa=np.linspace(85.0, 94.0, 4),
        workers=1,
        checkpoint_every=10,
        two_beam_top="Au",
        two_beam_floor="SiO2",
        truth_cd=40.0,
        truth_depth=40.0,
        truth_swa=89.45,
        pitch=80.0,
        angle_deg=70.0,
        include_swa=True,
        out_dir=tmp_path,
        dpi=80,
    )
    wl = spec.wavelengths[None, None, :]
    h = spec.depth[:, None]
    psi = 4.0 * np.pi * h * np.cos(np.deg2rad(70.0)) / wl
    R_depth = np.broadcast_to(0.12 + 0.08 * np.cos(psi), (2, spec.depth.size, spec.wavelengths.size)).copy()
    R_cd = 0.15 + 0.02 * np.linspace(-1, 1, spec.cd.size)[None, :, None] * np.ones(spec.wavelengths.size)
    R_cd = np.broadcast_to(R_cd, (2, spec.cd.size, spec.wavelengths.size)).copy()
    R_cd[1] = R_cd[1, :1, :]
    R_swa = 0.15 + 0.01 * np.linspace(-1, 1, spec.swa.size)[None, :, None]
    R_swa = np.broadcast_to(R_swa, (2, spec.swa.size, spec.wavelengths.size)).copy()
    filled_d = np.ones(R_depth.shape, dtype=bool)
    filled_c = np.ones(R_cd.shape, dtype=bool)
    filled_s = np.ones(R_swa.shape, dtype=bool)
    save_npz(tmp_path / "phi90_cd_sensitivity.npz", spec, R_depth, R_cd, R_swa, filled_d, filled_c, filled_s)
    summarize(cfg, spec, R_depth, R_cd, R_swa, filled_d, filled_c, filled_s)
    analog = tmp_path / "fig_phi90_cd_sensitivity.png"
    two = tmp_path / "fig_two_beam_depth90.png"
    js = tmp_path / "phi90_cd_sensitivity.json"
    assert analog.is_file() and analog.stat().st_size > 0
    assert two.is_file() and two.stat().st_size > 0
    assert js.is_file()
    analog.unlink()
    two.unlink()
    replot(tmp_path / "phi90_cd_sensitivity.npz", CFG)
    assert analog.is_file()
    assert two.is_file()
