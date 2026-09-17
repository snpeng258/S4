"""χ² landscape helpers (no S4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import GridAxisConfig, load_config
from order_collection import FIM_MASK_MODES, n_observables_full
from recipe import expand_measurement_conditions
from run_chi2_landscape import (
    inverse_objective_loss,
    inverse_objective_wsqrt,
    landscape_out_dir,
    parse_landscape,
    structure_grid,
)
from run_crlb_mc import apply_mask, fim_mask_pairs, layout_pairs


def _raw(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_inverse_loss_matches_solver_weights():
    cfg = apply_mask(load_config(Path(__file__).with_name("config_chi2_p80.yaml")), "decoupling")
    n = n_observables_full(cfg)
    r_meas = np.linspace(0.05, 0.4, n)
    wsqrt = inverse_objective_wsqrt(cfg, r_meas)
    assert abs(float(np.sum(wsqrt ** 2)) - 1.0) < 1e-12
    assert inverse_objective_loss(r_meas, r_meas, wsqrt, cfg) == 0.0
    r_off = r_meas + 0.01
    loss = inverse_objective_loss(r_off, r_meas, wsqrt, cfg)
    assert loss > 0.0


def test_p80_cd_depth_grid():
    path = Path(__file__).with_name("config_chi2_p80.yaml")
    cfg = load_config(path)
    spec = parse_landscape(cfg, _raw(path), None)
    assert len(expand_measurement_conditions(cfg)) == 20
    assert spec.slice == "cd_depth"
    assert spec.fixed == {"swa_deg": 89.45}
    xs, ys, params = structure_grid(spec)
    assert xs[0] == 32.0 and xs[-1] == 48.0
    assert ys[0] == 32.0 and ys[-1] == 48.0
    assert params.shape == (17 * 17, 3)
    assert abs(params[0, 2] - 89.45) < 1e-12
    mid = params[17 * 8 + 8]
    assert abs(mid[0] - 40.0) < 1e-12
    assert abs(mid[1] - 40.0) < 1e-12


def test_p300_cd_depth_grid():
    path = Path(__file__).with_name("config_chi2_p300.yaml")
    cfg = load_config(path)
    spec = parse_landscape(cfg, _raw(path), None)
    assert cfg.structure.pitch_nm == 300.0
    assert cfg.optical.NG == 61
    assert len(expand_measurement_conditions(cfg)) == 20
    assert spec.fixed == {"swa_deg": 89.45}
    xs, ys, params = structure_grid(spec)
    assert xs[0] == 134.0 and xs[-1] == 166.0
    assert len(xs) == 17
    assert len(ys) == 17
    assert params.shape == (17 * 17, 3)
    assert abs(cfg.structure.cd_nm - 150.0) < 1e-12
    mid = params[17 * 8 + 8]
    assert abs(mid[0] - 150.0) < 1e-12
    assert abs(mid[1] - 40.0) < 1e-12


def test_cd_swa_slice_fixes_depth():
    path = Path(__file__).with_name("config_chi2_p80.yaml")
    cfg = load_config(path)
    spec = parse_landscape(cfg, _raw(path), "cd_swa")
    assert spec.x_name == "cd_nm"
    assert spec.y_name == "swa_deg"
    assert spec.fixed == {"depth_nm": 40.0}
    xs, ys, params = structure_grid(spec)
    assert params.shape[0] == len(xs) * len(ys)
    assert np.allclose(params[:, 1], 40.0)


def test_chi2_configs_share_fim_masks_with_inverse():
    for name in ("config_chi2_p80.yaml", "config_chi2_p300.yaml"):
        cfg = load_config(Path(__file__).with_name(name))
        for mask in FIM_MASK_MODES:
            assert layout_pairs(cfg, mask) == fim_mask_pairs(cfg, mask), (name, mask)


def test_grid_axis_count():
    ax = GridAxisConfig(134.0, 166.0, 2.0)
    pts = ax.points()
    assert len(pts) == 17
    assert abs(pts[0] - 134.0) < 1e-12
    assert abs(pts[8] - 150.0) < 1e-12
    assert abs(pts[-1] - 166.0) < 1e-12


def test_noisy_output_dir_is_sibling():
    cfg = load_config(Path(__file__).with_name("config_chi2_p80.yaml"))
    clean = landscape_out_dir(cfg, True)
    noisy = landscape_out_dir(cfg, False)
    assert clean.name == "p80"
    assert noisy.name == "p80_noisy"
    assert noisy.parent == clean.parent
    cfg300 = load_config(Path(__file__).with_name("config_chi2_p300.yaml"))
    assert landscape_out_dir(cfg300, False).name == "p300_noisy"
