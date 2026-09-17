"""CRLB–MC helpers and FIM-mask collection (no S4)."""
from __future__ import annotations

from pathlib import Path

from config import load_config, role_weights_enabled
from order_collection import FIM_MASK_MODES, n_collectible
from run_crlb_mc import apply_experiment, apply_mask, fim_mask_pairs, layout_pairs, summarize_trials


def _cfg_p80():
    return load_config(Path(__file__).with_name("config_crlb_mc_p80.yaml"))


def _cfg_p300():
    return load_config(Path(__file__).with_name("config_crlb_mc_p300.yaml"))


def test_inverse_rows_match_fim_masks_p80():
    cfg = _cfg_p80()
    for mask in FIM_MASK_MODES:
        assert layout_pairs(cfg, mask) == fim_mask_pairs(cfg, mask), mask


def test_inverse_rows_match_fim_masks_p300():
    cfg = _cfg_p300()
    for mask in FIM_MASK_MODES:
        assert layout_pairs(cfg, mask) == fim_mask_pairs(cfg, mask), mask


def test_m0_all_has_no_first_orders():
    cfg = _cfg_p80()
    pairs = layout_pairs(cfg, "m0_all")
    assert pairs
    assert all(m == 0 for _az, m in pairs)


def test_only90_is_conical_subset():
    cfg = _cfg_p80()
    pairs = layout_pairs(cfg, "only90")
    assert pairs
    assert all(abs(az - 90.0) < 0.5 for az, _m in pairs)
    assert layout_pairs(cfg, "only90") < layout_pairs(cfg, "prop")


def test_decoupling_drops_non90_pm1():
    cfg = _cfg_p80()
    pairs = layout_pairs(cfg, "decoupling")
    assert (0.0, 0) in pairs
    assert (90.0, 0) in pairs
    assert (90.0, -1) in pairs or (90.0, 1) in pairs
    assert (0.0, -1) not in pairs
    assert (0.0, 1) not in pairs


def test_mode_a_disables_role_weights():
    cfg = _cfg_p80()
    work = apply_experiment(cfg, "decoupling", "A")
    assert work.inverse.use_ga is False
    assert work.inverse.use_role_weights is False
    assert role_weights_enabled(work) is False
    assert n_collectible(apply_mask(cfg, "decoupling")) == n_collectible(work)


def test_mode_b_decoupling_enables_solver_heuristics():
    cfg = _cfg_p80()
    work = apply_experiment(cfg, "decoupling", "B")
    assert work.inverse.use_ga is True
    assert work.inverse.use_role_weights is True
    assert work.inverse.decoupling.dynamic_lm.enabled is True


def test_summarize_efficiency():
    names = ["cd_nm", "depth_nm", "lswa_deg", "rswa_deg"]
    ref = {"cd_nm": 40.0, "depth_nm": 40.0, "lswa_deg": 89.0, "rswa_deg": 89.0}
    records = [
        {"p_est": {n: ref[n] + 0.1 for n in names}, "p_ref": ref, "forward_eval_count": 10, "timing": {"t_total": 1.0}},
        {"p_est": {n: ref[n] - 0.1 for n in names}, "p_ref": ref, "forward_eval_count": 12, "timing": {"t_total": 1.2}},
        {"p_est": {n: ref[n] for n in names}, "p_ref": ref, "forward_eval_count": 11, "timing": {"t_total": 1.1}},
    ]
    fim = {"crlb": {n: 0.1 for n in names}}
    out = summarize_trials(records, names, fim)
    assert out["n"] == 3
    assert out["std"]["cd_nm"] > 0
    assert out["efficiency_crlb_over_std"]["cd_nm"] > 0
