"""Unit tests for FIM masks and metrics (no S4)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from config import load_config
from fim_study import (
    ObservableRow,
    fisher_and_crlb,
    full_observable_layout,
    mask_rows,
    noise_with_flicker,
    order_lambda_phi_grid,
    write_layout_only,
)
from noise_model import observation_variance


def _row(**kwargs) -> ObservableRow:
    defaults = dict(
        flat_index=0,
        condition_index=0,
        order_m=0,
        wl_nm=13.5,
        angle_deg=70.0,
        azimuth_deg=0.0,
        harmonic_order=59,
        propagating=True,
        role="lateral",
    )
    defaults.update(kwargs)
    return ObservableRow(**defaults)


def _toy_layout() -> list[ObservableRow]:
    rows = []
    flat = 0
    for az, orders in (
        (0.0, ((0, "lateral", True), (-1, "aux", True), (1, "aux", False))),
        (30.0, ((0, "lateral", True),)),
        (45.0, ((0, "lateral", True),)),
        (60.0, ((0, "lateral", True), (-1, "aux", True))),
        (90.0, ((0, "depth_anchor", True), (-1, "swa", True), (1, "swa", True), (2, "aux", True))),
    ):
        for m, role, prop in orders:
            rows.append(
                _row(
                    flat_index=flat,
                    order_m=m,
                    azimuth_deg=az,
                    propagating=prop,
                    role=role,
                )
            )
            flat += 1
    return rows


WINDOWS = {
    "only90": [90.0],
    "near90": [60.0, 90.0],
    "far": [0.0, 30.0],
    "mid": [30.0, 45.0],
    "two_cam": [0.0, 30.0, 60.0, 90.0],
}


def test_masks_are_propagating_subsets():
    rows = _toy_layout()
    for name in (
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
    ):
        sel = mask_rows(rows, name, windows=WINDOWS, tol=0.5)
        for r, keep in zip(rows, sel):
            if keep:
                assert r.propagating, f"{name} kept evanescent {r}"
                assert r.order_m in (-1, 0, 1), f"{name} kept |m|>1 {r}"


def test_decoupling_drops_non90_pm1_and_aux():
    rows = _toy_layout()
    sel = mask_rows(rows, "decoupling", windows=WINDOWS, tol=0.5)
    picked = [r for r, k in zip(rows, sel) if k]
    assert {(r.azimuth_deg, r.order_m) for r in picked} == {
        (0.0, 0),
        (30.0, 0),
        (45.0, 0),
        (60.0, 0),
        (90.0, 0),
        (90.0, -1),
        (90.0, 1),
    }
    assert not any(r.role == "aux" for r in picked)


def test_prop_keeps_pm1_drops_m2():
    rows = _toy_layout()
    sel = mask_rows(rows, "prop", windows=WINDOWS, tol=0.5)
    picked = {(r.azimuth_deg, r.order_m) for r, k in zip(rows, sel) if k}
    assert (0.0, -1) in picked
    assert (90.0, 2) not in picked
    assert (0.0, 1) not in picked
    all_m = mask_rows(rows, "prop", windows=WINDOWS, tol=0.5, keep_orders=None)
    picked_all = {(r.azimuth_deg, r.order_m) for r, k in zip(rows, all_m) if k}
    assert (90.0, 2) in picked_all


def test_far_has_no_swa_channel():
    rows = _toy_layout()
    sel = mask_rows(rows, "far", windows=WINDOWS, tol=0.5)
    picked = [r for r, k in zip(rows, sel) if k]
    assert all(r.azimuth_deg in (0.0, 30.0) for r in picked)
    assert not any(r.role == "swa" for r in picked)


def test_fim_identity():
    J = np.eye(2)
    sigma = np.ones(2)
    scales = np.ones(2)
    out = fisher_and_crlb(J, sigma, scales)
    assert out["rank"] == 2
    np.testing.assert_allclose(out["crlb"], [1.0, 1.0])
    np.testing.assert_allclose(out["corr"], np.eye(2))
    assert math.isfinite(out["cond_scaled"])


def test_fim_unidentifiable_column():
    J = np.array([[1.0, 0.0], [2.0, 0.0]])
    sigma = np.ones(2)
    out = fisher_and_crlb(J, sigma, np.ones(2))
    assert out["rank"] == 1
    assert np.isfinite(out["crlb"][0])
    assert not np.isfinite(out["crlb"][1])
    assert out["singular"][1]


def test_fim_uses_raw_variance_not_normalized_weights():
    J = np.array([[1.0], [1.0]])
    sigma = np.array([1.0, 2.0])
    out = fisher_and_crlb(J, sigma, np.array([1.0]))
    # F = 1/1^2 + 1/2^2 = 1.25, CRLB = 1/sqrt(1.25)
    np.testing.assert_allclose(out["F"], [[1.25]])
    np.testing.assert_allclose(out["crlb"], [1.0 / np.sqrt(1.25)])


def test_observation_variance_matches_formula():
    from noise_model import ChannelNoiseConfig, DetectorNoiseConfig

    ch = ChannelNoiseConfig(
        relative_a=0.01,
        n0_electrons=1.0e6,
        readout_e_rms=0.0,
        n_roi_pixels=1,
        dark_e_per_pixel_s=0.0,
        t_exp_s=1.0,
    )
    noise = DetectorNoiseConfig(apply=True, zero=ch, first=ch)
    f = np.array([0.25])
    var = observation_variance(f, noise, np.array([0]))
    expect = (0.01 * 0.25) ** 2 + 0.25 / 1.0e6
    np.testing.assert_allclose(var, [expect])


def test_flicker_override_does_not_mutate_original():
    cfg = load_config(Path(__file__).with_name("config_fim.yaml"))
    a0 = cfg.inverse.noise.zero.relative_a
    muted = noise_with_flicker(cfg.inverse.noise, 0.0)
    assert muted.zero.relative_a == 0.0
    assert cfg.inverse.noise.zero.relative_a == a0


def test_load_config_fim_yaml():
    cfg = load_config(Path(__file__).with_name("config_fim.yaml"))
    assert cfg.eval.task == "fim_study"
    assert "decoupling" in cfg.fim.masks
    assert cfg.fim.flicker_a[0] is None
    assert cfg.fim.flicker_a[1] == 0.0
    assert cfg.fim.azimuth_windows["far"] == [0.0, 30.0]
    assert cfg.fim.keep_orders == [-1, 0, 1]


def test_layout_only_no_s4(tmp_path: Path | None = None):
    cfg = load_config(Path(__file__).with_name("config_fim.yaml"))
    rows = full_observable_layout(cfg)
    assert len(rows) == len(expand_n(cfg))
    n_prop = sum(1 for r in rows if r.propagating)
    assert n_prop > 0
    assert n_prop < len(rows)
    dec = mask_rows(
        rows, "decoupling", windows=cfg.fim.azimuth_windows, tol=cfg.fim.azimuth_tol_deg
    )
    prop = mask_rows(
        rows, "prop", windows=cfg.fim.azimuth_windows, tol=cfg.fim.azimuth_tol_deg
    )
    assert np.count_nonzero(dec) <= np.count_nonzero(prop)
    # decoupling never keeps non-90 ±1
    for r, keep in zip(rows, dec):
        if keep and r.order_m != 0:
            assert abs(r.azimuth_deg - 90.0) < 1.0
    for r, keep in zip(rows, prop):
        if keep:
            assert r.order_m in (-1, 0, 1)


def expand_n(cfg) -> list:
    from recipe import expand_measurement_conditions, n_orders

    return [0] * (len(expand_measurement_conditions(cfg)) * n_orders(cfg))


def test_write_layout_only(tmp_path=None):
    from pathlib import Path as P

    cfg = load_config(P(__file__).with_name("config_fim.yaml"))
    out = P(__file__).resolve().parent / "_fim_layout_test_out"
    try:
        payload = write_layout_only(cfg, out)
        assert payload["n_conditions"] == 20
        assert payload["n_orders_stored"] == 31
        assert payload["n_s4_forwards_for_J"] == 100
        assert payload["keep_orders"] == [-1, 0, 1]
        assert payload["n_measurable"] <= payload["n_propagating"]
        assert (out / "propagating_table.txt").is_file()
        assert (out / "mask_table.txt").is_file()
    finally:
        for name in ("propagating_table.txt", "mask_table.txt", "fim_layout.json"):
            p = out / name
            if p.exists():
                p.unlink()
        if out.exists():
            out.rmdir()


def test_lambda_phi_grid_cutoff_is_nan():
    rows = [
        _row(flat_index=0, order_m=0, wl_nm=13.0, azimuth_deg=0.0, propagating=True),
        _row(flat_index=1, order_m=0, wl_nm=13.0, azimuth_deg=90.0, propagating=False),
        _row(flat_index=2, order_m=1, wl_nm=14.0, azimuth_deg=90.0, propagating=True),
    ]
    values = np.array([0.2, 0.3, 0.4])
    wls, phis = [13.0, 14.0], [0.0, 90.0]
    g0 = order_lambda_phi_grid(rows, values, 0, wls, phis)
    assert g0[0, 0] == 0.2
    assert np.isnan(g0[0, 1])
    assert np.isnan(g0[1, 0])
    g1 = order_lambda_phi_grid(rows, values, 1, wls, phis)
    assert g1[1, 1] == 0.4
    assert np.isnan(g1[0, 0])


def main() -> None:
    tests = [
        test_masks_are_propagating_subsets,
        test_decoupling_drops_non90_pm1_and_aux,
        test_prop_keeps_pm1_drops_m2,
        test_far_has_no_swa_channel,
        test_fim_identity,
        test_fim_unidentifiable_column,
        test_fim_uses_raw_variance_not_normalized_weights,
        test_observation_variance_matches_formula,
        test_flicker_override_does_not_mutate_original,
        test_load_config_fim_yaml,
        test_layout_only_no_s4,
        test_write_layout_only,
        test_lambda_phi_grid_cutoff_is_nan,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
