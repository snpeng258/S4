"""Layer-1 Jacobian / FIM study: one full-order J, then row-mask layouts.

Does not run GA+LM and does not read the spectrum library. Σ is the detector
variance (observation_variance); decoupling role weights are not included.

    python3 fim_study.py --config config_fim.yaml --layout-only
    python3 fim_study.py --config config_fim.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from config import ScatterometryConfig, default_config_path, load_config
from decoupling import classify_role
from noise_model import DetectorNoiseConfig, observation_variance
from order_collection import order_m_values, propagating_orders
from recipe import expand_measurement_conditions, n_orders
from s4_runner import find_s4_binary
from sensitivity import full_stored_jacobian

SPECIAL_MASKS = ("prop", "decoupling", "m0_all", "no90", "drop_phi45")


@dataclass(frozen=True)
class ObservableRow:
    flat_index: int
    condition_index: int
    order_m: int
    wl_nm: float
    angle_deg: float
    azimuth_deg: float
    harmonic_order: int | None
    propagating: bool
    role: str


def az_near(azimuth_deg: float, target: float, tol: float) -> bool:
    d = abs((float(azimuth_deg) % 360.0) - (float(target) % 360.0))
    return min(d, 360.0 - d) < tol


def az_in(azimuth_deg: float, targets: list[float], tol: float) -> bool:
    return any(az_near(azimuth_deg, t, tol) for t in targets)


def full_observable_layout(cfg: ScatterometryConfig) -> list[ObservableRow]:
    conditions = expand_measurement_conditions(cfg)
    ms = order_m_values(cfg)
    rows: list[ObservableRow] = []
    flat = 0
    for c_idx, cond in enumerate(conditions):
        prop = set(
            propagating_orders(
                pitch_nm=cfg.structure.pitch_nm,
                wl_nm=cond.wl_nm,
                angle_deg=cond.angle_deg,
                azimuth_deg=cond.azimuth_deg,
                order_min=cfg.optical.order_min,
                order_max=cfg.optical.order_max,
            )
        )
        for m in ms:
            rows.append(
                ObservableRow(
                    flat_index=flat,
                    condition_index=c_idx,
                    order_m=int(m),
                    wl_nm=float(cond.wl_nm),
                    angle_deg=float(cond.angle_deg),
                    azimuth_deg=float(cond.azimuth_deg),
                    harmonic_order=cond.harmonic_order,
                    propagating=int(m) in prop,
                    role=classify_role(cond.azimuth_deg, m, cfg),
                )
            )
            flat += 1
    return rows


def mask_rows(
    rows: list[ObservableRow],
    name: str,
    *,
    windows: dict[str, list[float]],
    tol: float,
    drop_azimuth: float = 45.0,
) -> np.ndarray:
    """Boolean row mask. Every layout is a subset of propagating orders."""
    prop = np.array([r.propagating for r in rows], dtype=bool)
    m = np.array([r.order_m for r in rows], dtype=int)
    if name == "prop":
        return prop
    if name == "decoupling":
        return prop & np.array([r.role != "aux" for r in rows], dtype=bool)
    if name == "m0_all":
        return prop & (m == 0)
    if name == "no90":
        return prop & np.array([not az_near(r.azimuth_deg, 90.0, tol) for r in rows])
    if name == "drop_phi45":
        return prop & np.array(
            [not az_near(r.azimuth_deg, drop_azimuth, tol) for r in rows]
        )
    if name in windows:
        targets = windows[name]
        return prop & np.array([az_in(r.azimuth_deg, targets, tol) for r in rows])
    known = list(SPECIAL_MASKS) + list(windows)
    raise ValueError(f"unknown FIM mask {name!r}; known={known}")


def fisher_and_crlb(
    jac: np.ndarray,
    sigma: np.ndarray,
    scales: np.ndarray,
    rcond: float = 1e-12,
) -> dict:
    """Diagonal-Σ Gaussian FIM. CRLB in native units; cond uses column scales."""
    J = np.asarray(jac, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    sc = np.asarray(scales, dtype=float)
    if J.ndim != 2:
        raise ValueError("jac must be 2-D")
    n, p = J.shape
    if sig.shape != (n,):
        raise ValueError(f"sigma shape {sig.shape} != ({n},)")
    if sc.shape != (p,):
        raise ValueError(f"scales shape {sc.shape} != ({p},)")

    empty = {
        "F": np.zeros((p, p)),
        "F_scaled": np.zeros((p, p)),
        "cov": np.full((p, p), np.nan),
        "corr": np.full((p, p), np.nan),
        "crlb": np.full(p, np.inf),
        "cond_scaled": float("inf"),
        "rank": 0,
        "singular": np.ones(p, dtype=bool),
        "n_rows": n,
    }
    if n == 0 or p == 0:
        return empty

    w = 1.0 / np.maximum(sig, 1e-30)
    Jw = J * w[:, None]
    if not np.any(np.isfinite(Jw)):
        return empty
    F = Jw.T @ Jw
    Jw_s = Jw * sc[None, :]
    F_s = Jw_s.T @ Jw_s
    s_f = np.linalg.svd(F_s, compute_uv=False)
    smax = float(s_f[0]) if s_f.size else 0.0
    tol_s = smax * rcond if smax > 0 else rcond
    rank = int(np.sum(s_f > tol_s))
    smin = float(s_f[-1]) if s_f.size else 0.0
    cond_scaled = float(smax / smin) if smin > 0 else float("inf")

    _u, s, vt = np.linalg.svd(Jw, full_matrices=False)
    s0 = float(s[0]) if s.size else 0.0
    tol = s0 * rcond if s0 > 0 else rcond
    good = s > tol
    inv_s2 = np.zeros_like(s)
    inv_s2[good] = 1.0 / (s[good] ** 2)
    cov = (vt.T * inv_s2) @ vt
    singular = np.zeros(p, dtype=bool)
    col_n = np.linalg.norm(Jw, axis=0)
    col_ref = float(np.max(col_n)) if col_n.size else 0.0
    if col_ref <= 0:
        singular[:] = True
    else:
        singular |= col_n < col_ref * rcond
    if np.any(~good) and vt.size:
        V_null = vt[~good].T
        if V_null.size:
            singular |= np.sum(V_null**2, axis=1) > 0.1
    crlb = np.sqrt(np.maximum(np.diag(cov), 0.0))
    crlb[singular] = np.inf

    corr = np.full((p, p), np.nan)
    finite = np.isfinite(crlb) & (crlb > 0)
    if np.any(finite):
        d = np.where(finite, crlb, 1.0)
        c = cov / np.outer(d, d)
        mask2 = np.outer(finite, finite)
        corr[mask2] = c[mask2]
        for i in range(p):
            if finite[i]:
                corr[i, i] = 1.0
    return {
        "F": F,
        "F_scaled": F_s,
        "cov": cov,
        "corr": corr,
        "crlb": crlb,
        "cond_scaled": cond_scaled,
        "rank": rank,
        "singular": singular,
        "n_rows": n,
    }


def noise_with_flicker(noise: DetectorNoiseConfig, a: float | None) -> DetectorNoiseConfig:
    if a is None:
        return noise
    a = float(a)
    return replace(
        noise,
        zero=replace(noise.zero, relative_a=a),
        first=replace(noise.first, relative_a=a),
    )


def param_scales_vector(cfg: ScatterometryConfig) -> np.ndarray:
    names = list(cfg.inverse.param_names)
    scales = cfg.fim.param_scales
    return np.array([float(scales.get(n, 1.0)) for n in names], dtype=float)


def n0_list(cfg: ScatterometryConfig) -> list[float]:
    if cfg.fim.n0_electrons:
        return [float(v) for v in cfg.fim.n0_electrons]
    return [float(v) for v in cfg.eval.noise_n0_electrons]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return obj


def propagating_groups(rows: list[ObservableRow]) -> list[dict]:
    groups: dict[tuple[float, int], dict] = {}
    for r in rows:
        key = (round(float(r.azimuth_deg), 6), int(r.order_m))
        g = groups.get(key)
        if g is None:
            g = {
                "azimuth_deg": float(r.azimuth_deg),
                "order_m": int(r.order_m),
                "role": r.role,
                "n_wl": 0,
                "n_propagating": 0,
                "wavelengths_nm": [],
            }
            groups[key] = g
        g["n_wl"] += 1
        g["wavelengths_nm"].append(float(r.wl_nm))
        if r.propagating:
            g["n_propagating"] += 1
    out = []
    for key in sorted(groups):
        g = groups[key]
        g["propagating"] = g["n_propagating"] == g["n_wl"]
        g["partial"] = 0 < g["n_propagating"] < g["n_wl"]
        out.append(g)
    return out


def format_propagating_table(rows: list[ObservableRow]) -> str:
    lines = [
        f"{'phi':>8} {'m':>4} {'prop':>5} {'n_prop/n_wl':>12} {'role':<14}",
        "-" * 52,
    ]
    for g in propagating_groups(rows):
        flag = "yes" if g["propagating"] else ("part" if g["partial"] else "no")
        if flag == "no":
            continue
        lines.append(
            f"{g['azimuth_deg']:8g} {g['order_m']:4d} {flag:>5} "
            f"{g['n_propagating']:4d}/{g['n_wl']:<6d} {g['role']:<14}"
        )
    n_prop = sum(1 for r in rows if r.propagating)
    lines.append("-" * 52)
    lines.append(f"propagating rows: {n_prop} / {len(rows)} stored")
    return "\n".join(lines)


def equivalent_mask_names(masks: dict[str, np.ndarray]) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: set[str] = set()
    names = list(masks)
    for i, a in enumerate(names):
        if a in seen:
            continue
        same = [a]
        for b in names[i + 1 :]:
            if b not in seen and np.array_equal(masks[a], masks[b]):
                same.append(b)
        if len(same) > 1:
            seen.update(same)
            groups.append(same)
    return groups


def format_mask_table(
    rows: list[ObservableRow],
    masks: dict[str, np.ndarray],
) -> str:
    lines = [
        f"{'mask':<14} {'n':>5} {'m=0':>5} {'|m|>=1':>7} {'roles'}",
        "-" * 64,
    ]
    for name, sel in masks.items():
        picked = [r for r, keep in zip(rows, sel) if keep]
        n0 = sum(1 for r in picked if r.order_m == 0)
        n1 = len(picked) - n0
        role_n: dict[str, int] = defaultdict(int)
        for r in picked:
            role_n[r.role] += 1
        role_s = " ".join(f"{k}={v}" for k, v in sorted(role_n.items()))
        lines.append(f"{name:<14} {len(picked):5d} {n0:5d} {n1:7d} {role_s}")
    return "\n".join(lines)


def _mask_metrics(
    jac: np.ndarray,
    r0: np.ndarray,
    rows: list[ObservableRow],
    sel: np.ndarray,
    noise: DetectorNoiseConfig,
    scales: np.ndarray,
    names: list[str],
    rcond: float,
    n0: float | None = None,
) -> dict:
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        metrics = fisher_and_crlb(
            np.zeros((0, jac.shape[1])),
            np.zeros(0),
            scales,
            rcond=rcond,
        )
    else:
        order_m = np.array([rows[i].order_m for i in idx], dtype=int)
        sigma = observation_variance(
            r0[idx], noise, order_m, n0_electrons=n0
        )
        metrics = fisher_and_crlb(jac[idx], sigma, scales, rcond=rcond)
    crlb = {n: float(metrics["crlb"][j]) for j, n in enumerate(names)}
    singular = [n for n, flag in zip(names, metrics["singular"]) if flag]
    name_to_i = {n: i for i, n in enumerate(names)}
    rho = None
    if "depth_nm" in name_to_i and "cd_nm" in name_to_i:
        i, j = name_to_i["cd_nm"], name_to_i["depth_nm"]
        rho = float(metrics["corr"][i, j])
    picked = [rows[i] for i in idx]
    role_n: dict[str, int] = defaultdict(int)
    for r in picked:
        role_n[r.role] += 1
    return {
        "n_rows": int(idx.size),
        "n_m0": int(sum(1 for r in picked if r.order_m == 0)),
        "n_pm1_or_higher": int(sum(1 for r in picked if r.order_m != 0)),
        "roles": dict(role_n),
        "azimuths_deg": sorted({float(r.azimuth_deg) for r in picked}),
        "orders_m": sorted({int(r.order_m) for r in picked}),
        "crlb": crlb,
        "corr": metrics["corr"],
        "rho_depth_cd": rho,
        "cond_scaled": float(metrics["cond_scaled"]),
        "rank": int(metrics["rank"]),
        "singular_params": singular,
    }


def _true_params(cfg: ScatterometryConfig) -> tuple[list[str], np.ndarray]:
    names = list(cfg.inverse.param_names)
    p = np.array([cfg.structure.get_param(n) for n in names], dtype=float)
    return names, p


def build_masks(cfg: ScatterometryConfig, rows: list[ObservableRow]) -> dict[str, np.ndarray]:
    windows = cfg.fim.azimuth_windows
    tol = float(cfg.fim.azimuth_tol_deg)
    out = {}
    for name in cfg.fim.masks:
        out[name] = mask_rows(rows, name, windows=windows, tol=tol)
    return out


def layout_payload(cfg: ScatterometryConfig, rows: list[ObservableRow]) -> dict:
    names, p = _true_params(cfg)
    masks = build_masks(cfg, rows)
    return {
        "n_conditions": len(expand_measurement_conditions(cfg)),
        "n_orders_stored": n_orders(cfg),
        "n_stored_rows": len(rows),
        "n_propagating": int(sum(1 for r in rows if r.propagating)),
        "n_s4_forwards_for_J": len(expand_measurement_conditions(cfg)) * (1 + len(names)),
        "param_names": names,
        "param_true": {n: float(v) for n, v in zip(names, p)},
        "groups": propagating_groups(rows),
        "mask_counts": {k: int(np.count_nonzero(v)) for k, v in masks.items()},
        "equivalent_masks": equivalent_mask_names(masks),
        "note": (
            "FIM uses raw 1/σ_j^2 from observation_variance. "
            "Decoupling role weights and WLS sum(w)=1 are not in Σ^{-1}."
        ),
    }


def write_layout_only(cfg: ScatterometryConfig, out_dir: Path) -> dict:
    rows = full_observable_layout(cfg)
    table = format_propagating_table(rows)
    masks = build_masks(cfg, rows)
    mask_txt = format_mask_table(rows, masks)
    payload = layout_payload(cfg, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "propagating_table.txt").write_text(table + "\n", encoding="utf-8")
    (out_dir / "mask_table.txt").write_text(mask_txt + "\n", encoding="utf-8")
    (out_dir / "fim_layout.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    print(table)
    print()
    print(mask_txt)
    equiv = equivalent_mask_names(masks)
    if equiv:
        print()
        print("Identical row sets: " + "; ".join(" = ".join(g) for g in equiv))
    print()
    print(
        f"S4 forwards for one J: {payload['n_s4_forwards_for_J']} "
        f"({payload['n_conditions']} conditions × {1 + len(cfg.inverse.param_names)} FD points)"
    )
    print(f"Wrote {out_dir / 'propagating_table.txt'}")
    return payload


def _sorted_prop_indices(rows: list[ObservableRow]) -> np.ndarray:
    keyed = [
        (r.azimuth_deg, r.order_m, r.wl_nm, r.flat_index)
        for r in rows
        if r.propagating
    ]
    keyed.sort()
    return np.array([k[-1] for k in keyed], dtype=int)


def _heatmap_ylabels(rows: list[ObservableRow], idx: np.ndarray) -> list[str]:
    labels = []
    for i in idx:
        r = rows[int(i)]
        q = f"q{r.harmonic_order}" if r.harmonic_order is not None else f"{r.wl_nm:.3g}nm"
        labels.append(f"φ{r.azimuth_deg:g} m{r.order_m:+d} {q}")
    return labels


def _robust_lim(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(finite), 99.0))
    return v if v > 0 else 1.0


def plot_jacobian_heatmap(
    jac: np.ndarray,
    rows: list[ObservableRow],
    names: list[str],
    out_path: Path,
    dpi: int,
    *,
    sigma: np.ndarray | None = None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    idx = _sorted_prop_indices(rows)
    if idx.size == 0:
        return
    M = jac[idx]
    if sigma is not None:
        M = M / np.maximum(sigma[idx], 1e-30)[:, None]
    labels = _heatmap_ylabels(rows, idx)
    n_rows, n_p = M.shape
    h = min(max(5.0, 0.16 * n_rows), 18.0)
    fig, axes = plt.subplots(1, n_p, figsize=(3.1 * n_p, h), sharey=True)
    if n_p == 1:
        axes = [axes]
    vmax = _robust_lim(M)
    for ax, j, name in zip(axes, range(n_p), names):
        im = ax.imshow(
            M[:, j : j + 1],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks([0])
        ax.set_xticklabels([name], rotation=30, ha="right")
        ax.set_title(name, fontsize=10)
        if n_rows <= 48:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(labels, fontsize=6)
        else:
            # tick at first row of each (φ, m) block
            ticks = []
            tick_labels = []
            prev = None
            for k, lab_i in enumerate(idx):
                key = (rows[int(lab_i)].azimuth_deg, rows[int(lab_i)].order_m)
                if key != prev:
                    ticks.append(k)
                    tick_labels.append(f"φ{key[0]:g} m{key[1]:+d}")
                    prev = key
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("observable (propagating only)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_corr_heatmaps(
    records: dict[str, dict],
    names: list[str],
    mask_names: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    want = [m for m in mask_names if m in records]
    if not want:
        return
    n = len(want)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes_f = np.atleast_1d(axes).ravel()
    for ax, mname in zip(axes_f, want):
        corr = np.asarray(records[mname]["corr"], dtype=float)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticklabels(names)
        ax.set_title(mname)
        for i in range(len(names)):
            for j in range(len(names)):
                val = corr[i, j]
                txt = "—" if not np.isfinite(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_f[len(want) :]:
        ax.set_axis_off()
    fig.suptitle(r"Parameter correlation from $F^{-1}$ (not $J^\top J$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def _crlb_plot_vals(records: dict[str, dict], names: list[str], pname: str) -> tuple[list[str], np.ndarray]:
    labels = []
    vals = []
    for mname, rec in records.items():
        labels.append(mname)
        vals.append(rec["crlb"].get(pname, np.inf) if isinstance(rec["crlb"], dict) else np.inf)
    return labels, np.asarray(vals, dtype=float)


def plot_crlb_by_mask(records: dict[str, dict], names: list[str], out_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    n_p = len(names)
    ncols = 2
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols / 2, 3.6 * nrows))
    axes_f = np.atleast_1d(axes).ravel()
    for ax, pname in zip(axes_f, names):
        labels, vals = _crlb_plot_vals(records, names, pname)
        x = np.arange(len(labels))
        finite = np.isfinite(vals)
        bars = np.where(finite, vals, np.nan)
        ax.bar(x, bars, color="C0")
        if np.any(~finite):
            ymax = np.nanmax(bars) if np.any(finite) else 1.0
            ax.scatter(x[~finite], np.full(np.count_nonzero(~finite), ymax * 1.05 if ymax else 1.0),
                       marker="^", color="C3", zorder=3, label="unidentifiable")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel("CRLB")
        ax.set_title(pname)
        ax.grid(True, axis="y", alpha=0.3)
        if np.any(finite) and (np.nanmax(bars) / max(np.nanmin(bars[bars > 0]) if np.any(bars > 0) else 1.0, 1e-30) > 50):
            ax.set_yscale("log")
    for ax in axes_f[n_p:]:
        ax.set_axis_off()
    fig.suptitle("CRLB by layout (default $N_0$, YAML flicker $a$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_rho_cond(records: dict[str, dict], out_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    labels = list(records)
    rho = np.array([records[m]["rho_depth_cd"] if records[m]["rho_depth_cd"] is not None else np.nan for m in labels], dtype=float)
    cond = np.array([records[m]["cond_scaled"] for m in labels], dtype=float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, np.where(np.isfinite(rho), rho, np.nan))
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].set_ylabel(r"$\rho$(depth, CD)")
    axes[0].set_title("Depth–CD correlation")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, np.where(np.isfinite(cond), cond, np.nan))
    if np.any(np.isfinite(cond)) and np.nanmax(cond) / max(np.nanmin(cond[cond > 0]) if np.any(cond > 0) else 1.0, 1e-30) > 50:
        axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    axes[1].set_ylabel(r"cond($S F S$)  (1 nm / 1°)")
    axes[1].set_title("Scaled condition number")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_crlb_vs_n0(
    n0_vals: list[float],
    sweep: dict[str, dict[str, list[float]]],
    names: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n_p = len(names)
    ncols = 2
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols / 2, 3.8 * nrows))
    axes_f = np.atleast_1d(axes).ravel()
    x = np.asarray(n0_vals, dtype=float)
    for ax, pname in zip(axes_f, names):
        for mname, series in sweep.items():
            y = np.asarray(series[pname], dtype=float)
            ax.plot(x, y, "-o", ms=4, label=mname)
        ax.set_xscale("log")
        ax.set_xlabel(r"$N_0$ (photoelectrons at $R=1$)")
        ax.set_ylabel("CRLB")
        ax.set_title(pname)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        finite = np.concatenate([np.asarray(series[pname], dtype=float) for series in sweep.values()])
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size and np.max(finite) / np.min(finite) > 50:
            ax.set_yscale("log")
    for ax in axes_f[n_p:]:
        ax.set_axis_off()
    fig.suptitle(r"CRLB vs $N_0$ (YAML flicker $a$; same $J$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def plot_flicker_compare(
    a_labels: list[str],
    records_by_a: dict[str, dict[str, dict]],
    names: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    mask_names = []
    for recs in records_by_a.values():
        for k in recs:
            if k not in mask_names:
                mask_names.append(k)
    if not mask_names:
        return
    n_p = len(names)
    ncols = 2
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols / 2, 3.8 * nrows))
    axes_f = np.atleast_1d(axes).ravel()
    x = np.arange(len(mask_names))
    width = 0.8 / max(len(a_labels), 1)
    for ax, pname in zip(axes_f, names):
        for k, alab in enumerate(a_labels):
            recs = records_by_a[alab]
            vals = np.array([
                recs.get(m, {}).get("crlb", {}).get(pname, np.nan) for m in mask_names
            ], dtype=float)
            ax.bar(x + (k - len(a_labels) / 2 + 0.5) * width, vals, width, label=alab)
        ax.set_xticks(x)
        ax.set_xticklabels(mask_names, rotation=40, ha="right")
        ax.set_ylabel("CRLB")
        ax.set_title(pname)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes_f[n_p:]:
        ax.set_axis_off()
    fig.suptitle("CRLB: YAML $a$ vs $a=0$ (same $J$, default $N_0$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def run_fim_study(
    cfg: ScatterometryConfig,
    out_dir: Path,
    *,
    plot: bool | None = None,
    layout_only: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if layout_only:
        return write_layout_only(cfg, out_dir)

    find_s4_binary(cfg.paths.s4_bin)
    rows = full_observable_layout(cfg)
    names, p = _true_params(cfg)
    masks = build_masks(cfg, rows)
    print(format_propagating_table(rows))
    print()
    print(format_mask_table(rows, masks))
    equiv = equivalent_mask_names(masks)
    if equiv:
        print("Identical row sets: " + "; ".join(" = ".join(g) for g in equiv))
    n_cond = len(expand_measurement_conditions(cfg))
    n_fwd = n_cond * (1 + len(names))
    print(f"\nComputing full-order J: {n_fwd} S4 forwards ({n_cond} × {1 + len(names)}) ...")

    workers = int(getattr(cfg.library, "condition_workers", 1) or 1)
    jac, r0, steps = full_stored_jacobian(
        cfg, p, names, eps_frac=float(cfg.fim.eps_frac), condition_workers=workers
    )
    if jac.shape[0] != len(rows):
        raise RuntimeError(f"J rows {jac.shape[0]} != layout {len(rows)}")

    scales = param_scales_vector(cfg)
    rcond = float(cfg.fim.rcond)
    default_n0 = float(cfg.inverse.noise.zero.n0_electrons)
    yaml_noise = cfg.inverse.noise

    base_records = {}
    for mname, sel in masks.items():
        base_records[mname] = _mask_metrics(
            jac, r0, rows, sel, yaml_noise, scales, names, rcond, n0=None
        )

    n0_vals = n0_list(cfg)
    sweep_masks = [m for m in cfg.fim.n0_sweep_masks if m in masks]
    n0_sweep: dict[str, dict[str, list[float]]] = {
        m: {n: [] for n in names} for m in sweep_masks
    }
    for n0 in n0_vals:
        for mname in sweep_masks:
            rec = _mask_metrics(
                jac, r0, rows, masks[mname], yaml_noise, scales, names, rcond, n0=n0
            )
            for n in names:
                n0_sweep[mname][n].append(rec["crlb"][n])

    flicker_records: dict[str, dict[str, dict]] = {}
    a_labels: list[str] = []
    for a in cfg.fim.flicker_a:
        noise = noise_with_flicker(yaml_noise, a)
        label = "yaml_a" if a is None else f"a={a:g}"
        a_labels.append(label)
        flicker_records[label] = {}
        for mname in sweep_masks:
            flicker_records[label][mname] = _mask_metrics(
                jac, r0, rows, masks[mname], noise, scales, names, rcond, n0=None
            )

    order_m_full = np.array([r.order_m for r in rows], dtype=int)
    sigma_default = observation_variance(r0, yaml_noise, order_m_full)

    np.savez_compressed(
        out_dir / "jacobian.npz",
        J=jac,
        r0=r0,
        sigma=sigma_default,
        steps=steps,
        order_m=order_m_full,
        azimuth_deg=np.array([r.azimuth_deg for r in rows]),
        wl_nm=np.array([r.wl_nm for r in rows]),
        propagating=np.array([r.propagating for r in rows]),
        param_names=np.array(names),
    )

    payload = {
        "config": str(cfg.config_path) if cfg.config_path else None,
        "structure": {
            "pitch_nm": cfg.structure.pitch_nm,
            **{n: float(cfg.structure.get_param(n)) for n in names},
        },
        "optical": {
            "polarization": cfg.optical.polarization,
            "NG": cfg.optical.NG,
            "n_conditions": n_cond,
            "n_orders_stored": n_orders(cfg),
        },
        "eps_frac": float(cfg.fim.eps_frac),
        "fd_steps": {n: float(s) for n, s in zip(names, steps)},
        "noise": {
            "n0_electrons_default": default_n0,
            "zero": {
                "camera": yaml_noise.zero.camera,
                "relative_a": yaml_noise.zero.relative_a,
                "readout_e_rms": yaml_noise.zero.readout_e_rms,
            },
            "first": {
                "camera": yaml_noise.first.camera,
                "relative_a": yaml_noise.first.relative_a,
                "readout_e_rms": yaml_noise.first.readout_e_rms,
            },
            "sigma_is_observation_variance": True,
            "role_weights_in_FIM": False,
        },
        "layout": layout_payload(cfg, rows),
        "masks": {
            k: {kk: vv for kk, vv in rec.items() if kk != "corr"} | {"corr": rec["corr"]}
            for k, rec in base_records.items()
        },
        "n0_sweep": {"n0_electrons": n0_vals, "flicker": "yaml", "crlb": n0_sweep},
        "flicker_compare": flicker_records,
    }

    (out_dir / "propagating_table.txt").write_text(
        format_propagating_table(rows) + "\n", encoding="utf-8"
    )
    (out_dir / "mask_table.txt").write_text(format_mask_table(rows, masks) + "\n", encoding="utf-8")
    (out_dir / "fim_study.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )

    print("\nCRLB (default N0, YAML a):")
    hdr = f"{'mask':<14}" + "".join(f"{n:>14}" for n in names) + f"{'ρ_d,cd':>10} {'cond':>10}"
    print(hdr)
    for mname, rec in base_records.items():
        cells = []
        for n in names:
            v = rec["crlb"][n]
            cells.append(f"{'inf':>14}" if not np.isfinite(v) else f"{v:14.4g}")
        rho = rec["rho_depth_cd"]
        rho_s = "     nan" if rho is None or not np.isfinite(rho) else f"{rho:10.3f}"
        cond = rec["cond_scaled"]
        cond_s = f"{'inf':>10}" if not np.isfinite(cond) else f"{cond:10.3g}"
        print(f"{mname:<14}" + "".join(cells) + rho_s + cond_s)

    do_plot = cfg.eval.plot if plot is None else plot
    if do_plot:
        dpi = int(cfg.eval.plot_dpi)
        try:
            plot_jacobian_heatmap(
                jac, rows, names, out_dir / "fim_jacobian.png", dpi,
                title=r"Jacobian $\partial R/\partial p$ (propagating rows)",
            )
            plot_jacobian_heatmap(
                jac, rows, names, out_dir / "fim_jacobian_whitened.png", dpi,
                sigma=sigma_default,
                title=r"Whitened Jacobian $\Sigma^{-1/2} J$ (propagating rows)",
            )
            plot_corr_heatmaps(
                base_records, names, list(cfg.fim.corr_masks),
                out_dir / "fim_corr.png", dpi,
            )
            plot_crlb_by_mask(base_records, names, out_dir / "fim_crlb_by_mask.png", dpi)
            plot_rho_cond(base_records, out_dir / "fim_rho_cond.png", dpi)
            plot_crlb_vs_n0(n0_vals, n0_sweep, names, out_dir / "fim_crlb_vs_n0.png", dpi)
            plot_flicker_compare(a_labels, flicker_records, names, out_dir / "fim_flicker.png", dpi)
        except ImportError:
            print("matplotlib not installed; skipped plots (JSON/NPZ still written)")

    print(f"\nSaved: {out_dir / 'fim_study.json'}")
    print(f"Saved: {out_dir / 'jacobian.npz'} (local only; gitignored)")
    return payload


def parse_fim_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-1 Jacobian / FIM study")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Print (φ, m) propagating table and mask counts; do not call S4",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_fim_args(argv)
    cfg = load_config(args.config.resolve())
    out_dir = (Path(__file__).resolve().parent / cfg.paths.output_dir).resolve()
    print(f"Config: {args.config.resolve()}, eval.task={cfg.eval.task}")
    run_fim_study(cfg, out_dir, plot=not args.no_plot, layout_only=args.layout_only)


if __name__ == "__main__":
    main()
