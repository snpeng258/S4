#!/usr/bin/env python3
"""
Angular spectrum rearrangement (ASR) utilities — Python port of
/home/psn/Angular-spectrum-rearrangement (ScalarDiffraction_ASR_AP.m, mdft.m, midft.m).

Algorithm: Liu Xin / Hu Yiwen; licensed CC BY-NC 4.0 (see upstream repo).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse


def mdft(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    """2D forward DFT via matrix triple product (MTP), matching mdft.m."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    fx = np.asarray(fx, dtype=float).reshape(1, -1)
    fy = np.asarray(fy, dtype=float).reshape(-1, 1)
    mx = np.exp(-2j * np.pi * x * fx)
    my = np.exp(-2j * np.pi * fy * y)
    return my @ np.asarray(field, dtype=complex) @ mx


def midft(
    spectrum: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    """2D inverse DFT via MTP, matching midft.m."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    fx = np.asarray(fx, dtype=float).reshape(-1)
    fy = np.asarray(fy, dtype=float).reshape(-1)
    spec = spectrum.toarray() if sparse.issparse(spectrum) else np.asarray(spectrum, dtype=complex)
    n_out = x.size * y.size
    n_spec = fx.size * fy.size
    # Large k-grids (small aperture / wide far-field patch) need BLAS matmul, not dense MTP.
    if n_spec * n_out > 256 * 256 * 256:
        mx = np.exp(2j * np.pi * np.outer(fx, x))
        my = np.exp(2j * np.pi * np.outer(y, fy))
        return my @ spec @ mx
    x_row = x.reshape(1, -1)
    y_col = y.reshape(-1, 1)
    fx_col = fx.reshape(-1, 1)
    fy_row = fy.reshape(1, -1)
    mx = np.exp(2j * np.pi * fx_col * x_row)
    my = np.exp(2j * np.pi * y_col * fy_row)
    return my @ spec @ mx


def unique_tol(a: np.ndarray, num: int) -> tuple[np.ndarray, np.ndarray]:
    """Port of unique_tol.m — merge frequency samples with tolerance."""
    a = np.asarray(a, dtype=float).ravel()
    a_u = np.unique(a)
    n = a_u.size
    if n <= num:
        ic = np.searchsorted(a_u, a)
        return a_u, ic

    merged = _merge_closest_pair(a_u, num)
    ic = np.argmin(np.abs(a[:, None] - merged[None, :]), axis=1)
    return merged, ic


def _merge_closest_pair(a: np.ndarray, target: int) -> np.ndarray:
    """Repeatedly merge closest adjacent frequencies until len(a) <= target."""
    a = np.sort(a)
    while a.size > target:
        da = np.diff(a)
        i = int(np.argmin(da))
        merged = 0.5 * (a[i] + a[i + 1])
        a = np.concatenate([a[:i], [merged], a[i + 2 :]])
    return a


def k_rotate_asr(
    fxx: np.ndarray,
    fyy: np.ndarray,
    fzz: np.ndarray,
    theta2: float,
    phi2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate horizontal spectrum to tilted (u,v,w) frequencies."""
    c2, s2 = math.cos(theta2), math.sin(theta2)
    c1, s1 = math.cos(phi2), math.sin(phi2)
    fuu = c2 * c1 * fxx + c2 * s1 * fyy - s2 * fzz
    fvv = -s1 * fxx + c1 * fyy
    fww = s2 * c1 * fxx + s2 * s1 * fyy + c2 * fzz
    return fuu, fvv, fww


def asr_frequency_grid(
    *,
    pixel_size_um: float,
    n_x: int,
    n_y: int,
    wavelength_um: float,
    z_um: float,
    source_span_u_um: float,
    source_span_v_um: float,
    oversample: float = 2.0,
    max_n_freq: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fx, fy, fzz grids (1/um) per ScalarDiffraction_ASR_AP sampling rules."""
    lfx = 1.0 / pixel_size_um
    lfy = 1.0 / pixel_size_um
    fmax_fft = 1.0 / (2.0 * pixel_size_um)
    min_z = max(z_um, 1e-9)
    df_max1 = math.sqrt(max(1.0 - (wavelength_um * fmax_fft) ** 2, 0.0)) / (
        wavelength_um * min_z * max(fmax_fft, 1e-30)
    )
    dfx_max2 = 1.0 / max(source_span_u_um, 1e-9)
    dfy_max2 = 1.0 / max(source_span_v_um, 1e-9)
    dfx = min(dfx_max2, df_max1)
    dfy = min(dfy_max2, df_max1)
    lrfx = min(max(int(math.ceil(lfx / dfx * oversample)), n_x), max_n_freq)
    lrfy = min(max(int(math.ceil(lfy / dfy * oversample)), n_y), max_n_freq)
    fx = np.linspace(-lfx / 2, lfx / 2, lrfx)
    fy = np.linspace(-lfy / 2, lfy / 2, lrfy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fzz = np.sqrt(np.maximum(1.0 / wavelength_um**2 - fxx**2 - fyy**2, 0.0))
    fzz[wavelength_um**2 * (fxx**2 + fyy**2) > 1.0] = np.nan
    return fx, fy, fzz


def asr_rearrange_spectrum(
    fd: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fzz: np.ndarray,
    *,
    wavelength_um: float,
    theta2: float,
    phi2: float,
    number_u: int,
    number_v: int,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray, np.ndarray]:
    """ASR spectrum merge on tilted plane; returns (fw, fu_eff, fv_eff)."""
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fuu, fvv, fww = k_rotate_asr(fxx, fyy, fzz, theta2, phi2)
    del fww

    spa = wavelength_um**2 * (fxx**2 + fyy**2) <= 1.0
    fu_flat = fuu[spa]
    fv_flat = fvv[spa]
    fd_flat = fd[spa]

    if abs(theta2) < 1e-15 and abs(phi2) < 1e-15:
        return fd, fx, fy

    fu_temp, idx_fu = unique_tol(fu_flat, number_u)
    fv_temp, idx_fv = unique_tol(fv_flat, number_v)
    n_fu = fu_temp.size
    n_fv = fv_temp.size

    fw = sparse.coo_matrix((fd_flat, (idx_fv, idx_fu)), shape=(n_fv, n_fu)).tocsr()
    abs_fw = sparse.coo_matrix((np.abs(fd_flat), (idx_fv, idx_fu)), shape=(n_fv, n_fu)).tocsr()
    fu_fw = sparse.coo_matrix((np.abs(fd_flat) * fu_flat, (idx_fv, idx_fu)), shape=(n_fv, n_fu)).tocsr()
    fv_fw = sparse.coo_matrix((np.abs(fd_flat) * fv_flat, (idx_fv, idx_fu)), shape=(n_fv, n_fu)).tocsr()

    sum_abs = np.asarray(abs_fw.sum(axis=0)).ravel()
    sum_abs_row = np.asarray(abs_fw.sum(axis=1)).ravel()
    fu_eff = fu_temp.copy()
    fv_eff = fv_temp.copy()
    nz = sum_abs > 0
    if nz.any():
        fu_eff[nz] = np.asarray(fu_fw.sum(axis=0)).ravel()[nz] / sum_abs[nz]
    nzr = sum_abs_row > 0
    if nzr.any():
        fv_eff[nzr] = np.asarray(fv_fw.sum(axis=1)).ravel()[nzr] / sum_abs_row[nzr]
    return fw, fu_eff, fv_eff


def scalar_asr_propagate(
    e0: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    *,
    wavelength_um: float,
    z_um: float,
    us: np.ndarray,
    vs: np.ndarray,
    theta2: float,
    phi2: float,
    ws_um: float = 0.0,
    number_u: int = 512,
    number_v: int = 512,
    source_span_u_um: float | None = None,
    source_span_v_um: float | None = None,
    max_n_freq: int = 512,
) -> np.ndarray:
    """
    Scalar ASR: source plane field e0(x,y) -> observation plane eout(u,v).

    e0 shape (ny, nx) on grids yd (1, nx), xd (ny, 1) style meshgrid indexing='xy'.
    """
    ny, nx = e0.shape
    pixel_x = (float(np.max(xd)) - float(np.min(xd))) / max(nx - 1, 1)
    pixel_y = (float(np.max(yd)) - float(np.min(yd))) / max(ny - 1, 1)
    pixel_size = min(pixel_x, pixel_y)

    if source_span_u_um is None:
        source_span_u_um = (float(np.max(xd)) - float(np.min(xd))) or pixel_size
    if source_span_v_um is None:
        source_span_v_um = (float(np.max(yd)) - float(np.min(yd))) or pixel_size

    fx, fy, fzz = asr_frequency_grid(
        pixel_size_um=pixel_size,
        n_x=nx,
        n_y=ny,
        wavelength_um=wavelength_um,
        z_um=z_um,
        source_span_u_um=source_span_u_um,
        source_span_v_um=source_span_v_um,
        max_n_freq=max_n_freq,
    )

    f0 = mdft(e0, xd[0, :], yd[:, 0], fx, fy)
    h = np.exp(1j * 2 * np.pi * z_um * fzz)
    fww = fzz  # ws offset along w
    fd = f0 * h * np.exp(1j * 2 * np.pi * ws_um * fww)

    fw, fu_eff, fv_eff = asr_rearrange_spectrum(
        fd,
        fx,
        fy,
        fzz,
        wavelength_um=wavelength_um,
        theta2=theta2,
        phi2=phi2,
        number_u=number_u,
        number_v=number_v,
    )

    spec = fw.toarray() if sparse.issparse(fw) else fw
    return midft(spec, us, vs, fu_eff, fv_eff)


def observation_angles_from_k(
    kx_um: float,
    ky_um: float,
    kz_um: float,
) -> tuple[float, float]:
    """
    ASR (theta2, phi2) so observation-plane normal aligns with unit k_out.

    Normal from ASR: (sin(t2)cos(p2), sin(t2)sin(p2), cos(t2)).
    """
    kn = math.sqrt(kx_um**2 + ky_um**2 + kz_um**2)
    if kn < 1e-30:
        return 0.0, 0.0
    nx = kx_um / kn
    ny = ky_um / kn
    nz = kz_um / kn
    nz = max(min(nz, 1.0), -1.0)
    theta2 = math.acos(nz)
    phi2 = math.atan2(ny, nx)
    return theta2, phi2


def s4_k_to_spatial_freq_um(k_norm: float, period_nm: float) -> float:
    """S4 normalized k to ASR spatial frequency f [1/um]; k_phys=2πf."""
    k_phys_per_nm = k_norm * 2.0 * math.pi / period_nm
    k_phys_per_um = k_phys_per_nm * 1000.0
    return k_phys_per_um / (2.0 * math.pi)


def vacuum_propagating_fxy_from_mode(
    kx_norm: float,
    ky_norm: float,
    kz_norm: float,
    *,
    wavelength_um: float,
) -> tuple[float, float] | None:
    """
    Map S4 Floquet (kx, ky, kz) to vacuum ASR frequencies (fx, fy) in 1/μm.

    Scale |k| to k0=2π/λ in vacuum; propagating only when (λ·f)²<1 for mdft/ASR.
    """
    kn = math.sqrt(kx_norm**2 + ky_norm**2 + kz_norm**2)
    if kn < 1e-30 or kz_norm >= 0:
        return None
    k0 = 2.0 * math.pi / wavelength_um
    kx = k0 * kx_norm / kn
    ky = k0 * ky_norm / kn
    fx = kx / (2.0 * math.pi)
    fy = ky / (2.0 * math.pi)
    if wavelength_um**2 * (fx**2 + fy**2) >= 1.0:
        return None
    return fx, fy


def elliptical_aperture_spectrum(
    fxx: np.ndarray,
    fyy: np.ndarray,
    *,
    sigma_u_um: float,
    sigma_v_um: float,
    fx0: float = 0.0,
    fy0: float = 0.0,
) -> np.ndarray:
    """FT of amplitude Gaussian W=exp(-x²/2σ_u² - y²/2σ_v²); f in 1/μm."""
    return np.exp(
        -2.0 * math.pi**2 * (sigma_u_um**2 * (fxx - fx0) ** 2 + sigma_v_um**2 * (fyy - fy0) ** 2)
    )


def build_aperture_spectrum_mdft(
    fx: np.ndarray,
    fy: np.ndarray,
    *,
    amplitude: complex,
    sigma_u_um: float,
    sigma_v_um: float,
    n_aperture: int = 128,
) -> np.ndarray:
    """Angular spectrum of amplitude Gaussian aperture via mdft (matches ASR convention)."""
    half_u = max(3.0 * sigma_u_um, sigma_u_um)
    half_v = max(3.0 * sigma_v_um, sigma_v_um)
    x_um = np.linspace(-half_u, half_u, n_aperture)
    y_um = np.linspace(-half_v, half_v, n_aperture)
    xg, yg = np.meshgrid(x_um, y_um, indexing="xy")
    e0 = amplitude * np.exp(-0.5 * (xg / sigma_u_um) ** 2 - 0.5 * (yg / sigma_v_um) ** 2)
    return mdft(e0, xg[0, :], yg[:, 0], fx, fy)


def shift_spectrum_grid(
    f0: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fxx: np.ndarray,
    fyy: np.ndarray,
    *,
    fx_shift: float,
    fy_shift: float,
) -> np.ndarray:
    """F(f - f_m) by interpolating base spectrum on (fxx - fx_m, fyy - fy_m)."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (fy, fx),
        f0,
        bounds_error=False,
        fill_value=0.0,
    )
    pts = np.column_stack([(fyy - fy_shift).ravel(), (fxx - fx_shift).ravel()])
    return interp(pts).reshape(fyy.shape).astype(complex)


def order_spectrum_k(
    fxx: np.ndarray,
    fyy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    f_aperture: np.ndarray,
    *,
    fx_m: float,
    fy_m: float,
) -> np.ndarray:
    """F_m(f) = F_aperture(f - f_m) — k-domain convolution with shifted aperture spectrum."""
    return shift_spectrum_grid(f_aperture, fx, fy, fxx, fyy, fx_shift=fx_m, fy_shift=fy_m)


def observation_grid_asr_auto(
    us_um: np.ndarray,
    vs_um: np.ndarray,
    *,
    wavelength_um: float,
    z_um: float,
    theta2: float,
    phi2: float,
    aperture_span_u_um: float,
    aperture_span_v_um: float,
    ws_um: float = 0.0,
    oversample: float = 2.0,
    min_n_freq: int = 64,
    max_n_freq: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fx, fy, fzz (1/μm) per ScalarDiffraction_ASR_AP.m lines 52–95.

    us_um, vs_um: observation-plane sample coordinates (μm), m=0-normal ASR frame.
    """
    c2, s2 = math.cos(theta2), math.sin(theta2)
    c1, s1 = math.cos(phi2), math.sin(phi2)
    uus, vvs = np.meshgrid(us_um, vs_um, indexing="xy")
    xxs = c2 * c1 * uus - s1 * vvs + s2 * c1 * ws_um
    yys = c2 * s1 * uus + c1 * vvs + s2 * s1 * ws_um
    zzs = -s2 * uus + c2 * ws_um
    lx = float(np.max(xxs) - np.min(xxs))
    ly = float(np.max(yys) - np.min(yys))
    min_z = max(z_um + float(np.min(zzs)), 1e-9)

    pixel_u = max(aperture_span_u_um / max(min_n_freq - 1, 1), 1e-9)
    pixel_v = max(aperture_span_v_um / max(min_n_freq - 1, 1), 1e-9)
    pixel_size = min(pixel_u, pixel_v)

    lfx = 1.0 / pixel_size
    lfy = 1.0 / pixel_size
    fmax_fft = 1.0 / (2.0 * pixel_size)
    inv_lam = 1.0 / wavelength_um
    df_max1 = math.sqrt(max(1.0 - (wavelength_um * fmax_fft) ** 2, 0.0)) / (
        wavelength_um * min_z * max(fmax_fft, 1e-30)
    )
    dfx_max2 = 1.0 / max(lx, 1e-9)
    dfy_max2 = 1.0 / max(ly, 1e-9)
    dfx_ap = 1.0 / max(aperture_span_u_um, 1e-9)
    dfy_ap = 1.0 / max(aperture_span_v_um, 1e-9)
    dfx = min(dfx_max2, df_max1, dfx_ap)
    dfy = min(dfy_max2, df_max1, dfy_ap)
    lrfx = min(max(int(math.ceil(lfx / dfx * oversample)), min_n_freq), max_n_freq)
    lrfy = min(max(int(math.ceil(lfy / dfy * oversample)), min_n_freq), max_n_freq)
    fx = np.linspace(-lfx / 2, lfx / 2, lrfx)
    fy = np.linspace(-lfy / 2, lfy / 2, lrfy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fzz = np.sqrt(np.maximum(inv_lam**2 - fxx**2 - fyy**2, 0.0))
    fzz[wavelength_um**2 * (fxx**2 + fyy**2) > 1.0] = np.nan
    return fx, fy, fzz


def order_local_propagation_transfer(
    fx: np.ndarray,
    fy: np.ndarray,
    fzz: np.ndarray,
    *,
    fx_m: float,
    fy_m: float,
    wavelength_um: float,
    z_um: float,
) -> np.ndarray:
    """
    Order-local propagator: keep fzz curvature (divergence), drop linear tilt at (fx_m, fy_m).

    The linear part of fzz(fx,fy) around the order carrier is the gross beam redirect to the
    Floquet landing point (already in scr.x_center_mm). Applying it again in a patch centered
    on that landing double-counts O(z·fx/fzz) ~ mm shifts and breaks high-|m| spots.
    """
    inv_lam = 1.0 / wavelength_um
    fzz_m = math.sqrt(max(inv_lam**2 - fx_m**2 - fy_m**2, 0.0))
    if fzz_m < 1e-30:
        return np.exp(1j * 2.0 * math.pi * z_um * fzz)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fzz_lin = fzz_m + (-fx_m / fzz_m) * (fxx - fx_m) + (-fy_m / fzz_m) * (fyy - fy_m)
    dz = np.where(np.isfinite(fzz), fzz - fzz_lin, 0.0)
    return np.exp(1j * 2.0 * math.pi * z_um * dz)


def scalar_asr_from_spectrum(
    f0: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fzz: np.ndarray,
    us_um: np.ndarray,
    vs_um: np.ndarray,
    *,
    wavelength_um: float,
    z_um: float,
    theta2: float,
    phi2: float,
    number_u: int,
    number_v: int,
    ws_um: float = 0.0,
    fx_m: float | None = None,
    fy_m: float | None = None,
) -> np.ndarray:
    """k-domain spectrum F0 -> propagate -> ASR rotate (m0-normal) -> midft -> E(u,v)."""
    if fx_m is not None and fy_m is not None:
        h = order_local_propagation_transfer(
            fx,
            fy,
            fzz,
            fx_m=fx_m,
            fy_m=fy_m,
            wavelength_um=wavelength_um,
            z_um=z_um,
        )
    else:
        h = np.exp(1j * 2.0 * math.pi * z_um * fzz)
    fd = f0 * h * np.exp(1j * 2.0 * math.pi * ws_um * fzz)
    fd = np.where(np.isfinite(fd), fd, 0.0)

    fw, fu_eff, fv_eff = asr_rearrange_spectrum(
        fd,
        fx,
        fy,
        fzz,
        wavelength_um=wavelength_um,
        theta2=theta2,
        phi2=phi2,
        number_u=number_u,
        number_v=number_v,
    )
    spec = fw.toarray() if sparse.issparse(fw) else fw
    return midft(spec, us_um, vs_um, fu_eff, fv_eff)


def fu_fv_on_m0_plane(
    fx: float,
    fy: float,
    *,
    wavelength_um: float,
    theta2: float,
    phi2: float,
) -> tuple[float, float]:
    """Rotate horizontal spatial frequencies to m=0-normal (u,v) frame."""
    inv_lam = 1.0 / wavelength_um
    fzz = math.sqrt(max(inv_lam**2 - fx**2 - fy**2, 0.0))
    fu, fv, _ = k_rotate_asr(
        np.array([[fx]]), np.array([[fy]]), np.array([[fzz]]), theta2, phi2
    )
    return float(fu[0, 0]), float(fv[0, 0])


def gaussian_bandwidth_1_per_um(sigma_um: float) -> float:
    """1/e² bandwidth in spatial frequency (1/μm) for amplitude-Gaussian aperture."""
    return 1.0 / (math.pi * max(sigma_um, 1e-9))


def sigma_theta_rad(*, wavelength_um: float, waist_sigma_um: float) -> float:
    """
    Gaussian-beam 1/e² half-angle (rad): σ_θ ≈ λ / (π w₀).

    waist_sigma_um: amplitude 1/e² half-width at the source plane (μm).
    """
    if waist_sigma_um <= 0:
        return 0.0
    return wavelength_um / (math.pi * waist_sigma_um)


def far_field_spot_sigma_mm(
    *,
    z_mm: float,
    wavelength_nm: float,
    waist_1e2_full_um: float,
) -> float:
    """Far-field spot 1/e² half-width (mm): σ ≈ z · σ_θ at z ≫ z_R."""
    waist_sigma_um = 0.5 * waist_1e2_full_um
    if waist_sigma_um <= 0:
        return 0.0
    wl_um = wavelength_nm / 1000.0
    return abs(z_mm) * sigma_theta_rad(wavelength_um=wl_um, waist_sigma_um=waist_sigma_um)


def rayleigh_range_mm(*, wavelength_nm: float, waist_1e2_full_um: float) -> float:
    """Rayleigh range z_R = π w₀² / λ (mm)."""
    waist_sigma_um = 0.5 * waist_1e2_full_um
    if waist_sigma_um <= 0:
        return 0.0
    wl_um = wavelength_nm / 1000.0
    return math.pi * waist_sigma_um**2 / wl_um / 1000.0


def beam_divergence_info(
    *,
    wavelength_nm: float,
    aperture_u_um: float,
    aperture_v_um: float,
    z_mm: float,
) -> dict[str, float]:
    """Summary of σ_θ and far-field spot sizes from elliptical aperture waist."""
    wl_um = wavelength_nm / 1000.0
    sigma_u = 0.5 * aperture_u_um
    sigma_v = 0.5 * aperture_v_um
    st_u = sigma_theta_rad(wavelength_um=wl_um, waist_sigma_um=sigma_u)
    st_v = sigma_theta_rad(wavelength_um=wl_um, waist_sigma_um=sigma_v)
    spot_u = far_field_spot_sigma_mm(
        z_mm=z_mm, wavelength_nm=wavelength_nm, waist_1e2_full_um=aperture_u_um
    )
    spot_v = far_field_spot_sigma_mm(
        z_mm=z_mm, wavelength_nm=wavelength_nm, waist_1e2_full_um=aperture_v_um
    )
    z_r = rayleigh_range_mm(wavelength_nm=wavelength_nm, waist_1e2_full_um=min(aperture_u_um, aperture_v_um))
    return {
        "sigma_theta_u_mrad": st_u * 1e3,
        "sigma_theta_v_mrad": st_v * 1e3,
        "spot_sigma_u_mm": spot_u,
        "spot_sigma_v_mm": spot_v,
        "rayleigh_range_mm": z_r,
        "z_over_zr": abs(z_mm) / max(z_r, 1e-30),
    }


def order_spectrum_k_gaussian(
    fxx: np.ndarray,
    fyy: np.ndarray,
    *,
    sigma_u_um: float,
    sigma_v_um: float,
    fx_m: float,
    fy_m: float,
    amplitude: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """
    Per-order angular spectrum with Gaussian beam divergence.

    F_m(f) = A · exp[-2π²(σ_u²(fx-fx_m)² + σ_v²(fy-fy_m)²)]
    i.e. k-domain convolution of a delta at f_m with the aperture spectrum.
    """
    return amplitude * elliptical_aperture_spectrum(
        fxx,
        fyy,
        sigma_u_um=sigma_u_um,
        sigma_v_um=sigma_v_um,
        fx0=fx_m,
        fy0=fy_m,
    )


def build_order_propagation_k_grid(
    fx_m: float,
    fy_m: float,
    *,
    us_rel_um: np.ndarray,
    vs_rel_um: np.ndarray,
    wavelength_um: float,
    z_um: float,
    theta2: float,
    phi2: float,
    sigma_u_um: float,
    sigma_v_um: float,
    aperture_span_u_um: float,
    aperture_span_v_um: float,
    asr_n_u: int,
    asr_n_v: int,
    sigma_theta_span: float = 5.0,
    max_k_points: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k grid centered on order (fx_m, fy_m) with df from observation Nyquist (ASR auto grid).
    """
    u_max = float(np.max(np.abs(us_rel_um))) if us_rel_um.size else 1.0
    v_max = float(np.max(np.abs(vs_rel_um))) if vs_rel_um.size else 1.0
    df_nyq_u = 1.0 / (2.0 * max(u_max, 1e-9))
    df_nyq_v = 1.0 / (2.0 * max(v_max, 1e-9))

    inv_lam = 1.0 / wavelength_um
    bw_u = gaussian_bandwidth_1_per_um(sigma_u_um)
    bw_v = gaussian_bandwidth_1_per_um(sigma_v_um)
    margin_u = sigma_theta_span * bw_u
    margin_v = sigma_theta_span * bw_v
    fx_lo = max(fx_m - margin_u, -inv_lam * 0.99)
    fx_hi = min(fx_m + margin_u, inv_lam * 0.99)
    fy_lo = max(fy_m - margin_v, -inv_lam * 0.99)
    fy_hi = min(fy_m + margin_v, inv_lam * 0.99)

    span_fx = max(fx_hi - fx_lo, bw_u)
    span_fy = max(fy_hi - fy_lo, bw_v)
    nfx_req = max(int(math.ceil(span_fx / df_nyq_u)) | 1, 64)
    nfy_req = max(int(math.ceil(span_fy / df_nyq_v)) | 1, 64)
    max_n = min(max(asr_n_u, asr_n_v, nfx_req, nfy_req, 128), max_k_points)

    fx_obs, fy_obs, _ = observation_grid_asr_auto(
        us_rel_um,
        vs_rel_um,
        wavelength_um=wavelength_um,
        z_um=z_um,
        theta2=theta2,
        phi2=phi2,
        aperture_span_u_um=aperture_span_u_um,
        aperture_span_v_um=aperture_span_v_um,
        min_n_freq=64,
        max_n_freq=max_n,
    )
    dfx_obs = float(abs(fx_obs[1] - fx_obs[0])) if len(fx_obs) > 1 else 1e-9
    dfy_obs = float(abs(fy_obs[1] - fy_obs[0])) if len(fy_obs) > 1 else 1e-9

    dfx = min(dfx_obs, bw_u * 0.5, df_nyq_u)
    dfy = min(dfy_obs, bw_v * 0.5, df_nyq_v)
    dfx = max(dfx, 1e-12)
    dfy = max(dfy, 1e-12)
    nfx = min(max(int(math.ceil(span_fx / dfx)) | 1, 64), max_n)
    nfy = min(max(int(math.ceil(span_fy / dfy)) | 1, 64), max_n)
    fx = np.linspace(fx_lo, fx_hi, nfx)
    fy = np.linspace(fy_lo, fy_hi, nfy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fzz = np.sqrt(np.maximum(inv_lam**2 - fxx**2 - fyy**2, 0.0))
    fzz[wavelength_um**2 * (fxx**2 + fyy**2) > 1.0] = np.nan
    return fx, fy, fzz


def build_propagation_k_grid(
    fx_orders: list[float],
    fy_orders: list[float],
    *,
    wavelength_um: float,
    sigma_u_um: float,
    sigma_v_um: float,
    aperture_span_u_um: float,
    aperture_span_v_um: float,
    oversample: float = 2.0,
    min_n_freq: int = 128,
    max_n_freq: int = 512,
    sigma_theta_span: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    fx, fy grid (1/μm) covering each order peak ± σ_θ bandwidth in k-space.

    σ_θ = λ/(π w₀) ⇒ k-bandwidth Δf ≈ 1/(π w₀) (same as Gaussian aperture FT).
    """
    inv_lam = 1.0 / wavelength_um
    bw_u = gaussian_bandwidth_1_per_um(sigma_u_um)
    bw_v = gaussian_bandwidth_1_per_um(sigma_v_um)
    margin_u = sigma_theta_span * bw_u
    margin_v = sigma_theta_span * bw_v
    if fx_orders:
        fx_lo = min(fx_orders) - margin_u
        fx_hi = max(fx_orders) + margin_u
    else:
        fx_lo, fx_hi = -inv_lam * 0.1, inv_lam * 0.1
    if fy_orders:
        fy_lo = min(fy_orders) - margin_v
        fy_hi = max(fy_orders) + margin_v
    else:
        fy_lo, fy_hi = -inv_lam * 0.1, inv_lam * 0.1
    fx_lo = max(fx_lo, -inv_lam * 0.99)
    fx_hi = min(fx_hi, inv_lam * 0.99)
    fy_lo = max(fy_lo, -inv_lam * 0.99)
    fy_hi = min(fy_hi, inv_lam * 0.99)

    dfx = bw_u / max(oversample, 1.0)
    dfy = bw_v / max(oversample, 1.0)
    span_fx = max(fx_hi - fx_lo, dfx)
    span_fy = max(fy_hi - fy_lo, dfy)
    nfx = min(max(int(math.ceil(span_fx / dfx)) | 1, min_n_freq), max_n_freq)
    nfy = min(max(int(math.ceil(span_fy / dfy)) | 1, min_n_freq), max_n_freq)
    fx = np.linspace(fx_lo, fx_hi, nfx)
    fy = np.linspace(fy_lo, fy_hi, nfy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fzz = np.sqrt(np.maximum(inv_lam**2 - fxx**2 - fyy**2, 0.0))
    fzz[wavelength_um**2 * (fxx**2 + fyy**2) > 1.0] = np.nan
    return fx, fy, fzz


def observation_scope_uv_um(
    fx_orders: list[float],
    fy_orders: list[float],
    *,
    wavelength_um: float,
    z_um: float,
    theta2: float,
    phi2: float,
    sigma_u_um: float,
    sigma_v_um: float,
    obs_span_mm: float | None,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scope.us / Scope.vs (μm) on m=0-normal plane for midft output."""
    spot_hw = 8.0 * wavelength_um * z_um / (math.pi * max(min(sigma_u_um, sigma_v_um), 1e-9))
    if obs_span_mm is not None:
        half = 0.5 * obs_span_mm * 1000.0
        return np.linspace(-half, half, n_obs), np.linspace(-half, half, n_obs)

    inv_lam = 1.0 / wavelength_um
    u_pts: list[float] = [0.0]
    v_pts: list[float] = [0.0]
    for fx, fy in zip(fx_orders, fy_orders):
        if wavelength_um**2 * (fx**2 + fy**2) >= 1.0:
            continue
        fu, fv = fu_fv_on_m0_plane(fx, fy, wavelength_um=wavelength_um, theta2=theta2, phi2=phi2)
        u_pts.append(z_um * fu / inv_lam)
        v_pts.append(z_um * fv / inv_lam)
    u_half = max((max(u_pts) - min(u_pts)) * 0.5 + spot_hw, spot_hw, 5000.0)
    v_half = max((max(v_pts) - min(v_pts)) * 0.5 + spot_hw, spot_hw, 5000.0)
    return np.linspace(-u_half, u_half, n_obs), np.linspace(-v_half, v_half, n_obs)
