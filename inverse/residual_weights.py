"""SNR-based sqrt weights for inverse residuals (shared by GA/LM and decoupling)."""
from __future__ import annotations

import numpy as np


def snr_weights_block(r_meas: np.ndarray, inv) -> np.ndarray:
    n = len(r_meas)
    snr = r_meas / (inv.noise_level + 1e-30)
    w_snr = np.minimum(snr, 3.0)
    w_int = r_meas / (max(float(r_meas.max()), 1e-30))
    w_uni = np.ones(n) / n
    w = np.zeros(n)
    low = snr < inv.noise_threshold
    mid = (snr >= inv.noise_threshold) & (snr < 2 * inv.noise_threshold)
    high = snr >= 2 * inv.noise_threshold
    w[low] = 0.1 * w_uni[low] + 0.9 * w_snr[low]
    w[mid] = 0.3 * w_int[mid] + 0.7 * w_snr[mid]
    w[high] = 0.8 * w_int[high] + 0.2 * w_snr[high]
    w /= w.sum() + 1e-30
    return np.sqrt(w)


def snr_weights(
    r_meas: np.ndarray,
    inv,
    *,
    block_sizes: list[int] | None = None,
    n_orders: int | None = None,
) -> np.ndarray:
    if block_sizes is not None:
        parts: list[np.ndarray] = []
        offset = 0
        for size in block_sizes:
            parts.append(snr_weights_block(r_meas[offset : offset + size], inv))
            offset += size
        wsqrt = np.concatenate(parts)
    elif n_orders is None or len(r_meas) <= n_orders:
        return snr_weights_block(r_meas, inv)
    else:
        n_cond = len(r_meas) // n_orders
        parts = [
            snr_weights_block(r_meas[i * n_orders : (i + 1) * n_orders], inv)
            for i in range(n_cond)
        ]
        wsqrt = np.concatenate(parts)
    w = wsqrt ** 2
    w /= w.sum() + 1e-30
    return np.sqrt(w)
