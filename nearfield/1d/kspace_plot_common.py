"""K-space display helpers for S4 near-field plots."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def mirror_kspace_for_display(kx_axis, disp) -> tuple:
    """Flip spectrum + kx then sort ascending (optional display convention)."""
    import numpy as np

    kx = np.asarray(kx_axis, dtype=float)
    d = np.asarray(disp, dtype=float)
    kx = np.flip(kx)
    d = np.flip(d)
    order = np.argsort(kx)
    return kx[order], d[order]


def reflect_kx_about_zero(kx_axis, disp) -> tuple:
    """Mirror kx -> -kx (keep paired amplitudes), then sort ascending."""
    import numpy as np

    kx = np.asarray(kx_axis, dtype=float)
    d = np.asarray(disp, dtype=float)
    kx_new = -kx
    order = np.argsort(kx_new)
    return kx_new[order], d[order]
