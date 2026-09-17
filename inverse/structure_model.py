"""Trapezoidal grating slicing for layered RCWA."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import StructureConfig


@dataclass(frozen=True)
class SliceLayer:
    thickness_norm: float
    duty: float
    offset_norm: float


def _calcu_mid(top: float, bottom: float, num: int) -> np.ndarray:
    edges = np.linspace(top, bottom, num + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def slice_grating(struct: StructureConfig) -> list[SliceLayer]:
    pitch = struct.pitch_nm
    top_cd = struct.cd_nm
    height = struct.depth_nm
    n = max(1, int(struct.n_slices))
    swa = struct.swa_deg

    if swa == 90.0:
        return [
            SliceLayer(
                thickness_norm=height / pitch,
                duty=top_cd / pitch,
                offset_norm=0.0,
            )
        ]

    cot = 1.0 / math.tan(math.radians(swa)) if swa != 0 else 0.0
    bottom_cd = top_cd + height * (2.0 * cot)
    middle_cd = _calcu_mid(top_cd, bottom_cd, n)
    middle_offs = np.zeros(n, dtype=float)

    thickness_norm = (height / n) / pitch
    # Top slice first (nearest Air), matching S4 AddLayer order after superstrate.
    return [
        SliceLayer(
            thickness_norm=thickness_norm,
            duty=middle_cd[i] / pitch,
            offset_norm=middle_offs[i] / pitch,
        )
        for i in range(n)
    ]
