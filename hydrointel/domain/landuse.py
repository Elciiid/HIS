"""Land-use classes and their physical parameters.

Manning values are typical textbook ranges (Chow, *Open-Channel Hydraulics*,
1959, and common 2-D flood-model guidance). Horton rates are order-of-magnitude
values for the surface types listed. None of these are site-calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WATER, ROAD, DENSE_URBAN, RESIDENTIAL, BARE, GRASS, CROPLAND, MANGROVE = range(8)
N_CLASSES = 8


@dataclass(frozen=True)
class LandUseClass:
    code: int
    name: str
    manning: float          # s/m^(1/3)
    f0_mmh: float           # Horton initial rate
    fc_mmh: float           # Horton final rate
    k_per_h: float          # Horton decay constant
    storage_m: float        # baseline sub-grid retention (bunds etc.)
    exposure: float         # relative population/asset index (placeholder)
    color: str


# TODO: site-calibrate every row of this table.
LANDUSE_TABLE: tuple[LandUseClass, ...] = (
    # Chow (1959): clean natural channels 0.025-0.033
    LandUseClass(WATER, "Open water / channel", 0.030, 0.0, 0.0, 0.0, 0.0, 0.0, "#1f4e5f"),
    # asphalt/concrete surfaces 0.011-0.016
    LandUseClass(ROAD, "Paved road", 0.014, 0.0, 0.0, 0.0, 0.0, 0.10, "#9a9a9a"),
    # 2-D urban-block roughness is inflated to represent building blockage
    LandUseClass(DENSE_URBAN, "Dense urban / commercial", 0.050, 2.0, 0.5, 4.0, 0.0, 1.00, "#6b5b5b"),
    LandUseClass(RESIDENTIAL, "Residential / subdivision", 0.045, 10.0, 2.0, 4.0, 0.0, 0.60, "#b58f7a"),
    # bare soil 0.020-0.030
    LandUseClass(BARE, "Bare / cleared ground", 0.025, 40.0, 8.0, 3.0, 0.0, 0.02, "#c9b27c"),
    # short grass 0.030-0.035
    LandUseClass(GRASS, "Grass / open park", 0.035, 60.0, 12.0, 3.0, 0.0, 0.02, "#7fae5a"),
    # mature row crops 0.035-0.045; bunded paddies retain a shallow layer
    LandUseClass(CROPLAND, "Cropland / ricefield", 0.040, 30.0, 6.0, 2.0, 0.05, 0.05, "#b7cf6a"),
    # dense brush / mangrove 0.07-0.16
    LandUseClass(MANGROVE, "Mangrove / coastal vegetation", 0.120, 20.0, 5.0, 2.0, 0.0, 0.01, "#2f6b3a"),
)


def table_array(attr: str) -> np.ndarray:
    return np.array([getattr(c, attr) for c in LANDUSE_TABLE], dtype=np.float64)


def lookup(landuse: np.ndarray, attr: str) -> np.ndarray:
    """Map a class raster to a parameter raster; raises on unknown class codes."""
    lu = np.asarray(landuse)
    if lu.min() < 0 or lu.max() >= N_CLASSES:
        raise ValueError(f"land-use raster contains codes outside 0..{N_CLASSES - 1}: "
                         f"min={lu.min()}, max={lu.max()}")
    return table_array(attr)[lu.astype(np.int64)]


def horton_rate_mmh(landuse: np.ndarray, t_s: float) -> np.ndarray:
    """f(t) = fc + (f0 - fc) * exp(-k t)  [mm/h]."""
    f0, fc, k = lookup(landuse, "f0_mmh"), lookup(landuse, "fc_mmh"), lookup(landuse, "k_per_h")
    return fc + (f0 - fc) * np.exp(-k * t_s / 3600.0)


CLASS_NAMES = {c.code: c.name for c in LANDUSE_TABLE}
