"""Climate (RCP) and socioeconomic (SSP) pathways.

RCP changes the hazard (rainfall intensity, sea level). SSP changes the land
surface and exposure (urban expansion, population, baseline drainage), which in
turn changes roughness and infiltration. Both act on the physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import landuse as LU

# ---------------------------------------------------------------------------
# PLACEHOLDER ranges. TODO: replace with the client's adopted projections; the
# intended source is the IPCC AR6 SSP-RCP scenario pairing for the Philippines.
RCP_TABLE = {
    #          rain factor, SLR 2050 (m), SLR 2100 (m)
    "RCP2.6": (1.05, 0.18, 0.40),
    "RCP4.5": (1.10, 0.22, 0.55),
    "RCP6.0": (1.14, 0.24, 0.62),
    "RCP8.5": (1.22, 0.28, 0.85),
}
SLR_BASE_YEAR = 2020           # SLR taken as 0 at this year and interpolated to the 2050 anchor

# PLACEHOLDER socioeconomic modifiers. TODO: replace with the client's adopted values.
SSP_TABLE = {
    #        urban expansion (fraction of convertible land per decade), population factor, drainage investment
    "SSP1": (0.010, 1.10, 1.15),
    "SSP2": (0.020, 1.25, 1.05),
    "SSP3": (0.035, 1.45, 0.90),
    "SSP4": (0.025, 1.20, 0.95),
    "SSP5": (0.040, 1.35, 1.20),
}
# Conventional pairings. SSP3 is conventionally paired with RCP7.0, which this
# table does not carry, so SSP3 maps to the nearest available pathway, RCP6.0.
SSP_DEFAULT_RCP = {"SSP1": "RCP2.6", "SSP2": "RCP4.5", "SSP3": "RCP6.0", "SSP4": "RCP6.0", "SSP5": "RCP8.5"}
# ---------------------------------------------------------------------------

PATHWAY_NOTE = (
    "Forward-looking climate and socioeconomic pathways set the design tolerance for long-lifecycle assets: "
    "drainage, embankments and retention built now must still perform at the horizon year. Designing to "
    "present-day rainfall, sea level and land cover systematically under-engineers public safety "
    "infrastructure, because intensities, tide levels and impervious area all rise over the asset's life."
)


def slr_m(rcp: str, year: int) -> float:
    _, s50, s100 = RCP_TABLE[_rcp(rcp)]
    if year <= SLR_BASE_YEAR:
        return 0.0
    if year <= 2050:
        return s50 * (year - SLR_BASE_YEAR) / (2050 - SLR_BASE_YEAR)
    if year <= 2100:
        return s50 + (s100 - s50) * (year - 2050) / 50.0
    raise ValueError(f"horizon_year {year} beyond 2100 is outside the projection table")


def rain_factor(rcp: str) -> float:
    return RCP_TABLE[_rcp(rcp)][0]


def _rcp(rcp: str) -> str:
    if rcp not in RCP_TABLE:
        raise ValueError(f"unknown RCP '{rcp}' (expected one of {sorted(RCP_TABLE)})")
    return rcp


@dataclass(frozen=True)
class SSPState:
    ssp: str
    urban_expansion_rate: float
    population_factor: float
    drainage_investment_factor: float


def ssp_state(ssp: str) -> SSPState:
    if ssp not in SSP_TABLE:
        raise ValueError(f"unknown SSP '{ssp}' (expected one of {sorted(SSP_TABLE)})")
    return SSPState(ssp, *SSP_TABLE[ssp])


def expand_urban(landuse: np.ndarray, road: np.ndarray, core_xy, dx: float, rate_per_decade: float,
                 years: float, protected: np.ndarray) -> np.ndarray:
    """Convert cropland/bare/grass to residential (dense urban near the core),
    growing outward from the core along the road network. Deterministic."""
    lu = landuse.copy()
    convertible = np.isin(lu, [LU.CROPLAND, LU.BARE, LU.GRASS]) & ~protected
    n_convert = int(round(rate_per_decade * max(years, 0.0) / 10.0 * convertible.sum()))
    if n_convert <= 0:
        return lu
    ny, nx = lu.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    d_core = np.hypot((X + 0.5) * dx - core_xy[0], (Y + 0.5) * dx - core_xy[1])
    d_road = _distance_to_mask(road, dx)
    score = d_core + 3.0 * d_road                      # growth follows roads
    score = np.where(convertible, score, np.inf)
    flat = np.argsort(score, axis=None, kind="stable")[:n_convert]
    jj, ii = np.unravel_index(flat, lu.shape)
    dense = d_core[jj, ii] < np.percentile(d_core[jj, ii], 25)
    lu[jj, ii] = np.where(dense, LU.DENSE_URBAN, LU.RESIDENTIAL)
    return lu


def _distance_to_mask(mask: np.ndarray, dx: float) -> np.ndarray:
    """Chamfer (3-4) distance transform in pure numpy (two raster sweeps)."""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask) * dx
    except ImportError:
        pass
    big = 1e12
    d = np.where(mask, 0.0, big)
    ny, nx = d.shape
    for j in range(ny):
        for i in range(nx):
            if d[j, i]:
                best = d[j, i]
                if i: best = min(best, d[j, i - 1] + 3)
                if j:
                    best = min(best, d[j - 1, i] + 3)
                    if i: best = min(best, d[j - 1, i - 1] + 4)
                    if i < nx - 1: best = min(best, d[j - 1, i + 1] + 4)
                d[j, i] = best
    for j in range(ny - 1, -1, -1):
        for i in range(nx - 1, -1, -1):
            best = d[j, i]
            if i < nx - 1: best = min(best, d[j, i + 1] + 3)
            if j < ny - 1:
                best = min(best, d[j + 1, i] + 3)
                if i < nx - 1: best = min(best, d[j + 1, i + 1] + 4)
                if i: best = min(best, d[j + 1, i - 1] + 4)
            d[j, i] = best
    return d / 3.0 * dx
