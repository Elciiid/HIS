"""Procedural satellite-style basemap derived from the land-use and DEM rasters.

It is a rendering of the synthetic rasters, not imagery. Captions say so.
"""
from __future__ import annotations

import numpy as np

from ..config import numpy_rng
from ..domain import landuse as LU

BASEMAP_CAPTION = "Synthetic basemap (procedural)"

# RGB base tones per class (0-1)
_TONES = {
    LU.WATER: (0.10, 0.22, 0.24),
    LU.ROAD: (0.55, 0.55, 0.53),
    LU.DENSE_URBAN: (0.46, 0.44, 0.43),
    LU.RESIDENTIAL: (0.58, 0.52, 0.47),
    LU.BARE: (0.66, 0.58, 0.44),
    LU.GRASS: (0.33, 0.45, 0.24),
    LU.CROPLAND: (0.42, 0.52, 0.27),
    LU.MANGROVE: (0.12, 0.26, 0.15),
}


def hillshade(z: np.ndarray, dx: float, azimuth=315.0, altitude=45.0, exaggeration=3.0) -> np.ndarray:
    gy, gx = np.gradient(z * exaggeration, dx)
    # rows run south->north, so +gy is a northward slope
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def render(landuse: np.ndarray, dem: np.ndarray, dx: float, seed: int = 0,
           building: np.ndarray | None = None) -> np.ndarray:
    """Return an (ny, nx, 3) float RGB raster with origin at the south-west."""
    rng = numpy_rng(f"basemap/{seed}")
    ny, nx = landuse.shape
    rgb = np.zeros((ny, nx, 3))
    for c, tone in _TONES.items():
        rgb[landuse == c] = tone
    tex = rng.standard_normal((ny, nx))
    tex_s = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1) + np.roll(tex, -1, 0) + np.roll(tex, -1, 1)) / 5
    veg = np.isin(landuse, [LU.GRASS, LU.CROPLAND, LU.MANGROVE])
    urb = np.isin(landuse, [LU.DENSE_URBAN, LU.RESIDENTIAL])
    amp = np.where(veg, 0.07, np.where(urb, 0.09, np.where(landuse == LU.WATER, 0.02, 0.04)))
    rgb *= (1 + amp * tex_s)[..., None]
    if building is not None:
        roof = building & urb
        rgb[roof] = rgb[roof] * 0.8 + 0.2 * np.array([0.78, 0.74, 0.70])
    water = landuse == LU.WATER
    depth_tint = np.clip(-dem, 0, 3)[..., None] / 3.0
    rgb[water] = (rgb * (1 - 0.35 * depth_tint))[water]
    hs = hillshade(np.where(water, np.minimum(dem, 0.0), dem), dx)
    shade = 0.55 + 0.6 * hs
    rgb[~water] *= shade[~water][:, None]
    haze = np.array([0.78, 0.82, 0.86])
    rgb = 0.9 * rgb + 0.1 * haze
    return np.clip(rgb, 0, 1)


def plot_domain(domain, prov, path):
    """Stage-2 eyeball figure: DEM, land use, basemap, 1-D network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from .provenance import savefig

    ext = [0, domain.nx * domain.dx / 1000, 0, domain.ny * domain.dx / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(domain.dem, origin="lower", extent=ext, cmap="terrain", vmin=-3, vmax=30)
    fig.colorbar(im, ax=ax, shrink=0.8, label="elevation [m MSL]")
    ax.set_title("DEM (display, channels carved)")
    ax = axes[1]
    cmap = ListedColormap([c.color for c in LU.LANDUSE_TABLE])
    ax.imshow(domain.landuse, origin="lower", extent=ext, cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, LU.N_CLASSES), LU.N_CLASSES), interpolation="nearest")
    ax.legend(handles=[Patch(color=c.color, label=c.name) for c in LU.LANDUSE_TABLE],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    ax.set_title("Land use")
    ax = axes[2]
    ax.imshow(render(domain.landuse, domain.dem, domain.dx, building=domain.building),
              origin="lower", extent=ext)
    for r in domain.network.reaches:
        ax.plot(r.x / 1000, r.y / 1000, "-", lw=1.2, color="#00e5ff" if r.reach_id.startswith("main") else "#ffd54f")
        if r.up[0] == "inflow":
            ax.plot(r.x[0] / 1000, r.y[0] / 1000, "v", color="w", ms=8)
    for j in domain.network.junctions:
        ax.plot(j.x / 1000, j.y / 1000, "o", mfc="none", mec="w", ms=7)
    ax.plot(domain.urban_core[0] / 1000, domain.urban_core[1] / 1000, "*", color="#ff5252", ms=12)
    ax.set_title("Basemap + 1-D network (cyan main, yellow creeks)")
    for a in axes:
        a.set_xlabel("x [km]"); a.set_ylabel("y [km]  (north = sea)")
    return savefig(fig, path, prov, caption=BASEMAP_CAPTION)
