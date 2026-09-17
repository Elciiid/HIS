"""Flood depth maps: exact hazard classes, velocity arrows, discrete legend."""
from __future__ import annotations

import numpy as np

from ..config import HAZARD_CLASSES


def hazard_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap
    bounds = [c[1] for c in HAZARD_CLASSES] + [1e6]
    cmap = ListedColormap([c[3] for c in HAZARD_CLASSES])
    return cmap, BoundaryNorm(bounds, cmap.N)


def hazard_legend(ax, loc="upper left", anchor=(1.01, 1.0)):
    from matplotlib.patches import Patch
    handles = []
    for name, lo, hi, rgba in HAZARD_CLASSES:
        rng = f"< {hi:g} m" if lo == 0 else (f"> {lo:g} m" if not np.isfinite(hi) else f"{lo:g}–{hi:g} m")
        handles.append(Patch(facecolor=rgba if rgba[3] > 0 else (1, 1, 1, 1), edgecolor="0.4",
                             label=f"{name} ({rng})" + (" — transparent" if rgba[3] == 0 else "")))
    return ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=8, title="Flood hazard (depth)",
                     title_fontsize=8, frameon=True)


def extent_km(domain):
    return [0, domain.nx * domain.dx / 1000, 0, domain.ny * domain.dx / 1000]


def land_depth(domain, depth):
    """Depth with sea and 1-D channel cells masked: those are water bodies, not flooding."""
    d = np.array(depth, dtype=float, copy=True)
    d[domain.sea | domain.channel] = np.nan
    return d


def draw_depth(ax, domain, depth, basemap_rgb=None, u=None, v=None, quiver_every: int | None = None,
               title: str = "", ref_speed: float | None = None):
    cmap, norm = hazard_cmap()
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
    depth = land_depth(domain, depth)
    ext = extent_km(domain)
    if basemap_rgb is not None:
        ax.imshow(basemap_rgb, origin="lower", extent=ext, interpolation="bilinear")
    im = ax.imshow(np.asarray(depth), origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    if u is not None and v is not None:
        k = quiver_every or max(1, domain.nx // 40)
        U, V = np.asarray(u)[::k, ::k], np.asarray(v)[::k, ::k]
        spd = np.hypot(U, V)
        show = (np.nan_to_num(depth)[::k, ::k] > 0.05) & (spd > 0.05)
        X, Y = np.meshgrid(domain.x[::k] / 1000, domain.y[::k] / 1000)
        ref = ref_speed or max(float(np.percentile(spd[show], 95)) if show.any() else 1.0, 0.1)
        q = ax.quiver(X[show], Y[show], U[show], V[show], color="k", scale=ref * 30, width=0.0022, alpha=0.8)
        ax.quiverkey(q, 0.88, -0.07, ref, f"{ref:.2g} m/s", labelpos="E", coordinates="axes", fontproperties={"size": 8})
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km] (north = sea)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    return im


def monitoring_points(domain) -> list[dict]:
    """Deterministic monitoring points: city core, floodplain, channel bank, coast, upstream."""
    net = domain.network
    main = [r for r in net.reaches if r.reach_id.startswith("main")]
    mx = np.concatenate([r.x for r in main]); my = np.concatenate([r.y for r in main])
    pts = [("city_core", *domain.urban_core)]
    k = int(0.4 * len(mx))
    pts.append(("channel_bank", mx[k] + 1.5 * domain.cfg.channel_width_m + domain.dx, my[k]))
    k2 = int(0.55 * len(mx))
    pts.append(("floodplain", mx[k2] - 600.0, my[k2]))
    pts.append(("coast_mouth", mx[-1] - 400.0, my[-1] - 150.0))
    k3 = int(0.15 * len(mx))
    pts.append(("upstream_floodplain", mx[k3] + 500.0, my[k3]))
    out = []
    for name, x, y in pts:
        j, i = domain.cell_index(float(x), float(y))
        if domain.channel[j, i] or domain.sea[j, i]:
            # move to the nearest dry-land cell so the hydrograph is a floodplain one
            J, I = np.nonzero(~(domain.channel | domain.sea))
            d = (J - j) ** 2 + (I - i) ** 2
            j, i = int(J[d.argmin()]), int(I[d.argmin()])
        out.append({"name": name, "j": int(j), "i": int(i), "x": float(domain.x[i]), "y": float(domain.y[j])})
    return out


def corner_transects(domain, length_m: float = 1500.0) -> list[dict]:
    """Diagonal transects running inward from each corner."""
    n = int(length_m / domain.dx)
    out = []
    for name, (j0, i0, dj, di) in {"SW": (0, 0, 1, 1), "SE": (0, domain.nx - 1, 1, -1),
                                   "NW": (domain.ny - 1, 0, -1, 1), "NE": (domain.ny - 1, domain.nx - 1, -1, -1)}.items():
        js = np.clip(j0 + dj * np.arange(n), 0, domain.ny - 1)
        is_ = np.clip(i0 + di * np.arange(n), 0, domain.nx - 1)
        out.append({"name": name, "j": js, "i": is_, "dist_m": np.arange(n) * domain.dx * np.sqrt(2)})
    return out
