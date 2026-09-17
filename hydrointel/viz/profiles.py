"""1-D channel panels: water-surface profile and discharge along the main river."""
from __future__ import annotations

import numpy as np


def main_river_path(topo) -> tuple[np.ndarray, np.ndarray]:
    """Interior-cell positions (in the FloodResult channel arrays) along the main river,
    upstream to mouth, with continuous chainage through junctions."""
    interior = np.nonzero(topo.kind <= 1)[0]
    pos = {c: k for k, c in enumerate(interior)}
    reaches = sorted({int(r) for r in topo.reach_of[topo.kind == 0]})
    cells, chain = [], []
    offset = 0.0
    last_xy = None
    # main reaches are those whose id starts with "main" — they are consecutive in build order
    for ri in reaches:
        idx = np.nonzero((topo.reach_of == ri) & (topo.kind == 0))[0]
        if len(idx) == 0:
            continue
        xy = np.stack([topo.x[idx], topo.y[idx]], 1)
        if last_xy is not None and np.hypot(*(xy[0] - last_xy)) > 3 * topo.dx[idx].max():
            break      # not a continuation of the main stem (a creek)
        if last_xy is not None:
            offset += float(np.hypot(*(xy[0] - last_xy)))
        c = topo.chain[idx] - topo.chain[idx][0] + offset
        cells.extend(pos[i] for i in idx)
        chain.extend(c)
        offset = c[-1]
        last_xy = xy[-1]
    return np.array(cells), np.array(chain)


def draw_profiles(ax_wse, ax_q, topo, eta: np.ndarray, q: np.ndarray, label: str, style: str = "-",
                  draw_bed: bool = True):
    cells, chain = main_river_path(topo)
    interior = np.nonzero(topo.kind <= 1)[0]
    bed = topo.z[interior][cells]
    bank = topo.zbank[interior][cells]
    km = chain / 1000.0
    if draw_bed:
        ax_wse.fill_between(km, bed - 1.0, bed, color="#8d6e63", alpha=0.6, label="bed")
        ax_wse.plot(km, bank, color="#2e7d32", lw=1, label="bank crest")
    ax_wse.plot(km, eta[cells], style, color="#1565c0", lw=1.6, label=f"water level ({label})")
    ax_q.plot(km, q[cells], style, color="#6a1b9a", lw=1.6, label=f"discharge ({label})")
    ax_wse.set_ylabel("level [m MSL]")
    ax_q.set_ylabel("Q [m³/s]")
    ax_q.set_xlabel("main-river chainage [km] (upstream → mouth)")
    for a in (ax_wse, ax_q):
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="best")
