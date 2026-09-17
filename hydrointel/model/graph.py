"""Graph construction for GeoKAN-PINO.

Nodes
  2-D mesh   one per (coarse) grid cell, 4-connected (8 optional)
  1-D chain  one per interior channel cell (reach nodes + junctions)
Edges (stored as sparse index tensors; messages aggregated with index_add_)
  2-D:       [dz, distance, mean n, unit dx, unit dy]
  1-D:       [bed slope, width, n, gamma, distance]
  coupling:  [z_bank - z_bed, bank length]
No dense adjacency is ever built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..domain import landuse as LU


def d8_accumulation(z: np.ndarray, sink: np.ndarray) -> np.ndarray:
    """Upstream contributing cell count (D8, steepest descent). Sinks absorb flow."""
    ny, nx = z.shape
    acc = np.ones(ny * nx)
    zf = z.ravel()
    order = np.argsort(-zf, kind="stable")
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    zp = np.pad(z, 1, constant_values=np.inf)
    best = np.full((ny, nx), -1, dtype=np.int64)
    best_drop = np.zeros((ny, nx))
    J, I = np.mgrid[0:ny, 0:nx]
    for dj, di in offs:
        dist = np.hypot(dj, di)
        drop = (z - zp[1 + dj:ny + 1 + dj, 1 + di:nx + 1 + di]) / dist
        nb = (J + dj) * nx + (I + di)
        take = drop > best_drop
        best = np.where(take, nb, best)
        best_drop = np.where(take, drop, best_drop)
    best = best.ravel()
    sinkf = sink.ravel()
    for c in order:
        b = best[c]
        if b >= 0 and not sinkf[c]:
            acc[b] += acc[c]
    return acc.reshape(ny, nx)


def static_node_features(dom) -> tuple[np.ndarray, list[str]]:
    """Scenario-independent 2-D node features on the model grid, shape (F, ny, nx)."""
    z = dom.dem2d
    gy, gx = np.gradient(z, dom.dx)
    acc = d8_accumulation(z, dom.sea)
    feats = [z, np.clip(dom.dist_channel, 0, 5000) / 1000.0, np.clip(dom.dist_coast, -1000, 8000) / 1000.0,
             np.hypot(gx, gy), gx, gy, np.log10(acc * dom.dx ** 2), dom.sea.astype(float), dom.channel.astype(float)]
    names = ["z", "dist_channel_km", "dist_coast_km", "slope", "slope_x", "slope_y", "log10_upstream_area_m2",
             "sea", "channel"]
    return np.stack(feats).astype(np.float32), names


@dataclass
class Graph:
    n2: int
    ny: int
    nx: int
    e2_src: torch.Tensor
    e2_dst: torch.Tensor
    e2_static: torch.Tensor        # [dz, dist, ux, uy] (mean n added per sample)
    n1: int
    e1_src: torch.Tensor
    e1_dst: torch.Tensor
    e1_static: torch.Tensor        # [bed slope, width, dist] (n, gamma added per sample)
    e1_reach: torch.Tensor         # reach index per 1-D node
    c_node1: torch.Tensor          # coupling: 1-D node
    c_node2: torch.Tensor          # coupling: flat 2-D coarse cell
    c_static: torch.Tensor         # [bank height, bank length]
    node1_static: torch.Tensor     # [bed, width, side slope, n, chainage]
    node1_xy: np.ndarray

    def to(self, device):
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}
        return Graph(**kw)


def build_graph(dem_coarse: np.ndarray, cell: float, topo, connectivity: int = 4) -> Graph:
    ny, nx = dem_coarse.shape
    J, I = np.mgrid[0:ny, 0:nx]
    src, dst, ef = [], [], []
    offs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dj, di in offs:
        ok = (J + dj >= 0) & (J + dj < ny) & (I + di >= 0) & (I + di < nx)
        s = (J * nx + I)[ok]
        d = ((J + dj) * nx + (I + di))[ok]
        dist = np.hypot(dj, di) * cell
        dz = dem_coarse.ravel()[d] - dem_coarse.ravel()[s]
        src.append(s); dst.append(d)
        ef.append(np.stack([dz, np.full(s.shape, dist / cell), np.full(s.shape, di / np.hypot(dj, di)),
                            np.full(s.shape, dj / np.hypot(dj, di))], 1))
    src, dst, ef = np.concatenate(src), np.concatenate(dst), np.concatenate(ef)
    # 1-D chain on interior cells
    interior = np.nonzero(topo.kind <= 1)[0]
    remap = -np.ones(topo.n_cells, dtype=np.int64)
    remap[interior] = np.arange(len(interior))
    s1, d1, f1 = [], [], []
    for a, b in zip(topo.face_l, topo.face_r):
        if remap[a] < 0 or remap[b] < 0:
            continue
        dist = max(np.hypot(topo.x[a] - topo.x[b], topo.y[a] - topo.y[b]), 1.0)
        slope = (topo.z[a] - topo.z[b]) / dist
        for p, q, sg in ((a, b, 1.0), (b, a, -1.0)):
            s1.append(remap[p]); d1.append(remap[q])
            f1.append([sg * slope * 100.0, 0.5 * (topo.b[a] + topo.b[b]) / 50.0, dist / 100.0])
    ci = np.clip((topo.x[interior] / cell).astype(int), 0, nx - 1)
    cj = np.clip((topo.y[interior] / cell).astype(int), 0, ny - 1)
    c2 = cj * nx + ci
    bank_h = topo.zbank[interior] - topo.z[interior]
    n1s = np.stack([topo.z[interior], topo.b[interior] / 50.0, topo.m[interior], topo.n[interior] * 30.0,
                    topo.chain[interior] / 1000.0], 1)
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    return Graph(ny * nx, ny, nx, T(src, torch.long), T(dst, torch.long), T(ef),
                 len(interior), T(s1, torch.long), T(d1, torch.long), T(np.array(f1, dtype=np.float32).reshape(-1, 3)),
                 T(np.maximum(topo.reach_of[interior], 0), torch.long),
                 T(np.arange(len(interior)), torch.long), T(c2, torch.long),
                 T(np.stack([bank_h, 2 * topo.dx[interior] / 100.0], 1)), T(n1s),
                 np.stack([topo.x[interior], topo.y[interior]], 1))
