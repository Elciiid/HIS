"""1-D <-> 2-D lateral exchange and the coupled time loop.

Each interior 1-D cell (reach node or junction) is linked to the 2-D cell that
contains it. In the coupled DEM the channel footprint sits at the bank crest,
so the channel volume lives only in the 1-D model.

Exchange per link per step (positive = 2-D -> 1-D):
  free weir         Q = Cw L h_over^1.5, h_over = eta_up - z_bank
  submerged weir    Villemonte: Q = Q_free (1 - (d_down/d_up)^1.5)^0.385
Stability: |V| is limited to ``exchange_relax`` x the volume that would
equalise the two levels, and the total a donor gives per step to
``exchange_limit`` x its volume above the bank. The volume is moved as one
operator-split transfer before the hydrodynamic step, so what leaves one model
enters the other exactly (to round-off).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import SolverConfig
from .massbalance import MassBalance
from .swe1d import SWE1D
from .swe2d import SWE2D


@dataclass
class Links:
    """Link arrays live with the 1-D model (CPU); the 2-D side is addressed through
    the unique set of linked cells, kept on the 2-D device."""
    cell1d: torch.Tensor      # 1-D cell index per link
    zbank: torch.Tensor
    length: torch.Tensor      # weir length (both banks)
    z2: torch.Tensor          # 2-D bed at the linked cell, per link
    inv: torch.Tensor         # link -> position in the unique 2-D cell list
    uniq_dev: torch.Tensor    # unique flat 2-D cell indices (2-D device)
    n_unique: int


def build_links(one: SWE1D, two: SWE2D) -> Links:
    tp = one.topo
    idx = np.nonzero(tp.kind <= 1)[0]
    i = np.clip((tp.x[idx] / two.dx).astype(int), 0, two.nx - 1)
    j = np.clip((tp.y[idx] / two.dy).astype(int), 0, two.ny - 1)
    flat = j * two.nx + i
    uniq, inv = np.unique(flat, return_inverse=True)
    T = lambda a, dt=one.dtype: torch.as_tensor(a, dtype=dt, device=one.device)
    z2 = two.z.reshape(-1)[torch.as_tensor(flat, device=two.device)].to(device=one.device, dtype=one.dtype)
    return Links(T(idx, torch.long), T(tp.zbank[idx]), T(2.0 * tp.dx[idx]), z2, T(inv, torch.long),
                 torch.as_tensor(uniq, device=two.device), len(uniq))


def exchange(one: SWE1D, two: SWE2D, links: Links, dt: float, cfg: SolverConfig):
    """Move water between the models in place. Returns per-link volumes (m^3, + = 2-D -> 1-D)
    on the 1-D device."""
    hu_flat = two.h.reshape(-1)
    h_u = hu_flat[links.uniq_dev].to(device=one.device, dtype=one.dtype)          # one small transfer
    h2 = h_u[links.inv]
    c1 = links.cell1d
    eta2 = links.z2 + h2
    sec1 = one.sec
    y1 = sec1.depth(one.A)[c1]
    eta1 = one.z[c1] + y1
    zb = links.zbank
    up, dn = torch.maximum(eta1, eta2), torch.minimum(eta1, eta2)
    d_up = torch.clamp(up - zb, min=0.0)
    d_dn = torch.clamp(dn - zb, min=0.0)
    ratio = torch.where(d_up > 0, d_dn / torch.where(d_up > 0, d_up, torch.ones_like(d_up)), torch.zeros_like(d_up))
    q = cfg.weir_cw * links.length * d_up ** 1.5 * torch.clamp(1.0 - ratio ** 1.5, min=0.0) ** 0.385
    sign = torch.where(eta2 > eta1, 1.0, -1.0).to(q.dtype)
    V = q * dt
    # relaxation toward level equality
    area2 = two.area
    plan1 = torch.where(y1 <= sec1.yb[c1], sec1.b[c1] + 2 * sec1.m[c1] * y1, sec1.ts[c1]) * one.dx[c1]
    v_eq = (up - dn) / (1.0 / area2 + 1.0 / plan1)
    V = torch.minimum(V, cfg.exchange_relax * v_eq)
    # donor limits (aggregated over links sharing a donor)
    give2 = torch.where(sign > 0, V, torch.zeros_like(V))
    give1 = torch.where(sign < 0, V, torch.zeros_like(V))
    tot2 = torch.zeros_like(h_u).index_add(0, links.inv, give2)
    avail2 = cfg.exchange_limit * h_u * area2
    f2 = torch.where(tot2 > avail2, avail2 / torch.where(tot2 > 0, tot2, torch.ones_like(tot2)), torch.ones_like(tot2))
    tot1 = torch.zeros_like(one.A).index_add(0, c1, give1)
    avail1 = cfg.exchange_limit * torch.clamp(one.A - sec1.bank_area(), min=0.0) * one.dx
    f1 = torch.where(tot1 > avail1, avail1 / torch.where(tot1 > 0, tot1, torch.ones_like(tot1)), torch.ones_like(tot1))
    V = torch.where(sign > 0, V * f2[links.inv], V * f1[c1]) * sign
    # apply: 2-D side through the unique cells
    dV2 = torch.zeros_like(h_u).index_add(0, links.inv, V)
    h_new = h_u - dV2 / area2
    # water leaving a 2-D cell takes its momentum with it; arriving water brings none
    keep = torch.where(h_u > 0, torch.clamp(h_new / torch.where(h_u > 0, h_u, torch.ones_like(h_u)), max=1.0),
                       torch.zeros_like(h_u))
    idx = links.uniq_dev
    dev2 = dict(device=two.device, dtype=two.dtype)
    keep_d = keep.to(**dev2)
    shape = two.h.shape
    two.h = hu_flat.index_put((idx,), h_new.to(**dev2)).reshape(shape)
    hu = two.hu.reshape(-1)
    hv = two.hv.reshape(-1)
    two.hu = hu.index_put((idx,), hu[idx] * keep_d).reshape(shape)
    two.hv = hv.index_put((idx,), hv[idx] * keep_d).reshape(shape)
    one.A = one.A + torch.zeros_like(one.A).index_add(0, c1, V) / one.dx
    return V


@dataclass
class Snapshot:
    t: float
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    eta1: np.ndarray
    q1: np.ndarray
    ledger: dict = field(default_factory=dict)


@dataclass
class RunLog:
    steps: int = 0
    wall_s: float = 0.0
    dt_min: float = math.inf
    dt_max: float = 0.0
    n_sub_max: int = 0
    n_dt_clamped: int = 0
    exchanged_in_m3: float = 0.0      # 2-D -> 1-D
    exchanged_out_m3: float = 0.0     # 1-D -> 2-D
    series: list = field(default_factory=list)


class CoupledEngine:
    """Coupled 1-D/2-D time loop with mass accounting and snapshot output."""

    def __init__(self, two: SWE2D, one: SWE1D | None, cfg: SolverConfig, rain=None):
        self.two, self.one, self.cfg = two, one, cfg
        self.rain = rain or (lambda t: 0.0)
        self.links = build_links(one, two) if one is not None else None
        self.mb = MassBalance(two.device, self.total_volume())
        self.log = RunLog()
        dev1 = one.device if one is not None else two.device
        self._ex_in = torch.zeros((), dtype=torch.float64, device=dev1)
        self._ex_out = torch.zeros((), dtype=torch.float64, device=dev1)

    def total_volume(self) -> float:
        v = self.two.volume() + self.two.storage_volume()
        if self.one is not None:
            v = v + self.one.volume()
        return float(v)

    def step(self, t: float, t_limit: float) -> float:
        two, one = self.two, self.one
        dt = min(two.max_dt(), t_limit - t)
        if one is not None:
            dt1 = one.max_dt()
            n_sub = max(1, math.ceil(dt / dt1 - 1e-9))
            V = exchange(one, two, self.links, dt, self.cfg)
            self._ex_in += torch.clamp(V, min=0).to(torch.float64).sum()
            self._ex_out += torch.clamp(-V, min=0).to(torch.float64).sum()
        vol = two.step(dt, t, self.rain(t))
        mb = self.mb
        mb.add("rain", vol.rain); mb.add("infiltration", vol.infil); mb.add("clip", vol.clip)
        mb.add("bnd2d_in", vol.bnd_in); mb.add("bnd2d_out", vol.bnd_out)
        if one is not None:
            sub = dt / n_sub
            for k in range(n_sub):
                v1 = one.step(sub, t + k * sub)
                mb.add("bnd1d_in", v1.bnd_in); mb.add("bnd1d_out", v1.bnd_out)
                mb.add("clip", v1.clip)
            self.log.n_sub_max = max(self.log.n_sub_max, n_sub)
        self.log.steps += 1
        self.log.dt_min = min(self.log.dt_min, dt)
        self.log.dt_max = max(self.log.dt_max, dt)
        return t + dt

    def snapshot(self, t: float) -> Snapshot:
        two = self.two
        u, v = two.velocities(two.h, two.hu, two.hv)
        cast = lambda a: a.to(torch.float32).cpu().numpy()
        if self.one is not None:
            eta1, q1 = cast(self.one.level()), cast(self.one.Q)
        else:
            eta1 = q1 = np.zeros(0, np.float32)
        src = self.mb.sources()
        ledger = {"v2d": float(two.volume()), "v1d": float(self.one.volume()) if self.one is not None else 0.0,
                  "storage": float(two.storage_volume()), **src}
        return Snapshot(t, cast(two.h), cast(u), cast(v), eta1, q1, ledger)

    def run(self, t_end: float, out_times=(), t0: float = 0.0, on_snapshot=None, check_every: int = 500,
            label: str = "") -> float:
        """Advance to t_end, calling ``on_snapshot`` at each requested output time."""
        wall = time.perf_counter()
        t = t0
        outs = sorted(float(x) for x in out_times if t0 <= x <= t_end + 1e-9)
        k = 0
        if outs and abs(outs[0] - t0) < 1e-9:
            on_snapshot and on_snapshot(self.snapshot(t0))
            outs.pop(0)
        while t < t_end - 1e-9:
            limit = outs[0] if outs else t_end
            t = self.step(t, limit)
            if outs and t >= outs[0] - 1e-9:
                on_snapshot and on_snapshot(self.snapshot(t))
                outs.pop(0)
            k += 1
            if k % check_every == 0:
                self.mb.check(self.total_volume(), t)
        self.log.wall_s += time.perf_counter() - wall
        self.log.exchanged_in_m3 = float(self._ex_in)
        self.log.exchanged_out_m3 = float(self._ex_out)
        self.log.n_dt_clamped = self.two.n_dt_clamped
        return self.mb.assert_ok(self.total_volume(), t, self.cfg.mass_tol, label)
