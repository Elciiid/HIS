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

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import SolverConfig
from .massbalance import MassBalance
from .swe1d import SWE1D
from .swe2d import SWE2D

log = logging.getLogger("hydrointel.solver")

# Physical plausibility bounds. They detect a solver that has gone unstable; they do
# not judge accuracy. Mass balance cannot catch an instability that conserves
# volume -- found 2026-09-19, when the 1-D model's tidal mouth reach went unstable
# under channel dredging and parked 35 m of water on a cell at 0.96 m elevation with
# a mass error of 1e-14. Real peaks in this domain are < 5 m; the bounds sit well
# outside anything a flood here can produce.
MAX_PLAUSIBLE_DEPTH_2D_M = 10.0
MAX_PLAUSIBLE_RISE_1D_M = 10.0      # 1-D water level above the bank crest


class NumericalInstabilityError(RuntimeError):
    """The engine produced a physically impossible state. The run is invalid."""


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
    hu_flat = two.h.flatten(-2)
    h_u = hu_flat[..., links.uniq_dev].to(device=one.device, dtype=one.dtype)     # one small transfer
    h2 = h_u[..., links.inv]
    c1 = links.cell1d
    eta2 = links.z2 + h2
    sec1 = one.sec
    y1 = sec1.depth(one.A)[..., c1]
    eta1 = one.z[..., c1] + y1
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
    plan1 = torch.where(y1 <= sec1.yb[..., c1], sec1.b[..., c1] + 2 * sec1.m[..., c1] * y1, sec1.ts[..., c1]) \
        * one.dx[c1]
    v_eq = (up - dn) / (1.0 / area2 + 1.0 / plan1)
    V = torch.minimum(V, cfg.exchange_relax * v_eq)
    # donor limits (aggregated over links sharing a donor)
    give2 = torch.where(sign > 0, V, torch.zeros_like(V))
    give1 = torch.where(sign < 0, V, torch.zeros_like(V))
    tot2 = torch.zeros_like(h_u).index_add(-1, links.inv, give2)
    avail2 = cfg.exchange_limit * h_u * area2
    f2 = torch.where(tot2 > avail2, avail2 / torch.where(tot2 > 0, tot2, torch.ones_like(tot2)), torch.ones_like(tot2))
    tot1 = torch.zeros_like(one.A).index_add(-1, c1, give1)
    avail1 = cfg.exchange_limit * torch.clamp(one.A - sec1.bank_area(), min=0.0) * one.dx
    f1 = torch.where(tot1 > avail1, avail1 / torch.where(tot1 > 0, tot1, torch.ones_like(tot1)), torch.ones_like(tot1))
    V = torch.where(sign > 0, V * f2[..., links.inv], V * f1[..., c1]) * sign
    # apply: 2-D side through the unique cells
    dV2 = torch.zeros_like(h_u).index_add(-1, links.inv, V)
    h_new = h_u - dV2 / area2
    # water leaving a 2-D cell takes its momentum with it; arriving water brings none
    keep = torch.where(h_u > 0, torch.clamp(h_new / torch.where(h_u > 0, h_u, torch.ones_like(h_u)), max=1.0),
                       torch.zeros_like(h_u))
    idx = links.uniq_dev
    dev2 = dict(device=two.device, dtype=two.dtype)
    keep_d = keep.to(**dev2)
    shape = two.h.shape
    two.h = hu_flat.index_copy(-1, idx, h_new.to(**dev2).contiguous()).reshape(shape)
    hu = two.hu.flatten(-2)
    hv = two.hv.flatten(-2)
    two.hu = hu.index_copy(-1, idx, (hu[..., idx] * keep_d).contiguous()).reshape(shape)
    two.hv = hv.index_copy(-1, idx, (hv[..., idx] * keep_d).contiguous()).reshape(shape)
    one.A = one.A + torch.zeros_like(one.A).index_add(-1, c1, V) / one.dx
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
        self.batch = two.batch
        if one is not None and one.batch != two.batch:
            raise ValueError(f"the 2-D model is batched {two.batch} and the 1-D model {one.batch}; they must match")
        self.bshape = two.bshape
        self.mb = MassBalance(two.device, self.total_volume())
        self.log = RunLog()
        self._last_dt = 0.0
        dev1 = one.device if one is not None else two.device
        self._ex_in = torch.zeros(self.bshape, dtype=torch.float64, device=dev1)
        self._ex_out = torch.zeros(self.bshape, dtype=torch.float64, device=dev1)

    def check_plausible(self, t: float, label: str = "") -> None:
        """Raise NumericalInstabilityError if any member's state is physically impossible."""
        h = self.two.h
        dmax = h.flatten(-2).amax(-1) if h.dim() == 3 else h.max()
        bad2 = ~torch.isfinite(dmax) | (dmax > MAX_PLAUSIBLE_DEPTH_2D_M)
        bad1 = torch.zeros_like(bad2)
        rise = None
        if self.one is not None:
            one = self.one
            level = one.z + one.sec.depth(one.A)
            zb = torch.as_tensor(one.topo.zbank, dtype=level.dtype, device=level.device)
            rise = torch.where(one.interior, level - zb, torch.full_like(level, -1e9)).amax(-1)
            bad1 = (~torch.isfinite(rise) | (rise > MAX_PLAUSIBLE_RISE_1D_M)).to(bad2.device)
        bad = (bad2 | bad1).reshape(-1)
        if bool(bad.any()):
            members = torch.nonzero(bad).reshape(-1).tolist()
            d2 = dmax.reshape(-1).tolist()
            d1 = rise.reshape(-1).tolist() if rise is not None else [float("nan")] * len(d2)
            who = ", ".join(f"member {m}: max 2-D depth {d2[m]:.2f} m, max 1-D rise above bank {d1[m]:.2f} m"
                            for m in members) if self.batch is not None else \
                f"max 2-D depth {d2[0]:.2f} m, max 1-D rise above bank {d1[0]:.2f} m"
            raise NumericalInstabilityError(
                f"physically impossible state{(' in ' + label) if label else ''} at t = {t / 3600:.3f} h ({who}; "
                f"bounds {MAX_PLAUSIBLE_DEPTH_2D_M:g} m and {MAX_PLAUSIBLE_RISE_1D_M:g} m). The run is invalid.")

    def total_volume(self):
        """Total water in the coupled system; 0-d unbatched, one entry per member batched."""
        v = self.two.volume() + self.two.storage_volume()
        if self.one is not None:
            v = v + self.one.volume().to(v.device)
        return v if self.batch is not None else float(v)

    def step(self, t: float, t_limit: float, dt: float | None = None, n_sub: int | None = None) -> float:
        """``dt`` and ``n_sub`` override the CFL choice; the batching test drives a batched
        run and a sequential one through the identical step sequence that way."""
        two, one = self.two, self.one
        dt = min(two.max_dt() if dt is None else dt, t_limit - t)
        if one is not None:
            if n_sub is None:
                n_sub = max(1, math.ceil(dt / one.max_dt() - 1e-9))
            V = exchange(one, two, self.links, dt, self.cfg)
            self._ex_in += torch.clamp(V, min=0).to(torch.float64).sum(-1)
            self._ex_out += torch.clamp(-V, min=0).to(torch.float64).sum(-1)
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
        self._last_dt = dt
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
        num = (lambda x: x.detach().cpu().numpy()) if self.batch is not None else float
        ledger = {"v2d": num(two.volume()),
                  "v1d": num(self.one.volume()) if self.one is not None else 0.0,
                  "storage": num(two.storage_volume()), **src}
        return Snapshot(t, cast(two.h), cast(u), cast(v), eta1, q1, ledger)

    def run(self, t_end: float, out_times=(), t0: float = 0.0, on_snapshot=None, check_every: int = 500,
            label: str = "", progress_s: float = 120.0) -> float:
        """Advance to t_end, calling ``on_snapshot`` at each requested output time.

        Every ``progress_s`` seconds of wall time the run reports where it is and what
        it now expects to cost. A storm can take hours; a run that says nothing until
        it finishes cannot be managed, and its ETA cannot be checked against reality.
        """
        wall = time.perf_counter()
        last_report = wall
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
                self.check_plausible(t, label)          # never hand an impossible state to a caller
                on_snapshot and on_snapshot(self.snapshot(t))
                outs.pop(0)
            k += 1
            if k % check_every == 0:
                self.mb.check(self.total_volume(), t)
                self.check_plausible(t, label)
            if progress_s and time.perf_counter() - last_report >= progress_s:
                last_report = time.perf_counter()
                done = (t - t0) / (t_end - t0) if t_end > t0 else 1.0
                el = last_report - wall
                log.info("%s%.1f%% of %.2f h: t = %.2f h, %d steps (%.0f/s), dt now %.3f s, 1-D substeps up to %d, "
                         "%.0f s elapsed, %.0f s left at this rate", f"{label}: " if label else "",
                         100 * done, (t_end - t0) / 3600, t / 3600, self.log.steps,
                         self.log.steps / max(el, 1e-9), self._last_dt, self.log.n_sub_max, el,
                         el * (1 - done) / done if done > 0 else float("nan"))
        self.log.wall_s += time.perf_counter() - wall
        num = (lambda x: x.detach().cpu().numpy()) if self.batch is not None else float
        self.log.exchanged_in_m3 = num(self._ex_in)
        self.log.exchanged_out_m3 = num(self._ex_out)
        self.log.n_dt_clamped = self.two.n_dt_clamped
        self.check_plausible(t, label)
        return self.mb.assert_ok(self.total_volume(), t, self.cfg.mass_tol, label)
