"""Assemble and run the coupled engine for one (site state, scenario) pair.

Timeline (seconds from t = 0):
  [0, spinup)                 tide and baseflow only
  [spinup, spinup + D)        design storm
  [.., + recession)           recession
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import RunConfig
from .domain import landuse as LU
from .domain.channels import apply_gamma
from .domain.synthetic import Domain
from .forcing import scenarios as SC
from .forcing.catchment import InflowHydrograph, clark_hydrograph
from .forcing.hyetograph import Hyetograph, alternating_block, areal_reduction_factor
from .forcing.tide import Tide
from .solver.boundaries import EdgeBC
from .solver.coupling import CoupledEngine
from .solver.infiltration import Horton
from .solver.swe1d import SWE1D
from .solver.swe2d import SWE2D

log = logging.getLogger("hydrointel.engine")


@dataclass
class Forcing:
    rain: Hyetograph
    inflow: InflowHydrograph
    tide: Tide
    t_end: float
    storm_start: float
    storm_peak: float
    rain_factor: float
    slr_m: float
    arf_domain: float
    arf_catchment: float
    ssp: SC.SSPState

    def times(self, interval: float) -> np.ndarray:
        n = int(math.floor(self.t_end / interval + 1e-9))
        return np.arange(n + 1) * interval


def build_forcing(cfg: RunConfig, return_period: int, rcp: str, ssp: str, horizon_year: int,
                  tide_on: bool, storm_tide_offset_h: float, domain_area_km2: float) -> Forcing:
    fc = cfg.forcing
    if return_period not in (10, 25, 50, 100):
        raise ValueError(f"return_period_yr must be 10, 25, 50 or 100 (got {return_period})")
    start = fc.spinup_h * 3600.0
    base = alternating_block(return_period, fc.storm_duration_h, fc.storm_step_min, fc.idf, fc.peak_position, start)
    factor = SC.rain_factor(rcp)
    arf_d = areal_reduction_factor(domain_area_km2, fc.storm_duration_h) if fc.apply_arf else 1.0
    arf_c = areal_reduction_factor(fc.catchment.area_km2, fc.storm_duration_h) if fc.apply_arf else 1.0
    rain = base.scaled(factor * arf_d)
    t_end = start + fc.storm_duration_h * 3600.0 + cfg.solver.recession_h * 3600.0
    upstream = base.scaled(factor)
    inflow = clark_hydrograph(upstream, fc.catchment, t_end, arf=arf_c)
    peak = start + (int(np.argmax(base.intensity_mmh)) + 0.5) * base.step_s
    slr = SC.slr_m(rcp, horizon_year)
    tide = Tide(fc.tide, slr, tide_on).align(peak, storm_tide_offset_h)
    return Forcing(rain, inflow, tide, t_end, start, peak, factor, slr, arf_d, arf_c, SC.ssp_state(ssp))


def scenario_landuse(domain: Domain, ssp: SC.SSPState, horizon_year: int, base_year: int = 2025) -> np.ndarray:
    return SC.expand_urban(domain.landuse, domain.road, domain.urban_core, domain.dx, ssp.urban_expansion_rate,
                           horizon_year - base_year, protected=domain.sea | domain.channel)


def normal_depth(one: SWE1D, Q, slope_min: float = 1e-4) -> torch.Tensor:
    """Manning normal depth for discharge Q in every cell (bisection). Batched when the
    solver is: the bed, and therefore the slope, differs per member."""
    tp = one.topo
    z = one.z.detach().cpu().numpy()
    s = np.full(z.shape, slope_min)
    for ri in range(one.net.n_reaches):
        idx = np.nonzero(tp.reach_of == ri)[0]
        idx = idx[tp.kind[idx] == 0]
        if len(idx) > 1:
            dz = -np.gradient(z[..., idx], tp.chain[idx], axis=-1)
            s[..., idx] = np.maximum(dz, slope_min)
    S = torch.as_tensor(s, dtype=one.dtype, device=one.device)
    if torch.is_tensor(Q) or hasattr(Q, "shape"):
        Q = torch.as_tensor(np.asarray(Q, dtype=float).reshape(one.bshape + (1,)),
                            dtype=one.dtype, device=one.device)
    lo, hi = torch.zeros_like(S), one.sec.yb.expand_as(S).clone()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        A = one.sec.area(mid)
        R = A / one.sec.perimeter(mid)
        q = A * R ** (2 / 3) * torch.sqrt(S) / one.nman
        lo, hi = torch.where(q < Q, mid, lo), torch.where(q < Q, hi, mid)
    return 0.5 * (lo + hi)


class BatchForcing:
    """Evaluates B members' rainfall, tide and upstream inflow in one vectorised call.

    These are called several times per solver step, so the per-member Python loop
    this replaces would cost more than the arithmetic it performs. Members share the
    storm's time base (spin-up, block length and duration all come from the config),
    so only the intensities, the sea-level offsets, the tidal phase shifts and the
    hydrographs differ. The arithmetic is the same as each member's own forcing, in
    the same order, so the results agree to the last bit.
    """

    def __init__(self, forcings: list[Forcing]):
        from .forcing.hyetograph import MM_H_TO_M_S
        f0 = forcings[0]
        self.members = list(forcings)
        self.n = len(forcings)
        for f in forcings:
            if (f.rain.step_s, f.rain.t_start, len(f.rain.intensity_mmh)) != \
               (f0.rain.step_s, f0.rain.t_start, len(f0.rain.intensity_mmh)) or \
               (f.inflow.step_s, f.inflow.t0, len(f.inflow.q_m3s)) != \
               (f0.inflow.step_s, f0.inflow.t0, len(f0.inflow.q_m3s)) or f.t_end != f0.t_end:
                raise ValueError("batched forcings must share the storm time base (duration, block length, "
                                 "spin-up and recession); only their magnitudes may differ")
        self.t_end, self.storm_start, self.storm_peak = f0.t_end, f0.storm_start, f0.storm_peak
        self.rain_step, self.rain_t0 = f0.rain.step_s, f0.rain.t_start
        self.rain_mms = np.stack([f.rain.intensity_mmh for f in forcings]) * MM_H_TO_M_S
        self.q_step, self.q_t0 = f0.inflow.step_s, f0.inflow.t0
        self.q = np.stack([f.inflow.q_m3s for f in forcings]).astype(float)
        tides = [f.tide for f in forcings]
        cons = list(f0.tide.cfg.constituents.values())
        self.w = np.array([2 * np.pi / (c["period_h"] * 3600.0) for c in cons])
        self.amp = np.array([c["amp_m"] for c in cons])
        self.phase = np.radians(np.array([c["phase_deg"] for c in cons]))
        self.shift = np.array([td.shift_s for td in tides])
        self.on = np.array([td.on for td in tides])
        self.base = np.array([td.cfg.msl_offset_m + td.slr_m for td in tides])

    def times(self, interval: float) -> np.ndarray:
        n = int(math.floor(self.t_end / interval + 1e-9))
        return np.arange(n + 1) * interval

    def rain_rate(self, t: float) -> np.ndarray:
        k = int(np.floor((t - self.rain_t0) / self.rain_step + 1e-9))
        if k < 0 or k >= self.rain_mms.shape[1]:
            return np.zeros(self.n)
        return self.rain_mms[:, k]

    def tide(self, t: float) -> np.ndarray:
        tt = t + self.shift
        eta = np.zeros(self.n)
        for a, w, p in zip(self.amp, self.w, self.phase):
            eta = eta + a * np.cos(w * tt - p)
        return self.base + np.where(self.on, eta, 0.0)

    def inflow(self, t: float) -> np.ndarray:
        x = (t - self.q_t0) / self.q_step
        if x <= 0:
            return self.q[:, 0]
        if x >= self.q.shape[1] - 1:
            return self.q[:, -1]
        k = int(x)
        w = x - k
        return (1 - w) * self.q[:, k] + w * self.q[:, k + 1]


def build_engine_batch(cfg: RunConfig, domain: Domain, forcings: list[Forcing], storages, kappas, mannings,
                       gammas, landuses: list[np.ndarray], device=None) -> tuple[CoupledEngine, BatchForcing]:
    """Assemble one engine that advances ``len(forcings)`` storms together over the shared
    terrain. Every member keeps its own roughness, retention, infiltration, channel
    conveyance, rainfall, tide and upstream hydrograph, and its own mass balance."""
    device = device or cfg.resolved_device()
    dtype = cfg.dtype
    sv = cfg.solver
    b = len(forcings)
    if not (len(storages) == len(kappas) == len(mannings) == len(gammas) == len(landuses) == b):
        raise ValueError(f"batched inputs disagree in length: {b} forcings but "
                         f"{len(storages)}/{len(kappas)}/{len(mannings)}/{len(gammas)}/{len(landuses)} fields")
    fo = BatchForcing(forcings)
    nets = [apply_gamma(domain.network,
                        np.clip(np.asarray(g, float) * f.ssp.drainage_investment_factor, 0.5, 3.0))
            for g, f in zip(gammas, forcings)]
    bcs = {"west": EdgeBC("transmissive"), "east": EdgeBC("transmissive"),
           "south": EdgeBC("transmissive"), "north": EdgeBC("stage", fo.tide)}
    hortons = [Horton.from_landuse(lu, _np(k), device, dtype, t0=fo.storm_start)
               for lu, k in zip(landuses, kappas)]
    horton = Horton(torch.stack([h.f0 for h in hortons]), torch.stack([h.fc for h in hortons]),
                    torch.stack([h.k for h in hortons]), fo.storm_start)
    stack = lambda xs: np.stack([_np(x) for x in xs])
    two = SWE2D(domain.dem2d, stack(mannings), domain.dx, sv, bcs, device, dtype,
                storage=stack(storages), horton=horton, batch=b)
    one = SWE1D(nets, sv, torch.device("cpu"), dtype, inflow=fo.inflow, stage=fo.tide)
    eta0 = fo.tide(0.0)
    sea = torch.as_tensor(domain.sea, device=device)
    eta0_2d = torch.as_tensor(eta0, dtype=dtype, device=device).reshape(b, 1, 1)
    two.set_state(torch.where(sea, torch.clamp(eta0_2d - two.z, min=0.0),
                              torch.zeros_like(two.z)).expand(b, two.ny, two.nx))
    yn = normal_depth(one, fo.inflow(0.0))
    eta0_1d = torch.as_tensor(eta0, dtype=dtype, device=one.device).reshape(b, 1)
    one.set_level(torch.maximum(one.z + yn, eta0_1d.expand_as(one.z)))
    return CoupledEngine(two, one, sv, rain=fo.rain_rate), fo


def build_engine(cfg: RunConfig, domain: Domain, forcing: Forcing, storage, kappa, manning, gamma,
                 landuse: np.ndarray | None = None, device=None, closed: bool = False) -> CoupledEngine:
    """``closed=True`` walls every 2-D edge and every 1-D boundary and removes the
    upstream inflow (used by the closed-domain mass benchmark)."""
    device = device or cfg.resolved_device()
    dtype = cfg.dtype
    sv = cfg.solver
    lu = domain.landuse if landuse is None else landuse
    eff_gamma = np.clip(np.asarray(gamma, float) * forcing.ssp.drainage_investment_factor, 0.5, 3.0)
    net = apply_gamma(domain.network, eff_gamma)
    if closed:
        for r in net.reaches:
            r.up = r.up if r.up[0] == "junction" else ("closed",)
            r.down = r.down if r.down[0] == "junction" else ("closed",)
        bcs = {e: EdgeBC("closed") for e in ("west", "east", "south", "north")}
        inflow = None
    else:
        bcs = {"west": EdgeBC("transmissive"), "east": EdgeBC("transmissive"),
               "south": EdgeBC("transmissive"), "north": EdgeBC("stage", forcing.tide)}
        inflow = forcing.inflow
    horton = Horton.from_landuse(lu, _np(kappa), device, dtype, t0=forcing.storm_start)
    two = SWE2D(domain.dem2d, _np(manning), domain.dx, sv, bcs, device, dtype, storage=_np(storage), horton=horton)
    # the channel network is small (~10^2 cells): it runs on the CPU, where tiny
    # tensors are cheap, and only the linked cells cross the device boundary
    one = SWE1D(net, sv, torch.device("cpu"), dtype, inflow=inflow, stage=forcing.tide)
    # initial state: sea at the tide level, channels at baseflow normal depth (not below the tide)
    eta0 = forcing.tide(0.0)
    sea = torch.as_tensor(domain.sea, device=device)
    two.set_state(torch.where(sea, torch.clamp(eta0 - two.z, min=0.0), torch.zeros_like(two.z)))
    yn = normal_depth(one, forcing.inflow(0.0))
    level = torch.maximum(one.z + yn, torch.full_like(one.z, eta0 if not closed else -1e9))
    one.set_level(level)
    return CoupledEngine(two, one, sv, rain=forcing.rain.rate_ms)


def _np(a):
    if torch.is_tensor(a):
        return a.detach().to(torch.float64).cpu().numpy()
    return np.asarray(a, dtype=np.float64)
