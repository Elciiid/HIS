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


def normal_depth(one: SWE1D, Q: float, slope_min: float = 1e-4) -> torch.Tensor:
    """Manning normal depth for discharge Q in every cell (bisection)."""
    tp = one.topo
    s = np.full(tp.n_cells, slope_min)
    for ri in range(one.net.n_reaches):
        idx = np.nonzero(tp.reach_of == ri)[0]
        idx = idx[tp.kind[idx] == 0]
        if len(idx) > 1:
            dz = -np.gradient(tp.z[idx], tp.chain[idx])
            s[idx] = np.maximum(dz, slope_min)
    S = torch.as_tensor(s, dtype=one.dtype, device=one.device)
    lo, hi = torch.zeros_like(S), one.sec.yb.clone()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        A = one.sec.area(mid)
        R = A / one.sec.perimeter(mid)
        q = A * R ** (2 / 3) * torch.sqrt(S) / one.nman
        lo, hi = torch.where(q < Q, mid, lo), torch.where(q < Q, hi, mid)
    return 0.5 * (lo + hi)


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
