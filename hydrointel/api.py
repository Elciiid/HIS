"""The stable interface Part B builds against.

    predict(site, scenario)  -> FloodResult    GeoKAN-PINO surrogate (milliseconds)
    simulate(site, scenario) -> FloodResult    coupled 1-D/2-D engine (ground truth)
    baseline(scenario)       -> FloodResult    no interventions

``predict`` and ``simulate`` share signature and return type. Part B should
optimise with ``predict`` and re-verify the selected designs with ``simulate``,
reporting both.

Units: depths m, velocities m/s, discharge m^3/s, levels m MSL, times s from the
start of the simulation window (spin-up, storm, recession).
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import RunConfig
from .viz.provenance import DataProvenance

log = logging.getLogger("hydrointel.api")


@dataclass(frozen=True)
class SiteState:
    """The mutable physical state Part B is allowed to change."""
    storage_depth: Tensor      # S(x,y)  [m],  shape (ny, nx)
    infil_multiplier: Tensor   # kappa(x,y) [-], shape (ny, nx)
    manning: Tensor            # n(x,y)  [s/m^(1/3)], shape (ny, nx)
    channel_gamma: Tensor      # gamma(reach) [-], shape (n_reaches,)

    def replace(self, **kw) -> "SiteState":
        return dataclasses.replace(self, **kw)


@dataclass(frozen=True)
class Scenario:
    return_period_yr: int      # 10 | 25 | 50 | 100
    rcp: str                   # "RCP2.6" | "RCP4.5" | "RCP6.0" | "RCP8.5"
    ssp: str                   # "SSP1".."SSP5"
    horizon_year: int          # e.g. 2050
    tide_on: bool = True
    storm_tide_offset_h: float = 0.0


@dataclass(frozen=True)
class FloodResult:
    depth_max: Tensor          # (ny, nx) [m]
    depth_series: Tensor       # (nt, ny, nx) [m]
    u: Tensor                  # (nt, ny, nx) [m/s]
    v: Tensor                  # (nt, ny, nx) [m/s]
    channel_eta: Tensor        # (nt, n_nodes) [m MSL]
    channel_q: Tensor          # (nt, n_nodes) [m^3/s]
    times: Tensor              # (nt,) [s]
    infiltrated_volume_m3: float
    stored_volume_m3: float
    mass_balance_error: float
    provenance: DataProvenance
    in_distribution: bool
    distribution_warnings: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# context (domain, config, envelope, model) — built once per process
# ---------------------------------------------------------------------------
class _Context:
    def __init__(self, cfg: RunConfig, device=None):
        from .io.manifest import find_manifest, load
        from .domain.synthetic import generate
        self.cfg = cfg.validate()
        self.device = torch.device(device) if device else cfg.resolved_device()
        mp = find_manifest(cfg.outdir)
        if mp is not None:
            self.domain, self.real = load(mp, cfg.domain)
        else:
            self.domain, self.real = generate(cfg.domain, cfg.seed), None
        self._lu_cache: dict = {}
        self._model = None
        self._envelope = None

    @property
    def source(self) -> str:
        return self.domain.source

    def landuse(self, ssp: str, year: int) -> np.ndarray:
        from .engine import scenario_landuse
        from .forcing.scenarios import ssp_state
        key = (ssp, year)
        if key not in self._lu_cache:
            self._lu_cache[key] = scenario_landuse(self.domain, ssp_state(ssp), year)
        return self._lu_cache[key]

    def envelope(self):
        from .data.sampler import Envelope
        if self._envelope is None:
            p = dataset_dir(self.cfg) / "envelope.json"
            self._envelope = Envelope.load(p) if p.exists() else None
        return self._envelope

    def model(self):
        if self._model is None:
            from .model.geokan_pino import load_surrogate
            self._model = load_surrogate(self.cfg, self.domain, self.device)
        return self._model

    def provenance(self, engine: str, sections=("domain", "solver", "forcing")) -> DataProvenance:
        return DataProvenance(self.source, self.cfg.config_hash(*sections), self.cfg.seed,
                              device=str(self.device), precision=self.cfg.solver.precision, engine=engine)


_CTX: _Context | None = None


def configure(cfg: RunConfig | None = None, device=None) -> _Context:
    global _CTX
    _CTX = _Context(cfg or RunConfig(), device)
    return _CTX


def context() -> _Context:
    return _CTX or configure()


def dataset_dir(cfg: RunConfig) -> Path:
    return Path(cfg.outdir) / "dataset" / cfg.config_hash("domain", "solver", "forcing", "data")


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------
def baseline_site(scenario: Scenario) -> SiteState:
    """SiteState with no interventions on the scenario's (SSP-adjusted) land surface."""
    from .domain import landuse as LU
    ctx = context()
    dom = ctx.domain
    lu = ctx.landuse(scenario.ssp, scenario.horizon_year)
    manning = LU.lookup(lu, "manning") if ctx.real is None or ctx.real.manning is None else ctx.real.manning
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    return SiteState(t(LU.lookup(lu, "storage_m")), t(np.ones(dom.shape)), t(manning),
                     t(np.ones(dom.network.n_reaches)))


def check_site(site: SiteState, scenario: Scenario | None = None) -> tuple[bool, list[str]]:
    from .domain import landuse as LU
    ctx = context()
    dom = ctx.domain
    _validate_site(site, dom)
    env = ctx.envelope()
    if env is None:
        return False, ["no training envelope found (dataset not generated for this config); "
                       "surrogate reliability cannot be assessed"]
    lu = ctx.landuse(scenario.ssp, scenario.horizon_year) if scenario else dom.landuse
    active = ~(dom.sea | dom.channel)
    w = env.check(_np(site.storage_depth), LU.lookup(lu, "storage_m"), _np(site.infil_multiplier),
                  _np(site.manning), LU.lookup(lu, "manning"), _np(site.channel_gamma), dom.dist_coast, active,
                  scenario)
    return not w, w


def simulate(site: SiteState, scenario: Scenario, output_interval_s: float | None = None) -> FloodResult:
    """Run the coupled physics engine. Raises if mass balance fails."""
    from .engine import build_engine, build_forcing
    ctx = context()
    cfg, dom = ctx.cfg, ctx.domain
    _validate_scenario(scenario)
    ok, warns = check_site(site, scenario)
    fo = build_forcing(cfg, scenario.return_period_yr, scenario.rcp, scenario.ssp, scenario.horizon_year,
                       scenario.tide_on, scenario.storm_tide_offset_h, dom.nx * dom.ny * dom.dx ** 2 / 1e6)
    lu = ctx.landuse(scenario.ssp, scenario.horizon_year)
    eng = build_engine(cfg, dom, fo, _np(site.storage_depth), _np(site.infil_multiplier), _np(site.manning),
                       _np(site.channel_gamma), landuse=lu, device=ctx.device)
    snaps = []
    times = fo.times(output_interval_s or cfg.solver.output_interval_s)
    err = eng.run(fo.t_end, out_times=times, on_snapshot=snaps.append, label=_label(scenario))
    interior = torch.as_tensor(eng.one.topo.kind <= 1)
    st = lambda k: torch.as_tensor(np.stack([getattr(s, k) for s in snaps]))
    h = st("h")
    s = eng.mb.sources()
    prov = ctx.provenance("engine")
    return FloodResult(
        depth_max=h.amax(0), depth_series=h, u=st("u"), v=st("v"),
        channel_eta=st("eta1")[:, interior], channel_q=st("q1")[:, interior],
        times=torch.as_tensor(np.array([x.t for x in snaps])),
        infiltrated_volume_m3=s["infiltration"], stored_volume_m3=float(eng.two.storage_volume()),
        mass_balance_error=err, provenance=prov, in_distribution=ok, distribution_warnings=warns,
        extras={"forcing": fo, "run_log": eng.log, "mass_series": eng.mb.series, "sources": s,
                "channel_cells": np.nonzero(eng.one.topo.kind <= 1)[0], "topology": eng.one.topo,
                "ledger": [x.ledger for x in snaps],
                "boundary_in_m3": s["bnd2d_in"] + s["bnd1d_in"], "boundary_out_m3": s["bnd2d_out"] + s["bnd1d_out"],
                **exposure_metrics(h.amax(0), scenario)})


def predict(site: SiteState, scenario: Scenario, output_interval_s: float | None = None) -> FloodResult:
    """Evaluate the trained GeoKAN-PINO surrogate. Raises if no trained model exists."""
    ctx = context()
    _validate_scenario(scenario)
    ok, warns = check_site(site, scenario)
    model = ctx.model()
    t0 = time.perf_counter()
    out = model.predict_result(site, scenario, ctx, output_interval_s)
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train"))
    return dataclasses.replace(out, provenance=prov, in_distribution=ok, distribution_warnings=warns,
                               extras={**out.extras, "wall_s": time.perf_counter() - t0,
                                       **exposure_metrics(out.depth_max, scenario)})


def baseline(scenario: Scenario, engine: str = "predict") -> FloodResult:
    """Flood result with no interventions. ``engine`` is 'predict' or 'simulate'."""
    site = baseline_site(scenario)
    if engine == "predict":
        return predict(site, scenario)
    if engine == "simulate":
        return simulate(site, scenario)
    raise ValueError("engine must be 'predict' or 'simulate'")


# ---------------------------------------------------------------------------
def exposure_metrics(depth_max: Tensor, scenario: Scenario, threshold_m: float = 0.15) -> dict:
    """SSP-scaled exposure raster (relative index per cell; land-use table x population
    factor) and the exposure-weighted area flooded above ``threshold_m``."""
    from .domain import landuse as LU
    from .forcing.scenarios import ssp_state
    ctx = context()
    dom = ctx.domain
    lu = ctx.landuse(scenario.ssp, scenario.horizon_year)
    expo = LU.lookup(lu, "exposure") * ssp_state(scenario.ssp).population_factor
    wet = depth_max.detach().cpu().numpy() >= threshold_m
    land = ~(dom.sea | dom.channel)
    return {"exposure": expo.astype(np.float32),
            f"exposure_weighted_area_gt_{threshold_m:g}m_km2": float(np.sum(expo[wet & land]) * dom.dx ** 2 / 1e6)}


def _np(a) -> np.ndarray:
    if torch.is_tensor(a):
        return a.detach().to(torch.float64).cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _label(s: Scenario) -> str:
    return f"RP{s.return_period_yr} {s.rcp} {s.ssp} {s.horizon_year} tide={'on' if s.tide_on else 'off'}"


def _validate_scenario(s: Scenario) -> None:
    from .forcing import scenarios as SC
    if s.return_period_yr not in (10, 25, 50, 100):
        raise ValueError(f"return_period_yr must be 10, 25, 50 or 100 (got {s.return_period_yr})")
    if s.rcp not in SC.RCP_TABLE:
        raise ValueError(f"unknown rcp {s.rcp!r}")
    if s.ssp not in SC.SSP_TABLE:
        raise ValueError(f"unknown ssp {s.ssp!r}")
    if not 2020 <= s.horizon_year <= 2100:
        raise ValueError(f"horizon_year must be within 2020-2100 (got {s.horizon_year})")


def _validate_site(site: SiteState, dom) -> None:
    for name in ("storage_depth", "infil_multiplier", "manning"):
        a = getattr(site, name)
        if tuple(a.shape) != dom.shape:
            raise ValueError(f"SiteState.{name} has shape {tuple(a.shape)}, domain grid is {dom.shape} (ny, nx)")
        if not torch.isfinite(torch.as_tensor(a)).all():
            raise ValueError(f"SiteState.{name} contains non-finite values")
    if tuple(site.channel_gamma.shape) != (dom.network.n_reaches,):
        raise ValueError(f"SiteState.channel_gamma must have shape ({dom.network.n_reaches},)")
    if float(site.storage_depth.min()) < 0:
        raise ValueError("SiteState.storage_depth must be >= 0 m")
    if float(site.infil_multiplier.min()) < 1.0 - 1e-6:
        raise ValueError("SiteState.infil_multiplier must be >= 1")
    if float(site.manning.min()) <= 0:
        raise ValueError("SiteState.manning must be > 0")
    if float(site.channel_gamma.min()) <= 0:
        raise ValueError("SiteState.channel_gamma must be > 0")
