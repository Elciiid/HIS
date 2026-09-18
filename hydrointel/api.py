"""The stable interface Part B builds against.

    predict(sites, scenario)  -> [FloodSummary]   GeoKAN-PINO surrogate, a whole population per call
    predict(site, scenario)   -> FloodSummary     one candidate (delegates to the batched path)
    simulate(site, scenario)  -> FloodResult      coupled 1-D/2-D engine (ground truth)
    baseline(scenario)        -> no interventions

``predict(..., detail="full")`` returns the same FloodResult as ``simulate``;
the default ``detail="summary"`` returns only what an optimiser needs (peak
fields, point hydrographs, per-reach 1-D peaks, volumes, mass error), without
the (nt, ny, nx) histories. Every predict result carries ``max_depth_increase_m``
and ``area_worsened_ha``: how much worse the design makes the worst-affected
place, relative to baseline(scenario). A design can improve the city in
aggregate and still deepen someone's street; Part B must see that per design.
Part B should optimise with ``predict`` and re-verify the selected designs with
``simulate(..., compare_to_baseline=True)``, reporting both.

Units: depths m, velocities m/s, discharge m^3/s, levels m MSL, times s from the
start of the simulation window (spin-up, storm, recession).
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

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
    speed_max: Tensor | None = None            # (ny, nx) [m/s] peak velocity magnitude
    max_depth_increase_m: float | None = None  # vs baseline(scenario), land cells; see disbenefit()
    area_worsened_ha: float | None = None      # land where peak depth rose by more than 1 cm


@dataclass(frozen=True)
class FloodSummary:
    """What an optimiser needs from one candidate, without the full field history."""
    depth_max: Tensor                  # (ny, nx) [m]
    speed_max: Tensor                  # (ny, nx) [m/s] peak velocity magnitude
    times: Tensor                      # (nt,) [s]
    point_level: dict                  # monitoring point name -> (nt,) water level [m MSL]
    reach_peak_stage: Tensor           # (n_reaches,) [m MSL]
    reach_peak_discharge: Tensor       # (n_reaches,) [m^3/s], peak |Q| over the reach
    infiltrated_volume_m3: float
    stored_volume_m3: float
    mass_balance_error: float
    provenance: DataProvenance | None
    in_distribution: bool
    distribution_warnings: list[str] = field(default_factory=list)
    max_depth_increase_m: float | None = None
    area_worsened_ha: float | None = None
    extras: dict = field(default_factory=dict)


WORSENED_THRESHOLD_M = 0.01


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
    _BASELINE_PEAK.clear()                 # references belong to the old configuration
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


def simulate(site: SiteState, scenario: Scenario, output_interval_s: float | None = None,
             compare_to_baseline: bool = False) -> FloodResult:
    """Run the coupled physics engine. Raises if mass balance fails.

    ``compare_to_baseline`` fills ``max_depth_increase_m`` and ``area_worsened_ha``
    against an engine run of baseline(scenario). It is off by default because it costs
    a second engine run (cached per scenario); dataset generation never needs it,
    and Part B's final verification should always ask for it."""
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
    u_all, v_all = st("u"), st("v")
    res = FloodResult(
        depth_max=h.amax(0), depth_series=h, u=u_all, v=v_all,
        speed_max=torch.sqrt(u_all.double() ** 2 + v_all.double() ** 2).amax(0).float(),
        channel_eta=st("eta1")[:, interior], channel_q=st("q1")[:, interior],
        times=torch.as_tensor(np.array([x.t for x in snaps])),
        infiltrated_volume_m3=s["infiltration"], stored_volume_m3=float(eng.two.storage_volume()),
        mass_balance_error=err, provenance=prov, in_distribution=ok, distribution_warnings=warns,
        extras={"forcing": fo, "run_log": eng.log, "mass_series": eng.mb.series, "sources": s,
                "channel_cells": np.nonzero(eng.one.topo.kind <= 1)[0], "topology": eng.one.topo,
                "ledger": [x.ledger for x in snaps],
                "boundary_in_m3": s["bnd2d_in"] + s["bnd1d_in"], "boundary_out_m3": s["bnd2d_out"] + s["bnd1d_out"],
                **exposure_metrics(h.amax(0), scenario)})
    if compare_to_baseline:
        ref = _baseline_peak(scenario, "simulate", output_interval_s)
        res = dataclasses.replace(res, **disbenefit(res.depth_max, ref))
    return res


def simulate_batch(sites: list[SiteState], scenarios: list[Scenario],
                   output_interval_s: float | None = None) -> list[FloodResult]:
    """Run B storms together on one set of kernels and return one FloodResult each.

    The members share the terrain and the storm time base; everything else -- site
    state, return period, climate pathway, tide -- is per member, as is the mass
    balance, which is asserted per member so one bad storm cannot hide inside a
    batch average. They also share a time step (the smallest any member needs), so
    a batched run is not bit-identical to running the storms one at a time; see
    tests/test_batched_solver.py for the measured size of that difference.
    """
    from .engine import build_engine_batch, build_forcing
    ctx = context()
    cfg, dom = ctx.cfg, ctx.domain
    if len(sites) != len(scenarios):
        raise ValueError(f"{len(sites)} sites but {len(scenarios)} scenarios")
    if not sites:
        return []
    if len(sites) == 1:
        # a batch of one is the plain solver: use the path the benchmark suite verifies
        return [simulate(sites[0], scenarios[0], output_interval_s)]
    checks = []
    for s, sc in zip(sites, scenarios):
        _validate_scenario(sc)
        checks.append(check_site(s, sc))
    area = dom.nx * dom.ny * dom.dx ** 2 / 1e6
    fos = [build_forcing(cfg, sc.return_period_yr, sc.rcp, sc.ssp, sc.horizon_year, sc.tide_on,
                         sc.storm_tide_offset_h, area) for sc in scenarios]
    lus = [ctx.landuse(sc.ssp, sc.horizon_year) for sc in scenarios]
    eng, bf = build_engine_batch(cfg, dom, fos, [_np(s.storage_depth) for s in sites],
                                 [_np(s.infil_multiplier) for s in sites], [_np(s.manning) for s in sites],
                                 [_np(s.channel_gamma) for s in sites], lus, device=ctx.device)
    snaps = []
    times = bf.times(output_interval_s or cfg.solver.output_interval_s)
    err = eng.run(bf.t_end, out_times=times, on_snapshot=snaps.append,
                  label=f"batch of {len(sites)}: " + "; ".join(_label(sc) for sc in scenarios[:4]))
    interior = torch.as_tensor(eng.one.topo.kind <= 1)
    st = lambda k: torch.as_tensor(np.stack([getattr(s, k) for s in snaps]))
    h_all, u_all, v_all = st("h"), st("u"), st("v")
    eta_all, q_all = st("eta1")[:, :, interior], st("q1")[:, :, interior]
    t_all = torch.as_tensor(np.array([x.t for x in snaps]))
    src = eng.mb.sources()
    storage_v = eng.two.storage_volume().detach().cpu().numpy()
    prov = ctx.provenance("engine")
    out = []
    for b, (sc, fo) in enumerate(zip(scenarios, fos)):
        s_b = {k: float(v[b]) for k, v in src.items()}
        h = h_all[:, b]
        out.append(FloodResult(
            depth_max=h.amax(0), depth_series=h, u=u_all[:, b], v=v_all[:, b],
            channel_eta=eta_all[:, b], channel_q=q_all[:, b], times=t_all,
            infiltrated_volume_m3=s_b["infiltration"], stored_volume_m3=float(storage_v[b]),
            mass_balance_error=float(err[b]), provenance=prov,
            in_distribution=checks[b][0], distribution_warnings=checks[b][1],
            extras={"forcing": fo, "batch_index": b, "batch_size": len(sites),
                    # steps and wall time belong to the whole batch; the exchange totals are this member's
                    "run_log": dataclasses.replace(eng.log, exchanged_in_m3=float(eng.log.exchanged_in_m3[b]),
                                                   exchanged_out_m3=float(eng.log.exchanged_out_m3[b]),
                                                   series=[]),
                    "mass_series": [{k: (v[b] if getattr(v, "ndim", 0) else v) for k, v in rec.items()}
                                    for rec in eng.mb.series],
                    "sources": s_b, "channel_cells": np.nonzero(eng.one.topo.kind <= 1)[0],
                    "topology": eng.one.topo,
                    "ledger": [{k: (v[b] if getattr(v, "ndim", 0) else v) for k, v in x.ledger.items()}
                               for x in snaps],
                    "boundary_in_m3": s_b["bnd2d_in"] + s_b["bnd1d_in"],
                    "boundary_out_m3": s_b["bnd2d_out"] + s_b["bnd1d_out"],
                    **exposure_metrics(h.amax(0), sc)}))
    return out


def predict(site: SiteState | Sequence[SiteState], scenario: Scenario, output_interval_s: float | None = None,
            detail: Literal["full", "summary"] = "summary"):
    """Evaluate the trained GeoKAN-PINO surrogate. Raises if no trained model exists.

    ``site`` may be one SiteState or a sequence of them (an optimiser's population):
    a sequence is evaluated in one call and returns a list, one result per
    candidate, in order. A single SiteState returns a single result and goes
    through the same batched path.
    """
    single = isinstance(site, SiteState)
    sites = [site] if single else list(site)
    if not sites:
        return []
    out = _predict_core(sites, scenario, output_interval_s, detail, with_baseline=True)
    return out[0] if single else out


def _predict_core(sites, scenario, output_interval_s, detail, with_baseline: bool):
    ctx = context()
    _validate_scenario(scenario)
    checks = [check_site(s, scenario) for s in sites]
    model = ctx.model()
    key = (scenario, float(output_interval_s or ctx.cfg.solver.output_interval_s), "predict")
    need_base = with_baseline and key not in _BASELINE_PEAK
    batch = sites + ([baseline_site(scenario)] if need_base else [])
    t0 = time.perf_counter()
    raw = model.predict_many(batch, scenario, ctx, output_interval_s, detail=detail)
    wall = time.perf_counter() - t0
    if need_base:
        _BASELINE_PEAK[key] = raw.pop().depth_max
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train"))
    out = []
    for r, (ok, warns) in zip(raw, checks):
        extra = {**r.extras, "wall_s": wall / len(batch), "call_wall_s": wall, "call_batch": len(batch)}
        if detail == "full":
            extra.update(exposure_metrics(r.depth_max, scenario))
        else:
            em = exposure_metrics(r.depth_max, scenario)
            extra.update({k: v for k, v in em.items() if k != "exposure"})
        worse = disbenefit(r.depth_max, _BASELINE_PEAK[key]) if with_baseline else {}
        if with_baseline:
            extra["max_depth_increase_any_cell_m"] = float(torch.clamp(
                (r.depth_max.float() - _BASELINE_PEAK[key].float().cpu()).max(), min=0.0))
        out.append(dataclasses.replace(r, provenance=prov, in_distribution=ok, distribution_warnings=warns,
                                       extras=extra, **worse))
    return out


# peak depth of baseline(scenario), per (scenario, output interval, engine): an
# optimiser evaluates thousands of designs against the same reference
_BASELINE_PEAK: dict = {}


def _baseline_peak(scenario: Scenario, engine: str, output_interval_s=None) -> Tensor:
    ctx = context()
    key = (scenario, float(output_interval_s or ctx.cfg.solver.output_interval_s), engine)
    if key not in _BASELINE_PEAK:
        site = baseline_site(scenario)
        if engine == "predict":
            _BASELINE_PEAK[key] = _predict_core([site], scenario, output_interval_s, "summary", False)[0].depth_max
        else:
            _BASELINE_PEAK[key] = simulate(site, scenario, output_interval_s).depth_max
    return _BASELINE_PEAK[key]


def disbenefit(depth_max: Tensor, baseline_depth_max: Tensor) -> dict:
    """How much worse a design makes the worst-affected place.

    ``max_depth_increase_m``: largest rise in peak depth at any land cell relative to
    the baseline (0 if nowhere rises). ``area_worsened_ha``: land area where peak depth
    rose by more than 1 cm. Land excludes the sea and the 1-D channel footprint, which
    are water bodies, not places people live. ``predict`` also reports the all-cell
    maximum in ``extras["max_depth_increase_any_cell_m"]`` for transparency.
    """
    dom = context().domain
    land = torch.as_tensor(~(dom.sea | dom.channel))
    d = (depth_max.detach().float().cpu() - baseline_depth_max.detach().float().cpu())
    rise = torch.clamp(d[land].max(), min=0.0)
    return {"max_depth_increase_m": float(rise),
            "area_worsened_ha": float((d[land] > WORSENED_THRESHOLD_M).sum()) * dom.dx ** 2 / 1e4}


def baseline(scenario: Scenario, engine: str = "predict", detail: Literal["full", "summary"] = "summary"):
    """Flood result with no interventions. ``engine`` is 'predict' or 'simulate'."""
    site = baseline_site(scenario)
    if engine == "predict":
        return predict(site, scenario, detail=detail)
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
