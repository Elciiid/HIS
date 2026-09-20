"""Paired baselines: for every modified storm in the dataset, the same storm on the
site with no interventions.

Every modified sample needs a matching baseline (same scenario: return period,
RCP, SSP, horizon year, tide and storm-tide offset; so the same forcing) on the
SSP-adjusted land surface with nothing added. The surrogate can then be taught the
*effect* of a design, modified minus baseline, directly.

The modified runs are not regenerated. Baselines are written next to them:

  <dataset>/baseline/sim_XXXXX.pt    same record format as the modified storm XXXXX
  <dataset>/baseline/index.json      per-storm source, wall time, mass error, peak GPU memory

Sources, in order of preference (reported per storm):
  self        the storm is itself a baseline sample: its record is its own pair
  eval_cache  an engine baseline for the same scenario already run by the evaluation
              (peak depth only: test storms, which need no training record)
  engine      run now

The job is resumable: a storm whose record exists is skipped. Runs are ordered so
that a partial job still spans the return periods and both splits.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .. import api
from ..config import RunConfig, seed_everything
from . import sampler as SMP
from .generate import Budget, FAILED, _write_sim, check_capacity, measured_s_per_storm, record_path

log = logging.getLogger("hydrointel.data")


def baseline_sample(smp: SMP.Sample) -> SMP.Sample:
    """The same scenario with every intervention removed."""
    b = copy.deepcopy(smp)
    b.baseline = True
    b.patterns = []
    b.storage_delta = np.zeros_like(smp.storage_delta)
    b.kappa = np.ones_like(smp.kappa)
    b.manning_factor = np.ones_like(smp.manning_factor)
    b.manning_abs = np.full_like(smp.manning_abs, np.nan)
    b.gamma = np.ones_like(smp.gamma)
    return b


def pair_dir(cfg: RunConfig) -> Path:
    return api.dataset_dir(cfg) / "baseline"


def plan(cfg: RunConfig, dom, index: dict, eval_cache: Path | None) -> list[dict]:
    """One entry per usable storm: where its baseline comes from."""
    rows = []
    for k, meta in index.items():
        if meta.get("status") == FAILED:
            continue
        sid = int(k)
        src = "engine"
        if meta["baseline"]:
            src = "self"
        elif meta["split"] == "test":
            c = eval_cache / f"test_base_{sid}.pt" if eval_cache else None
            src = "eval_cache" if c is not None and c.exists() else "engine"
        rows.append({"sim": sid, "split": meta["split"], "rp": meta["return_period_yr"], "source": src})
    # engine runs interleaved by return period, validation first, so a partial job is balanced
    eng = [r for r in rows if r["source"] == "engine"]
    by = {}
    for r in sorted(eng, key=lambda r: (r["split"] != "val", r["sim"])):
        by.setdefault(r["rp"], []).append(r)
    order = []
    while any(by.values()):
        for rp in sorted(by):
            if by[rp]:
                order.append(by[rp].pop(0))
    return [r for r in rows if r["source"] != "engine"] + order


def generate_baselines(cfg: RunConfig, device=None, limit: int | None = None,
                       budget_hours: float | None = None) -> Path:
    from ..model.geokan_pino import model_dir
    from ..solver.coupling import NumericalInstabilityError
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    root = api.dataset_dir(cfg)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    out = pair_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ip = out / "index.json"
    pidx = json.loads(ip.read_text(encoding="utf-8")) if ip.exists() else {}
    f = cfg.store_factor
    rows = plan(cfg, dom, index, api.engine_cache_dir(cfg) / "effects")
    eng = [r for r in rows if r["source"] == "engine"]
    done = [r for r in eng if record_path(out, r["sim"]).exists()]
    todo = [r for r in eng if r not in done and pidx.get(str(r["sim"]), {}).get("status") != FAILED]
    if limit is not None:
        todo = todo[:limit]
    check_capacity(cfg, dom, len(index) + len(eng))
    log.info("paired baselines: %d storms; %d are baseline samples (self), %d reuse an evaluation run, "
             "%d need the engine (%d already done, %d to run now)", len(rows),
             sum(r["source"] == "self" for r in rows), sum(r["source"] == "eval_cache" for r in rows),
             len(eng), len(done), len(todo))
    for r in rows:
        if r["source"] != "engine":
            pidx[str(r["sim"])] = {**r, "record": ("../" + record_path(root, r["sim"]).name) if r["source"] == "self"
                                   else str(api.engine_cache_dir(cfg) / "effects" / f"test_base_{r['sim']}.pt")}
    ip.write_text(json.dumps(pidx, indent=1), encoding="utf-8")
    t_start = time.perf_counter()
    budget = Budget(budget_hours, assume_s=measured_s_per_storm(cfg))
    log.info("baseline budget: %s", budget.note())
    for n_done, r in enumerate(todo, 1):
        if not budget.room_for_another():
            log.warning("stopping cleanly with %d of %d baselines done: %s. Rerun to resume.",
                        n_done - 1, len(todo), budget.note())
            break
        sid = r["sim"]
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, r["split"])
        bs = baseline_sample(smp)
        scen = api.Scenario(bs.return_period_yr, bs.rcp, bs.ssp, bs.horizon_year, bs.tide_on, bs.storm_tide_offset_h)
        lu = ctx.landuse(bs.ssp, bs.horizon_year)
        storage, kappa, manning, gamma, _, _ = SMP.materialise(bs, dom, lu)
        ref = api.baseline_site(scen)
        land = ~(dom.sea | dom.channel)
        if not (np.allclose(storage[land], ref.storage_depth.numpy()[land]) and np.allclose(manning, ref.manning.numpy())
                and np.allclose(kappa, 1.0) and np.allclose(gamma, 1.0)):
            raise RuntimeError(f"storm {sid}: the zero-intervention sample differs from api.baseline_site")
        site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            res = api.simulate(site, scen)
        except NumericalInstabilityError as e:
            pidx[str(sid)] = {**r, "status": FAILED, "error": str(e), "wall_s": time.perf_counter() - t0}
            ip.write_text(json.dumps(pidx, indent=1), encoding="utf-8")
            log.error("baseline for storm %d: ENGINE UNSTABLE, pair excluded: %s", sid, e)
            continue
        wall = time.perf_counter() - t0
        sub = {}
        path = record_path(out, sid, cfg.data.compress_records)
        _write_sim(cfg, dom, f, sid, bs, lu, storage, kappa, manning, gamma, res, path, sub, wall, wall, 1)
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
        pidx[str(sid)] = {**r, **sub[str(sid)], "record": path.name, "pair_of": sid, "peak_gpu_mb": peak_mb}
        ip.write_text(json.dumps(pidx, indent=1), encoding="utf-8")
        budget.record(wall)
        el = time.perf_counter() - t_start
        log.info("baseline %d/%d for storm %d (%s, RP%d): %.0f s, mass err %.1e, peak GPU %.0f MB; %.2f h elapsed, "
                 "%.2f h to go at this rate", n_done, len(todo), sid, r["split"], r["rp"], wall,
                 res.mass_balance_error, peak_mb or 0.0, el / 3600, el / n_done * (len(todo) - n_done) / 3600)
        del res
    return ip
