"""Run the engine over sampled scenarios and write the training dataset.

Layout (``artifacts/dataset/<config-hash>/``):
  domain.npz           static rasters on the model grid and the coarse grid
  envelope.json        sampled ranges + coverage statistics (Part B clips to these)
  sim_00000.pt ...     one file per simulation (resumable: existing files are kept)
  index.json           per-simulation metadata, split, timing, mass error
  dataset_card.md      human-readable description
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .. import api
from ..config import RunConfig, seed_everything
from ..domain import landuse as LU
from ..viz.provenance import markdown_header
from . import sampler as SMP

log = logging.getLogger("hydrointel.data")


def coarsen_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    *lead, ny, nx = a.shape
    cy, cx = ny // f, nx // f
    return a[..., :cy * f, :cx * f].reshape(*lead, cy, f, cx, f).mean(axis=(-1, -3))


def require_benchmarks(cfg: RunConfig) -> None:
    p = Path(cfg.outdir) / "benchmark_results.json"
    if not p.exists():
        raise RuntimeError(f"{p} not found: run `python -m hydrointel.cli benchmark` first — the engine must pass "
                           "the analytical suite before it generates training data")
    d = json.loads(p.read_text(encoding="utf-8"))
    bad = [r["key"] for r in d["results"] if not r["passed"]]
    if bad:
        raise RuntimeError(f"benchmarks {bad} failed ({p}); refusing to generate training data")
    want = cfg.config_hash("solver", "domain", "forcing")
    if d["provenance"]["config_hash"] != want:
        raise RuntimeError(f"{p} was produced for config {d['provenance']['config_hash']}, current solver/domain/"
                           f"forcing config is {want}; rerun the benchmark suite")


def static_fields(dom, f: int) -> dict:
    from ..model.graph import static_node_features
    feats, names = static_node_features(dom)
    return {"dem": dom.dem.astype(np.float32), "dem2d": dom.dem2d.astype(np.float32),
            "landuse": dom.landuse, "sea": dom.sea, "channel": dom.channel,
            "dist_channel": dom.dist_channel.astype(np.float32), "dist_coast": dom.dist_coast.astype(np.float32),
            "coarse": {"factor": f, "features": coarsen_mean(feats, f).astype(np.float32), "names": names,
                       "dem2d": coarsen_mean(dom.dem2d, f).astype(np.float32),
                       "sea": coarsen_mean(dom.sea.astype(float), f) > 0.5}}


def generate(cfg: RunConfig, device=None, check_benchmarks: bool = True) -> Path:
    if check_benchmarks:
        require_benchmarks(cfg)
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    out = api.dataset_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    f = cfg.data.coarsen
    n = cfg.data.n_sims
    splits = SMP.assign_splits(n, cfg.data, cfg.seed, cfg.data.return_periods)
    torch.save(static_fields(dom, f), out / "domain.pt")
    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    t_start = time.perf_counter()
    done_now = 0
    for i in range(n):
        path = out / f"sim_{i:05d}.pt"
        if path.exists() and str(i) in index:
            continue
        smp = SMP.draw(i, dom, cfg.data, cfg.seed, splits[i])
        scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                            smp.storm_tide_offset_h)
        lu = ctx.landuse(smp.ssp, smp.horizon_year)
        storage, kappa, manning, gamma, base_S, base_n = SMP.materialise(smp, dom, lu)
        site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
        t0 = time.perf_counter()
        res = api.simulate(site, scen)
        wall = time.perf_counter() - t0
        fo = res.extras["forcing"]
        times = res.times.numpy()
        h, u, v = res.depth_series.numpy(), res.u.numpy(), res.v.numpy()
        hc = coarsen_mean(h, f)
        hc_safe = np.where(hc > 0, hc, 1.0)
        uc = np.where(hc > 0, coarsen_mean(h * u, f) / hc_safe, 0.0)
        vc = np.where(hc > 0, coarsen_mean(h * v, f) / hc_safe, 0.0)
        ms = res.extras["mass_series"]
        rec = {
            "meta": smp.meta(), "wall_s": wall, "mass_balance_error": res.mass_balance_error,
            "times": times.astype(np.float32),
            "h": hc.astype(np.float32), "u": uc.astype(np.float32), "v": vc.astype(np.float32),
            "depth_max_full": h.max(0).astype(np.float32),
            "eta1": res.channel_eta.numpy(), "q1": res.channel_q.numpy(),
            "fields": {"storage": coarsen_mean(storage, f).astype(np.float32),
                       "kappa": coarsen_mean(kappa, f).astype(np.float32),
                       "manning": coarsen_mean(manning, f).astype(np.float32),
                       "landuse_frac": coarsen_mean(np.eye(LU.N_CLASSES)[lu].transpose(2, 0, 1), f).astype(np.float32),
                       "gamma": gamma.astype(np.float32)},
            "forcing": {"rain_mmh": fo.rain.series(fo.t_end, cfg.model.forcing_steps).astype(np.float32),
                        "tide_m": np.array([fo.tide(t) for t in np.linspace(0, fo.t_end, cfg.model.forcing_steps)],
                                           np.float32),
                        "inflow_m3s": np.array([fo.inflow(t) for t in np.linspace(0, fo.t_end, cfg.model.forcing_steps)],
                                               np.float32),
                        "t_end": fo.t_end, "storm_start": fo.storm_start, "rain_factor": fo.rain_factor,
                        "slr_m": fo.slr_m, "arf": fo.arf_domain,
                        "horton_rate_mmh": _horton_series(lu, kappa, fo, times, f)},
            "volumes": {"infiltrated_m3": res.infiltrated_volume_m3, "stored_m3": res.stored_volume_m3,
                        "sources": res.extras["sources"], "series": ms},
            "ledger": _ledger(res, h, f, dom.dx),
            "run_log": {k: v for k, v in asdict(res.extras["run_log"]).items() if k != "series"},
        }
        torch.save(_to_tensors(rec), path)
        index[str(i)] = {**smp.meta(), "wall_s": wall, "mass_err": res.mass_balance_error,
                         "peak_depth_land": float(h.max(0)[~(dom.sea)].max()),
                         "in_distribution": res.in_distribution}
        index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
        done_now += 1
        rate = (time.perf_counter() - t_start) / done_now
        left = sum(1 for k in range(i + 1, n) if str(k) not in index)
        log.info("sim %d/%d %s: %.0f s, mass err %.1e, ETA %.1f h", i + 1, n, _desc(smp), wall,
                 res.mass_balance_error, left * rate / 3600)
    env = _envelope(cfg, dom, index, out, ctx)
    env.save(out / "envelope.json")
    write_card(cfg, out, index, env, ctx)
    return out


def _horton_series(lu, kappa, fo, times, f):
    """Domain-mean potential infiltration rate (mm/h) at the output times (for the physics loss)."""
    rates = []
    for t in times:
        r = LU.horton_rate_mmh(lu, max(t - fo.storm_start, 0.0)) * kappa
        rates.append(coarsen_mean(r, f))
    return np.stack(rates).astype(np.float32)


def _ledger(res, h, f, dx):
    """Engine volume ledger at the output times. ``v2d_coarse`` is the surface volume
    inside the coarse-grid extent (what the surrogate's global-mass loss can see)."""
    L = res.extras["ledger"]
    keys = ["v2d", "v1d", "storage", "rain", "infiltration", "bnd2d_in", "bnd2d_out", "bnd1d_in", "bnd1d_out"]
    out = {k: np.array([x[k] for x in L], np.float64) for k in keys}
    ny, nx = h.shape[-2:]
    out["v2d_coarse"] = h[:, :(ny // f) * f, :(nx // f) * f].astype(np.float64).sum(axis=(1, 2)) * dx * dx
    return out


def _to_tensors(obj):
    if isinstance(obj, dict):
        return {k: _to_tensors(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.dtype != object and obj.dtype.kind in "fiub":
        return torch.from_numpy(np.ascontiguousarray(obj))
    return obj


def _desc(s):
    return f"RP{s.return_period_yr} {s.rcp}/{s.ssp} {s.horizon_year} tide={'on' if s.tide_on else 'off'}"


def _envelope(cfg, dom, index, out, ctx):
    env = SMP.envelope_from(cfg.data)
    active = ~(dom.sea | dom.channel)
    for k, meta in index.items():
        if meta["split"] != "train":
            continue
        smp = SMP.draw(int(k), dom, cfg.data, cfg.seed, meta["split"])
        lu = ctx.landuse(smp.ssp, smp.horizon_year)
        storage, kappa, _, _, base_S, _ = SMP.materialise(smp, dom, lu)
        ds = (storage - base_S)[active]
        env.max_storage_cover = max(env.max_storage_cover, float(np.mean(ds > 0.01)))
        env.max_kappa_cover = max(env.max_kappa_cover, float(np.mean(kappa[active] > 1.01)))
        env.max_mean_storage_m = max(env.max_mean_storage_m, float(np.mean(np.clip(ds, 0, None))))
        env.max_mean_kappa_excess = max(env.max_mean_kappa_excess, float(np.mean(kappa[active] - 1.0)))
    return env


def write_card(cfg, out: Path, index: dict, env, ctx) -> Path:
    prov = ctx.provenance("engine", ("domain", "solver", "forcing", "data"))
    s = markdown_header(prov, "Dataset card: engine simulations for GeoKAN-PINO")
    rows = list(index.values())
    s += f"- simulations: {len(rows)} (train {sum(r['split'] == 'train' for r in rows)}, "
    s += f"val {sum(r['split'] == 'val' for r in rows)}, test {sum(r['split'] == 'test' for r in rows)}); "
    s += "split by whole scenario, stratified by return period\n"
    s += f"- grid: {ctx.domain.nx} x {ctx.domain.ny} @ {ctx.domain.dx:g} m; stored at coarsening factor "
    s += f"{cfg.data.coarsen}; output interval {cfg.solver.output_interval_s:g} s\n"
    s += f"- spatial order {cfg.solver.spatial_order}, precision {cfg.solver.precision}\n"
    if rows:
        w = np.array([r["wall_s"] for r in rows])
        me = np.array([r["mass_err"] for r in rows])
        s += f"- engine wall time per simulation: median {np.median(w):.0f} s, max {w.max():.0f} s\n"
        s += f"- mass-balance error: max {me.max():.2e} (every run asserted < {cfg.solver.mass_tol:g})\n"
        s += f"- baseline (no intervention) samples: {sum(r['baseline'] for r in rows)}\n"
        for rp in cfg.data.return_periods:
            s += f"- RP{rp}: {sum(r['return_period_yr'] == rp for r in rows)} simulations\n"
    s += "\n## Sampled intervention envelope (Part B must stay inside it)\n\n| field | range |\n|---|---|\n"
    s += f"| storage_depth S | {env.storage_range_m} m (absolute, after adding to the land-use baseline) |\n"
    s += f"| infil_multiplier kappa | {env.kappa_range} |\n"
    s += f"| manning | baseline x {env.manning_factor_range}; coastal planting within {env.coastal_band_m:g} m of " \
         f"the coast up to {env.manning_planted_range} |\n"
    s += f"| channel_gamma | {env.gamma_range} per reach (the SSP drainage factor multiplies it) |\n"
    s += f"| return period | {env.return_periods} yr |\n| RCP | {env.rcps} |\n| SSP | {env.ssps} |\n"
    s += f"| horizon year | {env.horizon_range} |\n| storm-tide offset | {env.offset_range_h} h |\n"
    s += f"| max storage coverage (train) | {env.max_storage_cover:.1%} of land |\n"
    s += f"| max infiltration-boost coverage (train) | {env.max_kappa_cover:.1%} of land |\n"
    s += f"| max mean added storage (train) | {env.max_mean_storage_m:.3f} m |\n"
    s += f"| max mean kappa excess (train) | {env.max_mean_kappa_excess:.3f} |\n"
    s += "\nPatterns: Gaussian blobs biased to urban land, road-aligned strips, coastal bands, thresholded smooth " \
         "noise; signed smooth Manning perturbations; per-reach uniform gamma.\n"
    p = out / "dataset_card.md"
    p.write_text(s, encoding="utf-8")
    (Path(cfg.outdir) / "dataset_card.md").write_text(s, encoding="utf-8")
    return p
