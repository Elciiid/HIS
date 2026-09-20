"""Score the surrogate against held-out engine runs, and (with real data)
the engine against observed high-water marks.

No accuracy statement exists anywhere in this package except the numbers this
module computes.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import api
from .config import PLACEHOLDER_IDF_ROXAS, RunConfig, seed_everything
from .data import sampler as SMP
from .data.dataset import SimDataset
from .data.generate import coarsen_mean
from .model.geokan_pino import model_dir
from .viz.depthmap import monitoring_points
from .viz.provenance import markdown_header, savefig, write_csv, write_json

log = logging.getLogger("hydrointel.evaluate")
THRESHOLDS = (0.15, 0.30, 0.50, 1.00)


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def err_stats(p, o, mask=None):
    d = (p - o) if mask is None else (p - o)[mask]
    if d.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "n": 0}
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d)),
            "n": int(d.size)}


def contingency(p, o, thr, mask):
    P, O = (p >= thr) & mask, (o >= thr) & mask
    H = int(np.sum(P & O)); M = int(np.sum(~P & O)); F = int(np.sum(P & ~O))
    return {"csi": H / (H + M + F) if H + M + F else float("nan"),
            "hit_rate": H / (H + M) if H + M else float("nan"),
            "far": F / (H + F) if H + F else float("nan"), "hits": H, "misses": M, "false_alarms": F}


def nse(p, o):
    den = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((p - o) ** 2) / den) if den > 1e-12 else float("nan")


def edge_score(smp: SMP.Sample, cfg) -> float:
    """0 = centre of the sampled ranges, 1 = at a range limit (max over intervention fields)."""
    d = cfg.data
    s = []
    if smp.storage_delta.max() > 0:
        s.append(smp.storage_delta.max() / d.storage_range_m[1])
    if smp.kappa.max() > 1:
        s.append((smp.kappa.max() - 1) / (d.kappa_range[1] - 1))
    g = smp.gamma
    if not np.allclose(g, 1):
        lo, hi = d.gamma_range
        s.append(float(np.max(np.abs(g - (lo + hi) / 2) / ((hi - lo) / 2))))
    s.append(float(np.max(np.abs(smp.manning_factor - 1)) / d.manning_perturb))
    return float(max(s)) if s else 0.0


def edge_probe(index: int, dom, cfg, seed: int) -> SMP.Sample:
    """A sample with intervention amplitudes pushed to the edge of the sampled ranges."""
    for k in range(1000):
        smp = SMP.draw(10_000_000 + index * 1000 + k, dom, cfg.data, seed, "probe")
        if not smp.baseline and "gamma" in smp.patterns and any(p.startswith("storage") for p in smp.patterns):
            break
    d = cfg.data
    land = ~(dom.sea | dom.channel)
    if smp.storage_delta.max() > 0:
        smp.storage_delta = smp.storage_delta / smp.storage_delta.max() * (d.storage_range_m[1] * 0.97) * land
    if smp.kappa.max() > 1:
        smp.kappa = 1 + (smp.kappa - 1) / (smp.kappa.max() - 1) * (d.kappa_range[1] - 1) * 0.97
    rng = np.random.default_rng(index)
    smp.gamma = np.where(rng.uniform(size=smp.gamma.shape) < 0.5, d.gamma_range[0] * 1.01, d.gamma_range[1] * 0.99)
    return smp


def site_from_sample(smp, dom, ctx):
    lu = ctx.landuse(smp.ssp, smp.horizon_year)
    storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
    site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
    scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on, smp.storm_tide_offset_h)
    return site, scen


# ---------------------------------------------------------------------------
def score(pred: api.FloodResult, eng: dict, dom, f: int, pts, builder) -> dict:
    """Compare a surrogate result with an engine record (dataset format)."""
    land = ~(dom.sea | dom.channel)
    h_e = eng["h"]                                              # (nt, nyc, nxc) coarse engine
    h_p = coarsen_mean(pred.depth_series.numpy(), f)
    nt = min(len(h_e), len(h_p))
    h_e, h_p = h_e[:nt], h_p[:nt]
    land_c = coarsen_mean(land.astype(float), f) > 0.5
    L = np.broadcast_to(land_c, h_e.shape)
    wet = L & (h_e > 0.05)
    out = {"depth_all": err_stats(h_p, h_e, L), "depth_wet": err_stats(h_p, h_e, wet)}
    dm_e = eng["depth_max_full"]
    dm_p = pred.depth_max.numpy()
    either = land & ((dm_e > 0.05) | (dm_p > 0.05))
    ad = np.abs(dm_p - dm_e)[either]
    out["peak_abs_err"] = {"median": float(np.median(ad)) if ad.size else float("nan"),
                           "p90": float(np.percentile(ad, 90)) if ad.size else float("nan"),
                           "max": float(ad.max()) if ad.size else float("nan")}
    out["depth_max_full"] = err_stats(dm_p, dm_e, either)
    out["csi"] = {str(t): contingency(dm_p, dm_e, t, land) for t in THRESHOLDS}
    # error introduced by coarse representation + bilinear upsampling alone (engine vs engine)
    up = builder.upsample(torch.as_tensor(h_e, dtype=torch.float32, device=builder.device)).amax(0).cpu().numpy()
    out["upsampling_only"] = err_stats(up, dm_e, either)
    # velocities
    u_e, v_e = eng["u"][:nt], eng["v"][:nt]
    u_p = coarsen_mean(pred.u.numpy() * pred.depth_series.numpy(), f)[:nt] / np.maximum(h_p, 1e-6)
    v_p = coarsen_mean(pred.v.numpy() * pred.depth_series.numpy(), f)[:nt] / np.maximum(h_p, 1e-6)
    s_e, s_p = np.hypot(u_e, v_e), np.hypot(u_p, v_p)
    out["speed_wet"] = err_stats(s_p, s_e, wet)
    both = wet & (s_e > 0.1) & (s_p > 0.1)
    ang = np.abs(np.angle(np.exp(1j * (np.arctan2(v_p, u_p) - np.arctan2(v_e, u_e)))))
    out["direction_err_deg"] = float(np.degrees(np.mean(ang[both]))) if both.any() else float("nan")
    # hydrographs at monitoring points (coarse cells)
    times = eng["times"][:nt]
    hyd = {}
    for p in pts:
        jc, ic = min(p["j"] // f, h_e.shape[1] - 1), min(p["i"] // f, h_e.shape[2] - 1)
        o, q = h_e[:, jc, ic], h_p[:, jc, ic]
        hyd[p["name"]] = {"nse": nse(q, o), "peak_timing_err_h": float((times[np.argmax(q)] - times[np.argmax(o)]) / 3600),
                          "peak_depth_err_m": float(q.max() - o.max()), "engine_peak_m": float(o.max()),
                          "obs": o.tolist(), "pred": q.tolist(), "t": times.tolist()}
    out["hydrographs"] = hyd
    # 1-D discharge
    q_e, q_p = eng["q1"][:nt], pred.channel_q.numpy()[:nt]
    out["q1"] = err_stats(q_p, q_e)
    pk_e, pk_p = np.abs(q_e).max(), np.abs(q_p).max()
    out["q1_peak_rel_err"] = float((pk_p - pk_e) / max(pk_e, 1e-9))
    out["volumes"] = {"infiltrated_rel_err": float((pred.infiltrated_volume_m3 - eng["infil"]) / max(eng["infil"], 1.0)),
                      "stored_rel_err": float((pred.stored_volume_m3 - eng["stored"]) / max(eng["stored"], 1.0))}
    return out


def _eng_record(rec, builder=None) -> dict:
    """The engine record as the model grid sees it. A dataset may store its snapshots on a
    finer grid than the model trains on, so they go through the same block-averaging the
    training inputs use (``builder.snapshots``) before anything is compared."""
    n = lambda a: a.numpy() if torch.is_tensor(a) else np.asarray(a)
    h, u, v = builder.snapshots(rec) if builder is not None else (n(rec["h"]), n(rec["u"]), n(rec["v"]))
    return {"h": n(h), "u": n(u), "v": n(v), "times": n(rec["times"]),
            "depth_max_full": n(rec["depth_max_full"]), "q1": n(rec["q1"]),
            "infil": float(rec["volumes"]["infiltrated_m3"]), "stored": float(rec["volumes"]["stored_m3"])}


def _eng_from_result(res: api.FloodResult, f: int) -> dict:
    h, u, v = res.depth_series.numpy(), res.u.numpy(), res.v.numpy()
    hc = coarsen_mean(h, f)
    safe = np.where(hc > 0, hc, 1.0)
    return {"h": hc, "u": np.where(hc > 0, coarsen_mean(h * u, f) / safe, 0), "v": np.where(hc > 0, coarsen_mean(h * v, f) / safe, 0),
            "times": res.times.numpy(), "depth_max_full": res.depth_max.numpy(), "q1": res.channel_q.numpy(),
            "infil": res.infiltrated_volume_m3, "stored": res.stored_volume_m3}


# ---------------------------------------------------------------------------
def evaluate(cfg: RunConfig, device=None, n_probe: int | None = None) -> Path:
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    model = ctx.model()
    from .model.batch import SampleBuilder
    builder = SampleBuilder.from_context(ctx, model)
    f = builder.f
    root = api.dataset_dir(cfg)
    test = SimDataset(root, "test")
    pts = monitoring_points(dom)
    out = Path(cfg.outdir)
    mdir = model_dir(cfg)
    ecache = api.engine_cache_dir(cfg)
    (ecache / "effects").mkdir(parents=True, exist_ok=True)
    rows = []
    warm = None
    for sid in test.ids:
        rec = test.load(sid)
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, "test")
        site, scen = site_from_sample(smp, dom, ctx)
        if warm is None:
            api.predict(site, scen, detail="full")          # warm-up (kernel selection, allocations)
            warm = True
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = api.predict(site, scen, detail="full")
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        sc = score(pred, _eng_record(rec, builder), dom, f, pts, builder)
        rows.append({"sim": sid, "kind": "test", "rp": smp.return_period_yr, "baseline": smp.baseline,
                     "edge_score": edge_score(smp, cfg), "engine_wall_s": float(rec["wall_s"]),
                     "surrogate_wall_s": wall, "engine_mass_err": float(rec["mass_balance_error"]),
                     "surrogate_mass_err": pred.mass_balance_error, "in_distribution": pred.in_distribution, **sc})
        log.info("test sim %d: depth RMSE (wet) %.3f m, CSI@0.3 %.2f", sid, sc["depth_wet"]["rmse"],
                 sc["csi"]["0.3"]["csi"])
    # edge-of-envelope probes (engine + surrogate, cached)
    n_probe = n_probe if n_probe is not None else (1 if cfg.quick else 4)
    unstable = []                  # designs the engine could not simulate
    for k in range(n_probe):
        smp = edge_probe(k, dom, cfg, cfg.seed)
        site, scen = site_from_sample(smp, dom, ctx)
        cache = ecache / f"probe_{k}.pt"
        if cache.exists():
            eng = torch.load(cache, weights_only=False)
        else:
            from .solver.coupling import NumericalInstabilityError
            t0 = time.perf_counter()
            try:
                res = api.simulate(site, scen)
            except NumericalInstabilityError as e:
                # the ground truth itself is invalid here: record it, score nothing against it
                unstable.append({"case": f"edge probe {k}", "error": str(e)})
                log.error("edge probe %d: engine unstable, not scored: %s", k, e)
                continue
            eng = _eng_from_result(res, f)
            eng.update({"wall_s": time.perf_counter() - t0, "mass_err": res.mass_balance_error})
            torch.save(eng, cache)
        t0 = time.perf_counter()
        pred = api.predict(site, scen, detail="full")
        wall = time.perf_counter() - t0
        sc = score(pred, eng, dom, f, pts, builder)
        rows.append({"sim": f"probe{k}", "kind": "edge_probe", "rp": smp.return_period_yr, "baseline": False,
                     "edge_score": edge_score(smp, cfg), "engine_wall_s": eng["wall_s"], "surrogate_wall_s": wall,
                     "engine_mass_err": eng["mass_err"], "surrogate_mass_err": pred.mass_balance_error,
                     "in_distribution": pred.in_distribution, **sc})
    directional = storage_direction_check(ctx)
    from . import evaluate_effects as EE
    eff = EE.run(cfg, ctx, test.ids, ecache, edge_probe(0, dom, cfg, cfg.seed))
    eff["rows"].append(EE.storage_case(cfg, ctx, ecache))
    unstable += eff.get("unstable", [])
    eng_check = EE.storage_check_engine(ctx, ecache)
    g1 = EE.gate1(eff["rows"], [r for r in rows if r["kind"] == "test"])
    r2 = EE.report2(eff["rows"])
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train"))
    write_json(out / "intervention_effects.json", {"gate1": g1, "report2": r2, "effects": eff,
                                                    "storage_check_engine": eng_check,
                                                    "storage_check_surrogate": directional,
                                                    "engine_unstable": unstable}, prov)
    eff["unstable"] = unstable
    report = write_validation_report(cfg, ctx, rows, directional, out, mdir, effects=(eff, eng_check),
                                     gates=(g1, r2))
    return report


def storage_direction_check(ctx) -> dict:
    """Adding retention storage must not raise the surrogate's predicted depth: not
    inside the storage footprint (beyond 5 mm), not by more than 2 cm anywhere, in at
    most 0.1% of land cells by more than 1 cm, and total flood volume must fall.
    (Same criteria as the engine test in tests/test_api_contract.py.)"""
    dom = ctx.domain
    scen = api.Scenario(100, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    X, Y = np.meshgrid(dom.x, dom.y)
    land = ~(dom.sea | dom.channel)
    blob = (np.hypot(X - dom.urban_core[0], Y - dom.urban_core[1]) < 1500) & land
    S = base.storage_depth.numpy().copy()
    S[blob] = np.maximum(S[blob], 0.6)
    more = base.replace(storage_depth=torch.as_tensor(S))
    a, b = api.predict([base, more], scen, detail="full")
    d = (b.depth_max - a.depth_max).numpy()
    vol_a, vol_b = float(a.depth_series.sum()), float(b.depth_series.sum())
    res = {"scenario": "RP100 RCP4.5/SSP2 2050, storage raised to 0.6 m within 1.5 km of the city core",
           "footprint_max_increase_m": float(d[blob].max()), "footprint_mean_change_m": float(d[blob].mean()),
           "land_max_increase_m": float(d[land].max()),
           "land_fraction_increase_gt_1cm": float(np.mean(d[land] > 0.01)),
           "total_depth_volume_ratio": vol_b / max(vol_a, 1e-9)}
    res["passed"] = bool(res["footprint_max_increase_m"] <= 0.005 and res["land_max_increase_m"] <= 0.02
                         and res["land_fraction_increase_gt_1cm"] <= 0.001 and vol_b < vol_a)
    return res


# ---------------------------------------------------------------------------
def _pool(rows, key_path):
    vals = []
    for r in rows:
        v = r
        for k in key_path:
            v = v[k]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def limitations(cfg: RunConfig, ctx) -> list[str]:
    from .domain.landuse import LANDUSE_TABLE
    out = []
    sv, dm = cfg.solver, ctx.domain
    if ctx.real is None:
        out.append("Terrain, land use and channel network are SYNTHETIC (procedurally generated). Nothing in this "
                   "report is a prediction for real Roxas City.")
    else:
        out.append(f"Real input dataset {ctx.real.dataset_id}; see calibration_report.md for agreement with observations.")
    if cfg.forcing.idf == PLACEHOLDER_IDF_ROXAS:
        out.append(f"Rainfall uses PLACEHOLDER_IDF_ROXAS (a={cfg.forcing.idf['a']}, b={cfg.forcing.idf['b']}, "
                   f"c={cfg.forcing.idf['c']}, e={cfg.forcing.idf['e']}), not PAGASA-published coefficients.")
    if ctx.real is None or ctx.real.obs_flood is None:
        out.append("No calibration against observed flood events has been performed (no obs_flood layer).")
    out.append(f"Infiltration (Horton f0/fc/k) and Manning values are uncalibrated table values for "
               f"{len(LANDUSE_TABLE)} land-use classes.")
    out.append("Tidal constituents, upstream catchment parameters (area "
               f"{cfg.forcing.catchment.area_km2:g} km², runoff coefficient {cfg.forcing.catchment.runoff_coeff:g}), "
               "RCP rainfall/SLR factors and SSP modifiers are placeholders.")
    out.append(f"Numerics: {'MUSCL + SSP-RK2' if sv.spatial_order == 2 else 'first-order'} finite volumes on a "
               f"{dm.nx} x {dm.ny} grid at {dm.dx:g} m, dry threshold {sv.h_dry:g} m, CFL {sv.cfl:g}, "
               f"{sv.precision}; 1-D channels first order, above-bank width {sv.slot_width_frac:g} x bankfull top width; "
               f"weir exchange Cw = {sv.weir_cw:g}.")
    out.append(f"The surrogate is trained on a {cfg.data.coarsen}x-coarsened grid and upsampled bilinearly; the "
               "error this adds is reported separately (upsampling_only).")
    br = Path(cfg.outdir) / "benchmark_results.json"
    if br.exists():
        res = {r["key"]: r for r in json.loads(br.read_text(encoding="utf-8"))["results"]}
        if "6" in res and "o1_rising_limb_nrmse" in res["6"]["metrics"]:
            out.append(f"Thin rain-on-grid sheets on steep cells: first-order hydrostatic reconstruction gave a "
                       f"{res['6']['metrics']['o1_rising_limb_nrmse']:.1%} rising-limb error on the tilted-plane "
                       f"benchmark versus {res['6']['metrics']['o2_rising_limb_nrmse']:.1%} with MUSCL.")
        if "4" in res and "lab_o2_relL2_per_period" in res["4"]["metrics"]:
            out.append(f"Dry threshold {sv.h_dry:g} m: on a 10 cm-deep lab-scale Thacker bowl it produces a "
                       f"{res['4']['metrics']['lab_o2_relL2_per_period'][-1]:.1%} error after 3 periods; very shallow "
                       "flows (centimetres) are represented less accurately than flood depths.")
    out.append("Antecedent conditions: every storm starts with empty retention storage and dry soil (Horton f0).")
    out.append("Rainfall is spatially uniform over the domain; the RCP rainfall factor is applied without "
               "horizon-year interpolation (only sea-level rise is interpolated).")
    out.append("Urban fabric follows one interpretation of the specification: streets are the lowest paths, "
               f"blocks are raised {dm.cfg.block_raise_range_m} m and buildings a further {dm.cfg.building_raise_m} m.")
    return out


def write_validation_report(cfg, ctx, rows, directional, out: Path, mdir: Path, effects=None, gates=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train"))
    test = [r for r in rows if r["kind"] == "test"]
    probes = [r for r in rows if r["kind"] == "edge_probe"]
    s = markdown_header(prov, "Validation report: GeoKAN-PINO surrogate vs. held-out engine runs")
    # the gate first: nothing below matters for Part B if the surrogate cannot reproduce
    # the engine's effect of a design
    if gates is not None:
        from .evaluate_effects import gate_report
        s += gate_report(*gates)
        if not gates[0]["passed"]:
            s += ("**This surrogate does not reproduce the engine's effect of a design. Part B must not be built "
                  "against it, however good the accuracy numbers below look.**\n\n")
    s += (f"Test set: **{len(test)} simulations** held out by scenario (whole storms, stratified by return period), "
          f"plus {len(probes)} edge-of-envelope probe runs. All numbers below are computed by `evaluate.py`.\n\n")

    def table(group, title):
        t = f"### {title}\n\n| metric | " + " | ".join(g for g, _ in group) + " |\n|---|" + "---|" * len(group) + "\n"
        spec = [("depth RMSE, wet cells [m]", ("depth_wet", "rmse")), ("depth MAE, wet cells [m]", ("depth_wet", "mae")),
                ("depth bias, wet cells [m]", ("depth_wet", "bias")), ("depth RMSE, all land cells [m]", ("depth_all", "rmse")),
                ("depth MAE, all land cells [m]", ("depth_all", "mae")), ("depth bias, all land cells [m]", ("depth_all", "bias")),
                ("peak-depth absolute error, median [m]", ("peak_abs_err", "median")), ("peak-depth absolute error, p90 [m]", ("peak_abs_err", "p90")),
                ("peak-depth absolute error, max [m]", ("peak_abs_err", "max"))]
        spec += [(f"CSI @ {th:g} m", ("csi", str(th), "csi")) for th in THRESHOLDS]
        spec += [(f"hit rate @ {th:g} m", ("csi", str(th), "hit_rate")) for th in THRESHOLDS]
        spec += [(f"false-alarm ratio @ {th:g} m", ("csi", str(th), "far")) for th in THRESHOLDS]
        spec += [("speed RMSE, wet cells [m/s]", ("speed_wet", "rmse")), ("flow direction error [deg]", ("direction_err_deg",)),
                 ("1-D discharge RMSE [m³/s]", ("q1", "rmse")), ("1-D peak discharge rel. error", ("q1_peak_rel_err",)),
                 ("infiltrated volume rel. error", ("volumes", "infiltrated_rel_err")),
                 ("stored volume rel. error", ("volumes", "stored_rel_err")),
                 ("coarse+upsampling-only depth-max RMSE [m]", ("upsampling_only", "rmse")),
                 ("surrogate implied mass-balance error", ("surrogate_mass_err",)),
                 ("engine mass-balance error", ("engine_mass_err",))]
        for label, path in spec:
            if path[0] == "upsampling_only" and cfg.data.coarsen == 1:
                t += f"| {label} | " + " | ".join("n/a (no coarsening)" for _ in group) + " |\n"
                continue
            t += f"| {label} | " + " | ".join(f"{_pool(rs, path):.4g}" for _, rs in group) + " |\n"
        return t + "\n"

    groups = [("pooled", test)] + [(f"RP{rp}", [r for r in test if r["rp"] == rp]) for rp in cfg.data.return_periods]
    groups = [(g, rs) for g, rs in groups if rs]
    s += "## Accuracy against the engine\n\n" + table(groups, "Held-out test simulations")
    if probes:
        s += table([("edge probes", probes)], "Edge-of-envelope probes (interventions at 97-99% of the sampled ranges)")
    # hydrographs
    names = list(test[0]["hydrographs"]) if test else []
    s += "### Hydrographs at monitoring points (test set)\n\n| point | mean NSE | median NSE | mean peak-timing error [h] | mean peak-depth error [m] |\n|---|---|---|---|---|\n"
    for n in names:
        ns = [r["hydrographs"][n]["nse"] for r in test if not math.isnan(r["hydrographs"][n]["nse"])]
        pt = [r["hydrographs"][n]["peak_timing_err_h"] for r in test]
        pd_ = [r["hydrographs"][n]["peak_depth_err_m"] for r in test]
        s += (f"| {n} | {np.mean(ns) if ns else float('nan'):.3f} | {np.median(ns) if ns else float('nan'):.3f} | "
              f"{np.mean(pt):.2f} | {np.mean(pd_):.3f} |\n")
    s += "\nNSE is undefined (NaN) where the engine hydrograph is constant (e.g. a point that stays dry).\n\n"
    # baseline versus modified states: accuracy on the storms Part B's designs look like
    s += "### Baseline states against modified states (test set)\n\n"
    s += "| group | n | depth RMSE wet [m] | peak-depth abs. error p90 [m] | CSI @ 0.3 m |\n|---|---|---|---|---|\n"
    for nm, rs in (("baseline site (no intervention)", [r for r in test if r["baseline"]]),
                   ("modified site", [r for r in test if not r["baseline"]])):
        if rs:
            s += (f"| {nm} | {len(rs)} | {_pool(rs, ('depth_wet', 'rmse')):.4g} | "
                  f"{_pool(rs, ('peak_abs_err', 'p90')):.4g} | {_pool(rs, ('csi', '0.3', 'csi')):.3f} |\n")
    s += "\n"
    if effects is not None:
        from .evaluate_effects import report as effects_report
        s += effects_report(effects[0], effects[1], directional)
    # speed
    ew = np.median([r["engine_wall_s"] for r in rows])
    sw = np.median([r["surrogate_wall_s"] for r in rows])
    s += "## Speed\n\n"
    s += (f"Same machine ({ctx.device}). Engine: median {ew:.1f} s per storm (from the dataset generation logs). "
          f"Surrogate: median {sw * 1000:.0f} ms per storm, measured end to end through `api.predict` "
          f"(forcing construction, feature coarsening, encoder, decoder for all output times, bilinear upsampling, "
          f"transfer to CPU). **Speedup factor: {ew / sw:.0f}x.**\n\n")
    # distribution
    s += "## Error versus distance from the training distribution\n\n"
    s += "`edge_score` is 0 at the centre of the sampled intervention ranges and 1 at a range limit.\n\n"
    s += "| group | n | mean edge score | depth RMSE wet [m] | CSI @ 0.3 m |\n|---|---|---|---|---|\n"
    bins = [("test, edge < 0.5", [r for r in test if r["edge_score"] < 0.5]),
            ("test, edge >= 0.5", [r for r in test if r["edge_score"] >= 0.5]), ("edge probes", probes)]
    for nm, rs in bins:
        if rs:
            s += (f"| {nm} | {len(rs)} | {np.mean([r['edge_score'] for r in rs]):.2f} | "
                  f"{_pool(rs, ('depth_wet', 'rmse')):.4g} | {_pool(rs, ('csi', '0.3', 'csi')):.3f} |\n")
    s += ("\nPart B must clip its search to the envelope in `dataset_card.md`; `FloodResult.in_distribution` and "
          "`distribution_warnings` flag queries outside it.\n\n")
    s += ("## Phase 1 storage-direction criteria (surrogate; reported for continuity, no longer a gate)\n\n"
          "| item | value |\n|---|---|\n")
    for k, v in directional.items():
        s += f"| {k} | {v} |\n"
    s += "\n"
    # figures
    figs = []
    if test:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        x = [r["edge_score"] for r in rows]
        ax[0].scatter(x, [r["depth_wet"]["rmse"] for r in rows],
                      c=["C3" if r["kind"] == "edge_probe" else "C0" for r in rows])
        ax[0].set_xlabel("edge score"); ax[0].set_ylabel("depth RMSE, wet [m]"); ax[0].set_title("error vs. envelope position")
        for k, th in enumerate(THRESHOLDS):
            ax[1].bar(k, _pool(test, ("csi", str(th), "csi")), color="C0")
        ax[1].set_xticks(range(len(THRESHOLDS)), [f"{t:g} m" for t in THRESHOLDS]); ax[1].set_ylim(0, 1)
        ax[1].set_title("Critical Success Index (test, pooled)")
        r0 = test[0]
        for k, (n, hd) in enumerate(r0["hydrographs"].items()):
            ax[2].plot(np.array(hd["t"]) / 3600, hd["obs"], "-", color=f"C{k}", label=f"{n} engine")
            ax[2].plot(np.array(hd["t"]) / 3600, hd["pred"], "--", color=f"C{k}", label=f"{n} surrogate")
        ax[2].set_title(f"hydrographs, test sim {r0['sim']}"); ax[2].set_xlabel("t [h]"); ax[2].set_ylabel("depth [m]")
        ax[2].legend(fontsize=6, ncol=2)
        figs.append(savefig(fig, out / "figures" / "validation_summary.png", prov))
    for p in figs:
        s += f"![validation]({Path(p).relative_to(out).as_posix()})\n\n"
    for extra in ("training_curves.png", "overfit_check.png"):
        if (out / "figures" / extra).exists():
            s += f"![{extra}](figures/{extra})\n\n"
    ov = mdir / "overfit_check.json"
    if ov.exists():
        o = json.loads(ov.read_text(encoding="utf-8"))
        s += (f"Overfit sanity check: data loss on {o.get('n_samples', '?')} simulations fell from "
              f"{o['initial_loss']:.4g} to {o['final_loss']:.4g} in {o.get('steps', '?')} steps "
              f"({'passed' if o['passed'] else 'FAILED'}).")
        if o.get("note"):
            s += f" Note: {o['note']}."
        s += "\n\n"
    from .forcing.scenarios import PATHWAY_NOTE, SSP_DEFAULT_RCP
    s += ("## Why the scenario pathways matter\n\n" + PATHWAY_NOTE + " In this system the RCP pathway scales "
          "rainfall and sets sea-level rise; the SSP pathway expands urban land along the road network (raising "
          "roughness, cutting infiltration and removing bund storage), scales exposure, and sets the baseline "
          "channel conveyance. Default pairings: " + ", ".join(f"{k}–{v}" for k, v in SSP_DEFAULT_RCP.items()) +
          " (SSP3 is conventionally paired with RCP7.0, which the placeholder table does not carry).\n\n")
    s += "## Limitations\n\n" + "".join(f"- {x}\n" for x in limitations(cfg, ctx))
    path = out / "validation_report.md"
    path.write_text(s, encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k != "hydrographs"} for r in rows]
    write_json(out / "validation_metrics.json", {"rows": slim, "directional": directional}, prov)
    write_csv(out / "validation_metrics.csv",
              ["sim", "kind", "rp", "edge_score", "depth_rmse_wet", "depth_rmse_all", "csi_0.3", "engine_wall_s",
               "surrogate_wall_s"],
              [(r["sim"], r["kind"], r["rp"], r["edge_score"], r["depth_wet"]["rmse"], r["depth_all"]["rmse"],
                r["csi"]["0.3"]["csi"], r["engine_wall_s"], r["surrogate_wall_s"]) for r in rows], prov)
    # the surrogate's contract tolerance, used by tests/test_api_contract.py
    tol = {"depth_max_rmse_m": _pool(test, ("depth_max_full", "rmse")) if test else None,
           "depth_max_p90_abs_m": _pool(test, ("peak_abs_err", "p90")) if test else None}
    (mdir / "contract_tolerance.json").write_text(json.dumps(tol), encoding="utf-8")
    log.info("validation report: %s", path)
    return path


# ---------------------------------------------------------------------------
def calibrate(cfg: RunConfig, device=None) -> Path:
    """Score the engine against observed high-water marks (needs real data)."""
    from .engine import Forcing, build_engine
    from .forcing.catchment import clark_hydrograph
    from .forcing.hyetograph import gauge_series
    from .forcing.scenarios import SSPState
    from .forcing.tide import GaugeTide
    from .io.loaders import read_table
    ctx = api.configure(cfg, device)
    out = Path(cfg.outdir) / "calibration_report.md"
    prov = ctx.provenance("engine")
    if ctx.real is None or ctx.real.obs_flood is None:
        msg = ("No real dataset with an `obs_flood` layer was found (artifacts/inputs/manifest.json). "
               "Nothing to calibrate against; no calibration metrics exist.\n")
        out.write_text(markdown_header(prov, "Calibration report") + msg, encoding="utf-8")
        log.warning(msg.strip())
        return out
    dom, real = ctx.domain, ctx.real
    rain = gauge_series(real.rain_gauge)
    t_end = rain.duration_s + cfg.solver.recession_h * 3600
    fo = Forcing(rain, clark_hydrograph(rain, cfg.forcing.catchment, t_end), GaugeTide.from_csv(real.tide_gauge),
                 t_end, 0.0, float(np.argmax(rain.intensity_mmh) * rain.step_s), 1.0, 0.0, 1.0, 1.0,
                 SSPState("observed", 0.0, 1.0, 1.0))
    site = api.baseline_site(api.Scenario(100, "RCP4.5", "SSP2", 2025))
    eng = build_engine(cfg, dom, fo, site.storage_depth.numpy(), site.infil_multiplier.numpy(),
                       site.manning.numpy(), site.channel_gamma.numpy(), device=ctx.device)
    hmax = None

    def snap(s):
        nonlocal hmax
        hmax = s.h if hmax is None else np.maximum(hmax, s.h)
    eng.run(t_end, out_times=np.arange(0, t_end + 1, 300.0), on_snapshot=snap, label="calibration event")
    obs = read_table(real.obs_flood, ["event_id", "x", "y", "max_depth", "source"])
    xmin, ymin = dom.report["bounds_m"][0], dom.report["bounds_m"][1]
    o, p = [], []
    for r in obs:
        j, i = dom.cell_index(float(r["x"]) - xmin, float(r["y"]) - ymin)
        o.append(float(r["max_depth"])); p.append(float(hmax[j, i]))
    o, p = np.array(o), np.array(p)
    st = err_stats(p, o)
    hit = float(np.mean((p >= 0.15) == (o >= 0.15)))
    s = markdown_header(prov, "Calibration report: engine vs observed high-water marks")
    s += (f"Observations: {len(o)} points from `{real.obs_flood.name}`.\n\n| metric | value |\n|---|---|\n"
          f"| RMSE [m] | {st['rmse']:.3f} |\n| bias [m] | {st['bias']:.3f} |\n| MAE [m] | {st['mae']:.3f} |\n"
          f"| flooded/not-flooded agreement at 0.15 m | {hit:.1%} |\n")
    out.write_text(s, encoding="utf-8")
    return out
