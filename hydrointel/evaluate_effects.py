"""How well does the surrogate predict the *change* an intervention makes?

Part B never asks the surrogate for a flood map on its own. It asks: if we add
this storage, raise this infiltration, plant this roughness, dredge this
channel -- what changes? Overall accuracy does not answer that; a surrogate can
match absolute depths well and still get the effect of a design wrong, even its
sign. So this module compares effects, not states:

    effect_engine    = engine(design)    - engine(baseline site)
    effect_surrogate = surrogate(design) - surrogate(baseline site)

on peak depth, for:

* every modified held-out test storm, paired with an engine run of the same
  scenario's baseline site (cached, one per scenario);
* single-field probes: storage alone, infiltration alone, roughness alone and
  conveyance alone, each at ~97% of its sampled range, on one scenario -- this is
  error as a function of the intervention field, and near the envelope edge.

Reported per case: effect RMSE and bias over affected land cells; the fraction of
cells the engine says changed by more than 1 cm where the surrogate gets the
direction right; correlation of the two effect fields; flooded-area change; and
the disbenefit measures Part B constrains on (largest rise, area worsened by more
than 1 cm), surrogate against engine.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from . import api
from .data import sampler as SMP

log = logging.getLogger("hydrointel.evaluate")
AFFECTED_M = 0.01
AREA_THRESHOLDS = (0.15, 0.30, 0.50)


def _engine_peak(site, scen, cache: Path) -> tuple[np.ndarray, float]:
    """Engine peak depth for (site, scenario), cached on disk: each costs a full storm."""
    if cache.exists():
        d = torch.load(cache, weights_only=False)
        return d["depth_max"], d["wall_s"]
    t0 = time.perf_counter()
    res = api.simulate(site, scen)
    out = {"depth_max": res.depth_max.numpy().astype(np.float32), "wall_s": time.perf_counter() - t0,
           "mass_err": res.mass_balance_error}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out["depth_max"], out["wall_s"]


def intensity(site, base, dom, cfg) -> dict:
    """How strongly a site departs from the baseline, per field (land cells)."""
    land = ~(dom.sea | dom.channel)
    n = lambda a: a.numpy() if torch.is_tensor(a) else np.asarray(a)
    dS = (n(site.storage_depth) - n(base.storage_depth))[land]
    k = n(site.infil_multiplier)[land]
    nr = (n(site.manning) / n(base.manning))[land]
    g = n(site.channel_gamma)
    d = cfg.data
    out = {"storage_added_mean_m": float(np.mean(np.clip(dS, 0, None))),
           "storage_cover": float(np.mean(dS > 0.01)),
           "kappa_excess_mean": float(np.mean(k - 1.0)),
           "manning_change_mean": float(np.mean(np.abs(nr - 1.0))),
           "gamma_change_mean": float(np.mean(np.abs(g - 1.0)))}
    # one number for sorting: each field's mean change over its sampled span, summed
    out["total"] = (out["storage_added_mean_m"] / d.storage_range_m[1]
                    + out["kappa_excess_mean"] / (d.kappa_range[1] - 1.0)
                    + out["manning_change_mean"] / d.manning_perturb
                    + out["gamma_change_mean"] / max(d.gamma_range[1] - 1.0, 1.0 - d.gamma_range[0]))
    return out


def effect_metrics(d_eng: np.ndarray, d_sur: np.ndarray, pk_eng: np.ndarray, pk_eng_base: np.ndarray,
                   pk_sur: np.ndarray, pk_sur_base: np.ndarray, land: np.ndarray, dx: float) -> dict:
    aff = land & ((np.abs(d_eng) > AFFECTED_M) | (np.abs(d_sur) > AFFECTED_M))
    eng_changed = land & (np.abs(d_eng) > AFFECTED_M)
    e = (d_sur - d_eng)[aff]
    out = {"n_affected": int(aff.sum()), "n_engine_changed": int(eng_changed.sum()),
           "effect_rmse_m": float(np.sqrt(np.mean(e ** 2))) if e.size else 0.0,
           "effect_bias_m": float(np.mean(e)) if e.size else 0.0,
           "sign_agreement": float(np.mean(np.sign(d_sur[eng_changed]) == np.sign(d_eng[eng_changed])))
           if eng_changed.any() else float("nan"),
           "effect_corr": float(np.corrcoef(d_sur[aff], d_eng[aff])[0, 1]) if aff.sum() > 2 and
           d_sur[aff].std() > 0 and d_eng[aff].std() > 0 else float("nan"),
           "mean_effect_engine_m": float(d_eng[land].mean()), "mean_effect_surrogate_m": float(d_sur[land].mean())}
    cell_ha = dx * dx / 1e4
    for thr in AREA_THRESHOLDS:
        out[f"area_change_{thr:g}_engine_ha"] = float(np.sum(land & (pk_eng >= thr)) - np.sum(land & (pk_eng_base >= thr))) * cell_ha
        out[f"area_change_{thr:g}_surrogate_ha"] = float(np.sum(land & (pk_sur >= thr)) - np.sum(land & (pk_sur_base >= thr))) * cell_ha
    for who, d in (("engine", d_eng), ("surrogate", d_sur)):
        out[f"max_rise_{who}_m"] = float(max(d[land].max(), 0.0))
        out[f"area_worsened_{who}_ha"] = float(np.sum(d[land] > AFFECTED_M)) * cell_ha
    return out


def single_field_sites(smp, dom, ctx, cfg):
    """Storage, infiltration, roughness and conveyance applied one at a time, each at the
    amplitude of an edge-of-envelope probe."""
    lu = ctx.landuse(smp.ssp, smp.horizon_year)
    storage, kappa, manning, gamma, base_S, base_n = SMP.materialise(smp, dom, lu)
    scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on, smp.storm_tide_offset_h)
    base = api.baseline_site(scen)
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    sites = {"storage only": base.replace(storage_depth=T(storage)),
             "infiltration only": base.replace(infil_multiplier=T(kappa)),
             "roughness only": base.replace(manning=T(manning)),
             "conveyance only": base.replace(channel_gamma=T(gamma))}
    changed = {k: v for k, v in sites.items()
               if intensity(v, base, dom, cfg)["total"] > 1e-6}
    return scen, base, changed


def run(cfg, ctx, test_ids, mdir: Path, probe_sample) -> dict:
    """All effect comparisons. Engine runs are cached under ``mdir``."""
    from .solver.coupling import NumericalInstabilityError
    dom = ctx.domain
    land = ~(dom.sea | dom.channel)
    rows, unstable = [], []

    def engine(site, scen, cache, case):
        try:
            return _engine_peak(site, scen, cache)[0]
        except NumericalInstabilityError as e:
            unstable.append({"case": case, "error": str(e)})
            log.error("%s: engine unstable, not scored: %s", case, e)
            return None
    for sid in test_ids:
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, "test")
        if smp.baseline:
            continue
        from .evaluate import site_from_sample
        site, scen = site_from_sample(smp, dom, ctx)
        base = api.baseline_site(scen)
        from .data.dataset import SimDataset
        rec = SimDataset(api.dataset_dir(cfg), "test").load(sid)
        pk_e = np.asarray(rec["depth_max_full"])
        pk_e0 = engine(base, scen, mdir / "effects" / f"test_base_{sid}.pt", f"test {sid} baseline")
        if pk_e0 is None:
            continue
        sur = api.predict([site, base], scen)
        pk_s, pk_s0 = sur[0].depth_max.numpy(), sur[1].depth_max.numpy()
        m = effect_metrics(pk_e - pk_e0, pk_s - pk_s0, pk_e, pk_e0, pk_s, pk_s0, land, dom.dx)
        rows.append({"case": f"test {sid}", "kind": "test", "patterns": smp.patterns,
                     "intensity": intensity(site, base, dom, cfg), **m})
        log.info("effect, test %d: RMSE %.3f m, sign agreement %.2f", sid, m["effect_rmse_m"], m["sign_agreement"])
    scen, base, sites = single_field_sites(probe_sample, dom, ctx, cfg)
    pk_e0 = engine(base, scen, mdir / "effects" / "single_base.pt", "single-field baseline")
    if pk_e0 is None:
        sites = {}
    names = list(sites)
    sur = api.predict([sites[n] for n in names] + [base], scen)
    pk_s0 = sur[-1].depth_max.numpy()
    for k, name in enumerate(names):
        pk_e = engine(sites[name], scen, mdir / "effects" / f"single_{name.split()[0]}.pt", name)
        if pk_e is None:
            continue
        pk_s = sur[k].depth_max.numpy()
        m = effect_metrics(pk_e - pk_e0, pk_s - pk_s0, pk_e, pk_e0, pk_s, pk_s0, land, dom.dx)
        rows.append({"case": name, "kind": "single field", "patterns": [name],
                     "intensity": intensity(sites[name], base, dom, cfg), **m})
        log.info("effect, %s: RMSE %.3f m, sign agreement %.2f", name, m["effect_rmse_m"], m["sign_agreement"])
    return {"rows": rows, "unstable": unstable, "single_field_scenario": {"return_period_yr": scen.return_period_yr, "rcp": scen.rcp,
                                                    "ssp": scen.ssp, "horizon_year": scen.horizon_year,
                                                    "tide_on": scen.tide_on}}


def storage_case(cfg, ctx, ecache: Path) -> dict:
    """The storage-direction design (0.6 m of storage within 1.5 km of the city core, RP100
    RCP4.5/SSP2 2050) as one more effect case: engine against surrogate."""
    dom = ctx.domain
    scen = api.Scenario(100, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    X, Y = np.meshgrid(dom.x, dom.y)
    land = ~(dom.sea | dom.channel)
    blob = (np.hypot(X - dom.urban_core[0], Y - dom.urban_core[1]) < 1500) & land
    S = base.storage_depth.numpy().copy()
    S[blob] = np.maximum(S[blob], 0.6)
    more = base.replace(storage_depth=torch.as_tensor(S))
    a, _ = _engine_peak(base, scen, ecache / "effects" / "storage_check_base.pt")
    b, _ = _engine_peak(more, scen, ecache / "effects" / "storage_check_more.pt")
    sur = api.predict([more, base], scen)
    pk_s, pk_s0 = sur[0].depth_max.numpy(), sur[1].depth_max.numpy()
    m = effect_metrics(b - a, pk_s - pk_s0, b, a, pk_s, pk_s0, land, dom.dx)
    m["footprint_max_rise_engine_m"] = float((b - a)[blob].max())
    m["footprint_max_rise_surrogate_m"] = float((pk_s - pk_s0)[blob].max())
    return {"case": "storage check", "kind": "storage check", "patterns": ["storage:disc 0.6 m"],
            "intensity": intensity(more, base, dom, cfg), **m}


# ---------------------------------------------------------------------------
# Gate 1 (model quality, the gate for Part B) and Report 2 (physical finding)
# ---------------------------------------------------------------------------
# Acceptance targets fixed 2026-09-20 before the paired surrogate was trained. Every
# effect case must meet the per-case targets.
ACCEPT = {"sign_agreement_min": 0.85, "effect_corr_min": 0.70, "area_change_rel_tol": 0.30,
          "mass_err_max": 0.02, "nse_min": 0.50, "local_rise_rel_tol": 0.30}


def gate1(rows: list, test_rows: list) -> dict:
    """Does the surrogate's predicted effect agree with the engine's? The engine is the
    reference by definition, so it cannot fail this; the surrogate is judged on reproducing
    the engine's effect, including its local depth increase, however large that is."""
    per = []
    tol = ACCEPT
    for r in rows:
        chk = {}
        chk["sign agreement >= 0.85"] = (r["sign_agreement"], r["sign_agreement"] >= tol["sign_agreement_min"])
        c = r["effect_corr"]
        chk["effect correlation >= 0.7"] = (c, bool(np.isfinite(c) and c >= tol["effect_corr_min"]))
        for t in ("0.15", "0.3"):
            e, s_ = r[f"area_change_{t}_engine_ha"], r[f"area_change_{t}_surrogate_ha"]
            rel = abs(s_ - e) / abs(e) if e != 0 else (0.0 if s_ == 0 else float("inf"))
            chk[f"area change @{t} m within 30%"] = (rel, rel <= tol["area_change_rel_tol"])
        e, s_ = r["max_rise_engine_m"], r["max_rise_surrogate_m"]
        rel = abs(s_ - e) / e if e > 0 else (0.0 if s_ == 0 else float("inf"))
        chk["local depth increase within 30%"] = (rel, rel <= tol["local_rise_rel_tol"])
        per.append({"case": r["case"], "checks": {k: {"value": float(v), "passed": bool(p)} for k, (v, p) in chk.items()},
                    "passed": all(p for _, p in chk.values())})
    mass = [r["surrogate_mass_err"] for r in test_rows]
    glob = {"implied mass-balance error < 2% (test mean)": {"value": float(np.mean(mass)),
                                                           "passed": bool(np.mean(mass) < tol["mass_err_max"])},
            "implied mass-balance error, worst test storm (reported)": {"value": float(np.max(mass)), "passed": None}}
    names = list(test_rows[0]["hydrographs"]) if test_rows else []
    for n in names:
        ns = [r["hydrographs"][n]["nse"] for r in test_rows if np.isfinite(r["hydrographs"][n]["nse"])]
        v = float(np.mean(ns)) if ns else float("nan")
        glob[f"hydrograph NSE > 0.5 at {n} (test mean)"] = {"value": v, "passed": bool(np.isfinite(v) and v > tol["nse_min"])}
    passed = all(p["passed"] for p in per) and all(g["passed"] for g in glob.values() if g["passed"] is not None)
    return {"targets": ACCEPT, "cases": per, "global": glob, "passed": passed,
            "n_cases_passed": sum(p["passed"] for p in per), "n_cases": len(per)}


def report2(rows: list) -> list:
    """What each design does to the place it makes worst, according to the engine. A property
    of the design that an LGU must be told, never a pass/fail on any model."""
    return [{"case": r["case"], "max_depth_increase_m": r["max_rise_engine_m"],
             "area_worsened_ha": r["area_worsened_engine_ha"],
             **{f"area_change_{t:g}_ha": r[f"area_change_{t:g}_engine_ha"] for t in AREA_THRESHOLDS}}
            for r in rows]


def gate_report(g1: dict, r2: list) -> str:
    s = (f"## Gate 1 (model quality, the gate for Part B): {'PASSED' if g1['passed'] else 'FAILED'}\n\n"
         "Does the surrogate's predicted effect of a design agree with the engine's? Per case: sign agreement "
         ">= 0.85 on cells the engine moved by more than 1 cm; effect correlation >= 0.7 and positive; flooded-area "
         "change within 30% of the engine's at 0.15 and 0.30 m; the engine's largest local depth increase reproduced "
         "within 30%. Overall: implied mass-balance error < 2%; hydrograph NSE > 0.5 at every monitoring point. "
         f"**{g1['n_cases_passed']} of {g1['n_cases']} cases pass every per-case target.**\n\n")
    keys = list(g1["cases"][0]["checks"]) if g1["cases"] else []
    s += "| case | " + " | ".join(keys) + " | all |\n|---|" + "---|" * (len(keys) + 1) + "\n"
    for c in g1["cases"]:
        s += f"| {c['case']} | " + " | ".join(
            f"{c['checks'][k]['value']:.2f} {'ok' if c['checks'][k]['passed'] else '**FAIL**'}" for k in keys) + \
            f" | {'PASS' if c['passed'] else '**FAIL**'} |\n"
    s += "\n| overall target | value | result |\n|---|---|---|\n"
    for k, v in g1["global"].items():
        s += f"| {k} | {v['value']:.3f} | {'-' if v['passed'] is None else ('ok' if v['passed'] else '**FAIL**')} |\n"
    s += ("\nFor sign agreement and correlation the value shown is the metric itself; for the area and local-rise "
          "targets it is the relative difference |surrogate - engine| / |engine|.\n\n")
    s += ("## Report 2 (physical finding per design; not a gate)\n\nWhat the engine says each design does to the "
          "place it makes worst. This is a property of the design, which an LGU must be told and which Part B "
          "constrains on as a design objective; it says nothing about the surrogate.\n\n"
          "| design | largest local depth increase [m] | land worsened by > 1 cm [ha] | flooded-area change "
          "@0.15 / 0.30 / 0.50 m [ha] |\n|---|---|---|---|\n")
    for r in r2:
        s += (f"| {r['case']} | {r['max_depth_increase_m']:.3f} | {r['area_worsened_ha']:.1f} | "
              f"{r['area_change_0.15_ha']:+.1f} / {r['area_change_0.3_ha']:+.1f} / {r['area_change_0.5_ha']:+.1f} |\n")
    s += ("\n**Deliberate change from Phase 1.** Phase 1 gated Part B on a storage-direction sign test that mixed "
          "two questions: whether the surrogate reproduces the engine, and whether the design is harmless. The "
          "engine itself failed it (a 40 mm local rise where the storage was added), which is impossible for a "
          "test of agreement with the engine and shows the test was measuring the design. The two are now "
          "separate: Gate 1 judges the model against the engine, which is the reference by definition; Report 2 "
          "states what each design physically does. The engine's 40 mm rise is now a reported finding, and the "
          "surrogate is judged on whether it reproduces that 40 mm, not on whether it is small.\n\n")
    return s


def storage_check_engine(ctx, mdir: Path) -> dict:
    """The surrogate's storage-direction test, run on the engine at full resolution with the
    same scenario and criteria, so the tolerance is shown to be one the physics can meet."""
    dom = ctx.domain
    scen = api.Scenario(100, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    X, Y = np.meshgrid(dom.x, dom.y)
    land = ~(dom.sea | dom.channel)
    blob = (np.hypot(X - dom.urban_core[0], Y - dom.urban_core[1]) < 1500) & land
    S = base.storage_depth.numpy().copy()
    S[blob] = np.maximum(S[blob], 0.6)
    more = base.replace(storage_depth=torch.as_tensor(S))
    a, _ = _engine_peak(base, scen, mdir / "effects" / "storage_check_base.pt")
    b, _ = _engine_peak(more, scen, mdir / "effects" / "storage_check_more.pt")
    d = b - a
    res = {"footprint_max_increase_m": float(d[blob].max()), "land_max_increase_m": float(d[land].max()),
           "land_fraction_increase_gt_1cm": float(np.mean(d[land] > 0.01)),
           "peak_volume_ratio": float(b[land].sum() / max(a[land].sum(), 1e-9))}
    res["passed"] = bool(res["footprint_max_increase_m"] <= 0.005 and res["land_max_increase_m"] <= 0.02
                         and res["land_fraction_increase_gt_1cm"] <= 0.001 and res["peak_volume_ratio"] < 1.0)
    return res


def report(eff: dict, eng_check: dict, sur_check: dict) -> str:
    rows = eff["rows"]
    s = "## Does the surrogate predict the effect of an intervention?\n\n"
    s += ("This is the question Part B asks. Effect = peak depth with the design minus peak depth on the baseline "
          "site, same scenario; engine against surrogate. 'Affected' cells are land cells where either model says "
          f"the design moved peak depth by more than {AFFECTED_M * 100:.0f} cm. **Sign agreement** is the share of "
          "cells the engine says changed where the surrogate gets the direction right: below ~0.5 the surrogate "
          "is no better than a coin at saying whether a design helps a place or hurts it.\n\n")
    s += ("| case | intensity | engine-changed cells | effect RMSE [m] | effect bias [m] | sign agreement | "
          "corr | max rise eng / sur [m] | area worsened eng / sur [ha] |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        s += (f"| {r['case']} | {r['intensity']['total']:.2f} | {r['n_engine_changed']} | {r['effect_rmse_m']:.3f} | "
              f"{r['effect_bias_m']:+.3f} | {r['sign_agreement']:.2f} | {r['effect_corr']:.2f} | "
              f"{r['max_rise_engine_m']:.3f} / {r['max_rise_surrogate_m']:.3f} | "
              f"{r['area_worsened_engine_ha']:.1f} / {r['area_worsened_surrogate_ha']:.1f} |\n")
    tests = [r for r in rows if r["kind"] == "test"]
    if len(tests) >= 2:
        med = np.median([r["intensity"]["total"] for r in tests])
        for label, grp in (("moderately modified (intensity <= median)", [r for r in tests if r["intensity"]["total"] <= med]),
                           ("heavily modified (intensity > median)", [r for r in tests if r["intensity"]["total"] > med])):
            if grp:
                s += (f"\n- **{label}**, {len(grp)} storms: mean effect RMSE "
                      f"{np.mean([r['effect_rmse_m'] for r in grp]):.3f} m, mean sign agreement "
                      f"{np.nanmean([r['sign_agreement'] for r in grp]):.2f}")
        s += "\n"
    s += ("\nArea change at thresholds (engine / surrogate, ha): " + "; ".join(
        f"{r['case']}: " + ", ".join(f"{t:g} m {r[f'area_change_{t:g}_engine_ha']:+.1f}/{r[f'area_change_{t:g}_surrogate_ha']:+.1f}"
                                     for t in AREA_THRESHOLDS) for r in rows) + "\n\n")
    if eff.get("unstable"):
        s += ("**Designs the engine could not simulate** (physical-plausibility check; not scored, because "
              "there is no valid ground truth to score against):" + chr(10) * 2
              + "".join(f"- {u['case']}: {u['error'][:200]}" + chr(10) for u in eff["unstable"]) + chr(10))
    s += ("### Storage-direction test on the engine itself (same grid, scenario and criteria)\n\n"
          "| item | engine | surrogate |\n|---|---|---|\n")
    for k in ("footprint_max_increase_m", "land_max_increase_m", "land_fraction_increase_gt_1cm", "passed"):
        s += f"| {k} | {eng_check.get(k)} | {sur_check.get(k)} |\n"
    s += (f"| volume ratio (design / baseline) | {eng_check['peak_volume_ratio']:.4f} (peak-depth sum) | "
          f"{sur_check.get('total_depth_volume_ratio', float('nan')):.4f} (depth-series sum) |\n\n")
    return s
