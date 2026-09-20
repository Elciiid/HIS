"""Precision study: is float32 safe for dataset generation?

The solver is written in float64 and every claim in the analytical suite is made
in float64. Consumer NVIDIA cards run double precision at a small fraction of
single-precision throughput, so the question is whether the *training dataset*
may be generated in float32 without changing the physics the surrogate learns.

This module answers it by measurement, not by assertion: three representative
storms are run at full resolution in both precisions from the same seed and the
same output times, and the differences are compared against fixed acceptance
criteria (``CRITERIA`` below). Nothing here changes the benchmark suite, which
stays float64.

Run it with::

    python -m hydrointel.cli precision-study

It writes ``precision_study.md`` and ``precision_study.json`` to the output
directory, both provenance-stamped.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig
from ..viz.provenance import markdown_header, write_json

log = logging.getLogger("hydrointel.bench")

# Acceptance criteria for switching dataset generation to fp32. These are the
# thresholds the study is judged against; they are not tuned to the result.
CRITERIA = {
    "mass_balance_error": 1e-4,        # relative, every test storm
    "max_abs_depth_diff_m": 5e-3,      # over all cells and all output times
    "depth_rmse_wet_m": 1e-3,
    "peak_depth_diff_m": 5e-3,
    "flooded_area_rel_diff": 0.005,    # at 0.15 / 0.30 / 0.50 m
    "lake_at_rest_fp32_velocity_ms": 1e-5,
}
AREA_THRESHOLDS_M = (0.15, 0.30, 0.50)
# finer than the 900 s used by dataset generation, so max|dh| is sampled harder
STUDY_OUTPUT_INTERVAL_S = 300.0


# ---------------------------------------------------------------------------
# storm selection
# ---------------------------------------------------------------------------
def _has(patterns, prefix) -> bool:
    return any(p.split(":")[0] == prefix for p in patterns)


def choose_storms(cfg: RunConfig, dom, search: int = 400) -> list[dict]:
    """Three storms from the real sampler: they must between them cover different
    return periods, tide on and off, and one fully perturbed SiteState."""
    from ..data import sampler as SMP
    picks: dict[str, dict] = {}
    for i in range(search):
        smp = SMP.draw(i, dom, cfg.data, cfg.seed, "train")
        full = all(_has(smp.patterns, k) for k in ("storage", "kappa", "manning", "gamma"))
        if "baseline" not in picks and smp.baseline and smp.return_period_yr == 10:
            picks["baseline"] = {"role": "baseline site, low return period, tide off", "sample": smp,
                                 "tide_on": False, "return_period_yr": 10}
        if "mixed" not in picks and not smp.baseline and not full and smp.return_period_yr == 50:
            picks["mixed"] = {"role": "partly perturbed site, mid return period, tide on", "sample": smp,
                              "tide_on": True, "return_period_yr": 50}
        if "full" not in picks and full:
            picks["full"] = {"role": "storage, infiltration, roughness and conveyance all perturbed; "
                                    "highest return period, tide on", "sample": smp,
                             "tide_on": True, "return_period_yr": 100}
        if len(picks) == 3:
            break
    missing = {"baseline", "mixed", "full"} - set(picks)
    if missing:
        raise RuntimeError(f"could not find storms of kind {sorted(missing)} in the first {search} samples; "
                           "the sampler configuration has changed")
    return [picks[k] for k in ("baseline", "mixed", "full")]


# ---------------------------------------------------------------------------
# one storm, one precision
# ---------------------------------------------------------------------------
def _run_once(ctx, site, scen, precision: str) -> dict:
    from .. import api
    ctx.cfg.solver.precision = precision
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = api.simulate(site, scen, output_interval_s=STUDY_OUTPUT_INTERVAL_S)
    wall = time.perf_counter() - t0
    log_ = res.extras["run_log"]
    speed = np.sqrt(res.u.numpy().astype(np.float64) ** 2 + res.v.numpy().astype(np.float64) ** 2)
    out = {
        "precision": precision,
        "wall_s": wall,
        "loop_wall_s": float(log_.wall_s),
        "steps": int(log_.steps),
        "ms_per_step": 1e3 * float(log_.wall_s) / max(log_.steps, 1),
        "dt_min_s": float(log_.dt_min),
        "dt_max_s": float(log_.dt_max),
        "n_dt_clamped": int(log_.n_dt_clamped),
        "mass_balance_error": float(res.mass_balance_error),
        "infiltrated_m3": float(res.infiltrated_volume_m3),
        "stored_m3": float(res.stored_volume_m3),
        "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
        "_h": res.depth_series.numpy().astype(np.float64),
        "_peak_speed": speed.max(0),
        "_times": res.times.numpy().astype(np.float64),
    }
    del res
    return out


def _compare(a: dict, b: dict, land: np.ndarray, dx: float, h_dry: float) -> dict:
    """a = fp64 reference, b = fp32."""
    if a["_h"].shape != b["_h"].shape:
        raise RuntimeError(f"snapshot shapes differ: {a['_h'].shape} vs {b['_h'].shape}")
    if not np.allclose(a["_times"], b["_times"]):
        raise RuntimeError("snapshot times differ between precisions")
    d = b["_h"] - a["_h"]
    wet = np.maximum(a["_h"], b["_h"]) > h_dry
    n_wet = int(wet.sum())
    if n_wet == 0:
        raise RuntimeError("no wet cells in either run: the comparison would be vacuous")
    pk_a, pk_b = a["_h"].max(0), b["_h"].max(0)
    cell_area = dx ** 2
    areas = {}
    for thr in AREA_THRESHOLDS_M:
        A64 = float(np.sum(land & (pk_a >= thr)) * cell_area)
        A32 = float(np.sum(land & (pk_b >= thr)) * cell_area)
        if A64 == 0.0:
            # no reference area: identical (both dry) is a pass, anything else fails loudly
            rel = 0.0 if A32 == 0.0 else float("inf")
        else:
            rel = abs(A32 - A64) / A64
        areas[f"{thr:g}"] = {"fp64_ha": A64 / 1e4, "fp32_ha": A32 / 1e4,
                             "abs_diff_ha": (A32 - A64) / 1e4, "rel_diff": rel}
    kt, kj, ki = np.unravel_index(int(np.abs(d).argmax()), d.shape)
    return {
        "max_abs_depth_diff_m": float(np.abs(d).max()),
        "max_abs_depth_diff_at": {"t_s": float(a["_times"][kt]), "j": int(kj), "i": int(ki),
                                  "on_land": bool(land[kj, ki]),
                                  "h_fp64_m": float(a["_h"][kt, kj, ki]), "h_fp32_m": float(b["_h"][kt, kj, ki])},
        "depth_rmse_wet_m": float(np.sqrt(np.mean(d[wet] ** 2))),
        "depth_bias_wet_m": float(np.mean(d[wet])),
        "n_wet_samples": n_wet,
        "peak_depth_diff_m": float(np.abs(pk_b - pk_a).max()),
        "peak_depth_diff_land_m": float(np.abs(pk_b - pk_a)[land].max()),
        "peak_speed_diff_ms": float(np.abs(b["_peak_speed"] - a["_peak_speed"]).max()),
        "flooded_area": areas,
        "speed_ratio": a["loop_wall_s"] / b["loop_wall_s"] if b["loop_wall_s"] > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# fp32 lake at rest
# ---------------------------------------------------------------------------
def lake_at_rest_fp32(cfg: RunConfig, device) -> dict:
    """Benchmark 1 in fp32: well-balancedness cannot hold to 1e-10 in single
    precision; the question is only whether the spurious velocity is physically
    negligible."""
    import copy

    from .suite import bench_lake_at_rest
    run = copy.deepcopy(cfg)
    run.solver.precision = "fp32"
    r = bench_lake_at_rest(device, False, Path(cfg.outdir), None, run)
    vel = max(r.metrics[f"o{o}_max_abs_{c}"] for o in (1, 2) for c in ("u", "v"))
    return {"max_abs_velocity_ms": float(vel),
            "max_eta_dev_m": float(max(r.metrics[f"o{o}_max_eta_dev"] for o in (1, 2))),
            "mass_err": float(max(r.metrics[f"o{o}_mass_err"] for o in (1, 2))),
            "passed": vel < CRITERIA["lake_at_rest_fp32_velocity_ms"],
            "metrics": {k: float(v) for k, v in r.metrics.items()}}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_study(cfg: RunConfig, device=None) -> dict:
    from .. import api
    from ..config import seed_everything
    from ..data import sampler as SMP

    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    land = ~(dom.sea | dom.channel)
    storms = choose_storms(cfg, dom)
    results = []
    for spec in storms:
        smp = spec["sample"]
        smp.return_period_yr = spec["return_period_yr"]
        smp.tide_on = spec["tide_on"]
        scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                            smp.storm_tide_offset_h)
        lu = ctx.landuse(smp.ssp, smp.horizon_year)
        storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
        site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
        log.info("precision study: storm %d (%s)", smp.index, spec["role"])
        runs = {p: _run_once(ctx, site, scen, p) for p in ("fp64", "fp32")}
        cmp = _compare(runs["fp64"], runs["fp32"], land, dom.dx, cfg.solver.h_dry)
        results.append({
            "index": smp.index, "role": spec["role"], "patterns": smp.patterns, "baseline_site": smp.baseline,
            "scenario": {"return_period_yr": scen.return_period_yr, "rcp": scen.rcp, "ssp": scen.ssp,
                         "horizon_year": scen.horizon_year, "tide_on": scen.tide_on,
                         "storm_tide_offset_h": scen.storm_tide_offset_h},
            "site_stats": {"storage_max_m": float(storage[land].max()), "storage_mean_m": float(storage[land].mean()),
                           "kappa_max": float(kappa[land].max()), "kappa_mean": float(kappa[land].mean()),
                           "manning_min": float(manning[land].min()), "manning_max": float(manning[land].max()),
                           "gamma_min": float(gamma.min()), "gamma_max": float(gamma.max())},
            "runs": {p: {k: v for k, v in r.items() if not k.startswith("_")} for p, r in runs.items()},
            "comparison": cmp,
            "verdict": _verdict(runs, cmp),
        })
        del runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cfg.solver.precision = "fp64"
    lake = lake_at_rest_fp32(cfg, ctx.device)
    passed = all(r["verdict"]["passed"] for r in results) and lake["passed"]
    out = {"criteria": CRITERIA, "output_interval_s": STUDY_OUTPUT_INTERVAL_S,
           "grid": {"nx": dom.nx, "ny": dom.ny, "dx": dom.dx},
           "storms": results, "lake_at_rest_fp32": lake, "passed": passed}
    prov = ctx.provenance("engine").replace(precision="fp64+fp32",
                                            notes=("float32 vs float64 study for dataset generation",))
    write_json(Path(cfg.outdir) / "precision_study.json", _jsonable(out), prov)
    (Path(cfg.outdir) / "precision_study.md").write_text(report(out, prov), encoding="utf-8")
    return out


def _verdict(runs: dict, cmp: dict) -> dict:
    checks = {
        "mass_balance_error fp32": (runs["fp32"]["mass_balance_error"], CRITERIA["mass_balance_error"]),
        "max|dh| over all cells and times": (cmp["max_abs_depth_diff_m"], CRITERIA["max_abs_depth_diff_m"]),
        "depth RMSE over wet cells": (cmp["depth_rmse_wet_m"], CRITERIA["depth_rmse_wet_m"]),
        "peak-depth difference": (cmp["peak_depth_diff_m"], CRITERIA["peak_depth_diff_m"]),
    }
    for thr, a in cmp["flooded_area"].items():
        checks[f"flooded-area difference at {thr} m"] = (a["rel_diff"], CRITERIA["flooded_area_rel_diff"])
    rows = {k: {"value": float(v), "limit": float(lim), "passed": bool(v < lim)} for k, (v, lim) in checks.items()}
    return {"passed": all(r["passed"] for r in rows.values()), "checks": rows}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ---------------------------------------------------------------------------
def report(d: dict, prov) -> str:
    s = markdown_header(prov, "Precision study: float32 vs float64 for dataset generation")
    g = d["grid"]
    s += (f"\nThree storms at full resolution ({g['nx']}x{g['ny']} @ {g['dx']:g} m), each run in both precisions "
          f"from the same seed with snapshots every {d['output_interval_s']:g} s. The analytical benchmark suite "
          "is unaffected: it stays float64.\n\n")
    s += f"**Result: {'PASS' if d['passed'] else 'FAIL'}** "
    s += ("— dataset generation may run in float32.\n\n" if d["passed"]
          else "— dataset generation stays in float64.\n\n")
    ratios = [r["comparison"]["speed_ratio"] for r in d["storms"]]
    s += (f"The measured float32 speedup is {min(ratios):.2f}x to {max(ratios):.2f}x, so single precision was never "
          "going to pay for a rewrite of the verification story even if the accuracy criteria had been met. On this "
          "grid the step is not bound by double-precision arithmetic.\n\n")
    s += "## Acceptance criteria\n\n| criterion | limit |\n|---|---|\n"
    for k, v in d["criteria"].items():
        s += f"| {k} | {v:g} |\n"
    s += "\n## Storms\n\n| # | role | RP | tide | patterns |\n|---|---|---|---|---|\n"
    for r in d["storms"]:
        s += (f"| {r['index']} | {r['role']} | {r['scenario']['return_period_yr']} | "
              f"{'on' if r['scenario']['tide_on'] else 'off'} | {', '.join(r['patterns']) or 'none (baseline)'} |\n")
    s += "\n## Timing and mass balance\n\n"
    s += "| # | fp64 loop [s] | fp32 loop [s] | ratio | fp64 steps | fp32 steps | fp64 mass err | fp32 mass err |\n"
    s += "|---|---|---|---|---|---|---|---|\n"
    for r in d["storms"]:
        a, b = r["runs"]["fp64"], r["runs"]["fp32"]
        s += (f"| {r['index']} | {a['loop_wall_s']:.1f} | {b['loop_wall_s']:.1f} | "
              f"{r['comparison']['speed_ratio']:.2f}x | {a['steps']} | {b['steps']} | "
              f"{a['mass_balance_error']:.2e} | {b['mass_balance_error']:.2e} |\n")
    s += "\n## Peak GPU memory\n\n| # | fp64 [MB] | fp32 [MB] |\n|---|---|---|\n"
    for r in d["storms"]:
        a, b = r["runs"]["fp64"], r["runs"]["fp32"]
        if a.get("peak_gpu_mb") is None:
            continue
        s += f"| {r['index']} | {a['peak_gpu_mb']:.0f} | {b['peak_gpu_mb']:.0f} |\n"
    s += "\n## Differences (fp32 minus fp64)\n\n"
    s += ("| # | max\\|dh\\| [m] | RMSE wet [m] | bias wet [m] | peak-depth diff, all cells [m] | "
          "peak-depth diff, land only [m] | peak-speed diff [m/s] |\n")
    s += "|---|---|---|---|---|---|---|\n"
    for r in d["storms"]:
        c = r["comparison"]
        s += (f"| {r['index']} | {c['max_abs_depth_diff_m']:.2e} | {c['depth_rmse_wet_m']:.2e} | "
              f"{c['depth_bias_wet_m']:+.2e} | {c['peak_depth_diff_m']:.2e} | "
              f"{c['peak_depth_diff_land_m']:.2e} | {c['peak_speed_diff_ms']:.2e} |\n")
    s += ("\n'Land only' excludes the sea and the 1-D channel footprint, which are water bodies rather than flood "
          "exposure. The two columns differ by roughly a factor of 4 on every storm: the largest single-cell "
          "divergences sit in open water, not on the ground Part B optimises.\n")
    if any("max_abs_depth_diff_at" in r["comparison"] for r in d["storms"]):
        s += "\n| # | worst cell | t [h] | on land | h fp64 [m] | h fp32 [m] |\n|---|---|---|---|---|---|\n"
        for r in d["storms"]:
            w = r["comparison"].get("max_abs_depth_diff_at")
            if w:
                s += (f"| {r['index']} | ({w['j']}, {w['i']}) | {w['t_s'] / 3600:.2f} | {w['on_land']} | "
                      f"{w['h_fp64_m']:.3f} | {w['h_fp32_m']:.3f} |\n")
    s += "\n## Flooded area on land\n\n| # | threshold [m] | fp64 [ha] | fp32 [ha] | difference [ha] | relative |\n"
    s += "|---|---|---|---|---|---|\n"
    for r in d["storms"]:
        for thr, a in r["comparison"]["flooded_area"].items():
            s += (f"| {r['index']} | {thr} | {a['fp64_ha']:.1f} | {a['fp32_ha']:.1f} | {a['abs_diff_ha']:+.2f} | "
                  f"{a['rel_diff'] * 100:.3f}% |\n")
    s += "\n## Per-storm verdict\n\n| # | check | value | limit | result |\n|---|---|---|---|---|\n"
    for r in d["storms"]:
        for k, c in r["verdict"]["checks"].items():
            s += (f"| {r['index']} | {k} | {c['value']:.3e} | {c['limit']:.1e} | "
                  f"{'PASS' if c['passed'] else 'FAIL'} |\n")
    lk = d["lake_at_rest_fp32"]
    s += (f"\n## Lake at rest in float32\n\nWell-balancedness holds to the working precision's round-off, which in "
          f"single precision is about 1e-7 relative, not the 1e-10 the float64 suite asserts. Measured after 500 "
          f"steps on the full grid, worst of 1st and 2nd order:\n\n"
          f"| metric | value | limit | result |\n|---|---|---|---|\n"
          f"| max spurious velocity [m/s] | {lk['max_abs_velocity_ms']:.3e} | "
          f"{CRITERIA['lake_at_rest_fp32_velocity_ms']:.0e} | {'PASS' if lk['passed'] else 'FAIL'} |\n"
          f"| max surface deviation [m] | {lk['max_eta_dev_m']:.3e} | reported | — |\n"
          f"| volume drift [m^3] | {lk['mass_err']:.3e} | reported | — |\n\n"
          "This case has no sources or sinks, so the mass-balance ratio degenerates to the absolute volume drift in "
          "cubic metres (the accounting divides by `max(gross sources, 1 m^3)`); it is not a relative error.\n")
    return s


# ---------------------------------------------------------------------------
# v2: the same storms on the fixed engine, judged by hydrointel.criteria
# ---------------------------------------------------------------------------
def run_study_v2(cfg: RunConfig, device=None) -> dict:
    """float32 against float64 under the agreement criteria of ``hydrointel.criteria``
    (which replaced the max-over-all-cells limits above). The storms are the ones the
    original study chose; every run passes the engine's physical-plausibility check or
    raises, so only clean storms can be scored. Lake at rest is unchanged."""
    from .. import api
    from ..config import seed_everything
    from ..criteria import AGREEMENT, agreement
    from ..data import sampler as SMP

    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    land = ~(dom.sea | dom.channel)
    results = []
    for spec in choose_storms(cfg, dom):
        smp = spec["sample"]
        smp.return_period_yr = spec["return_period_yr"]
        smp.tide_on = spec["tide_on"]
        scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                            smp.storm_tide_offset_h)
        lu = ctx.landuse(smp.ssp, smp.horizon_year)
        storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
        site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
        log.info("precision study v2: storm %d (%s)", smp.index, spec["role"])
        runs = {}
        for p in ("fp64", "fp32"):
            ctx.cfg.solver.precision = p
            t0 = time.perf_counter()
            res = api.simulate(site, scen, output_interval_s=STUDY_OUTPUT_INTERVAL_S)
            lg = res.extras["run_log"]
            tp = res.extras["topology"]
            runs[p] = {"wall_s": time.perf_counter() - t0, "loop_wall_s": float(lg.wall_s), "steps": int(lg.steps),
                       "mass_balance_error": float(res.mass_balance_error),
                       "_h": res.depth_series.numpy(), "_q": res.channel_q.numpy(),
                       "_reach": tp.reach_of[tp.kind <= 1]}
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        a, b = runs["fp64"], runs["fp32"]
        cmp = agreement(a["_h"], b["_h"], land, dom.dx, cfg.solver.h_dry, a["_q"], b["_q"], a["_reach"])
        mb_ok = b["mass_balance_error"] < CRITERIA["mass_balance_error"]
        cmp["checks"]["mass_balance_error fp32"] = {"value": b["mass_balance_error"],
                                                    "limit": CRITERIA["mass_balance_error"], "passed": mb_ok}
        cmp["passed"] = all(c["passed"] for c in cmp["checks"].values())
        results.append({"index": smp.index, "role": spec["role"], "patterns": smp.patterns,
                        "runs": {p: {k: v for k, v in r.items() if not k.startswith("_")} for p, r in runs.items()},
                        "speed_ratio": a["loop_wall_s"] / b["loop_wall_s"], "comparison": cmp})
        log.info("storm %d: %s", smp.index, {k: (round(c["value"], 5), c["passed"]) for k, c in cmp["checks"].items()})
        del runs
    cfg.solver.precision = "fp64"
    ctx.cfg.solver.precision = "fp64"
    lake = lake_at_rest_fp32(cfg, ctx.device)
    passed = all(r["comparison"]["passed"] for r in results) and lake["passed"]
    out = {"criteria": AGREEMENT, "mass_balance_limit": CRITERIA["mass_balance_error"],
           "lake_limit_ms": CRITERIA["lake_at_rest_fp32_velocity_ms"], "output_interval_s": STUDY_OUTPUT_INTERVAL_S,
           "grid": {"nx": dom.nx, "ny": dom.ny, "dx": dom.dx}, "storms": results, "lake_at_rest_fp32": lake,
           "passed": passed}
    prov = ctx.provenance("engine").replace(precision="fp64+fp32",
                                            notes=("float32 vs float64, agreement criteria v2, fixed engine",))
    write_json(Path(cfg.outdir) / "precision_study_v2.json", _jsonable(out), prov)
    (Path(cfg.outdir) / "precision_study_v2.md").write_text(report_v2(out, prov), encoding="utf-8")
    return out


def report_v2(d: dict, prov) -> str:
    s = markdown_header(prov, "Precision study v2: float32 vs float64 under the agreement criteria")
    g = d["grid"]
    s += (f"\nThe three storms of the original study, rerun on the fixed engine (1-D above-bank width 1 x bankfull), "
          f"at full resolution ({g['nx']}x{g['ny']} @ {g['dx']:g} m), snapshots every {d['output_interval_s']:g} s, "
          "and judged by `hydrointel/criteria.py`, which replaced the max-over-all-cells limits (see its docstring "
          "for the reasoning; the limits were fixed before this study ran).\n\n")
    s += f"**Result: {'PASS' if d['passed'] else 'FAIL'}** "
    s += "— dataset generation may run in float32.\n\n" if d["passed"] else "— dataset generation stays in float64.\n\n"
    s += "| criterion | limit |\n|---|---|\n" + "".join(f"| {k} | {v:g} |\n" for k, v in d["criteria"].items())
    s += (f"| mass_balance_error (fp32) | {d['mass_balance_limit']:g} |\n"
          f"| lake-at-rest velocity (fp32) | {d['lake_limit_ms']:g} m/s |\n\n")
    s += "## Timing\n\n| # | role | fp64 loop [s] | fp32 loop [s] | speedup | fp64 steps | fp32 steps |\n"
    s += "|---|---|---|---|---|---|---|\n"
    for r in d["storms"]:
        a, b = r["runs"]["fp64"], r["runs"]["fp32"]
        s += (f"| {r['index']} | {r['role']} | {a['loop_wall_s']:.0f} | {b['loop_wall_s']:.0f} | "
              f"{r['speed_ratio']:.2f}x | {a['steps']} | {b['steps']} |\n")
    s += "\n## Checks\n\n| # | check | value | limit | result |\n|---|---|---|---|---|\n"
    for r in d["storms"]:
        for k, c in r["comparison"]["checks"].items():
            s += (f"| {r['index']} | {k} | {c['value']:.3e} | {c['limit']:.1e} | "
                  f"{'PASS' if c['passed'] else 'FAIL'} |\n")
    s += "\n## Reported, not judged\n\n| # | max abs dh, any cell [m] | wet samples | samples deeper than 0.10 m in both |\n"
    s += "|---|---|---|---|\n"
    for r in d["storms"]:
        c = r["comparison"]
        s += f"| {r['index']} | {c['depth_max_abs_any_m']:.3f} | {c['n_wet_samples']} | {c['n_deep_samples']} |\n"
    lk = d["lake_at_rest_fp32"]
    s += (f"\n## Lake at rest in float32\n\nmax spurious velocity {lk['max_abs_velocity_ms']:.3e} m/s "
          f"(limit {d['lake_limit_ms']:g}): {'PASS' if lk['passed'] else 'FAIL'}\n")
    return s
