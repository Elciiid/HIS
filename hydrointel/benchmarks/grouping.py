"""Does batching pay if the members have similar time-step demand?

Task 2 found that a batch runs as many steps as its most restricted member needs:
one storm needing 92k steps dragged a batch of eight (typically ~42k each) to
90,682 steps and made batching 0.82x the speed of running the storms alone. This
study asks whether grouping storms by *estimated* step demand recovers the
per-step gain, and at what cost in accuracy.

Estimating demand before running
    A storm's step demand is estimated by rehearsing it on an 80 m grid: the same
    scenario, the same intervention fields block-averaged 4x4, the same channel
    conveyance, run start to end. The 80 m step count is the estimate. Whether it
    ranks storms correctly is measured against the true 20 m step counts of the
    storms that are also run alone here (and, for free, against the Phase 1
    dataset later, which runs each storm alone and records its steps).

The test
    The tightest-agreeing group of ``GROUP`` storms (by estimate) is run as one
    batch at full resolution over the whole storm, and each member is run alone.
    Reported: seconds per storm both ways; the true step counts; and the same
    accuracy measures the precision study used (max |d peak depth| over all cells
    and on land only, depth RMSE over wet cells, flooded-area difference at
    0.15/0.30/0.50 m) so the numbers are comparable with Tasks 1 and 2.

Decision rule, fixed before the measurement
    Grouped batching is used for generation only if it is faster than one storm at
    a time by more than 5% AND every member meets the 1 mm peak-depth requirement
    Task 2 set. Otherwise generation runs one storm at a time.

    python -m hydrointel.cli grouping-study
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig
from ..viz.provenance import markdown_header, write_json

log = logging.getLogger("hydrointel.bench")

N_CANDIDATES = 16
GROUP = 4
SPEEDUP_REQUIRED = 1.05
PEAK_DEPTH_REQUIREMENT_M = 1e-3
AREA_THRESHOLDS_M = (0.15, 0.30, 0.50)
REHEARSAL_CELL_M = 80.0
# 20 m step counts already measured for these sampler indices, run alone, full storm
# (Task 1 precision study and Task 2 full-length stage). Only used as extra
# ground truth for the estimator, never as a substitute for a run in this study.
KNOWN_STEPS_20M = {0: 42494, 1: 42200, 6: 92280, 20: 40230}


def _rehearsal_cfg(cfg: RunConfig) -> RunConfig:
    """The production config on an 80 m grid: same storm, same solver, coarser cells.
    Street geometry follows quick_config so the coarse city is resolvable."""
    c = copy.deepcopy(cfg)
    c.domain.cell_m = REHEARSAL_CELL_M
    c.domain.street_spacing_m = 320.0
    c.domain.street_width_m = 80.0
    c.domain.channel_node_spacing_m = 160.0
    return c


def _coarse(a: np.ndarray, f: int, shape) -> np.ndarray:
    ny, nx = shape
    a = a[:ny * f, :nx * f]
    return a.reshape(ny, f, nx, f).mean(axis=(1, 3))


def estimate_demand(cfg: RunConfig, dom20, members: list[dict], device) -> list[dict]:
    """80 m rehearsal step count for each member (the demand estimate)."""
    from ..domain.synthetic import generate
    from ..engine import build_engine, build_forcing, scenario_landuse
    from ..forcing.scenarios import ssp_state
    rc = _rehearsal_cfg(cfg)
    dom80 = generate(rc.domain, cfg.seed)
    f = int(round(REHEARSAL_CELL_M / dom20.dx))
    if dom80.network.n_reaches != dom20.network.n_reaches:
        raise RuntimeError(f"80 m network has {dom80.network.n_reaches} reaches, 20 m has "
                           f"{dom20.network.n_reaches}: gamma cannot be carried over")
    area = dom80.nx * dom80.ny * dom80.dx ** 2 / 1e6
    out = []
    for m in members:
        smp = m["smp"]
        fo = build_forcing(rc, smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                           smp.storm_tide_offset_h, area)
        lu = scenario_landuse(dom80, ssp_state(smp.ssp), smp.horizon_year)
        S, k, n = (_coarse(a, f, dom80.shape) for a in (m["S"], m["k"], m["n"]))
        water = dom80.sea | dom80.channel
        S = np.where(water, 0.0, S)
        k = np.where(water, 1.0, np.maximum(k, 1.0))
        n = np.where(water, dom80.manning(), n)
        eng = build_engine(rc, dom80, fo, S, k, n, m["g"], landuse=lu, device=device)
        t0 = time.perf_counter()
        eng.run(fo.t_end, label=f"rehearsal {m['i']}", progress_s=0)
        out.append({"i": m["i"], "est_steps_80m": int(eng.log.steps), "rehearsal_s": time.perf_counter() - t0,
                    "return_period_yr": smp.return_period_yr, "rcp": smp.rcp, "tide_on": smp.tide_on,
                    "rain_total_mm": float(m["fo"].rain.total_mm), "gamma_mean": float(np.mean(m["g"])),
                    "baseline": bool(smp.baseline)})
        log.info("rehearsal storm %d: %d steps at 80 m (%.0f s)", m["i"], out[-1]["est_steps_80m"],
                 out[-1]["rehearsal_s"])
        del eng
    return out


def pick_group(est: list[dict], size: int) -> list[int]:
    """The ``size`` storms whose estimated demand is closest together (smallest spread)."""
    order = sorted(est, key=lambda e: e["est_steps_80m"])
    best, best_spread = None, None
    for s in range(len(order) - size + 1):
        win = order[s:s + size]
        spread = win[-1]["est_steps_80m"] / max(win[0]["est_steps_80m"], 1)
        if best_spread is None or spread < best_spread:
            best, best_spread = win, spread
    return [e["i"] for e in best]


def _accuracy(pk_b: np.ndarray, pk_a: np.ndarray, h_b: np.ndarray, h_a: np.ndarray, land, dx, h_dry) -> dict:
    d = h_b - h_a
    wet = np.maximum(h_a, h_b) > h_dry
    areas = {}
    for thr in AREA_THRESHOLDS_M:
        A = float(np.sum(land & (pk_a >= thr))) * dx * dx
        Bv = float(np.sum(land & (pk_b >= thr))) * dx * dx
        areas[f"{thr:g}"] = 0.0 if A == Bv == 0 else (abs(Bv - A) / A if A > 0 else float("inf"))
    return {"max_peak_diff_m": float(np.abs(pk_b - pk_a).max()),
            "max_peak_diff_land_m": float(np.abs(pk_b - pk_a)[land].max()),
            "depth_rmse_wet_m": float(np.sqrt(np.mean(d[wet] ** 2))) if wet.any() else 0.0,
            "flooded_area_rel_diff": areas}


def _series(eng, t_end, out_times, label):
    snaps = []
    err = eng.run(t_end, out_times=out_times, on_snapshot=lambda s: snaps.append(np.asarray(s.h, np.float32)),
                  label=label)
    return np.stack(snaps), err


def run_study(cfg: RunConfig, device=None) -> dict:
    from .. import api
    from ..config import seed_everything
    from ..engine import build_engine, build_engine_batch
    from .throughput import _members, _warm_up
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom, dev = ctx.domain, ctx.device
    land = ~(dom.sea | dom.channel)
    members = _members(cfg, ctx, dom, list(range(N_CANDIDATES)))
    est = estimate_demand(cfg, dom, members, dev)
    group = pick_group(est, GROUP)
    log.info("chosen group (by estimated demand): %s", group)
    gm = [m for m in members if m["i"] in group]
    t_end = gm[0]["fo"].t_end
    out_times = gm[0]["fo"].times(cfg.solver.output_interval_s)

    _warm_up(cfg, dom, gm, dev)
    eng, _ = build_engine_batch(cfg, dom, [m["fo"] for m in gm], [m["S"] for m in gm], [m["k"] for m in gm],
                                [m["n"] for m in gm], [m["g"] for m in gm], [m["lu"] for m in gm], device=dev)
    t0 = time.perf_counter()
    hb, err_b = _series(eng, t_end, out_times, f"grouped batch of {len(gm)}")
    batch_wall, batch_steps = time.perf_counter() - t0, int(eng.log.steps)
    del eng
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    alone = []
    for j, m in enumerate(gm):
        e = build_engine(cfg, dom, m["fo"], m["S"], m["k"], m["n"], m["g"], landuse=m["lu"], device=dev)
        t0 = time.perf_counter()
        ha, err_a = _series(e, t_end, out_times, f"storm {m['i']} alone")
        acc = _accuracy(hb[:, j].max(0), ha.max(0), hb[:, j], ha, land, dom.dx, cfg.solver.h_dry)
        alone.append({"i": m["i"], "wall_s": time.perf_counter() - t0, "steps": int(e.log.steps),
                      "mass_error": float(err_a), **acc})
        log.info("storm %d alone: %.0f s, %d steps; max |d peak| %.1f mm (land %.1f mm)", m["i"],
                 alone[-1]["wall_s"], alone[-1]["steps"], acc["max_peak_diff_m"] * 1e3,
                 acc["max_peak_diff_land_m"] * 1e3)
        del e
    s_alone = float(np.mean([a["wall_s"] for a in alone]))
    s_batch = batch_wall / len(gm)
    speedup = s_alone / s_batch
    acc_ok = all(a["max_peak_diff_m"] < PEAK_DEPTH_REQUIREMENT_M for a in alone)
    use = speedup > SPEEDUP_REQUIRED and acc_ok

    truth = dict(KNOWN_STEPS_20M)
    truth.update({a["i"]: a["steps"] for a in alone})
    pairs = [(e["est_steps_80m"], truth[e["i"]]) for e in est if e["i"] in truth]
    out = {"estimates": est, "group": group, "batch": {"wall_s": batch_wall, "steps": batch_steps,
                                                       "s_per_storm": s_batch,
                                                       "max_mass_error": float(np.max(err_b))},
           "alone": alone, "s_per_storm_alone": s_alone, "speedup": speedup,
           "accuracy_requirement_m": PEAK_DEPTH_REQUIREMENT_M, "accuracy_met": acc_ok,
           "use_grouped_batching": use, "truth_steps_20m": truth, "estimate_vs_truth": pairs,
           "severity": _severity(est)}
    prov = ctx.provenance("engine").replace(notes=("grouped batching by estimated step demand",))
    write_json(Path(cfg.outdir) / "grouping_study.json", out, prov)
    (Path(cfg.outdir) / "grouping_study.md").write_text(report(out, prov), encoding="utf-8")
    return out


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _severity(est: list[dict]) -> dict:
    """Does estimated demand track storm severity? If it does, grouping by demand also
    groups by severity, and any error the batching introduces would correlate with it."""
    s = np.array([e["est_steps_80m"] for e in est], float)
    return {"spearman_demand_vs_return_period": _spearman(s, [e["return_period_yr"] for e in est]),
            "spearman_demand_vs_rain_total": _spearman(s, [e["rain_total_mm"] for e in est]),
            "spearman_demand_vs_gamma_mean": _spearman(s, [e["gamma_mean"] for e in est])}


def report(d: dict, prov) -> str:
    s = markdown_header(prov, "Grouped batching by estimated time-step demand")
    s += (f"\n**Decision: {'use grouped batching' if d['use_grouped_batching'] else 'generate one storm at a time'}.** "
          f"Grouped batch speedup {d['speedup']:.2f}x (needs > {SPEEDUP_REQUIRED:.2f}x); peak-depth requirement "
          f"{d['accuracy_requirement_m'] * 1e3:.0f} mm {'met' if d['accuracy_met'] else 'NOT met'}.\n\n")
    s += "## (a) Estimating step demand before running\n\n"
    s += ("Each storm is rehearsed start to end on an 80 m grid with its intervention fields block-averaged; the "
          "80 m step count is the estimate.\n\n| storm | RP | tide | rain [mm] | mean gamma | 80 m steps | "
          "20 m steps (truth) | rehearsal [s] |\n|---|---|---|---|---|---|---|---|\n")
    truth = {int(k): v for k, v in d["truth_steps_20m"].items()}
    for e in d["estimates"]:
        tr = truth.get(e["i"])
        s += (f"| {e['i']} | {e['return_period_yr']} | {'on' if e['tide_on'] else 'off'} | {e['rain_total_mm']:.0f} | "
              f"{e['gamma_mean']:.2f} | {e['est_steps_80m']} | {tr if tr else '—'} | {e['rehearsal_s']:.0f} |\n")
    pairs = d["estimate_vs_truth"]
    if len(pairs) >= 3:
        est_, tru = zip(*pairs)
        s += (f"\nAgainst the {len(pairs)} storms with a measured 20 m step count: Spearman rank correlation "
              f"{_spearman(est_, tru):.2f}. That is few storms to judge an estimator by; the Phase 1 dataset "
              "records the true step count of every storm it runs, and the final report re-checks the "
              "estimate against those.\n")
    s += "\n## Grouped batch against the same storms run alone\n\n"
    b = d["batch"]
    s += (f"Group {d['group']}: batch of {len(d['group'])} ran {b['steps']} steps in {b['wall_s']:.0f} s "
          f"({b['s_per_storm']:.0f} s per storm); alone, {d['s_per_storm_alone']:.0f} s per storm. "
          f"**Speedup {d['speedup']:.2f}x.** Worst batched mass error {b['max_mass_error']:.1e}.\n\n")
    s += ("| storm | steps alone | max \\|d peak\\| all cells [mm] | on land [mm] | RMSE wet [mm] | "
          "area diff 0.15/0.30/0.50 m |\n|---|---|---|---|---|---|\n")
    for a in d["alone"]:
        ar = a["flooded_area_rel_diff"]
        s += (f"| {a['i']} | {a['steps']} | {a['max_peak_diff_m'] * 1e3:.2f} | {a['max_peak_diff_land_m'] * 1e3:.2f} | "
              f"{a['depth_rmse_wet_m'] * 1e3:.3f} | "
              + " / ".join(f"{100 * ar[k]:.3f}%" for k in ("0.15", "0.3", "0.5")) + " |\n")
    s += ("\n## (b) Is grouped agreement better than ungrouped?\n\nUngrouped (Task 2, batch of 8 including a "
          "92k-step storm): max |d peak| over all cells 115.0 and 129.0 mm for storms 0 and 1. ")
    worst = max(a["max_peak_diff_m"] for a in d["alone"])
    s += (f"Grouped: worst {worst * 1e3:.2f} mm. " + (
        "Grouping agrees better, as expected: the shared step is closer to each member's own.\n"
        if worst < 0.115 else
        "**Suspicious:** grouping was expected to agree better than ungrouped batching, because the shared step "
        "is closer to each member's own, and it does not. Treat the batching result with caution.\n"))
    sv = d["severity"]
    s += ("\n## (c) Does grouping sort storms by severity?\n\nRank correlation of estimated demand with: return "
          f"period {sv['spearman_demand_vs_return_period']:.2f}, total rain {sv['spearman_demand_vs_rain_total']:.2f}, "
          f"mean channel gamma {sv['spearman_demand_vs_gamma_mean']:.2f}. A strong correlation means demand-grouped "
          "batches are also severity-homogeneous, so any error the shared step introduces would differ "
          "systematically between severe and mild storms -- a pattern the surrogate could learn as if it were "
          "physics. That risk only matters if grouped batching is used.\n")
    return s
