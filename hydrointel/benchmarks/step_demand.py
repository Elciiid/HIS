"""What predicts a storm's time-step demand? (grouping follow-up)

The grouping study found that one storm in eight can need twice the steps of the
rest and drag a whole batch with it, and that an 80 m rehearsal cannot see it
coming (Spearman 0.09). Phase 1 runs every storm alone and records its true step
count, so this checks, against those counts, whether anything known *before* a
run -- return period, rainfall, tide, the intervention fields -- predicts demand.
If something does, grouped batching has a usable estimator; if not, it has none.

    python -m hydrointel.cli --config artifacts/phase1.json step-demand
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig
from ..viz.provenance import markdown_header


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(cfg: RunConfig) -> dict:
    from .. import api
    root = api.dataset_dir(cfg)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = []
    for k, meta in sorted(index.items(), key=lambda kv: int(kv[0])):
        rec = torch.load(root / f"sim_{int(k):05d}.pt", weights_only=False)
        rl = rec["run_log"]
        f = rec["fields"]
        g = np.asarray(f["gamma"])
        rows.append({"sim": int(k), "steps": int(rl["steps"]), "dt_min_s": float(rl["dt_min"]),
                     "n_sub_max": int(rl["n_sub_max"]), "wall_s": float(meta["wall_s"]),
                     "return_period_yr": meta["return_period_yr"], "tide_on": bool(meta["tide_on"]),
                     "rain_total_mm": float(np.asarray(rec["forcing"]["rain_mmh"]).mean()
                                            * float(rec["forcing"]["t_end"]) / 3600.0),
                     "gamma_mean": float(g.mean()), "gamma_max": float(g.max()),
                     "storage_mean_m": float(np.asarray(f["storage"]).mean()),
                     "kappa_mean": float(np.asarray(f["kappa"]).mean()),
                     "baseline": bool(meta["baseline"]), "peak_depth_land_m": float(meta["peak_depth_land"])})
    steps = np.array([r["steps"] for r in rows])
    med = float(np.median(steps))
    predictors = ("return_period_yr", "rain_total_mm", "gamma_mean", "gamma_max", "storage_mean_m", "kappa_mean",
                  "tide_on", "peak_depth_land_m")
    corr = {p: _spearman([r[p] for r in rows], steps) for p in predictors}
    outliers = [r for r in rows if r["steps"] > 1.5 * med]
    est = {}
    gp = Path(cfg.outdir) / "grouping_study.json"
    if gp.exists():
        g = json.loads(gp.read_text(encoding="utf-8"))
        e80 = {e["i"]: e["est_steps_80m"] for e in g["estimates"]}
        pairs = [(e80[r["sim"]], r["steps"]) for r in rows if r["sim"] in e80]
        if len(pairs) >= 3:
            est = {"n": len(pairs), "spearman_80m_vs_true": _spearman(*zip(*pairs))}
    return {"n": len(rows), "median_steps": med, "rows": rows, "spearman_vs_steps": corr,
            "outliers": [r["sim"] for r in outliers], "rehearsal_estimator": est}


def write(cfg: RunConfig) -> Path:
    from .. import api
    d = analyse(cfg)
    ctx = api.configure(cfg)
    prov = ctx.provenance("engine", ("domain", "solver", "forcing", "data")).replace(
        notes=("time-step demand against pre-run predictors",))
    s = markdown_header(prov, "What predicts a storm's time-step demand?")
    s += (f"\n{d['n']} storms, each run alone at full resolution. Median {d['median_steps']:.0f} steps. "
          f"Outliers (> 1.5x median): {d['outliers'] or 'none'}.\n\n")
    s += "| predictor (known before the run) | Spearman with true step count |\n|---|---|\n"
    for k, v in sorted(d["spearman_vs_steps"].items(), key=lambda kv: -abs(kv[1]) if kv[1] == kv[1] else 0):
        s += f"| {k} | {v:+.2f} |\n"
    if d["rehearsal_estimator"]:
        e = d["rehearsal_estimator"]
        s += f"\n80 m rehearsal estimate against the true count, {e['n']} storms: Spearman {e['spearman_80m_vs_true']:+.2f}.\n"
    s += "\n| sim | steps | min dt [s] | RP | tide | mean gamma | max gamma | rain [mm] | peak land depth [m] |\n"
    s += "|---|---|---|---|---|---|---|---|---|\n"
    for r in sorted(d["rows"], key=lambda r: -r["steps"])[:15]:
        s += (f"| {r['sim']} | {r['steps']} | {r['dt_min_s']:.3f} | {r['return_period_yr']} | "
              f"{'on' if r['tide_on'] else 'off'} | {r['gamma_mean']:.2f} | {r['gamma_max']:.2f} | "
              f"{r['rain_total_mm']:.0f} | {r['peak_depth_land_m']:.2f} |\n")
    s += "\n(The 15 storms with the most steps.)\n"
    p = Path(cfg.outdir) / "step_demand.md"
    p.write_text(s, encoding="utf-8")
    (Path(cfg.outdir) / "step_demand.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    return p
