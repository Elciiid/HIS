"""Re-cost the run from recorded measurements (Task 4).

Every input is read from a file a measurement wrote; nothing here is typed in.
The one exception is flagged where it is used: the training step time, which is
the full-size probe measured on 2026-09-17 before a dataset existed, and is
replaced by the rate Phase 1 actually achieves once training has run.

    python -m hydrointel.cli cost
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import RunConfig
from ..viz.provenance import DataProvenance, markdown_header

SIZES = (60, 150, 300, 400)
# training steps per dataset size: Phase 1 and Phase 2 as planned, then the spec's 40k
STEPS = {60: 10_000, 150: 20_000, 300: 30_000, 400: 40_000}
PROBE_TRAIN_S_PER_STEP = 0.95          # full-size probe, 2026-09-17 (see module docstring)
EVALUATIONS_PER_SCENARIO = (10_000, 30_000, 100_000)


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def project(cfg: RunConfig) -> dict:
    out = Path(cfg.outdir)
    bt = _read(out / "batch_throughput.json")
    gs = _read(out / "grouping_study.json")
    ps = _read(out / "predict_speed.json")
    if bt is None or bt.get("full_length") is None:
        raise FileNotFoundError(f"{out / 'batch_throughput.json'} with a full-length stage is required")
    solo = float(bt["full_length"]["s_per_storm_sequential"])
    gen_rate, gen_source = solo, "one storm at a time, measured full-length (batch_throughput.json)"
    if gs and gs.get("use_grouped_batching"):
        gen_rate = float(gs["batch"]["s_per_storm"])
        gen_source = "grouped batching, measured full-length (grouping_study.json)"
    train_rate, train_source = PROBE_TRAIN_S_PER_STEP, "full-size probe of 2026-09-17 (no dataset yet)"
    tr = _read(out / "training_speed.json")
    if tr:
        train_rate, train_source = float(tr["s_per_step"]), "measured in Phase 1 training (training_speed.json)"
    rows = [{"sims": n, "generation_h": n * gen_rate / 3600, "train_steps": STEPS[n],
             "training_h": STEPS[n] * train_rate / 3600} for n in SIZES]
    for r in rows:
        r["total_h"] = r["generation_h"] + r["training_h"]
    pred = None
    if ps:
        summ = [r for r in ps["rows"] if r["detail"] == "summary" and not r["skipped"]]
        best = min(summ, key=lambda r: r["s_per_candidate"])
        pred = {"s_per_candidate": best["s_per_candidate"], "batch": best["batch"], "weights": ps["weights"],
                "per_scenario_h": {str(n): n * best["s_per_candidate"] / 3600 for n in EVALUATIONS_PER_SCENARIO}}
    return {"generation_s_per_storm": gen_rate, "generation_source": gen_source,
            "train_s_per_step": train_rate, "train_source": train_source, "rows": rows, "prediction": pred}


def report(d: dict, prov: DataProvenance) -> str:
    s = markdown_header(prov, "Run cost from measured rates")
    s += (f"\nGeneration: **{d['generation_s_per_storm']:.0f} s per storm** ({d['generation_source']}). "
          f"Training: **{d['train_s_per_step']:.2f} s per step** ({d['train_source']}).\n\n")
    s += "| simulations | generation | training steps | training | total |\n|---|---|---|---|---|\n"
    for r in d["rows"]:
        s += (f"| {r['sims']} | {r['generation_h']:.1f} h | {r['train_steps']:,} | {r['training_h']:.1f} h | "
              f"**{r['total_h']:.1f} h** |\n")
    s += ("\nTraining steps per size follow the plan (Phase 1 ~10k, Phase 2 ~20k) and then the spec's 40k at 400; "
          "they are a plan, not a measured need. Evaluation adds roughly 1.5-2 h of engine runs per phase "
          "(effect pairs, probes, the engine storage check).\n\n")
    p = d["prediction"]
    if p:
        s += (f"**Part B cost per scenario** at the measured surrogate rate, {p['s_per_candidate']:.4f} s per "
              f"candidate (summary mode, batch {p['batch']}, {p['weights']} weights):\n\n"
              "| evaluations per scenario | wall time |\n|---|---|\n")
        for n, h in p["per_scenario_h"].items():
            s += f"| {int(n):,} | {h * 60:.0f} min |\n"
        ratio = d["generation_s_per_storm"] / p["s_per_candidate"]
        s += (f"\nThe surrogate is ~{ratio:.0f}x faster than the engine per evaluation, which is what makes Part B "
              f"possible at all: 10,000 engine evaluations would take "
              f"{10_000 * d['generation_s_per_storm'] / 86400:.0f} days. It is still decoder-bound -- batching "
              "buys ~1.2x and the summary mode ~7%, because every output time is decoded at every node -- so a "
              "100,000-evaluation search costs over half a day per scenario on this GPU. Part B's search budget "
              "(population x generations) has to be planned around that, or the surrogate needs a cheaper "
              "peak-only output path.\n")
    return s


def write(cfg: RunConfig) -> Path:
    d = project(cfg)
    prov = DataProvenance("SYNTHETIC", cfg.config_hash("domain", "solver", "forcing", "data"), cfg.seed,
                          engine="engine+surrogate", notes=("projection from recorded measurements",))
    out = Path(cfg.outdir)
    (out / "cost_projection.json").write_text(json.dumps({"provenance": prov.to_dict(), **d}, indent=1),
                                              encoding="utf-8")
    p = out / "cost_projection.md"
    p.write_text(report(d, prov), encoding="utf-8")
    return p
