"""Support for running this project on a Kaggle GPU session (see kaggle/run.py).

Three jobs no other module wants:

* ``compare_benchmarks`` -- the same analytical suite run on two machines, compared metric
  by metric. This is the cross-check for the local machine's faulty RAM: if the numbers
  agree, the local measurements stand; if they differ beyond round-off, every local number
  in the project has to be re-derived.
* ``hardware_study`` -- seconds per storm and per training step, and peak VRAM, on whatever
  GPU the session was given, so the run plans can be sized from measurement.
* ``storage_check`` -- what the storage format costs: the model's own targets built from a
  reduced record against the same targets from a full-fidelity one.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from . import api
from .config import RunConfig, seed_everything

log = logging.getLogger("hydrointel.kaggle")

# Two machines running the same fp64 code do not agree bit for bit: different GPUs use
# different kernels and reduction orders. These separate round-off from a real difference.
ROUNDOFF_REL = 1e-6
MEANINGFUL_REL = 1e-3
TINY_ABS = 1e-12


# ---------------------------------------------------------------------------
# K0.3: does this machine reproduce the local machine's benchmark numbers?
# ---------------------------------------------------------------------------
def _flat_metrics(res: dict) -> dict:
    out = {}
    for r in res.get("results", []):
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{r['key']}.{k}"] = float(v)
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if isinstance(x, (int, float)) and not isinstance(x, bool):
                        out[f"{r['key']}.{k}[{i}]"] = float(x)
        out[f"{r['key']}.passed"] = float(bool(r.get("passed")))
    return out


def compare_benchmarks(local_path: Path, here_path: Path, info: dict | None = None) -> dict:
    """Compare two benchmark_results.json files metric by metric."""
    local_path, here_path = Path(local_path), Path(here_path)
    if not here_path.exists():
        raise FileNotFoundError(f"{here_path}: run the benchmark suite on this machine first")
    if not local_path.exists():
        raise FileNotFoundError(f"{local_path}: the local benchmark record is missing from the repo")
    L, H = json.loads(local_path.read_text(encoding="utf-8")), json.loads(here_path.read_text(encoding="utf-8"))
    lm, hm = _flat_metrics(L), _flat_metrics(H)
    rows = []
    for k in sorted(set(lm) | set(hm)):
        a, b = lm.get(k), hm.get(k)
        if a is None or b is None:
            rows.append({"metric": k, "local": a, "kaggle": b, "rel": None, "verdict": "missing on one side"})
            continue
        scale = max(abs(a), abs(b))
        if scale <= TINY_ABS:
            # e.g. the lake-at-rest velocities, ~1e-14: the method's own round-off floor, where
            # a relative change carries no information. Reported, not counted.
            rows.append({"metric": k, "local": a, "kaggle": b, "rel": None, "verdict": "both negligible"})
            continue
        rel = abs(b - a) / scale
        rows.append({"metric": k, "local": a, "kaggle": b, "rel": rel,
                     "verdict": "identical" if rel == 0.0 else
                                "round-off" if rel <= ROUNDOFF_REL else
                                "differs" if rel <= MEANINGFUL_REL else "MEANINGFUL DIFFERENCE"})
    bad = [r for r in rows if r["verdict"] in ("MEANINGFUL DIFFERENCE", "missing on one side")]
    mid = [r for r in rows if r["verdict"] == "differs"]
    pass_local = {r["key"]: bool(r["passed"]) for r in L.get("results", [])}
    pass_here = {r["key"]: bool(r["passed"]) for r in H.get("results", [])}
    flipped = sorted(k for k in set(pass_local) & set(pass_here) if pass_local[k] != pass_here[k])
    verdict = ("the local benchmark numbers are NOT reproduced on this machine" if bad or flipped else
               "this machine reproduces the local benchmark numbers to the precision they are reported at")
    md = ["# Benchmark cross-check: Kaggle against the local machine\n",
          "The local machine has confirmed faulty RAM, so every number it produced is suspect. The engine's "
          "analytical suite is deterministic in fp64, so running it here and comparing metric by metric says "
          "whether bit flips reached the results.\n",
          f"**Verdict: {verdict}.**\n",
          f"- metrics compared: {len(rows)}",
          f"- identical: {sum(r['verdict'] == 'identical' for r in rows)}",
          f"- both below the reporting floor ({TINY_ABS:g}), so uninformative: "
          f"{sum(r['verdict'] == 'both negligible' for r in rows)}",
          f"- round-off only (<= {ROUNDOFF_REL:g} relative): {sum(r['verdict'] == 'round-off' for r in rows)}",
          f"- differing between {ROUNDOFF_REL:g} and {MEANINGFUL_REL:g}: {len(mid)}",
          f"- **meaningful differences (> {MEANINGFUL_REL:g}) or missing: {len(bad)}**",
          f"- pass/fail flips: {flipped or 'none'}\n"]
    if info:
        md.append(f"Local: {L.get('provenance', {}).get('device')} / "
                  f"{L.get('provenance', {}).get('precision')}, commit "
                  f"{L.get('provenance', {}).get('git_commit')}. "
                  f"Here: {info.get('gpu')} / torch {info.get('torch')}, commit {info.get('repo_head', '')[:7]}.\n")
    md.append("| metric | local | Kaggle | relative difference | verdict |")
    md.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["rel"] or 0)):
        rel = "n/a" if r["rel"] is None else f"{r['rel']:.2e}"
        md.append(f"| {r['metric']} | {r['local']!r} | {r['kaggle']!r} | {rel} | {r['verdict']} |")
    md.append("")
    if bad or flipped:
        md.append("**Every measurement made on the local machine must be re-derived here.** The engine ran the "
                  "same code on both machines; a difference beyond round-off in a deterministic fp64 benchmark "
                  "means the local memory corrupted the computation.\n")
    else:
        md.append("The suite agrees, so the local dataset and model were probably not corrupted -- but that is an "
                  "argument about this suite's arrays, not a proof about every array in a 6-hour generation run. "
                  "The dataset is regenerated here regardless; the surrogate's failure to learn effects rests on "
                  "reasoning that this cross-check supports rather than replaces.\n")
    return {"verdict": verdict, "reproduced": not (bad or flipped), "rows": rows, "flipped": flipped,
            "markdown": "\n".join(md), "local_provenance": L.get("provenance"), "here_provenance": H.get("provenance")}


# ---------------------------------------------------------------------------
# K0.4: what does this GPU do with this code?
# ---------------------------------------------------------------------------
def time_one_storm(cfg: RunConfig, precision: str, device=None, sim_id: int = 0) -> dict:
    """One full storm at full resolution, timed. Uses the dataset sampler's storm ``sim_id``,
    so the work is representative of generation rather than a toy."""
    from .data import sampler as SMP
    import copy
    run = copy.deepcopy(cfg)
    run.solver.precision = precision
    ctx = api.configure(run, device)
    dom = ctx.domain
    smp = SMP.draw(sim_id, dom, run.data, run.seed, "train")
    lu = ctx.landuse(smp.ssp, smp.horizon_year)
    storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
    site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
    scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                        smp.storm_tide_offset_h)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = api.simulate(site, scen)
    wall = time.perf_counter() - t0
    lg = res.extras["run_log"]
    out = {"precision": precision, "sim": sim_id, "wall_s": wall, "loop_s": float(lg.wall_s), "steps": int(lg.steps),
           "ms_per_step": 1e3 * float(lg.wall_s) / max(int(lg.steps), 1),
           "mass_balance_error": float(res.mass_balance_error),
           "peak_depth_m": float(res.depth_max.max()),
           "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None}
    del res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("storm timing %s", out)
    return out


def hardware_study(cfg: RunConfig, device=None, storms: int = 1) -> dict:
    """K0.4: storm cost in both precisions, and the training step at each candidate resolution
    with and without gradient checkpointing, in fp16 and bf16."""
    from .diagnostics import measure_training_step
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dev = ctx.device
    info = {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
            if dev.type == "cuda" else None,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if dev.type == "cuda" else False,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    storm_rows = []
    for p in ("fp64", "fp32"):
        for k in range(storms):
            storm_rows.append(time_one_storm(cfg, p, device, sim_id=k))
    api.configure(cfg, device)                                    # back to the configured precision
    step_rows = []
    have_dataset = (api.dataset_dir(cfg) / "index.json").exists()
    if have_dataset:
        for factor in (2.0, 1.5, 1.0):
            for ck in (True, False) if factor == 2.0 else (True,):
                for dt in (torch.float16, torch.bfloat16):
                    step_rows.append(measure_training_step(cfg, factor, ck, dt, dev))
    else:
        log.warning("no dataset yet: training-step timings skipped (run the generate stage, then this one again)")
    med = lambda rows, p: float(np.median([r["loop_s"] for r in rows if r["precision"] == p])) if rows else None
    res = {"info": info, "storms": storm_rows, "training_steps": step_rows,
           "s_per_storm_fp64": med(storm_rows, "fp64"), "s_per_storm_fp32": med(storm_rows, "fp32"),
           "fp32_speedup": (med(storm_rows, "fp64") / med(storm_rows, "fp32"))
           if med(storm_rows, "fp32") else None,
           "dataset_present": have_dataset}
    ok = [r for r in step_rows if r["status"] == "ok"]
    if ok:
        best = min((r for r in ok if r["factor"] == 1.0), key=lambda r: r["s_per_step"], default=None)
        res["full_resolution_fits"] = best is not None
        res["full_resolution_row"] = best
    from .viz.provenance import write_json
    write_json(Path(cfg.outdir) / "kaggle_hardware.json", res, ctx.provenance("engine"))
    (Path(cfg.outdir) / "kaggle_hardware.md").write_text(hardware_report(res), encoding="utf-8")
    return res


def hardware_report(res: dict) -> str:
    i = res["info"]
    s = [f"# Hardware measurement: {i['gpu']}\n",
         f"{i['vram_gb']} GB VRAM, torch {i['torch']} (CUDA {i['torch_cuda']}), bf16 supported: "
         f"{i['bf16_supported']}.\n",
         "## One storm at 400 x 325, 20 m\n",
         "| precision | loop [s] | steps | ms/step | mass error | peak VRAM [MB] |",
         "|---|---|---|---|---|---|"]
    for r in res["storms"]:
        s.append(f"| {r['precision']} | {r['loop_s']:.0f} | {r['steps']} | {r['ms_per_step']:.2f} | "
                 f"{r['mass_balance_error']:.1e} | {(r['peak_gpu_mb'] or 0):.0f} |")
    if res.get("fp32_speedup"):
        s.append(f"\nfp32 is {res['fp32_speedup']:.2f}x the fp64 rate on this GPU. Dataset generation runs in "
                 "fp64 unless the precision study under the agreement criteria says otherwise.\n")
    if res["training_steps"]:
        s += ["## One training step\n",
              "| grid | cell [m] | nodes | checkpointing | autocast | status | s/step | peak VRAM [MB] |",
              "|---|---|---|---|---|---|---|---|"]
        for r in res["training_steps"]:
            head = (f"| {r['grid'][0]}x{r['grid'][1]} | {r['cell_m']:.0f} | {r['nodes']} | {r['checkpointing']} | "
                    f"{r['autocast']} | {r['status']} | ")
            if r["status"] == "ok":
                s.append(head + f"{r['s_per_step']:.2f} | {r['peak_allocated_mb']:.0f} |")
            else:
                s.append(head + f"- | {(r.get('peak_allocated_mb_before_oom') or 0):.0f} (at failure) |")
        if res.get("full_resolution_fits"):
            r = res["full_resolution_row"]
            s.append(f"\n**Full 20 m resolution fits**: {r['s_per_step']:.2f} s/step at "
                     f"{r['peak_allocated_mb']:.0f} MB. A5's workaround hunting and C4 are unnecessary; train at "
                     "full resolution and answer A1 by direct comparison instead of by the oracle argument.\n")
        elif res["training_steps"]:
            s.append("\n**Full 20 m resolution does not fit** on this GPU: the 2x coarse grid stands, and A1's "
                     "oracle argument remains the available evidence.\n")
    else:
        s.append("\n_Training-step timings need a dataset: run the generate stage first, then this stage again._\n")
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------------------
# K1 sizing: how many pairs fit in one session?
# ---------------------------------------------------------------------------
def pair_budget(s_per_storm: float, budget_hours: float, margin: float = 0.2,
                overhead_s: float = 600.0) -> dict:
    """How many (modified, baseline) pairs a session of ``budget_hours`` fits, keeping a
    ``margin`` fraction in reserve for the session ending early, plus a fixed overhead for
    the clone, the benchmark suite and writing the results out."""
    usable = budget_hours * 3600.0 * (1.0 - margin) - overhead_s
    per_pair = 2.0 * s_per_storm
    n = int(max(0, usable // per_pair))
    return {"s_per_storm": s_per_storm, "budget_hours": budget_hours, "margin": margin,
            "overhead_s": overhead_s, "usable_s": usable, "s_per_pair": per_pair, "pairs": n,
            "storms": 2 * n, "projected_hours": 2 * n * s_per_storm / 3600.0}


# ---------------------------------------------------------------------------
# K0.2 verification: what does the storage format cost?
# ---------------------------------------------------------------------------
def storage_check(cfg: RunConfig, device=None, sim_id: int | None = None) -> dict:
    """Run one storm once, write it at full fidelity and at the reduced settings, and compare
    what the model is actually trained on: the depth and velocity targets on the model grid,
    the peak-depth field, and the flooded areas the metrics are computed from."""
    import copy
    from .data import sampler as SMP
    from .data.generate import _write_sim, coarsen_mean, load_record, record_path, static_fields
    from .viz.provenance import write_json
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    sim_id = 0 if sim_id is None else sim_id
    smp = SMP.draw(sim_id, dom, cfg.data, cfg.seed, "train")
    lu = ctx.landuse(smp.ssp, smp.horizon_year)
    storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
    site = api.SiteState(*(torch.as_tensor(a, dtype=torch.float32) for a in (storage, kappa, manning, gamma)))
    scen = api.Scenario(smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                        smp.storm_tide_offset_h)
    t0 = time.perf_counter()
    res = api.simulate(site, scen)
    wall = time.perf_counter() - t0
    out = Path(cfg.outdir) / "storage_check"
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    # always compare the two real options, whichever the config picked: full fidelity (20 m
    # snapshots, which can train a model at any coarser grid) against the coarse fallback
    # (stored on the model grid, a quarter of the bytes, and unusable for a finer model).
    fallback = max(2, cfg.data.coarsen)
    for name, sf, compress in (("full_fidelity", 1, cfg.data.compress_records),
                               ("coarse_fallback", fallback, cfg.data.compress_records)):
        run = copy.deepcopy(cfg)
        run.data.store_coarsen = sf
        run.data.compress_records = compress
        idx: dict = {}
        p = record_path(out / name, sim_id, compress)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_sim(run, dom, sf, sim_id, smp, lu, storage, kappa, manning, gamma, res, p, idx, wall, wall, 1)
        torch.save(static_fields(dom, sf), p.parent / "domain.pt")
        written[name] = {"path": str(p), "bytes": p.stat().st_size, "store_coarsen": sf, "compressed": compress}
    # the model's own view of each record, on the model grid
    from .model.batch import SampleBuilder
    views, model_f = {}, max(cfg.data.coarsen, fallback)
    for name, w in written.items():
        rec = load_record(Path(w["path"]))
        b = SampleBuilder.__new__(SampleBuilder)
        b.k = model_f // w["store_coarsen"]
        b.store_f, b.f = w["store_coarsen"], model_f
        h, u, v = b.snapshots(rec)
        views[name] = {"h": h, "u": u, "v": v, "peak": np.asarray(rec["depth_max_full"])}
    a, c = views["full_fidelity"], views["coarse_fallback"]
    land = ~(dom.sea | dom.channel)
    land_c = coarsen_mean(land.astype(float), model_f) > 0.5
    wet = (np.maximum(a["h"], c["h"]) > 0.05) & land_c[None]
    cell_ha = (dom.dx * model_f) ** 2 / 1e4
    areas = {}
    for thr in (0.15, 0.30, 0.50):
        pa = float(np.sum((a["h"].max(0) >= thr) & land_c)) * cell_ha
        pc = float(np.sum((c["h"].max(0) >= thr) & land_c)) * cell_ha
        areas[f"{thr:g}"] = {"full_ha": pa, "reduced_ha": pc, "rel": abs(pc - pa) / max(pa, 1e-9)}
    d = c["h"] - a["h"]
    metrics = {"depth_rmse_wet_m": float(np.sqrt(np.mean(d[wet] ** 2))) if wet.any() else 0.0,
               "depth_max_abs_m": float(np.abs(d).max()),
               "speed_rmse_wet_ms": float(np.sqrt(np.mean((np.hypot(c["u"], c["v"]) - np.hypot(a["u"], a["v"]))[wet] ** 2)))
               if wet.any() else 0.0,
               "peak_field_max_abs_m": float(np.abs(c["peak"] - a["peak"]).max()),
               "flooded_area": areas, "snapshots": int(a["h"].shape[0]),
               "identical": bool(np.array_equal(a["h"], c["h"]) and np.array_equal(a["peak"], c["peak"]))}
    res_d = {"sim": sim_id, "model_coarsen": model_f, "configured_coarsen": cfg.data.coarsen,
             "configured_store_coarsen": cfg.store_factor, "fallback_store_coarsen": fallback,
             "fallback_can_train_at_20m": fallback == 1, "store_coarsen": cfg.store_factor,
             "compress_records": cfg.data.compress_records, "written": written, "metrics": metrics,
             "storm_wall_s": wall,
             "size_ratio": written["coarse_fallback"]["bytes"] / max(written["full_fidelity"]["bytes"], 1)}
    write_json(Path(cfg.outdir) / "storage_check.json", res_d, ctx.provenance("engine"))
    (Path(cfg.outdir) / "storage_check.md").write_text(storage_report(res_d), encoding="utf-8")
    log.info("storage check: %s", {k: v for k, v in metrics.items() if k != "flooded_area"})
    return res_d


def storage_report(d: dict) -> str:
    m = d["metrics"]
    w = d["written"]
    s = ["# What the storage format costs\n",
         f"One storm (sim {d['sim']}, {d['storm_wall_s']:.0f} s) written twice: at full fidelity "
         f"(store_coarsen=1, snapshots on the 20 m engine grid) and at the coarse fallback "
         f"(store_coarsen={d['fallback_store_coarsen']}), both with compression={d['compress_records']}. Both were "
         f"read back through the training input builder onto the {d['model_coarsen']}x model grid and compared on "
         f"what the model is trained on. This dataset is configured to store at "
         f"{d['configured_store_coarsen']}x and train at {d['configured_coarsen']}x.\n",
         f"- record size: {w['full_fidelity']['bytes'] / 2 ** 20:.1f} MB full fidelity, "
         f"{w['coarse_fallback']['bytes'] / 2 ** 20:.1f} MB at the fallback ({100 * d['size_ratio']:.0f}%)",
         f"- snapshots per storm: {m['snapshots']}",
         f"- depth RMSE over wet cells: {m['depth_rmse_wet_m']:.3e} m",
         f"- largest single-cell depth difference: {m['depth_max_abs_m']:.3e} m",
         f"- speed RMSE over wet cells: {m['speed_rmse_wet_ms']:.3e} m/s",
         f"- peak-depth field (stored at full fidelity in both): {m['peak_field_max_abs_m']:.3e} m",
         "- flooded area on land: " + ", ".join(
             f"{t} m {v['full_ha']:.1f} -> {v['reduced_ha']:.1f} ha ({100 * v['rel']:.3f}%)"
             for t, v in m["flooded_area"].items()) + "\n"]
    if m["identical"]:
        s.append(f"**On the {d['model_coarsen']}x model grid the two give the same arrays, exactly.** Block "
                 "averaging composes, so averaging a 20 m record onto the model grid reproduces what the generator "
                 "would have written had it stored there directly; the peak-depth field is kept at full fidelity "
                 "either way; and gzip is lossless. So the fallback's cost is not accuracy but capability: a record "
                 f"stored at {d['fallback_store_coarsen']}x **cannot train a model at 20 m at all**, which is the "
                 "comparison A1 needs. Full fidelity costs "
                 f"{1 / max(d['size_ratio'], 1e-9):.1f}x the bytes "
                 f"({w['full_fidelity']['bytes'] / 2 ** 20:.1f} MB per record), and the capacity guard keeps the "
                 "projected total under 15 GB before a single storm runs.\n")
    else:
        s.append("**The two differ** by the amounts above, which should not happen: block averaging composes and "
                 "compression is lossless, so a difference means a bug in the storage path or the input builder, "
                 "not a cost of the format. Judge the size against the intervention effect the system must resolve "
                 "(20-60 mm) and treat any depth RMSE near that as disqualifying.\n")
    return "\n".join(s) + "\n"
