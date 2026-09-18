"""How many storms should advance together?

The precision study measured a single full-resolution storm using 50-67 MB of a
6 GB card at ~6.6 ms per step. That is a card left idle: the grid is too small
to fill it, so the step is paced by kernel-launch overhead rather than by work.
Batching B storms into (B, ny, nx) tensors amortises that overhead.

Two stages, because they answer different questions:

* ``scaling`` runs a fixed set of storms over a short window (spin-up plus one
  hour) at each batch size, so every B does identical work and the wall times
  are directly comparable. It gives the speedup curve and the memory curve.
* ``full`` runs one batch over the whole storm at the chosen B. Only this stage
  measures the price of the shared time step honestly: over a short window the
  members' CFL limits have not yet diverged, so ``scaling`` flatters the batch.

Run it with::

    python -m hydrointel.cli batch-study
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig
from ..viz.provenance import markdown_header, write_json

log = logging.getLogger("hydrointel.bench")

BATCHES = (1, 2, 4, 8, 16)
N_STORMS = 16
WINDOW_H = 1.0            # simulated hours past the start of the run, for the scaling stage
MEMORY_HEADROOM = 0.5     # a batch may not claim more than this fraction of the card
N_SEQUENTIAL = 4          # storms also run alone at full length, for the accuracy comparison
# The shared time step may not move a peak depth by more than the solver's own dry
# threshold. Applied at the production resolution, which is where the dataset is made.
PEAK_DEPTH_BOUND_M = 1e-3


def _members(cfg: RunConfig, ctx, dom, idx: list[int]):
    """Build the per-storm inputs for sampler indices ``idx``."""
    from ..data import sampler as SMP
    from ..engine import build_forcing
    area = dom.nx * dom.ny * dom.dx ** 2 / 1e6
    out = []
    for i in idx:
        smp = SMP.draw(i, dom, cfg.data, cfg.seed, "train")
        lu = ctx.landuse(smp.ssp, smp.horizon_year)
        storage, kappa, manning, gamma, _, _ = SMP.materialise(smp, dom, lu)
        fo = build_forcing(cfg, smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                           smp.storm_tide_offset_h, area)
        out.append({"i": i, "smp": smp, "lu": lu, "fo": fo, "S": storage, "k": kappa, "n": manning, "g": gamma})
    return out


_WARMED: set[int] = set()


def _warm_up(cfg, dom, mem: list[dict], device) -> None:
    """Compile the step for this batch shape before anything is timed.

    Each batch size is a new shape, so torch.compile fires once per size and would
    otherwise land inside the first chunk's measured loop and dominate it.
    """
    from ..engine import build_engine_batch
    b = len(mem)
    if b in _WARMED or torch.device(device).type != "cuda":
        _WARMED.add(b)
        return
    t0 = time.perf_counter()
    eng, bf = build_engine_batch(cfg, dom, [m["fo"] for m in mem], [m["S"] for m in mem], [m["k"] for m in mem],
                                 [m["n"] for m in mem], [m["g"] for m in mem], [m["lu"] for m in mem], device=device)
    t = 0.0
    for _ in range(3):
        t = eng.step(t, bf.t_end)
    del eng
    torch.cuda.empty_cache()
    _WARMED.add(b)
    log.info("warmed up batch %d in %.0f s (compilation, excluded from the timings)", b, time.perf_counter() - t0)


def _run_chunk(cfg, dom, mem: list[dict], t_end: float, device) -> dict:
    from ..engine import build_engine_batch
    _warm_up(cfg, dom, mem, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    eng, bf = build_engine_batch(cfg, dom, [m["fo"] for m in mem], [m["S"] for m in mem], [m["k"] for m in mem],
                                 [m["n"] for m in mem], [m["g"] for m in mem], [m["lu"] for m in mem], device=device)
    t0 = time.perf_counter()
    err = eng.run(t_end, label=f"batch of {len(mem)}")
    wall = time.perf_counter() - t0
    res = {"n": len(mem), "wall_s": wall, "loop_wall_s": float(eng.log.wall_s), "steps": int(eng.log.steps),
           "dt_min_s": float(eng.log.dt_min), "dt_max_s": float(eng.log.dt_max),
           "n_dt_clamped": int(eng.log.n_dt_clamped), "n_sub_max": int(eng.log.n_sub_max),
           "max_mass_error": float(np.max(err)),
           "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None}
    del eng
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def scaling(cfg: RunConfig, ctx, dom, device, batches=BATCHES, n_storms=N_STORMS, window_h=WINDOW_H) -> list[dict]:
    """The same ``n_storms`` storms over the same short window at every batch size."""
    cached = _cache_read(cfg, n_storms, window_h)
    if cached and {r["batch"] for r in cached} >= {b for b in batches if b <= n_storms}:
        return cached
    mem = _members(cfg, ctx, dom, list(range(n_storms)))
    t_end = mem[0]["fo"].storm_start + window_h * 3600.0
    rows = []
    for b in batches:
        if b > n_storms:
            continue
        chunks = [mem[c:c + b] for c in range(0, n_storms, b)]
        runs = [_run_chunk(cfg, dom, c, t_end, device) for c in chunks]
        total = sum(r["loop_wall_s"] for r in runs)
        rows.append({
            "batch": b, "n_storms": n_storms, "n_chunks": len(runs), "window_h": window_h,
            "total_loop_s": total, "s_per_storm": total / n_storms,
            "steps_per_chunk": [r["steps"] for r in runs],
            "peak_gpu_mb": max((r["peak_gpu_mb"] or 0.0) for r in runs) if runs[0]["peak_gpu_mb"] else None,
            "max_mass_error": max(r["max_mass_error"] for r in runs),
        })
        log.info("batch %d: %.1f s for %d storms (%.1f s/storm), peak %s MB", b, total, n_storms,
                 total / n_storms, rows[-1]["peak_gpu_mb"])
        _cache_write(cfg, {"window_h": window_h, "n_storms": n_storms, "rows": rows})
    return rows


def _measured(cfg, key: str):
    """A previously measured stage, if one is on record for exactly these settings."""
    p = Path(cfg.outdir) / "batch_measured.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if key in d:
        log.info("reusing the %s stage measured earlier (%s)", key, p)
        return d[key]
    return None


def _record(cfg, key: str, value) -> None:
    p = Path(cfg.outdir) / "batch_measured.json"
    d = {}
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            d = {}
    d[key] = value
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _cache_path(cfg) -> Path:
    return Path(cfg.outdir) / "batch_scaling_cache.json"


def _cache_write(cfg, obj) -> None:
    """Persist the scaling rows as they are measured: the stage costs tens of minutes
    and an interrupted run should not throw away what it already measured."""
    _cache_path(cfg).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _cache_read(cfg, n_storms: int, window_h: float):
    p = _cache_path(cfg)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if d.get("n_storms") != n_storms or d.get("window_h") != window_h:
        log.warning("%s was measured for %s storms over %s h, not %s/%s; re-measuring",
                    p, d.get("n_storms"), d.get("window_h"), n_storms, window_h)
        return None
    log.info("reusing %d scaling rows measured earlier (%s)", len(d["rows"]), p)
    return d["rows"]


def profile_steps(cfg: RunConfig, ctx, dom, device, batches=(1, 8), n_steps=400, warm=60) -> list[dict]:
    """Where a coupled step's time goes, at each batch size.

    Synchronised around each call, so GPU work is attributed to the call that issued
    it. This is the measurement that says which part of the step batching can
    amortise and which it cannot.
    """
    from ..engine import build_engine_batch
    from ..solver.coupling import exchange
    sync = (lambda: torch.cuda.synchronize()) if torch.device(device).type == "cuda" else (lambda: None)
    out = []
    for b in batches:
        mem = _members(cfg, ctx, dom, list(range(b)))
        _warm_up(cfg, dom, mem, device)
        eng, bf = build_engine_batch(cfg, dom, [m["fo"] for m in mem], [m["S"] for m in mem], [m["k"] for m in mem],
                                     [m["n"] for m in mem], [m["g"] for m in mem], [m["lu"] for m in mem],
                                     device=device)
        acc = dict(dt_2d=0.0, dt_1d=0.0, exchange=0.0, step_2d=0.0, step_1d=0.0, accounting=0.0)
        t = 0.0
        for k in range(warm + n_steps):
            rec = k >= warm
            if rec:
                sync(); t0 = time.perf_counter()
            dt = eng.two.max_dt()
            if rec:
                sync(); t1 = time.perf_counter(); acc["dt_2d"] += t1 - t0
            n_sub = max(1, int(np.ceil(dt / eng.one.max_dt() - 1e-9)))
            if rec:
                t2 = time.perf_counter(); acc["dt_1d"] += t2 - t1
            exchange(eng.one, eng.two, eng.links, dt, eng.cfg)
            if rec:
                sync(); t3 = time.perf_counter(); acc["exchange"] += t3 - t2
            vol = eng.two.step(dt, t, eng.rain(t))
            if rec:
                sync(); t4 = time.perf_counter(); acc["step_2d"] += t4 - t3
            for j in range(n_sub):
                eng.one.step(dt / n_sub, t + j * dt / n_sub)
            if rec:
                t5 = time.perf_counter(); acc["step_1d"] += t5 - t4
                eng.mb.add("rain", vol.rain)
                sync(); acc["accounting"] += time.perf_counter() - t5
            t += dt
        row = {"batch": b, **{k: 1e3 * v / n_steps for k, v in acc.items()}}
        row["total"] = sum(v for k, v in row.items() if k != "batch")
        row["per_storm"] = row["total"] / b
        out.append(row)
        log.info("profile B=%d: %.2f ms/step (%.2f per storm-step); 2-D %.2f, 1-D %.2f, exchange %.2f",
                 b, row["total"], row["per_storm"], row["step_2d"], row["step_1d"], row["exchange"])
        del eng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _peak_depths(eng, t_end: float, out_times, label: str):
    """Run to the end, keeping the running maximum depth at the output times."""
    peak = [None]

    def keep(s):
        h = np.asarray(s.h, dtype=np.float64)
        peak[0] = h if peak[0] is None else np.maximum(peak[0], h)
    err = eng.run(t_end, out_times=out_times, on_snapshot=keep, label=label)
    return peak[0], err


def full_length(cfg: RunConfig, ctx, dom, device, batch: int, n_storms: int, n_sequential: int,
                t_end_h: float | None = None) -> dict:
    """Full storms at the chosen batch size, plus the same storms one at a time.

    This is the stage that both prices the run and measures the accuracy cost of the
    shared time step, at the resolution the dataset is actually generated at. Over a
    short window the members' CFL limits have barely diverged; only a whole storm,
    through the peak, shows what sharing a step is worth.
    """
    from ..engine import build_engine, build_engine_batch
    mem = _members(cfg, ctx, dom, list(range(n_storms)))
    t_end = mem[0]["fo"].t_end if t_end_h is None else min(t_end_h * 3600.0, mem[0]["fo"].t_end)
    out_times = [t for t in mem[0]["fo"].times(cfg.solver.output_interval_s) if t <= t_end + 1e-9]
    runs, peaks = [], {}
    for c in range(0, n_storms, batch):
        chunk = mem[c:c + batch]
        _warm_up(cfg, dom, chunk, device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        eng, bf = build_engine_batch(cfg, dom, [m["fo"] for m in chunk], [m["S"] for m in chunk],
                                     [m["k"] for m in chunk], [m["n"] for m in chunk], [m["g"] for m in chunk],
                                     [m["lu"] for m in chunk], device=device)
        t0 = time.perf_counter()
        pk, err = _peak_depths(eng, t_end, out_times, f"batch of {len(chunk)}")
        runs.append({"n": len(chunk), "loop_wall_s": time.perf_counter() - t0, "steps": int(eng.log.steps),
                     "max_mass_error": float(np.max(err)),
                     "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2 ** 20)
                     if torch.cuda.is_available() else None})
        for j, m in enumerate(chunk):
            peaks[m["i"]] = pk[j]
        del eng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("full-length batch of %d: %.0f s, %d steps", len(chunk), runs[-1]["loop_wall_s"], runs[-1]["steps"])
    total = sum(r["loop_wall_s"] for r in runs)

    seq = []
    for m in mem[:n_sequential]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        eng = build_engine(cfg, dom, m["fo"], m["S"], m["k"], m["n"], m["g"], landuse=m["lu"], device=device)
        t0 = time.perf_counter()
        pk, err = _peak_depths(eng, t_end, out_times, f"storm {m['i']}")
        d = float(np.abs(peaks[m["i"]] - pk).max())
        seq.append({"i": m["i"], "loop_wall_s": time.perf_counter() - t0, "steps": int(eng.log.steps),
                    "mass_error": float(err), "peak_depth_diff_m": d,
                    "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2 ** 20)
                    if torch.cuda.is_available() else None})
        log.info("storm %d alone: %.0f s, %d steps, peak-depth difference vs the batch %.4f mm",
                 m["i"], seq[-1]["loop_wall_s"], seq[-1]["steps"], d * 1e3)
        del eng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    s_seq = sum(r["loop_wall_s"] for r in seq) / len(seq) if seq else None
    return {"batch": batch, "n_storms": n_storms, "t_end_s": t_end, "runs": runs,
            "total_loop_s": total, "s_per_storm": total / n_storms,
            "peak_gpu_mb": max((r["peak_gpu_mb"] or 0.0) for r in runs) if runs[0]["peak_gpu_mb"] else None,
            "max_mass_error": max(r["max_mass_error"] for r in runs),
            "sequential": seq, "s_per_storm_sequential": s_seq,
            "speedup": (s_seq / (total / n_storms)) if s_seq else None,
            "worst_peak_depth_diff_m": max((r["peak_depth_diff_m"] for r in seq), default=None),
            "peak_depth_bound_m": PEAK_DEPTH_BOUND_M,
            "shared_timestep_within_bound": all(r["peak_depth_diff_m"] < PEAK_DEPTH_BOUND_M for r in seq)}


def choose_batch(rows: list[dict], budget_mb: float | None) -> int:
    """Largest batch that both pays off and leaves the card room to breathe."""
    ok = [r for r in rows if budget_mb is None or (r["peak_gpu_mb"] or 0.0) <= budget_mb]
    if not ok:
        raise RuntimeError(f"no batch size fits in {budget_mb:.0f} MB")
    best = min(ok, key=lambda r: r["s_per_storm"])
    # prefer a smaller batch when a larger one buys less than 5%
    for r in sorted(ok, key=lambda r: r["batch"]):
        if r["s_per_storm"] <= best["s_per_storm"] * 1.05:
            return r["batch"]
    return best["batch"]


def run_study(cfg: RunConfig, device=None, batches=BATCHES, n_storms=N_STORMS, window_h=WINDOW_H,
              n_sequential=N_SEQUENTIAL, batch=None, full_storms=None, full_t_end_h=None,
              stage: str = "all") -> dict:
    from .. import api
    from ..config import seed_everything
    if cfg.quick:                       # plumbing check only, not a throughput result
        batches, n_storms, window_h, n_sequential = (1, 2), 2, 0.15, 1
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    dev = ctx.device
    total_mb = (torch.cuda.get_device_properties(dev).total_memory / 2 ** 20) if dev.type == "cuda" else None
    budget = total_mb * MEMORY_HEADROOM if total_mb else None
    rows = scaling(cfg, ctx, dom, dev, batches, n_storms, window_h) if stage in ("all", "scaling") else []
    if stage == "full" and not batch:
        raise ValueError("stage 'full' needs an explicit --batch: it does not measure the scaling curve")
    chosen = int(batch) if batch else choose_batch(rows, budget)
    log.info("chosen batch size %d", chosen)
    key = f"full_b{chosen}_n{int(full_storms or chosen)}_s{n_sequential}_t{full_t_end_h}"
    full = None if stage == "scaling" else _measured(cfg, key) or full_length(cfg, ctx, dom, dev, chosen, int(full_storms or chosen), n_sequential, full_t_end_h)
    if full is not None:
        _record(cfg, key, full)
    prof = _measured(cfg, "profile")
    if prof is None and stage != "scaling":
        prof = profile_steps(cfg, ctx, dom, dev)
        _record(cfg, "profile", prof)
    single = next((r for r in rows if r["batch"] == 1), None)
    if full is None:
        out = {"grid": {"nx": dom.nx, "ny": dom.ny, "dx": dom.dx}, "device": str(dev), "total_gpu_mb": total_mb,
               "memory_budget_mb": budget, "scaling": rows, "chosen_batch": chosen, "stage": stage,
               "profile": prof}
        prov = ctx.provenance("engine").replace(notes=(f"batched throughput: {stage} stage only",))
        write_json(Path(cfg.outdir) / "batch_throughput.json", out, prov)
        (Path(cfg.outdir) / "batch_throughput.md").write_text(report(out, prov), encoding="utf-8")
        return out
    out = {"grid": {"nx": dom.nx, "ny": dom.ny, "dx": dom.dx},
           "device": str(dev), "total_gpu_mb": total_mb, "memory_budget_mb": budget,
           "stage": stage, "scaling": rows, "chosen_batch": chosen, "full_length": full, "profile": prof,
           "speedup_short_window": (single["s_per_storm"] / next(r["s_per_storm"] for r in rows
                                                                 if r["batch"] == chosen)) if single else None,
           "projection": {str(n): full["s_per_storm"] * n / 3600.0 for n in (60, 150, 300, 400)},
           "projection_sequential": {str(n): full["s_per_storm_sequential"] * n / 3600.0
                                     for n in (60, 150, 300, 400)} if full["s_per_storm_sequential"] else None}
    prov = ctx.provenance("engine").replace(notes=("batched dataset-generation throughput study",))
    write_json(Path(cfg.outdir) / "batch_throughput.json", out, prov)
    (Path(cfg.outdir) / "batch_throughput.md").write_text(report(out, prov), encoding="utf-8")
    return out


def report(d: dict, prov) -> str:
    g = d["grid"]
    s = markdown_header(prov, "Batched dataset generation: measured throughput")
    s += (f"\nGrid {g['nx']}x{g['ny']} @ {g['dx']:g} m on {d['device']}"
          + (f" ({d['total_gpu_mb']:.0f} MB, budget {d['memory_budget_mb']:.0f} MB)\n\n" if d["total_gpu_mb"]
             else "\n\n"))
    if d.get("scaling"):
        r0 = d["scaling"][0]
        s += (f"**Scaling stage.** The same {r0['n_storms']} storms, each advanced over the same "
              f"{r0['window_h']:g} h window, at every batch size. Identical work, so the wall times compare "
              "directly. This stage measures per-step efficiency and memory only: it stops before the storm "
              "peak, so it cannot see what the shared time step costs.\n\n")
        s += "| batch | chunks | total loop [s] | s per storm | speedup | peak GPU [MB] | max mass error |\n"
        s += "|---|---|---|---|---|---|---|\n"
        base = r0["s_per_storm"]
        for r in d["scaling"]:
            mem = f"{r['peak_gpu_mb']:.0f}" if r["peak_gpu_mb"] else "n/a"
            s += (f"| {r['batch']} | {r['n_chunks']} | {r['total_loop_s']:.1f} | {r['s_per_storm']:.1f} | "
                  f"{base / r['s_per_storm']:.2f}x | {mem} | {r['max_mass_error']:.1e} |\n")
    if d.get("profile"):
        p0, pn = d["profile"][0], d["profile"][-1]
        s += ("\n**Step breakdown**, synchronised so GPU work is charged to the call that issued it "
              "(ms per step). The last column is the cost per storm-step at the largest batch relative to "
              "the smallest: below 1 means batching amortised that part.\n\n| part | "
              + " | ".join(f"B = {p['batch']}" for p in d["profile"]) + " | per-storm |\n"
              + "|---" * (len(d["profile"]) + 2) + "|\n")
        for k in sorted((k for k in p0 if k not in ("batch", "per_storm")), key=lambda k: -pn[k]):
            gain = (pn[k] / pn["batch"]) / max(p0[k] / p0["batch"], 1e-12)
            s += f"| {k} | " + " | ".join(f"{p[k]:.2f}" for p in d["profile"]) + f" | {gain:.2f}x |\n"
    if not d.get("full_length"):
        return s
    f = d["full_length"]
    s += (f"\n**Full-length stage.** {f['n_storms']} storms run start to end "
          f"({f['t_end_s'] / 3600:.1f} h of storm) at batch {f['batch']}, and {len(f['sequential'])} of them "
          "also run alone. This is the stage that prices a dataset and the only one that measures the shared "
          "time step honestly: over a short window the members' CFL limits have barely diverged.\n\n")
    s += "| metric | value |\n|---|---|\n"
    s += f"| batch size | {f['batch']} |\n| storms | {f['n_storms']} |\n"
    s += f"| total loop time | {f['total_loop_s'] / 60:.1f} min |\n"
    s += f"| **seconds per storm, batched** | **{f['s_per_storm']:.0f} s** |\n"
    if f["s_per_storm_sequential"]:
        s += (f"| seconds per storm, one at a time | {f['s_per_storm_sequential']:.0f} s "
              f"(mean of {len(f['sequential'])}) |\n| **speedup** | **{f['speedup']:.2f}x** |\n")
    if f["peak_gpu_mb"]:
        s += f"| peak GPU memory | {f['peak_gpu_mb']:.0f} MB |\n"
    s += f"| worst mass-balance error | {f['max_mass_error']:.2e} |\n"
    s += "| steps per chunk | " + ", ".join(str(r["steps"]) for r in f["runs"]) + " |\n"
    if f["sequential"]:
        s += (f"\n**Cost of the shared time step**, on the {g['nx']}x{g['ny']} @ {g['dx']:g} m grid above. Each "
              f"storm's peak depth field from the batch, against the same storm run alone with its own step:\n\n"
              "| storm | steps alone | max \\|peak depth difference\\| [mm] | within "
              f"{f['peak_depth_bound_m'] * 1e3:.1f} mm |\n|---|---|---|---|\n")
        for r in f["sequential"]:
            s += (f"| {r['i']} | {r['steps']} | {r['peak_depth_diff_m'] * 1e3:.4f} | "
                  f"{'yes' if r['peak_depth_diff_m'] < f['peak_depth_bound_m'] else 'NO'} |\n")
        s += (f"\nWorst case {f['worst_peak_depth_diff_m'] * 1e3:.4f} mm against a "
              f"{f['peak_depth_bound_m'] * 1e3:.1f} mm bound: "
              f"**{'within bound' if f['shared_timestep_within_bound'] else 'OVER BOUND'}**. "
              f"The batch ran {f['runs'][0]['steps']} steps.\n")
    if f["speedup"]:
        s += "\n**Verdict.** " + (
            f"Batching pays: {f['speedup']:.2f}x per storm at batch {f['batch']}.\n" if f["speedup"] > 1.05 else
            f"Batching does not pay on this machine. At batch {f['batch']} a storm costs "
            f"{f['s_per_storm']:.0f} s against {f['s_per_storm_sequential']:.0f} s run alone "
            f"({f['speedup']:.2f}x). The step breakdown shows why it should have worked -- a storm-step costs "
            f"about half as much inside a batch -- and the step counts show why it did not: the shared time step "
            f"made the batch run {f['runs'][0]['steps']} steps where a member alone needs about "
            f"{int(np.mean([r['steps'] for r in f['sequential']]))}, because the whole batch is dragged down to "
            "whichever member is most restricted at that moment. The per-step gain and the extra steps very "
            "nearly cancel. `data.batch` therefore defaults to 1; the batched solver stays available, tested "
            "and documented, for batches of storms with similar time-step demand.\n")
    s += "\n**Projected dataset generation** at the measured full-length rate:\n\n"
    s += "| simulations | batched | one at a time |\n|---|---|---|\n"
    for n, h in d["projection"].items():
        seq = d["projection_sequential"]
        s += f"| {n} | {h:.1f} h | " + (f"{seq[n]:.1f} h |\n" if seq else "n/a |\n")
    return s
