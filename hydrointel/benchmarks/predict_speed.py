"""Surrogate prediction cost per candidate, by population size and output detail.

Part B is an evolutionary optimiser: it evaluates a whole population per
generation against one scenario, tens of thousands of candidates per scenario.
This measures what that costs: seconds per candidate at batch sizes 1, 8, 32 and
64 in both output modes, with peak GPU memory and the host memory the results
occupy.

Timing does not depend on weight values. If no trained model exists for the
configuration, a full-size model with untrained weights is timed instead and the
report says so; re-run after training for the trained model's numbers.

    python -m hydrointel.cli predict-speed
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

BATCHES = (1, 8, 32, 64)
REPEATS = 3
HOST_MEMORY_FRACTION = 0.5      # a full-detail batch may not claim more than this share of free RAM


def _free_ram_bytes() -> float | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return float(m.ullAvailPhys)
    except (AttributeError, OSError):
        return None


def _untrained_model(ctx):
    """A full-size GeoKAN-PINO with untrained weights, for timing only."""
    from ..data.dataset import Scales, node_dynamic
    from ..data.generate import coarsen_mean, static_fields
    from ..domain import landuse as LU
    from ..model.batch import SampleBuilder
    from ..model.geokan_pino import GeoKANPINO
    from ..model.graph import build_graph
    from ..solver.swe1d import build_topology
    cfg, dom, dev = ctx.cfg, ctx.domain, ctx.device
    f = cfg.data.coarsen
    static = static_fields(dom, f)
    n_static = static["coarse"]["features"].shape[0]
    probe = {"fields": {k: torch.zeros(dom.ny // f, dom.nx // f) for k in ("manning", "storage", "kappa")}}
    probe["fields"]["landuse_frac"] = torch.as_tensor(
        coarsen_mean(np.eye(LU.N_CLASSES)[dom.landuse].transpose(2, 0, 1), f))
    n_feat = n_static + node_dynamic(probe).shape[0]
    # neutral scales: they set units, not cost
    sc = Scales(H0=1.0, L=dom.nx * dom.dx, U0=3.13, T0=2555.0, Q0=100.0, z_mean=0.0, z_std=1.0,
                feat_mean=[0.0] * n_feat, feat_std=[1.0] * n_feat, n1_mean=[0.0], n1_std=[1.0],
                inflow_scale=100.0, rain_scale=100.0, V_scale=1e6)
    graph = build_graph(np.asarray(static["coarse"]["dem2d"]), dom.dx * f, build_topology(dom.network)).to(dev)
    model = GeoKANPINO(cfg.model, n_feat, graph, sc).to(dev).eval()
    model._builder_cache = SampleBuilder(cfg, dom, static, sc, graph, dev)
    model._builder_key = id(ctx)
    return model


def run_study(cfg: RunConfig, device=None, batches=BATCHES) -> dict:
    from .. import api
    from ..config import seed_everything
    seed_everything(cfg.seed)
    ctx = api.configure(cfg, device)
    dev = ctx.device
    try:
        ctx.model()
        weights = "trained"
    except FileNotFoundError:
        ctx._model = _untrained_model(ctx)
        weights = "untrained (timing only)"
    log.info("timing the surrogate with %s weights", weights)
    dom = ctx.domain
    scen = api.Scenario(100, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    land = ~(dom.sea | dom.channel)
    rng = np.random.default_rng(cfg.seed)
    pool = []
    for _ in range(max(batches)):
        S = base.storage_depth.numpy().copy()
        S[land & (rng.random(dom.shape) < 0.05)] += rng.uniform(0.1, 1.0)
        pool.append(base.replace(storage_depth=torch.as_tensor(S)))
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    api.predict(pool[0], scen)                   # warm-up; also caches the baseline reference
    api.predict(pool[0], scen, detail="full")
    rows = []
    for detail in ("summary", "full"):
        for b in batches:
            host_need = None
            if detail == "full":
                t_count = int(np.floor((ctx.cfg.forcing.spinup_h + ctx.cfg.forcing.storm_duration_h
                                        + ctx.cfg.solver.recession_h) * 3600 / ctx.cfg.solver.output_interval_s)) + 1
                host_need = 3.0 * b * t_count * dom.ny * dom.nx * 4
                free = _free_ram_bytes()
                if free is not None and host_need > HOST_MEMORY_FRACTION * free:
                    rows.append({"detail": detail, "batch": b, "skipped": True,
                                 "reason": f"needs {host_need / 2 ** 30:.1f} GB of host RAM for the histories; "
                                           f"{free / 2 ** 30:.1f} GB free"})
                    log.info("full, batch %d: skipped (%s)", b, rows[-1]["reason"])
                    continue
            walls = []
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for _ in range(REPEATS):
                sync(); t0 = time.perf_counter()
                res = api.predict(pool[:b], scen, detail=detail)
                sync(); walls.append(time.perf_counter() - t0)
                del res
            w = float(np.median(walls))
            rows.append({"detail": detail, "batch": b, "skipped": False, "call_s": w, "s_per_candidate": w / b,
                         "calls": REPEATS, "host_result_gb": (host_need / 2 ** 30) if host_need else None,
                         "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None})
            log.info("%s, batch %d: %.3f s per call, %.4f s per candidate, peak %s MB", detail, b, w, w / b,
                     rows[-1]["peak_gpu_mb"])
    out = {"weights": weights, "grid": {"nx": dom.nx, "ny": dom.ny, "dx": dom.dx}, "device": str(dev),
           "encode_batch": int(getattr(ctx.model(), "encode_batch", 4)), "rows": rows}
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train")).replace(
        notes=(f"surrogate prediction cost, {weights}",))
    write_json(Path(cfg.outdir) / "predict_speed.json", out, prov)
    (Path(cfg.outdir) / "predict_speed.md").write_text(report(out, prov), encoding="utf-8")
    return out


def report(d: dict, prov) -> str:
    g = d["grid"]
    s = markdown_header(prov, "Surrogate prediction cost per candidate")
    s += (f"\nGrid {g['nx']}x{g['ny']} @ {g['dx']:g} m on {d['device']}; model weights: **{d['weights']}**. "
          f"Candidates are encoded {d['encode_batch']} at a time and decoded in time chunks; the scenario's "
          "forcing is encoded once per call. Median of the repeated calls.\n\n")
    s += "| detail | batch | s per call | **s per candidate** | peak GPU [MB] | result on host [GB] |\n"
    s += "|---|---|---|---|---|---|\n"
    for r in d["rows"]:
        if r["skipped"]:
            s += f"| {r['detail']} | {r['batch']} | skipped: {r['reason']} | | | |\n"
            continue
        gpu = f"{r['peak_gpu_mb']:.0f}" if r["peak_gpu_mb"] else "n/a"
        host = f"{r['host_result_gb']:.2f}" if r["host_result_gb"] else "—"
        s += (f"| {r['detail']} | {r['batch']} | {r['call_s']:.3f} | **{r['s_per_candidate']:.4f}** | "
              f"{gpu} | {host} |\n")
    summ = {r["batch"]: r for r in d["rows"] if r["detail"] == "summary" and not r["skipped"]}
    full = {r["batch"]: r for r in d["rows"] if r["detail"] == "full" and not r["skipped"]}
    common = sorted(set(summ) & set(full))
    if common:
        b = common[-1]
        s += (f"\nSummary against full detail at batch {b}: {full[b]['s_per_candidate'] / summ[b]['s_per_candidate']:.2f}x "
              "faster per candidate. Summary mode never builds the (nt, ny, nx) histories -- each time chunk "
              "updates running maxima and point series and is dropped -- so it saves memory and host transfer, "
              "not decoder arithmetic: a peak depth still needs the decoder at every output time.\n")
    if 1 in summ:
        best = min(summ.values(), key=lambda r: r["s_per_candidate"])
        s += (f"\nBatching: {summ[1]['s_per_candidate']:.4f} s per candidate alone against "
              f"{best['s_per_candidate']:.4f} s at batch {best['batch']} "
              f"({summ[1]['s_per_candidate'] / best['s_per_candidate']:.2f}x).\n")
    return s
