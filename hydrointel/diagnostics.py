"""Task A diagnostics: why did Phase 1 fail its effect and storage tests?

Each part answers one question from existing runs where it can, runs the engine
only where it must, and writes its numbers to ``artifacts/diagnostics/<part>.json``.
``report`` collects them into ``artifacts/diagnostics_A.md``.

  a1  is the 2x coarse grid the binding constraint?  (oracle round trip vs model)
  a3  where does the engine's storage-induced rise come from?  (plane, far, near)
  a4  did the global-mass term ever carry weight?  (lambda trace, train/test residual)
  a5  what resolution fits the card?  (peak memory and s/step per option)

A2 (does reported depth include retained water?) is answered by
tests/test_retained_water.py, not here.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from . import api

log = logging.getLogger("hydrointel.diagnostics")

STORAGE_M = 0.6                 # storage depth placed by every storage-direction test
DISC_R_M = 1500.0               # radius of the storage disc on the city grid


def out_dir(cfg) -> Path:
    p = Path(cfg.outdir) / "diagnostics"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(cfg, name: str, data: dict, ctx) -> Path:
    from .viz.provenance import write_json
    p = out_dir(cfg) / f"{name}.json"
    write_json(p, data, ctx.provenance("engine+surrogate", ("domain", "solver", "forcing", "data", "model", "train")))
    return p


# ---------------------------------------------------------------------------
# A3: storage-direction test, three ways
# ---------------------------------------------------------------------------
def storage_criteria(base: np.ndarray, more: np.ndarray, footprint: np.ndarray, land: np.ndarray,
                     dist_channel: np.ndarray | None = None) -> dict:
    """The storage-direction criteria exactly as the evaluation applies them (footprint
    max rise <= 5 mm, land max rise <= 2 cm, share of land rising > 1 cm <= 0.1%,
    peak-depth volume falls), plus the distribution behind the maxima: a max over
    ~10^4 cells is set by the single worst cell, so how many cells rise, by how much,
    and where they sit is reported alongside."""
    d = more - base
    res = {"footprint_max_increase_m": float(d[footprint].max()), "land_max_increase_m": float(d[land].max()),
           "land_fraction_increase_gt_1cm": float(np.mean(d[land] > 0.01)),
           "peak_volume_ratio": float(more[land].sum() / max(base[land].sum(), 1e-9))}
    res["passed"] = bool(res["footprint_max_increase_m"] <= 0.005 and res["land_max_increase_m"] <= 0.02
                         and res["land_fraction_increase_gt_1cm"] <= 0.001 and res["peak_volume_ratio"] < 1.0)
    res["failed_criteria"] = [k for k, ok in (
        ("footprint max rise <= 5 mm", res["footprint_max_increase_m"] <= 0.005),
        ("land max rise <= 20 mm", res["land_max_increase_m"] <= 0.02),
        ("land share rising > 1 cm <= 0.1%", res["land_fraction_increase_gt_1cm"] <= 0.001),
        ("peak-depth volume falls", res["peak_volume_ratio"] < 1.0)) if not ok]
    res.update({"n_footprint": int(footprint.sum()), "n_land": int(land.sum()),
                "n_land_rise_gt_5mm": int((d[land] > 0.005).sum()), "n_land_rise_gt_1cm": int((d[land] > 0.01).sum()),
                "n_land_fall_gt_1cm": int((d[land] < -0.01).sum()),
                "footprint_mean_change_m": float(d[footprint].mean()),
                "footprint_p99_change_m": float(np.percentile(d[footprint], 99)),
                "land_p99_9_change_m": float(np.percentile(d[land], 99.9))})
    if dist_channel is not None:
        m = land & (d > 0.005)
        res["rise_gt_5mm_dist_channel_m"] = sorted(float(x) for x in dist_channel[m])
        res["rise_gt_5mm_base_depth_m"] = [float(x) for x in base[m][np.argsort(dist_channel[m])]]
    return res


def plane_run(storage: np.ndarray, cfg, device, nx: int = 150, ny: int = 100, dx: float = 20.0,
              slope: float = 0.002, micro_amp: float = 0.0) -> dict:
    """2-D solver alone on a tilted plane (no channel, no urban fabric): Phase 1's RP100
    RCP4.5/SSP2 2050 hyetograph, residential Horton infiltration and roughness, outlet at
    the low (east) edge, walls elsewhere. Returns peak depth and the volume ledger."""
    from .domain import landuse as LU
    from .engine import build_forcing
    from .solver.boundaries import EdgeBC
    from .solver.infiltration import Horton
    from .solver.massbalance import MassBalance
    from .solver.swe2d import SWE2D
    fo = build_forcing(cfg, 100, "RCP4.5", "SSP2", 2050, False, 0.0, nx * ny * dx * dx / 1e6)
    x = (np.arange(nx) + 0.5) * dx
    z = np.broadcast_to((x.max() - x) * slope, (ny, nx)).copy()
    if micro_amp > 0:
        rng = np.random.default_rng(12345)
        z += micro_amp * rng.standard_normal((ny, nx))
    lu = np.full((ny, nx), 3, dtype=np.int64)            # residential / subdivision
    dtype = cfg.dtype
    bcs = {"west": EdgeBC("closed"), "east": EdgeBC("transmissive"), "south": EdgeBC("closed"),
           "north": EdgeBC("closed")}
    horton = Horton.from_landuse(lu, np.ones((ny, nx)), device, dtype, t0=fo.storm_start)
    sim = SWE2D(z, LU.lookup(lu, "manning"), dx, cfg.solver, bcs, device, dtype, storage=storage, horton=horton)
    mb = MassBalance(device, 0.0)
    t, hmax, steps = 0.0, torch.zeros_like(sim.h), 0
    while t < fo.t_end - 1e-9:
        dt = min(sim.max_dt(), fo.t_end - t)
        vol = sim.step(dt, t, fo.rain.rate_ms(t))
        mb.add("rain", vol.rain); mb.add("infiltration", vol.infil); mb.add("clip", vol.clip)
        mb.add("bnd2d_in", vol.bnd_in); mb.add("bnd2d_out", vol.bnd_out)
        hmax = torch.maximum(hmax, sim.h)
        t += dt
        steps += 1
    err = mb.check(float(sim.volume() + sim.storage_volume()), t)
    return {"depth_max": hmax.cpu().numpy(), "mass_err": float(err), "steps": steps,
            "stored_m3": float(sim.storage_volume()), "sources": mb.sources()}


def a3_plane(cfg, device, micro_amp: float = 0.0) -> dict:
    nx, ny, dx = 150, 100, 20.0
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx)
    disc = np.hypot(X - nx * dx / 2, Y - ny * dx / 2) < 500.0
    t0 = time.perf_counter()
    a = plane_run(np.zeros((ny, nx)), cfg, device, nx, ny, dx, micro_amp=micro_amp)
    b = plane_run(np.where(disc, STORAGE_M, 0.0), cfg, device, nx, ny, dx, micro_amp=micro_amp)
    land = np.ones((ny, nx), bool)
    res = storage_criteria(a["depth_max"], b["depth_max"], disc, land)
    res.update({"geometry": f"{nx} x {ny} cells @ {dx:g} m, slope {0.002:g} toward an open east edge, walls "
                            f"elsewhere, micro-topography std {micro_amp:g} m; storage {STORAGE_M:g} m in a 500 m "
                            "disc at the centre; RP100 RCP4.5/SSP2 2050 rain, residential infiltration and roughness",
                "mass_err_base": a["mass_err"], "mass_err_more": b["mass_err"], "steps": [a["steps"], b["steps"]],
                "stored_m3_more": b["stored_m3"], "wall_s": time.perf_counter() - t0,
                "baseline_peak_max_m": float(a["depth_max"].max())})
    return res


def far_disc(dom, min_dist_m: float = 500.0) -> tuple[np.ndarray, tuple[float, float]]:
    """A storage disc of the same radius as the evaluation's, placed on the city grid
    where every land cell in it is at least ``min_dist_m`` from the channel network,
    choosing the centre that covers the most urban land."""
    X, Y = np.meshgrid(dom.x, dom.y)
    land = ~(dom.sea | dom.channel)
    lu = api.context().landuse("SSP2", 2050)
    urban = np.isin(lu, (2, 3))
    best = None
    for cy in range(0, dom.ny, 5):
        for cx in range(0, dom.nx, 5):
            disc = (np.hypot(X - dom.x[cx], Y - dom.y[cy]) < DISC_R_M) & land
            if disc.sum() < 0.8 * np.pi * DISC_R_M ** 2 / dom.dx ** 2 or dom.dist_channel[disc].min() < min_dist_m:
                continue
            score = int((urban & disc).sum())
            if best is None or score > best[0]:
                best = (score, cx, cy)
    if best is None:
        raise RuntimeError(f"no {DISC_R_M:g} m disc on this domain stays {min_dist_m:g} m from the channels")
    _, cx, cy = best
    return (np.hypot(X - dom.x[cx], Y - dom.y[cy]) < DISC_R_M) & land, (float(dom.x[cx]), float(dom.y[cy]))


def a3_city(ctx, mdir: Path, where: str) -> dict:
    """``where`` = 'near' (the evaluation's disc round the city core, which the channel
    crosses) or 'far' (same radius, >= 500 m from any channel). Engine runs are cached."""
    from .evaluate_effects import _engine_peak
    dom = ctx.domain
    scen = api.Scenario(100, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    land = ~(dom.sea | dom.channel)
    X, Y = np.meshgrid(dom.x, dom.y)
    if where == "near":
        disc = (np.hypot(X - dom.urban_core[0], Y - dom.urban_core[1]) < DISC_R_M) & land
        centre = (float(dom.urban_core[0]), float(dom.urban_core[1]))
        cache = mdir / "effects" / "storage_check_more.pt"
    else:
        disc, centre = far_disc(dom)
        cache = out_dir(ctx.cfg) / "a3_far_more.pt"
    S = base.storage_depth.numpy().copy()
    S[disc] = np.maximum(S[disc], STORAGE_M)
    t0 = time.perf_counter()
    a, _ = _engine_peak(base, scen, mdir / "effects" / "storage_check_base.pt")
    b, _ = _engine_peak(base.replace(storage_depth=torch.as_tensor(S)), scen, cache)
    res = storage_criteria(a, b, disc, land, dom.dist_channel)
    res.update({"centre_m": centre, "disc_min_dist_channel_m": float(dom.dist_channel[disc].min()),
                "wall_s": time.perf_counter() - t0})
    return res


def run_a3(cfg, device=None) -> dict:
    from .model.geokan_pino import model_dir
    ctx = api.configure(cfg, device)
    mdir = api.engine_cache_dir(cfg)
    res = {}
    res["plane"] = a3_plane(cfg, ctx.device)
    log.info("A3 plane: %s", {k: res["plane"][k] for k in ("passed", "footprint_max_increase_m", "land_max_increase_m")})
    res["plane_micro"] = a3_plane(cfg, ctx.device, micro_amp=0.10)
    log.info("A3 plane + micro-topography: %s",
             {k: res["plane_micro"][k] for k in ("passed", "footprint_max_increase_m", "land_max_increase_m")})
    res["city_far"] = a3_city(ctx, mdir, "far")
    log.info("A3 city far: %s", {k: res["city_far"][k] for k in ("passed", "footprint_max_increase_m")})
    res["city_near"] = a3_city(ctx, mdir, "near")
    p1, p2, p3 = (res[k]["passed"] for k in ("plane", "city_far", "city_near"))
    if not p1:
        branch = "A3-BUG"
    elif not p2 and not p3:
        branch = "re-routing (physical)"
    elif p2 and not p3:
        branch = "A3-CHANNEL"
    else:
        branch = "none (all pass)" if p3 else "unclassified"
    res["branch_by_table"] = branch
    _save(cfg, "a3", res, ctx)
    return res


# ---------------------------------------------------------------------------
# A1: is the 2x coarse grid the binding constraint?
# ---------------------------------------------------------------------------
THR = (0.15, 0.30, 0.50)


def oracle_peak(peak_full: np.ndarray, builder) -> np.ndarray:
    """The engine's own peak depth, coarsened 2x (block mean) and upsampled bilinearly back
    to 20 m with the surrogate's own upsampler: the best a perfect surrogate on the coarse
    grid could return."""
    from .data.generate import coarsen_mean
    c = coarsen_mean(np.asarray(peak_full, np.float64), builder.f).astype(np.float32)
    t = torch.as_tensor(c, device=builder.device)[None]
    return builder.upsample(t)[0].cpu().numpy()


def abs_metrics(pred: np.ndarray, eng: np.ndarray, land: np.ndarray, cell_ha: float) -> dict:
    from .evaluate import contingency
    either = land & ((eng > 0.05) | (pred > 0.05))
    d = (pred - eng)[either]
    out = {"peak_rmse_m": float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")}
    for t in THR:
        out[f"csi_{t:g}"] = contingency(pred, eng, t, land)["csi"]
        a_p, a_e = float(np.sum(land & (pred >= t))) * cell_ha, float(np.sum(land & (eng >= t))) * cell_ha
        out[f"area_{t:g}_ha"] = a_p
        out[f"area_{t:g}_engine_ha"] = a_e
        out[f"area_{t:g}_rel_err"] = (a_p - a_e) / max(a_e, 1e-9)
    return out


def effect_block(d_eng, d_x, pk_eng, pk_eng0, pk_x, pk_x0, land, dx) -> dict:
    """Effect metrics (the evaluation's own definitions) plus how much of the true effect
    survives: magnitude ratio sum|d_x| / sum|d_eng| and the projection coefficient
    sum(d_x d_eng) / sum(d_eng^2), both over land cells the engine moved by > 1 cm; and the
    effect RMSE on that common mask, so model and oracle are also compared on identical cells."""
    from .evaluate_effects import AFFECTED_M, effect_metrics
    m = effect_metrics(d_eng, d_x, pk_eng, pk_eng0, pk_x, pk_x0, land, dx)
    ch = land & (np.abs(d_eng) > AFFECTED_M)
    m["magnitude_ratio"] = float(np.abs(d_x[ch]).sum() / max(np.abs(d_eng[ch]).sum(), 1e-12))
    m["projection_coef"] = float((d_x[ch] * d_eng[ch]).sum() / max((d_eng[ch] ** 2).sum(), 1e-12))
    m["effect_rmse_engine_changed_m"] = float(np.sqrt(np.mean((d_x - d_eng)[ch] ** 2))) if ch.any() else float("nan")
    return m


def run_a1(cfg, device=None) -> dict:
    from .data import sampler as SMP
    from .data.dataset import SimDataset
    from .evaluate import edge_probe, site_from_sample
    from .evaluate_effects import _engine_peak, single_field_sites
    from .model.batch import SampleBuilder
    from .model.geokan_pino import model_dir
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    model = ctx.model()
    builder = SampleBuilder.from_context(ctx, model)
    land = ~(dom.sea | dom.channel)
    cell_ha = dom.dx ** 2 / 1e4
    mdir = api.engine_cache_dir(cfg)
    test = SimDataset(api.dataset_dir(cfg), "test")
    cases = []

    def one(name, kind, pk_e, pk_e0, pk_s, pk_s0, series_oracle=None):
        orc, orc0 = oracle_peak(pk_e, builder), oracle_peak(pk_e0, builder)
        row = {"case": name, "kind": kind}
        for who, (p, p0) in (("model", (pk_s, pk_s0)), ("oracle", (orc, orc0))):
            row[who] = {"modified": abs_metrics(p, pk_e, land, cell_ha),
                        "baseline": abs_metrics(p0, pk_e0, land, cell_ha),
                        "effect": effect_block(pk_e - pk_e0, p - p0, pk_e, pk_e0, p, p0, land, dom.dx)}
        if series_oracle is not None:
            row["series_oracle_modified"] = abs_metrics(series_oracle, pk_e, land, cell_ha)
        row["R"] = row["oracle"]["effect"]["effect_rmse_m"] / max(row["model"]["effect"]["effect_rmse_m"], 1e-12)
        row["R_engine_changed_mask"] = (row["oracle"]["effect"]["effect_rmse_engine_changed_m"]
                                        / max(row["model"]["effect"]["effect_rmse_engine_changed_m"], 1e-12))
        cases.append(row)
        log.info("A1 %s: effect RMSE model %.4f oracle %.4f (R %.2f); sign model %.2f oracle %.2f", name,
                 row["model"]["effect"]["effect_rmse_m"], row["oracle"]["effect"]["effect_rmse_m"], row["R"],
                 row["model"]["effect"]["sign_agreement"], row["oracle"]["effect"]["sign_agreement"])

    for sid in test.ids:
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, "test")
        if smp.baseline:
            continue
        site, scen = site_from_sample(smp, dom, ctx)
        base = api.baseline_site(scen)
        rec = test.load(sid)
        pk_e = np.asarray(rec["depth_max_full"])
        cache = mdir / "effects" / f"test_base_{sid}.pt"
        if not cache.exists():
            raise FileNotFoundError(f"{cache}: the engine baseline for test {sid} is not cached; run evaluate first")
        pk_e0, _ = _engine_peak(base, scen, cache)
        sur = api.predict([site, base], scen)
        h = torch.as_tensor(np.asarray(rec["h"]), dtype=torch.float32, device=builder.device)
        ser = builder.upsample(h).amax(0).cpu().numpy()
        one(f"test {sid}", "test", pk_e, pk_e0, sur[0].depth_max.numpy(), sur[1].depth_max.numpy(), ser)
    scen, base, sites = single_field_sites(edge_probe(0, dom, cfg, cfg.seed), dom, ctx, cfg)
    pk_e0, _ = _engine_peak(base, scen, mdir / "effects" / "single_base.pt")
    names = list(sites)
    sur = api.predict([sites[n] for n in names] + [base], scen)
    for k, n in enumerate(names):
        pk_e, _ = _engine_peak(sites[n], scen, mdir / "effects" / f"single_{n.split()[0]}.pt")
        one(n, "single field", pk_e, pk_e0, sur[k].depth_max.numpy(), sur[-1].depth_max.numpy())
    tests = [c for c in cases if c["kind"] == "test"]
    mean = lambda rows, who, key: float(np.nanmean([r[who]["effect"][key] for r in rows]))
    pooled = {"n_test_pairs": len(tests)}
    for key in ("effect_rmse_m", "effect_rmse_engine_changed_m", "sign_agreement", "effect_corr",
                "magnitude_ratio", "projection_coef"):
        for who in ("model", "oracle"):
            pooled[f"{who}_{key}"] = mean(tests, who, key)
    pooled["R"] = pooled["oracle_effect_rmse_m"] / pooled["model_effect_rmse_m"]
    pooled["R_engine_changed_mask"] = (pooled["oracle_effect_rmse_engine_changed_m"]
                                       / pooled["model_effect_rmse_engine_changed_m"])
    pooled["R_per_case"] = [r["R"] for r in tests]
    R = pooled["R"]
    pooled["branch"] = ("do C4 (resolution work)" if R >= 0.7 else
                        "skip C4; spend the budget on C3" if R <= 0.3 else "do C4 cheapest option; still C2/C3")
    res = {"definition": "oracle = engine peak depth, 2x block-mean coarsened and bilinearly upsampled to 20 m with "
                         "the surrogate's upsampler; effect = modified - baseline peak depth; R = mean oracle effect "
                         "RMSE / mean model effect RMSE over the held-out test pairs (effect RMSE as "
                         "evaluate_effects defines it)",
           "pooled": pooled, "cases": cases}
    _save(cfg, "a1", res, ctx)
    return res


# ---------------------------------------------------------------------------
# A4: why is the surrogate's implied mass balance 15.6% off?
# ---------------------------------------------------------------------------
LAMBDA_FLOOR = 0.25               # x lambda_data (= 1): the floor the task prescribes if lambda_mass collapsed


def lambda_trace(curves: list) -> dict:
    tr = [r for r in curves if "total" in r]
    names = sorted(k[len("lambda_"):] for k in tr[-1] if k.startswith("lambda_"))
    phys = [r for r in tr if r["ramp"] > 0]
    out = {"steps": [r["step"] for r in tr], "ramp": [r["ramp"] for r in tr], "terms": {}}
    for n in names:
        lam = np.array([r[f"lambda_{n}"] for r in tr])
        loss = np.array([r.get(f"loss_{n}", np.nan) for r in tr], float)
        w = lam * np.array([1.0 if n == "initial" else r["ramp"] for r in tr])
        share = np.array([w[k] * loss[k] / r["total"] if np.isfinite(loss[k]) else np.nan for k, r in enumerate(tr)])
        pl = np.array([r[f"lambda_{n}"] for r in phys])
        ps = share[[k for k, r in enumerate(tr) if r["ramp"] > 0]]
        out["terms"][n] = {"lambda": lam.tolist(), "loss": loss.tolist(), "share_of_total": share.tolist(),
                           "physics_phase": {"lambda_min": float(pl.min()), "lambda_median": float(np.median(pl)),
                                             "lambda_max": float(pl.max()), "lambda_last": float(pl[-1]),
                                             "share_median": float(np.nanmedian(ps)),
                                             "share_max": float(np.nanmax(ps))}}
    return out


@torch.no_grad()
def volume_pieces(model, builder, site, scen, times) -> dict:
    """The quantities the surrogate's implied mass error is built from, for one site."""
    b, fo, _ = builder.from_sites([site], scen)
    dom, sc = builder.domain, model.scales
    x2, x1, g = model.encode(b)
    vols = model.volumes(x2, x1, g)[0]
    t = torch.as_tensor(times, dtype=torch.float32, device=b["feat2"].device)
    hs, ys = [], []
    for q0 in range(0, len(t), 8):
        ts = t[q0:q0 + 8] / sc.T0
        o2 = model._dec2(x2, b, ts[:, None, None].expand(-1, x2.shape[-2], 1))
        o1 = model._dec1(x1, b, ts[:, None, None].expand(-1, x1.shape[-2], 1))
        h, _, _, y1, _ = model.dimensional(o2, o1)
        hs.append(h[0]); ys.append(y1[0])
    h, y1 = torch.cat(hs), torch.cat(ys)
    bs = SampleBuilderSelect(builder, b)
    v = h.sum(-1) * builder.cell_c ** 2 + builder.volume1d(y1, bs)
    rain_vol = float(b["rain_mmh"].sum()) / 1000.0 / 3600.0 * (b["t_end"] / len(b["rain_mmh"])) \
        * dom.nx * dom.ny * dom.dx ** 2
    return {"dV": float(v[-1] - v[0]), "rain": rain_vol, "infil": float(vols[0]), "stored": float(vols[1]),
            "bnd_net": float(vols[2])}


def SampleBuilderSelect(builder, b):
    """The first candidate of a stacked input, in the shape volume1d expects."""
    from .model.batch import SampleBuilder
    one = SampleBuilder.select(b, slice(0, 1))
    from .solver.swe1d import Sections
    one["sec1"] = Sections(*(getattr(one["sec1"], a)[0] for a in ("b", "m", "yb", "ts")))
    one["dx1"] = one["dx1"][0]
    return one


def mass_decomposition(pred: dict, rec: dict) -> dict:
    """Split |dV - (rain + boundary - infiltration - storage)| / gross into what the depth
    field gets wrong and what the event-total head gets wrong, against the engine ledger."""
    led = rec["ledger"]
    n = lambda a: np.asarray(a.numpy() if torch.is_tensor(a) else a, np.float64)
    v_eng = n(led["v2d_coarse"]) + n(led["v1d"])
    eng = {"dV": float(v_eng[-1] - v_eng[0]), "rain": float(n(led["rain"])[-1]),
           "infil": float(rec["volumes"]["infiltrated_m3"]), "stored": float(rec["volumes"]["stored_m3"]),
           "bnd_net": float(n(led["bnd2d_in"])[-1] - n(led["bnd2d_out"])[-1] + n(led["bnd1d_in"])[-1]
                            - n(led["bnd1d_out"])[-1])}
    net = lambda s: s["rain"] + s["bnd_net"] - s["infil"] - s["stored"]
    gross = lambda s: s["rain"] + abs(s["bnd_net"]) + s["infil"] + s["stored"]
    err = lambda dv, s: abs(dv - net(s)) / max(gross(s), 1.0)
    return {"reported": err(pred["dV"], pred),
            "field_vs_true_sources": err(pred["dV"], eng),
            "true_field_vs_predicted_sources": err(eng["dV"], pred),
            "engine_coarse_floor": err(eng["dV"], eng),
            "dV_rel_err": (pred["dV"] - eng["dV"]) / max(gross(eng), 1.0),
            "infil_rel_err": (pred["infil"] - eng["infil"]) / max(eng["infil"], 1.0),
            "stored_rel_err": (pred["stored"] - eng["stored"]) / max(eng["stored"], 1.0),
            "bnd_net_err_of_gross": (pred["bnd_net"] - eng["bnd_net"]) / max(gross(eng), 1.0),
            "rain_rel_err": (pred["rain"] - eng["rain"]) / max(eng["rain"], 1.0),
            "pred": pred, "engine": eng}


def run_a4(cfg, device=None) -> dict:
    from .data import sampler as SMP
    from .data.dataset import SimDataset
    from .evaluate import site_from_sample
    from .model.batch import SampleBuilder
    from .model.geokan_pino import model_dir
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    model = ctx.model()
    builder = SampleBuilder.from_context(ctx, model)
    curves = json.loads((model_dir(cfg) / "curves.json").read_text(encoding="utf-8"))
    trace = lambda_trace(curves)
    res = {"trace": trace, "splits": {}}
    for split in ("train", "val", "test"):
        ds = SimDataset(api.dataset_dir(cfg), split)
        rows = []
        for sid in ds.ids:
            rec = ds.load(sid)
            smp = SMP.draw(sid, dom, cfg.data, cfg.seed, split)
            site, scen = site_from_sample(smp, dom, ctx)
            pred = volume_pieces(model, builder, site, scen, np.asarray(rec["times"], np.float64))
            rows.append({"sim": sid, **mass_decomposition(pred, rec)})
        keys = ("reported", "field_vs_true_sources", "true_field_vs_predicted_sources", "engine_coarse_floor",
                "dV_rel_err", "infil_rel_err", "stored_rel_err", "bnd_net_err_of_gross", "rain_rel_err")
        res["splits"][split] = {"n": len(rows), "rows": rows,
                                "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                                "mean_abs": {k: float(np.mean([abs(r[k]) for r in rows])) for k in keys}}
        log.info("A4 %s: %s", split, {k: round(v, 4) for k, v in res["splits"][split]["mean_abs"].items()})
    gm = trace["terms"]["global_mass"]["physics_phase"]
    res["lambda_mass_collapsed"] = bool(gm["lambda_median"] < LAMBDA_FLOOR)
    res["branch"] = ("floor lambda_mass at 0.25 x lambda_data" if res["lambda_mass_collapsed"] else
                     "lambda_mass carried weight: post-decoder volume correction (revert if depth RMSE worsens > 20%)")
    _save(cfg, "a4", res, ctx)
    return res


# ---------------------------------------------------------------------------
# A5: what resolution fits the card?
# ---------------------------------------------------------------------------
BUDGET_MB = 4.5 * 1024
TRAIN_CACHE = 17                  # batches train.Trainer keeps on the device (evicts above 16)


def resample_batch(b: dict, graph_from, graph_to, cell_to: float) -> dict:
    """A training batch built at one 2-D resolution, resampled to another: every per-node
    2-D array is interpolated bilinearly (masks nearest), 1-D arrays are unchanged. The
    numbers are only as good as bilinear interpolation, which is all a memory and speed
    measurement needs: the shapes, the graph and the loss code are the real ones."""
    import torch.nn.functional as F
    ny0, nx0, ny1, nx1 = graph_from.ny, graph_from.nx, graph_to.ny, graph_to.nx
    n0 = ny0 * nx0

    def rs(a, mode="bilinear"):
        lead = a.shape[:-1]
        x = a.reshape(-1, 1, ny0, nx0).float()
        kw = {"align_corners": False} if mode == "bilinear" else {}
        y = F.interpolate(x, size=(ny1, nx1), mode=mode, **kw)
        return y.reshape(*lead, ny1 * nx1).to(a.dtype)
    out = dict(b)
    for k in ("h", "u", "v", "horton_mmh", "manning2", "z2"):
        out[k] = rs(b[k])
    out["feat2"] = rs(b["feat2"].T).T.contiguous()
    for k in ("resid_mask2", "bc_mask2"):
        out[k] = rs(b[k].float(), "nearest") > 0.5
    n = out["manning2"] * 30.0
    out["n_edge2"] = 0.5 * (n[graph_to.e2_src] + n[graph_to.e2_dst])
    out["v_target"] = b["v_target"]              # a volume: unchanged by the grid
    assert b["h"].shape[-1] == n0
    return out


def _losses(model, b, qi, cell, volume1d, dtype):
    """Trainer.losses with the autocast dtype as a parameter (the full physics loss)."""
    from .model import physics_loss as PL
    from .model.geokan_pino import interp_series
    sc = model.scales
    dev = b["feat2"].device
    with torch.autocast(device_type=dev.type, dtype=dtype, enabled=dtype != torch.float32):
        x2, x1, g = model.encode(b)
    x2, x1, g = x2.float(), x1.float(), g.float()
    o2, o1, d2, d1 = model.decode(x2, x1, b, b["times"][qi], need_dt=True)
    ld, _ = PL.data_loss(o2, o1, b, qi, sc, model.y1_scale)
    terms = {"data": ld + ((model.volumes(x2, x1, g) - b["vol_target"]) / sc.V_scale).pow(2).mean()}
    terms.update(PL.residuals_2d(o2, d2, b, qi, sc, model.graph, cell))
    terms.update(PL.residuals_1d(o1, d1, b, sc, model.graph, model.y1_scale))
    terms["boundary"] = PL.boundary_loss(o2, b, qi, sc, lambda t: interp_series(b["series"][1], t, b["t_end"]))
    terms["global_mass"] = PL.global_mass_loss(o2, o1, b, qi, sc, cell, volume1d, model.y1_scale).mean()
    return sum(terms.values())


def measure_training_step(cfg, factor: float, checkpointing: bool, dtype, device, steps: int = 6) -> dict:
    """Peak GPU memory and s/step of one full training step (data + physics losses, backward,
    optimiser) on a 2-D grid at ``factor`` x the 20 m cell."""
    import gc
    from .data.dataset import SimDataset
    from .model.batch import SampleBuilder, build_model
    from .model.graph import build_graph
    from .model.geokan_pino import GeoKANPINO
    from .solver.swe1d import build_topology
    ctx = api.context()
    dom = ctx.domain
    root = api.dataset_dir(cfg)
    ds = SimDataset(root, "train")
    base_model = build_model(cfg, root, device)
    builder = SampleBuilder(cfg, dom, ds.static, base_model.scales, base_model.graph, device)
    b = builder.from_record(ds.load(ds.ids[0]))
    ny, nx = (dom.ny // 2, dom.nx // 2) if factor == 2.0 else (int(round(dom.ny / factor)), int(round(dom.nx / factor)))
    cell = dom.dx * factor
    import torch.nn.functional as F
    z = F.interpolate(torch.as_tensor(dom.dem2d, dtype=torch.float32)[None, None], size=(ny, nx),
                      mode="area")[0, 0].numpy()
    graph = build_graph(z, cell, build_topology(dom.network)).to(device)
    bt = resample_batch(b, base_model.graph, graph, cell) if (ny, nx) != (base_model.graph.ny, base_model.graph.nx) \
        else b
    del base_model
    model = GeoKANPINO(cfg.model, len(builder.mean), graph, builder.scales).to(device)
    model.use_checkpoint = checkpointing
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # as in training: a gradient scaler with float16 autocast (bfloat16 has float32's range and needs none)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    qi = torch.as_tensor([0, 5, 10, 15, 20, 30][:cfg.train.query_times], device=device)
    batch_mb = sum(v.numel() * v.element_size() for v in bt.values() if torch.is_tensor(v)) / 2 ** 20
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res = {"factor": factor, "grid": [ny, nx], "nodes": ny * nx, "cell_m": cell, "checkpointing": checkpointing,
           "autocast": str(dtype).replace("torch.", ""), "batch_mb": batch_mb}
    try:
        walls = []
        for k in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = _losses(model, bt, qi, cell, builder.volume1d, dtype)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        res.update({"status": "ok", "s_per_step": float(np.median(walls[2:])),
                    "peak_allocated_mb": peak, "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
                    # the training loop also keeps up to TRAIN_CACHE prepared batches on the card
                    "projected_training_peak_mb": peak + (TRAIN_CACHE - 1) * batch_mb,
                    "finite_loss": bool(torch.isfinite(loss).item())})
    except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
        res.update({"status": "OOM", "error": str(e).split("\n")[0][:300],
                    "peak_allocated_mb_before_oom": torch.cuda.max_memory_allocated() / 2 ** 20})
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    log.info("A5 %s", {k: v for k, v in res.items() if k != "error"})
    return res


def run_a5(cfg, device=None) -> dict:
    ctx = api.configure(cfg, device)
    dev = ctx.device
    if dev.type != "cuda":
        raise RuntimeError("A5 measures GPU memory: it needs the CUDA device")
    bf16_ok = torch.cuda.is_bf16_supported()
    try:
        bf16_native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        bf16_native = bf16_ok
    # rows already measured (artifacts/diagnostics/a5_rows.jsonl) are kept: a configuration that
    # spills past the card's memory runs at minutes per step, so the job is resumable
    part = out_dir(cfg) / "a5_rows.jsonl"
    rows = [json.loads(x) for x in part.read_text(encoding="utf-8").splitlines() if x.strip()] if part.exists() else []
    seen = {(r["factor"], r["checkpointing"], r["autocast"]) for r in rows}
    # the task's configurations: every resolution with checkpointing, in bf16 (and fp16, which
    # Phase 1 trained with); without checkpointing only where it could plausibly fit (2x)
    plan = [(2.0, True), (2.0, False), (1.5, True), (1.0, True)]
    for factor, ck in plan:
        for dt in (torch.float16, torch.bfloat16):
            if (factor, ck, str(dt).replace("torch.", "")) in seen:
                continue
            r = measure_training_step(cfg, factor, ck, dt, dev)
            rows.append(r)
            with part.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
    res = {"budget_mb": BUDGET_MB, "gpu": torch.cuda.get_device_name(0),
           "total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
           "bf16_supported_including_emulation": bf16_ok, "bf16_native": bf16_native, "rows": rows}
    res["choice"] = a5_choice(rows)
    _save(cfg, "a5", res, ctx)
    return res


def a5_choice(rows: list) -> dict:
    """The task's decision rule, applied to the calibrated training-loop peak: the harness's
    projection (step peak + batch cache) scaled by Phase 1's measured training peak over the
    harness's projection for the same configuration (2x, checkpointing, fp16)."""
    from .diagnostics_report import PHASE1_TRAIN_PEAK_MB
    base = next(r for r in rows if r["factor"] == 2.0 and r["checkpointing"] and r["autocast"] == "float16"
                and r["status"] == "ok")
    cal = PHASE1_TRAIN_PEAK_MB / base["projected_training_peak_mb"]
    fits = lambda r: r["status"] == "ok" and r["projected_training_peak_mb"] * cal <= BUDGET_MB
    full = [r for r in rows if r["factor"] == 1.0 and r["checkpointing"] and r["autocast"] == "bfloat16" and fits(r)]
    mid = [r for r in rows if r["factor"] == 1.5 and fits(r)]
    two = [r for r in rows if r["factor"] == 2.0 and fits(r)]
    note = ("option 2 (patch-based training at native 20 m with a coarse global branch) was not implemented, so it "
            "could not be measured; ")
    if full:
        return {"option": 1, "what": "full 20 m, gradient checkpointing, bf16", "row": full[0], "calibration": cal}
    if mid:
        return {"option": 3, "what": "1.5x coarse (30 m)", "row": min(mid, key=lambda r: r["s_per_step"]),
                "calibration": cal, "note": note + "1.5x fits the budget"}
    return {"option": 4, "what": "stay at 2x coarse (40 m): resolution is capped by the hardware",
            "calibration": cal, "fits_at_2x": [f"{r['autocast']}, checkpointing={r['checkpointing']}" for r in two],
            "note": note + "no finer grid fits the 4.5 GB budget on this card"}


# ---------------------------------------------------------------------------
# C5: post-decoder volume correction (A4 branch: lambda_mass carried weight)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_coarse(model, builder, site, scen, times):
    """Coarse-grid depth series (nt, N), 1-D depth (nt, n1) and event totals for one site
    (single-stage or two-stage model)."""
    b, fo, _ = builder.from_sites([site], scen)
    model.eval()
    model._prepare_call([site], scen, builder)
    state, vols = model._group_state(b)
    t = torch.as_tensor(times, dtype=torch.float32, device=b["feat2"].device) / model.scales.T0
    hs, ys = [], []
    for q0 in range(0, len(t), 8):
        o2, o1 = model._group_decode(state, b, t[q0:q0 + 8])
        h, _, _, y1, _ = model.dimensional(o2, o1)
        hs.append(h[0]); ys.append(y1[0])
    model._release_call()
    return torch.cat(hs), torch.cat(ys), vols[0], b


def volume_scale(h, y1, vols, b, builder, land_c) -> float:
    """The factor s that, applied to land-cell depths at every output time, makes the
    change in predicted water volume equal the forcing-implied net inflow
    (rain + boundary - infiltration - storage, the last three from the volume head)."""
    area = builder.cell_c ** 2
    one = SampleBuilderSelect(builder, b)
    v_land = (h * land_c).sum(-1) * area
    v_rest = (h * ~land_c).sum(-1) * area + builder.volume1d(y1, one)
    dom = builder.domain
    rain = float(b["rain_mmh"].sum()) / 1000.0 / 3600.0 * (b["t_end"] / len(b["rain_mmh"])) * dom.nx * dom.ny * dom.dx ** 2
    net = rain + float(vols[2]) - float(vols[0]) - float(vols[1])
    d_land = float(v_land[-1] - v_land[0])
    if d_land <= 0:
        raise ValueError(f"predicted land volume does not grow over the storm (dV = {d_land:.3g} m^3): "
                         "a volume rescale is undefined")
    return (net - float(v_rest[-1] - v_rest[0])) / d_land


def run_c5(cfg, device=None) -> dict:
    """Measure the correction on the Phase 1 model's test storms before enabling it: depth
    accuracy, effects, and both mass measures (the implied error, which the correction
    zeroes by construction, and the depth field against the engine's true sources)."""
    from .data import sampler as SMP
    from .data.dataset import SimDataset
    from .data.generate import coarsen_mean
    from .evaluate import site_from_sample
    from .evaluate_effects import _engine_peak, effect_metrics
    from .model.batch import SampleBuilder
    from .model.geokan_pino import model_dir
    ctx = api.configure(cfg, device)
    dom = ctx.domain
    model = ctx.model()
    builder = SampleBuilder.from_context(ctx, model)
    f = builder.f
    land = ~(dom.sea | dom.channel)
    land_c = torch.as_tensor((coarsen_mean(land.astype(float), f) > 0.5).reshape(-1), device=builder.device)
    test = SimDataset(api.dataset_dir(cfg), "test")
    rows = []
    for sid in test.ids:
        rec = test.load(sid)
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, "test")
        site, scen = site_from_sample(smp, dom, ctx)
        times = np.asarray(rec["times"], np.float64)
        h, y1, vols, b = decode_coarse(model, builder, site, scen, times)
        s = volume_scale(h, y1, vols, b, builder, land_c)
        hc = h * torch.where(land_c, s, 1.0)
        h_e = torch.as_tensor(np.asarray(rec["h"]), device=builder.device).reshape(len(times), -1)
        L = land_c[None].expand_as(h_e)
        wet = L & (h_e > 0.05)
        row = {"sim": sid, "scale": s}
        pieces_eng = mass_decomposition({"dV": 0, "rain": 0, "infil": 0, "stored": 0, "bnd_net": 0}, rec)["engine"]
        for tag, hh in (("before", h), ("after", hc)):
            pk = builder.upsample(hh.reshape(len(times), builder.graph.ny, builder.graph.nx)).amax(0).cpu().numpy()
            pk_e = np.asarray(rec["depth_max_full"])
            either = land & ((pk_e > 0.05) | (pk > 0.05))
            one = SampleBuilderSelect(builder, b)
            v = hh.sum(-1) * builder.cell_c ** 2 + builder.volume1d(y1, one)
            pred = {"dV": float(v[-1] - v[0]), "rain": float(b["rain_mmh"].sum()) / 1000.0 / 3600.0
                    * (b["t_end"] / len(b["rain_mmh"])) * dom.nx * dom.ny * dom.dx ** 2,
                    "infil": float(vols[0]), "stored": float(vols[1]), "bnd_net": float(vols[2])}
            md = mass_decomposition(pred, rec)
            from .evaluate import contingency
            row[tag] = {"depth_rmse_wet_m": float(((hh - h_e)[wet] ** 2).mean().sqrt()),
                        "depth_rmse_land_m": float(((hh - h_e)[L] ** 2).mean().sqrt()),
                        "peak_rmse_m": float(np.sqrt(np.mean((pk - pk_e)[either] ** 2))),
                        "csi_0.3": contingency(pk, pk_e, 0.3, land)["csi"],
                        "implied_mass_err": md["reported"], "field_vs_true_sources": md["field_vs_true_sources"],
                        "_pk": pk}
        if not smp.baseline:
            base = api.baseline_site(scen)
            pk_e0, _ = _engine_peak(base, scen, api.engine_cache_dir(cfg) / "effects" / f"test_base_{sid}.pt")
            h0, y10, vols0, b0 = decode_coarse(model, builder, base, scen, times)
            s0 = volume_scale(h0, y10, vols0, b0, builder, land_c)
            for tag, hh0 in (("before", h0), ("after", h0 * torch.where(land_c, s0, 1.0))):
                pk0 = builder.upsample(hh0.reshape(len(times), builder.graph.ny, builder.graph.nx)).amax(0).cpu().numpy()
                pk = row[tag]["_pk"]
                em = effect_metrics(np.asarray(rec["depth_max_full"]) - pk_e0, pk - pk0, np.asarray(rec["depth_max_full"]),
                                    pk_e0, pk, pk0, land, dom.dx)
                row[tag].update({k: em[k] for k in ("effect_rmse_m", "sign_agreement", "effect_corr")})
            row["scale_baseline"] = s0
        for tag in ("before", "after"):
            row[tag].pop("_pk")
        rows.append(row)
        log.info("C5 sim %d: scale %.3f; %s", sid, s, {t: {k: round(v, 4) for k, v in row[t].items()}
                                                      for t in ("before", "after")})
    pooled = {tag: {k: float(np.nanmean([r[tag][k] for r in rows if k in r[tag]]))
                    for k in rows[0]["before"] if k not in ("sim",)} for tag in ("before", "after")}
    for tag in ("before", "after"):
        for k in ("effect_rmse_m", "sign_agreement", "effect_corr"):
            vals = [r[tag][k] for r in rows if k in r[tag]]
            pooled[tag][k] = float(np.nanmean(vals)) if vals else float("nan")
    worse = pooled["after"]["depth_rmse_wet_m"] / pooled["before"]["depth_rmse_wet_m"] - 1.0
    res = {"rows": rows, "pooled": pooled, "depth_rmse_wet_change": worse,
           "enable": bool(worse <= 0.20),
           "rule": "enable the correction unless it worsens the pooled wet-cell depth RMSE by more than 20%"}
    _save(cfg, "c5", res, ctx)
    return res


# ---------------------------------------------------------------------------
# C3: what does a candidate cost when stage 1 is cached?
# ---------------------------------------------------------------------------
def run_c3_speed(cfg, device=None, sizes=(1, 8, 32)) -> dict:
    """Per-candidate inference cost of the two-stage model, with stage 1 encoded once per
    scenario as Part B would use it. Weights are untrained (this measures the architecture,
    not accuracy), so it can be run before E."""
    from .data import sampler as SMP
    from .evaluate import site_from_sample
    from .model.batch import SampleBuilder, build_model
    from .model.two_stage import TwoStage
    import copy
    two_cfg = copy.deepcopy(cfg)
    two_cfg.model.two_stage = True
    ctx = api.configure(two_cfg, device)
    dom = ctx.domain
    base = build_model(two_cfg, api.dataset_dir(cfg), ctx.device)
    model = TwoStage(two_cfg.model, len(base.scales.feat_mean), base.graph, base.scales).to(ctx.device)
    model.eval()
    del base
    builder = SampleBuilder.from_context(ctx, model)
    sites, scen = [], None
    for sid in range(40):
        smp = SMP.draw(sid, dom, cfg.data, cfg.seed, "train")
        site, sc = site_from_sample(smp, dom, ctx)
        if scen is None:
            scen = sc
        if sc == scen:
            sites.append(site)
    if len(sites) < max(sizes):
        sites = (sites * (max(sizes) // max(len(sites), 1) + 1))[:max(sizes)]
    sync = lambda: torch.cuda.synchronize() if ctx.device.type == "cuda" else None
    rows = []
    for n in sizes:
        pop = sites[:n]
        model.predict_many(pop[:1], scen, ctx, None, detail="summary")          # warm-up
        sync()
        t0 = time.perf_counter()
        model._prepare_call(pop, scen, builder)
        sync()
        t_stage1 = time.perf_counter() - t0
        model._release_call()
        sync()
        t0 = time.perf_counter()
        model.predict_many(pop, scen, ctx, None, detail="summary")
        sync()
        total = time.perf_counter() - t0
        rows.append({"n_candidates": n, "total_s": total, "stage1_s": t_stage1,
                     "per_candidate_s": (total - t_stage1) / n, "per_candidate_including_stage1_s": total / n})
        log.info("C3 speed %s", rows[-1])
    res = {"scenario": api._label(scen), "rows": rows, "peak_gpu_mb":
           torch.cuda.max_memory_allocated() / 2 ** 20 if ctx.device.type == "cuda" else None,
           "note": "untrained weights; stage 1 is encoded once per scenario and cached, so the marginal cost of a "
                   "candidate is stage 2's encode plus both decoders"}
    _save(cfg, "c3_speed", res, ctx)
    return res


# ---------------------------------------------------------------------------
def report(cfg) -> Path:
    """Collect every part that has run into artifacts/diagnostics_A.md."""
    from .diagnostics_report import write
    return write(cfg)
