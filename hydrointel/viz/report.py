"""Scenario figures and the interactive HTML report (engine and/or surrogate)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .. import api
from ..config import HAZARD_CLASSES
from ..forcing.scenarios import PATHWAY_NOTE
from . import basemap as BM
from .depthmap import corner_transects, draw_depth, hazard_cmap, hazard_legend, land_depth, monitoring_points
from .html_report import fig_b64, png_b64, write_html
from .profiles import draw_profiles
from .provenance import savefig, write_csv

log = logging.getLogger("hydrointel.viz")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _tag(s: api.Scenario) -> str:
    return f"rp{s.return_period_yr}_{s.rcp}_{s.ssp}_{s.horizon_year}_tide{'on' if s.tide_on else 'off'}"


def scenario_figures(cfg, scen: api.Scenario, engine: str = "both", site: api.SiteState | None = None) -> dict:
    plt = _plt()
    ctx = api.configure(cfg)
    dom = ctx.domain
    site = site or api.baseline_site(scen)
    results = {}
    if engine in ("simulate", "both"):
        results["engine"] = api.simulate(site, scen)
    if engine in ("predict", "both"):
        try:
            results["surrogate"] = api.predict(site, scen, detail="full")
        except FileNotFoundError as e:
            if engine == "predict":
                raise
            log.warning("surrogate not available (%s); figures show the engine only", e)
    if not results:
        raise RuntimeError("no results to plot")
    out = Path(cfg.outdir) / "figures"
    tag = _tag(scen)
    rgb = BM.render(ctx.landuse(scen.ssp, scen.horizon_year), dom.dem, dom.dx, cfg.seed, dom.building)
    topo = next(iter(results.values())).extras.get("topology")
    if topo is None:
        from ..solver.swe1d import build_topology
        topo = build_topology(dom.network)
    pts = monitoring_points(dom)
    written = {"domain": str(BM.plot_domain(dom, ctx.provenance("engine", ("domain",)), out / "domain_overview.png"))}
    for name, r in results.items():
        prov = r.provenance
        h = r.depth_series.numpy()
        kpk = int(np.argmax(h.reshape(len(h), -1).sum(1)))
        fig = plt.figure(figsize=(15, 17), constrained_layout=True)
        gs = fig.add_gridspec(4, 4, height_ratios=[3.2, 1.1, 1.1, 1.0])
        ax = fig.add_subplot(gs[0, :3])
        draw_depth(ax, dom, r.depth_max.numpy(), rgb, r.u[kpk].numpy(), r.v[kpk].numpy(),
                   title=f"{name}: max depth on land; arrows = velocity at t = {r.times[kpk] / 3600:.2f} h\n"
                         f"RP{scen.return_period_yr} · {scen.rcp}/{scen.ssp} · {scen.horizon_year} · "
                         f"tide {'on' if scen.tide_on else 'off'}")
        for p in pts:
            ax.plot(p["x"] / 1000, p["y"] / 1000, "o", mfc="w", mec="k", ms=6)
            ax.annotate(p["name"], (p["x"] / 1000, p["y"] / 1000), xytext=(4, 4), textcoords="offset points", fontsize=7)
        lax = fig.add_subplot(gs[0, 3]); lax.axis("off")
        hazard_legend(lax, loc="upper left", anchor=(0.0, 1.0))
        stats = _stats(dom, r)
        lax.text(0.0, 0.35, "\n".join(f"{k}: {v}" for k, v in stats.items()), fontsize=8, va="top", family="monospace")
        a1, a2 = fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[2, :2])
        kq = int(np.argmax(np.abs(r.channel_q.numpy()).max(1)))
        draw_profiles(a1, a2, topo, r.channel_eta[kq].numpy(), r.channel_q[kq].numpy(),
                      f"t = {r.times[kq] / 3600:.2f} h")
        a1.set_title("main river at peak discharge", fontsize=9)
        ah = fig.add_subplot(gs[1:3, 2:])
        for k, p in enumerate(pts):
            c = f"C{k}"
            for nm, rr in results.items():
                ah.plot(rr.times / 3600, rr.depth_series[:, p["j"], p["i"]], "-" if nm == "engine" else "--",
                        color=c, label=f"{p['name']} ({nm})")
        ah.set_xlabel("t [h]"); ah.set_ylabel("depth [m]"); ah.set_title("hydrographs at monitoring points", fontsize=9)
        ah.legend(fontsize=7, ncol=2); ah.grid(alpha=0.3)
        for k, tr in enumerate(corner_transects(dom)):
            axc = fig.add_subplot(gs[3, k])
            axc.plot(tr["dist_m"], r.depth_max.numpy()[tr["j"], tr["i"]], color="#1565c0", label="max depth")
            axg = axc.twinx()
            axg.plot(tr["dist_m"], dom.dem2d[tr["j"], tr["i"]], color="#8d6e63", lw=1)
            axg.set_ylabel("ground [m MSL]", fontsize=7, color="#8d6e63")
            axc.set_title(f"{tr['name']} corner transect", fontsize=9)
            axc.set_xlabel("distance from corner [m]"); axc.set_ylabel("max depth [m]"); axc.grid(alpha=0.3)
        f = savefig(fig, out / f"scenario_{tag}_{name}.png", prov, caption=BM.BASEMAP_CAPTION)
        written[name] = str(f)
        rows = [(p["name"], float(t), float(r.depth_series[k2, p["j"], p["i"]]))
                for p in pts for k2, t in enumerate(r.times.numpy())]
        write_csv(out / f"hydrographs_{tag}_{name}.csv", ["point", "t_s", "depth_m"], rows, prov)
    written["html"] = str(_html(cfg, dom, scen, results, rgb, pts, tag))
    return written


def _stats(dom, r) -> dict:
    land = ~(dom.sea | dom.channel)
    dm = r.depth_max.numpy()
    area = dom.dx ** 2 / 1e6
    out = {}
    for name, lo, hi, _ in HAZARD_CLASSES[1:]:
        out[f"{name} [km²]"] = f"{np.sum(land & (dm >= lo) & (dm < hi)) * area:.2f}"
    out["peak depth on land [m]"] = f"{dm[land].max():.2f}"
    out["infiltrated [10³ m³]"] = f"{r.infiltrated_volume_m3 / 1e3:.0f}"
    out["stored [10³ m³]"] = f"{r.stored_volume_m3 / 1e3:.0f}"
    out["mass-balance error"] = f"{r.mass_balance_error:.2e}"
    out["in training envelope"] = str(r.in_distribution)
    return out


def _masked_rgba(cmap, norm, d):
    rgba = cmap(norm(np.nan_to_num(d)))
    rgba[np.isnan(d)] = 0.0
    return rgba


def _html(cfg, dom, scen, results, rgb, pts, tag):
    plt = _plt()
    cmap, norm = hazard_cmap()
    layers = []
    for k, (name, r) in enumerate(results.items()):
        layers.append({"id": f"depth_{name}", "name": f"max depth ({name})",
                       "png": png_b64(_masked_rgba(cmap, norm, land_depth(dom, r.depth_max.numpy()))),
                       "visible": k == 0, "opacity": 0.9})
    if len(results) == 2:
        d = np.nan_to_num(land_depth(dom, (results["surrogate"].depth_max - results["engine"].depth_max).numpy()))
        lim = max(float(np.percentile(np.abs(d), 99)), 0.05)
        import matplotlib.cm as cm
        rgba = cm.RdBu_r((np.clip(d, -lim, lim) + lim) / (2 * lim))
        rgba[..., 3] = np.where(np.abs(d) > 0.02, 0.85, 0.0)
        layers.append({"id": "diff", "name": f"surrogate − engine (±{lim:.2f} m)", "png": png_b64(rgba),
                       "visible": False, "opacity": 0.9})
    markers = []
    for k, p in enumerate(pts):
        lines = []
        info = {"cell (j, i)": f"{p['j']}, {p['i']}", "ground [m MSL]": f"{dom.dem2d[p['j'], p['i']]:.2f}"}
        for nm, r in results.items():
            lines.append({"label": nm, "x": (r.times / 3600).tolist(),
                          "y": r.depth_series[:, p["j"], p["i"]].tolist(), "color": "#1565c0" if nm == "engine" else "#e65100",
                          "dash": nm != "engine"})
            info[f"max depth ({nm}) [m]"] = f"{float(r.depth_max[p['j'], p['i']]):.2f}"
        markers.append({"x": p["x"], "y": p["y"], "label": p["name"], "info": info,
                        "series": [{"title": "depth [m]", "xlabel": "t [h]", "lines": lines}]})
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.axis("off")
    hazard_legend(ax, loc="upper left", anchor=(0, 1))
    legend = fig_b64(fig)
    base = next(iter(results.values()))
    return write_html(Path(cfg.outdir) / f"report_{tag}.html",
                      f"Hydrointelligence System — RP{scen.return_period_yr} {scen.rcp}/{scen.ssp} {scen.horizon_year}",
                      png_b64(rgb), (dom.nx * dom.dx, dom.ny * dom.dx), (dom.nx, dom.ny), layers, markers, [],
                      base.provenance, legend, {**_stats(dom, base), "results shown": ", ".join(results)},
                      [BM.BASEMAP_CAPTION, "Depths are computed by the engine (ground truth) and, where shown, the "
                       "GeoKAN-PINO surrogate; the difference layer shows surrogate minus engine.",
                       "Depth colours cover land only; the sea and the 1-D channel footprint are water bodies.",
                       PATHWAY_NOTE])
