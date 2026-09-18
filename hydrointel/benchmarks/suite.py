"""Analytical verification suite. Writes artifacts/benchmark_report.md and figures.

Every number in the report is computed here from the solver and the exact
solutions in ``analytical.py``.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig, SolverConfig
from ..solver.boundaries import EdgeBC, all_edges
from ..solver.massbalance import MassBalance
from ..solver.swe2d import SWE2D
from . import analytical as A

log = logging.getLogger("hydrointel.bench")


@dataclass
class BenchResult:
    key: str
    name: str
    passed: bool
    criterion: str
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    seconds: float = 0.0


def run_2d(sim: SWE2D, t_end: float | None = None, n_steps: int | None = None, rain=None,
           mb: MassBalance | None = None, dt_fixed: float | None = None, t0: float = 0.0,
           max_steps: int = 5_000_000):
    t, k = t0, 0
    while True:
        if n_steps is not None and k >= n_steps:
            break
        if t_end is not None and t >= t_end - 1e-12:
            break
        dt = dt_fixed or sim.max_dt()
        if t_end is not None:
            dt = min(dt, t_end - t)
        r = rain(t) if rain else 0.0
        vol = sim.step(dt, t, r)
        if mb is not None:
            mb.add("rain", vol.rain); mb.add("infiltration", vol.infil); mb.add("clip", vol.clip)
            mb.add("bnd2d_in", vol.bnd_in); mb.add("bnd2d_out", vol.bnd_out)
        t += dt
        k += 1
        if k > max_steps:
            raise RuntimeError("run_2d exceeded max_steps")
    return t, k


def _scfg(order=1, compile=False, **kw) -> SolverConfig:
    # small 1-row benchmark grids run eagerly: recompiling per grid costs more than it saves
    s = SolverConfig(spatial_order=order, compile=compile)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _order(errs, ns):
    return [math.log(errs[i] / errs[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(errs) - 1)]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# 1. lake at rest over the irregular synthetic DEM
# ---------------------------------------------------------------------------
def bench_lake_at_rest(dev, fast, out, prov, cfg: RunConfig):
    from ..config import quick_config
    from ..domain.synthetic import generate
    dcfg = quick_config().domain if fast else cfg.domain
    dom = generate(dcfg, cfg.seed)
    eta0 = 3.0
    # well-balancedness can only hold to the working precision's round-off: 1e-10 is a
    # float64 statement. The float32 tolerance is used by the precision study, never by
    # the suite, which runs float64.
    prec = cfg.solver.precision
    tol = {"fp64": 1e-10, "fp32": 1e-5}[prec]
    res = {}
    for order in (1, 2):
        sim = SWE2D(dom.dem, dom.manning(), dom.dx, _scfg(order, compile=True), all_edges("closed"), dev,
                    dtype=cfg.dtype)
        sim.set_level(eta0)
        wet0 = (sim.h > 0)
        mb = MassBalance(dev, float(sim.volume()))
        run_2d(sim, n_steps=500, mb=mb)
        u, v = sim.velocities(sim.h, sim.hu, sim.hv)
        eta = sim.h + sim.z
        res[order] = {
            "max_abs_u": float(u.abs().max()), "max_abs_v": float(v.abs().max()),
            "max_eta_dev": float(((eta - eta0).abs() * wet0).max()),
            "mass_err": mb.check(float(sim.volume()), 0.0),
            "wet_fraction": float(wet0.double().mean()),
        }
    ok = all(r["max_abs_u"] < tol and r["max_abs_v"] < tol and r["max_eta_dev"] < tol for r in res.values())
    m = {f"o{o}_{k}": v for o, r in res.items() for k, v in r.items()}
    return BenchResult("1", "Lake at rest (well-balancedness)", ok,
                       f"max|u|,max|v| < {tol:g} m/s and max|eta-eta0| < {tol:g} m after 500 steps "
                       f"(1st and 2nd order, {prec})",
                       m, [f"grid {dom.nx}x{dom.ny} @ {dom.dx:g} m, eta0 = {eta0} m, partially dry, closed edges"])


# ---------------------------------------------------------------------------
# 2/3. dam breaks with convergence study
# ---------------------------------------------------------------------------
def _dambreak(dev, n, order, hR, t_end=6.0, L=100.0, x0=50.0, hL=1.0, **kw):
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    sim = SWE2D(np.zeros((1, n)), 0.0, dx, _scfg(order, **kw), all_edges("closed", west="transmissive", east="transmissive"), dev)
    sim.set_state(np.where(x < x0, hL, hR)[None, :])
    mb = MassBalance(dev, float(sim.volume()))
    run_2d(sim, t_end=t_end, mb=mb)
    h = sim.h[0].cpu().numpy()
    if hR == 0:
        he, ue = A.ritter(x, t_end, x0, hL)
    else:
        he, ue = A.riemann(x, t_end, x0, hL, 0.0, hR, 0.0)
    l1 = np.abs(h - he).sum() / np.abs(he).sum()
    l2 = np.sqrt(((h - he) ** 2).sum() / (he ** 2).sum())
    return x, h, he, l1, l2, mb.check(float(sim.volume()), t_end)


def bench_dambreak(dev, fast, out, prov, cfg, key, wet: bool):
    hR = 0.1 if wet else 0.0
    ns = [100, 200, 400] if fast else [200, 400, 800, 1600]
    res, profiles = {}, {}
    for order in (1, 2):
        l1s, l2s = [], []
        for n in ns:
            x, h, he, l1, l2, me = _dambreak(dev, n, order, hR)
            l1s.append(l1); l2s.append(l2)
            profiles[(order, n)] = (x, h, he)
        res[order] = {"L1": l1s, "L2": l2s, "order_L1": _order(l1s, ns), "order_L2": _order(l2s, ns), "mass_err": me}
    # scheme-isolated run: dry threshold and KP epsilon lowered so that the
    # operational 1 mm cut-off does not dominate the convergence measurement
    iso = {}
    for order in (1, 2):
        e = [_dambreak(dev, n, order, hR, h_dry=1e-6, kp_eps=1e-24)[3] for n in ns]
        iso[order] = {"L1": e, "order_L1": _order(e, ns)}
    mono = all(all(np.diff(res[o]["L1"]) < 0) for o in (1, 2)) and res[2]["L1"][-1] < res[1]["L1"][-1]
    i1, i2 = iso[1]["order_L1"][-1], iso[2]["order_L1"][-1]
    thr1, thr2 = (0.7, 0.9) if wet else (0.6, 0.9)
    ok = mono and i1 >= thr1 and i2 >= thr2
    crit = (f"operational thresholds (h_dry=1e-3): L1 error decreases monotonically under refinement for both "
            f"orders and MUSCL beats 1st order at the finest grid; scheme-isolated (h_dry=1e-6, KP eps=1e-24): "
            f"observed L1 order >= {thr1} (1st) and >= {thr2} (MUSCL). The spec's 0.8 / 1.5 thresholds are "
            f"enforced on the smooth problem (2b): this solution contains "
            + ("a shock, which caps any scheme's L1 order at 1" if wet else
               "rarefaction kinks and a dry front with sqrt-type velocity, which cap attainable L1 order near 1"))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for order, st in ((1, "-"), (2, "--")):
        x, h, he = profiles[(order, ns[-1])]
        ax[0].plot(x, h, st, label=f"solver order {order}, N={ns[-1]}")
    ax[0].plot(x, he, "k:", lw=1.5, label="exact")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("h [m]"); ax[0].legend(); ax[0].set_title("profile at t = 6 s")
    for order, mk in ((1, "o-"), (2, "s-")):
        ax[1].loglog([100 / n for n in ns], res[order]["L1"], mk, label=f"order {order}: p={res[order]['order_L1'][-1]:.2f}")
    ax[1].set_xlabel("dx [m]"); ax[1].set_ylabel("relative L1 error"); ax[1].legend(); ax[1].grid(True, which="both", alpha=.3)
    from ..viz.provenance import savefig
    name = "stoker" if wet else "ritter"
    f = savefig(fig, Path(out) / "figures" / f"bench_{name}.png", prov)
    m = {"grids": ns}
    for o in (1, 2):
        for k, v in res[o].items():
            m[f"o{o}_{k}"] = v
        for k, v in iso[o].items():
            m[f"isolated_o{o}_{k}"] = v
    return BenchResult(key, "Stoker dam-break (wet bed)" if wet else "Ritter dam-break (dry bed)", ok, crit, m,
                       ["domain 100 m, dam at 50 m, h_L = 1 m" + (", h_R = 0.1 m" if wet else ", dry right"),
                        "frictionless, t = 6 s, strip of 1 row with closed side walls"], [str(f)])


# ---------------------------------------------------------------------------
# 2b. smooth-solution convergence (formal order of accuracy)
# ---------------------------------------------------------------------------
def _hump(dev, n, order, t_end=3.0, L=100.0):
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    sim = SWE2D(np.zeros((1, n)), 0.0, dx, _scfg(order), all_edges("closed"), dev)
    sim.set_state((1.0 + 0.05 * np.exp(-((x - 50.0) / 5.0) ** 2))[None, :])
    run_2d(sim, t_end=t_end)
    return sim.h[0].cpu().numpy()


def bench_smooth(dev, fast, out, prov, cfg):
    ns = [100, 200, 400] if fast else [100, 200, 400, 800]
    nref = 3200 if fast else 6400
    ref = _hump(dev, nref, 2)
    res = {}
    for order in (1, 2):
        errs = []
        for n in ns:
            h = _hump(dev, n, order)
            r = ref.reshape(n, nref // n).mean(axis=1)          # reference cell averages
            errs.append(float(np.abs(h - r).sum() / np.abs(r - 1.0).sum()))
        res[order] = {"L1": errs, "order_L1": _order(errs, ns)}
    ok = res[1]["order_L1"][-1] >= 0.8 and res[2]["order_L1"][-1] >= 1.5
    m = {"grids": ns, "reference_grid": nref}
    for o in (1, 2):
        for k, v in res[o].items():
            m[f"o{o}_{k}"] = v
    return BenchResult("2b", "Smooth-wave self-convergence (formal order)", ok,
                       "observed L1 order >= 0.8 (1st order) and >= 1.5 (MUSCL) on a smooth solution, finest pair",
                       m, [f"Gaussian hump (5 cm on 1 m) in a 100 m closed channel, t = 3 s (before shock formation); "
                           f"errors against cell averages of a {nref}-cell MUSCL reference, normalised by the "
                           "perturbation magnitude",
                           "This is where the spec's 0.8 / 1.5 order thresholds are mathematically meaningful; "
                           "the dam-break solutions contain kinks, a dry front or a shock that cap attainable "
                           "L1 order near 1 for any scheme."])


# ---------------------------------------------------------------------------
# 4. Thacker planar oscillation
# ---------------------------------------------------------------------------
def _thacker(dev, n, order, L, a, h0, eta, periods=3, compile=False, **kw):
    dx = L / n
    xc = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc)
    h0a, u0, v0, z, T = A.thacker_planar(X, Y, 0.0, a, h0, eta, L)
    sim = SWE2D(z, 0.0, dx, _scfg(order, compile=compile, **kw), all_edges("closed"), dev)
    sim.set_state(h0a, h0a * u0, h0a * v0)
    mb = MassBalance(dev, float(sim.volume()))
    v0 = float(sim.volume())
    errs = []
    for p in range(1, periods + 1):
        run_2d(sim, t_end=p * T, t0=(p - 1) * T, mb=mb)
        he = A.thacker_planar(X, Y, p * T, a, h0, eta, L)[0]
        h = sim.h.cpu().numpy()
        errs.append(float(np.sqrt(((h - he) ** 2).sum() / (he ** 2).sum())))
    prof = (xc, h[n // 2], he[n // 2], z[n // 2])
    return {"n": n, "dx_m": dx, "period_s": float(T), "relL2_per_period": errs,
            # closed domain, no sources: drift relative to the initial volume
            "mass_err": abs(float(sim.volume()) - v0) / v0, "clip_m3": mb.sources()["clip"]}, prof


def bench_thacker(dev, fast, out, prov, cfg):
    # Field scale (pass criterion): 2 m deep, 1 km radius bowl on 20 m cells, the
    # operational resolution and dry threshold of the city model.
    field = dict(L=4000.0, a=1000.0, h0=2.0, eta=500.0)
    lab = dict(L=4.0, a=1.0, h0=0.1, eta=0.5)
    res, profs = {}, {}
    d = cfg.solver.spatial_order
    for order in ((d,) if fast else (1, 2)):
        res[f"field_o{order}"], profs[order] = _thacker(dev, 110 if fast else 200, order, compile=True, **field)
    res["lab_o2"], _ = _thacker(dev, 100 if fast else 200, 2, **lab)
    if not fast:
        res["lab_o2_hdry1e-6"], _ = _thacker(dev, 200, 2, h_dry=1e-6, kp_eps=1e-24, **lab)
    r = res[f"field_o{d}"]
    ok = r["relL2_per_period"][-1] < 0.05 and r["mass_err"] < 1e-10
    plt = _plt()
    fig, ax = plt.subplots(1, len(profs), figsize=(6 * len(profs), 4), constrained_layout=True, squeeze=False)
    ax = ax[0]
    for i, o in enumerate(sorted(profs)):
        xc, h, he, z = profs[o]
        ax[i].plot(xc, z, "k", lw=1, label="bed")
        ax[i].plot(xc, np.where(h > 1e-3, h + z, np.nan), "b", label="solver eta")
        ax[i].plot(xc, np.where(he > 0, he + z, np.nan), "r:", label="exact eta")
        ax[i].set_title(f"field scale, order {o}, dx={res[f'field_o{o}']['dx_m']:g} m, centre line after 3 periods")
        ax[i].legend(); ax[i].set_xlabel("x [m]")
    from ..viz.provenance import savefig
    f = savefig(fig, Path(out) / "figures" / "bench_thacker.png", prov)
    m = {f"{k}_{kk}": vv for k, v in res.items() for kk, vv in v.items()}
    return BenchResult("4", "Thacker planar oscillation in a paraboloid", ok,
                       f"field-scale bowl, configured order ({d}): relative L2 error of h < 5% after 3 periods "
                       "and mass drift < 1e-10",
                       m, [f"field scale: a = 1000 m, h0 = 2 m, eta = 500 m, {res[f'field_o{d}']['n']}x"
                           f"{res[f'field_o{d}']['n']} cells (dx = {res[f'field_o{d}']['dx_m']:g} m), "
                           "h_dry = 1 mm, frictionless, closed edges",
                           "lab scale (a = 1 m, h0 = 0.1 m, key lab_o2) is reported for reference: there the "
                           "1 mm dry threshold is 1% of the whole water column and dominates the error; the "
                           "full suite also runs it with h_dry = 1e-6 m (key lab_o2_hdry1e-6) to show that the "
                           "scheme itself converges",
                           "first order is too diffusive for this test (reported above); the engine defaults "
                           "to MUSCL for this reason"], [str(f)])


# ---------------------------------------------------------------------------
# 5. steady flow over a bump
# ---------------------------------------------------------------------------
def _bump(dev, case, n, order, t_end):
    L = 25.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    z = A.bump_bed(x)
    he, q, info = A.bump_exact(x, case)
    # stage BC switches itself to free outflow once the outflow is supercritical
    east = EdgeBC("stage", info.get("h_out", 0.66))
    sim = SWE2D(z[None, :], 0.0, dx, _scfg(order), all_edges("closed", west=EdgeBC("discharge", q), east=east), dev)
    h_init = (info.get("h_out", 0.4)) - z if case != "transcritical" else np.maximum(0.66 - z, 0)
    sim.set_state(np.maximum(h_init, 0)[None, :])
    run_2d(sim, t_end=t_end)
    h = sim.h[0].cpu().numpy()
    hu = sim.hu[0].cpu().numpy()
    l1 = np.abs(h - he).sum() / np.abs(he).sum()
    if "x_shock" in info:      # a captured shock is smeared over a few cells; skip them in the q check
        keep = np.abs(x - info["x_shock"]) > 4 * dx
    else:
        keep = np.ones_like(x, bool)
    return x, z, h, he, l1, float(np.abs(hu - q)[keep].max() / q)


def bench_bump(dev, fast, out, prov, cfg):
    n = 40 if fast else 125
    t_end = 150.0 if fast else 400.0
    tol = {"subcritical": 0.01, "transcritical": 0.02, "shock": 0.05}
    res, ok = {}, True
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, case in zip(axes, tol):
        x, z, h, he, l1, qerr = _bump(dev, case, n, cfg.solver.spatial_order, t_end)
        res[case] = {"relL1_h": float(l1), "max_rel_q_error": qerr}
        ok &= l1 < tol[case] and qerr < 0.05
        ax.plot(x, z, "k", lw=1); ax.plot(x, h + z, "b", label="solver"); ax.plot(x, he + z, "r:", label="exact")
        ax.set_title(f"{case}: L1={l1:.2e}"); ax.set_xlabel("x [m]"); ax.legend()
    from ..viz.provenance import savefig
    f = savefig(fig, Path(out) / "figures" / "bench_bump.png", prov)
    return BenchResult("5", "Steady flow over a bump (sub-, trans-, trans+shock)", bool(ok),
                       "relative L1 error of h below 1% / 2% / 5% and steady discharge within 5% of q",
                       res, [f"25 m channel, N={n}, order {cfg.solver.spatial_order}, frictionless, "
                             f"run to t={t_end:g} s"], [str(f)])


# ---------------------------------------------------------------------------
# 6. rainfall-runoff on a tilted plane
# ---------------------------------------------------------------------------
def _plane(dev, n, order, length=200.0, slope=0.05, nman=0.03, rain=100.0 / 3.6e6, duration=1500.0):
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    z = (length - x) * slope
    sim = SWE2D(z[None, :], nman, dx, _scfg(order), all_edges("closed", east="transmissive"), dev)
    mb = MassBalance(dev, 0.0)
    ts, qs = [], []
    t, vout = 0.0, 0.0
    while t < duration - 1e-9:
        t0 = t
        t, _ = run_2d(sim, t_end=min(t + 10.0, duration), t0=t, rain=lambda _t: rain, mb=mb)
        v = mb.sources()["bnd2d_out"]
        ts.append(t)
        qs.append((v - vout) / ((t - t0) * sim.dy))     # outlet-face discharge per unit width, window mean
        vout = v
    ts, qs = np.array(ts), np.array(qs)
    tm = ts - 5.0                                        # window mid-points
    qe, te = A.kinematic_plane(tm, length, slope, nman, rain, duration)
    q_eq = rain * length
    eq_err = abs(qs[tm > 1.5 * te].mean() - q_eq) / q_eq
    rise = tm < te
    nrmse = float(np.sqrt(np.mean((qs[rise] - qe[rise]) ** 2)) / q_eq)
    return tm, qs, qe, te, float(eq_err), nrmse, mb.check(float(sim.volume() + 0.0), t)


def bench_plane(dev, fast, out, prov, cfg):
    length, slope, nman = 200.0, 0.05, 0.03
    n = 100 if fast else 400
    res, curves = {}, {}
    for order in (1, 2):
        tm, qs, qe, te, eq, nr, me = _plane(dev, n, order)
        res[order] = {"equilibrium_rel_error": eq, "rising_limb_nrmse": nr, "mass_err": me}
        curves[order] = (tm, qs)
    d = cfg.solver.spatial_order
    r = res[d]
    ok = r["equilibrium_rel_error"] < 0.02 and r["rising_limb_nrmse"] < 0.10 and r["mass_err"] < 1e-10
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for o, st in ((1, "c-"), (2, "b-")):
        ax.plot(curves[o][0] / 60, curves[o][1] * 1e3, st, label=f"solver order {o} (outlet face)")
    ax.plot(tm / 60, qe * 1e3, "r:", lw=2, label="kinematic wave (exact)")
    ax.set_xlabel("t [min]"); ax.set_ylabel("q [L/s per m]"); ax.legend()
    from ..viz.provenance import savefig
    f = savefig(fig, Path(out) / "figures" / "bench_plane.png", prov)
    m = {"t_equilibrium_s": float(te), "default_order": d}
    for o in (1, 2):
        for k, v in res[o].items():
            m[f"o{o}_{k}"] = v
    return BenchResult("6", "Rainfall-runoff on a tilted plane (kinematic wave)", ok,
                       f"for the configured order ({d}): equilibrium discharge within 2%, rising-limb RMSE < 10% "
                       "of q_eq, mass error < 1e-10",
                       m, [f"L={length} m, S0={slope}, n={nman}, i=100 mm/h, N={n} (bed step S0*dx = "
                           f"{slope * length / n:.3f} m vs kinematic equilibrium depth "
                           f"{(100.0 / 3.6e6 * length * nman / np.sqrt(slope)) ** 0.6:.4f} m)",
                           "First-order hydrostatic reconstruction under-represents the slope force when the "
                           "bed step between cells exceeds the flow depth (thin rain-on-grid sheets on steep "
                           "cells); the MUSCL variant reconstructs the bed linearly and removes most of this. "
                           "Order-1 numbers are reported for transparency."], [str(f)])


# ---------------------------------------------------------------------------
# 7. 1-D/2-D coupling mass closure
# ---------------------------------------------------------------------------
def _overtopping_case(dev, dx=20.0, compile=False):
    from ..domain.channels import ChannelNetwork, Reach
    from ..solver.coupling import CoupledEngine
    from ..solver.swe1d import SWE1D
    lx, ly = 2000.0, 1000.0
    nx, ny = int(lx / dx), int(ly / dx)
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    yc = 500.0
    bank = 5.0 - 0.001 * X                           # floodplain/bank falls 2 m over the reach
    z = bank + 0.002 * np.abs(Y - yc)                # floodplain rises gently away from the channel
    xs = np.arange(20.0, lx, 40.0)
    zb = 5.0 - 0.001 * xs
    r = Reach("ch", xs, np.full(len(xs), yc), zb - 2.0, np.full(len(xs), 15.0), np.full(len(xs), 1.0),
              np.full(len(xs), 0.03), zb, ("inflow",), ("stage",))
    net = ChannelNetwork([r], [], "ch")
    peak, t_peak, t_base = 180.0, 1800.0, 5400.0

    def q(t):
        return 5.0 + (peak * max(0.0, 1.0 - abs(t - t_peak) / t_peak) if t < t_base else 0.0)
    cfg = _scfg(2, compile=compile)
    one = SWE1D(net, cfg, dev, inflow=q, stage=lambda t: float(zb[-1] - 1.0))
    one.set_level(one.z + 0.8)
    two = SWE2D(z, 0.04, dx, cfg, all_edges("closed"), dev)
    return CoupledEngine(two, one, cfg), one, two


def bench_coupling(dev, fast, out, prov, cfg):
    from ..solver.coupling import exchange
    dx = 40.0 if fast else 20.0
    t_run = (2.0 if fast else 3.0) * 3600.0
    # (a) exchange term alone: random states on the same link layout
    eng, one, two = _overtopping_case(dev, dx)
    g = torch.Generator(device="cpu").manual_seed(7)
    worst = 0.0
    negative = False
    for _ in range(50):
        two.set_state(torch.rand(two.z.shape, generator=g, dtype=torch.float64).to(dev) * 1.5)
        one.A = one.sec.area(torch.rand(one.z.shape, generator=g, dtype=torch.float64).to(dev) * 4.0)
        v1, v2 = float(one.volume()), float(two.volume())
        V = exchange(one, two, eng.links, 5.0, eng.cfg)
        d1, d2 = float(one.volume()) - v1, float(two.volume()) - v2
        scale = max(float(V.abs().sum()), 1e-30)
        worst = max(worst, abs(d1 + d2) / scale)
        negative |= float(two.h.min()) < 0.0 or float(one.A.min()) < 0.0
    # (b) full overtopping event with return flow
    eng, one, two = _overtopping_case(dev, dx)
    series = []

    def snap(s):
        series.append((s.t, float(two.volume()), float(one.volume())))
    err = eng.run(t_run, out_times=np.arange(0, t_run + 1, 300.0), on_snapshot=snap, label="coupling")
    ex_in, ex_out = eng.log.exchanged_in_m3, eng.log.exchanged_out_m3
    ok = worst < 1e-12 and not negative and err < 1e-10 and ex_in > 0 and ex_out > 0
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    s = np.array(series)
    ax.plot(s[:, 0] / 3600, s[:, 1] / 1e3, label="2-D floodplain volume")
    ax.plot(s[:, 0] / 3600, s[:, 2] / 1e3, label="1-D channel volume")
    ax.set_xlabel("t [h]"); ax.set_ylabel("volume [1000 m^3]"); ax.legend()
    from ..viz.provenance import savefig
    f = savefig(fig, Path(out) / "figures" / "bench_coupling.png", prov)
    return BenchResult("7", "1-D/2-D coupling mass closure (overtopping and return flow)", ok,
                       "exchange term alone conserves volume to < 1e-12 of the exchanged volume with no negative "
                       "state; full event mass error < 1e-10; exchange occurred in both directions",
                       {"exchange_only_worst_rel": worst, "exchange_only_negative_state": negative,
                        "event_mass_err": err, "exchanged_2d_to_1d_m3": ex_in, "exchanged_1d_to_2d_m3": ex_out,
                        "peak_2d_volume_m3": float(s[:, 1].max()), "final_2d_volume_m3": float(s[-1, 1])},
                       [f"2 km x 1 km floodplain ({dx:g} m cells), 15 m trapezoidal channel with 2 m banks, "
                        "triangular inflow hydrograph (5 m^3/s base, 180 m^3/s peak at 30 min), "
                        f"downstream stage 1 m above bed, {t_run / 3600:g} h simulated"], [str(f)])


# ---------------------------------------------------------------------------
# 8. mass conservation, closed domain, design storm
# ---------------------------------------------------------------------------
def bench_closed_storm(dev, fast, out, prov, cfg):
    import copy
    from ..config import quick_config
    from ..domain import landuse as LU
    from ..domain.synthetic import generate
    from ..engine import build_engine, build_forcing
    run = quick_config() if fast else copy.deepcopy(cfg)
    run.solver.spatial_order = cfg.solver.spatial_order
    dom = generate(run.domain, cfg.seed)
    area = dom.nx * dom.ny * dom.dx ** 2 / 1e6
    fo = build_forcing(run, 100, "RCP8.5", "SSP5", 2050, True, 0.0, area)
    res = {}
    for prec in (("fp64",) if fast else ("fp64", "fp32")):
        run.solver.precision = prec
        storage = dom.base_storage() + 0.3 * (dom.landuse == LU.GRASS)     # park retention exercises S
        eng = build_engine(run, dom, fo, storage, np.ones(dom.shape), dom.manning(),
                           np.ones(dom.network.n_reaches), device=dev, closed=True)
        err = eng.run(fo.t_end, label=f"closed storm {prec}")
        s = eng.mb.sources()
        res[prec] = {"mass_err": err, "rain_m3": s["rain"], "infiltration_m3": s["infiltration"],
                     "stored_m3": float(eng.two.storage_volume()), "exchanged_2d_to_1d_m3": eng.log.exchanged_in_m3,
                     "exchanged_1d_to_2d_m3": eng.log.exchanged_out_m3, "steps": eng.log.steps,
                     "wall_s": eng.log.wall_s, "clip_m3": s["clip"]}
    ok = all(r["mass_err"] < 1e-3 for r in res.values()) and res["fp64"]["rain_m3"] > 0
    m = {f"{p}_{k}": v for p, r in res.items() for k, v in r.items()}
    return BenchResult("8", "Mass conservation, closed domain, design storm (coupled, infiltration, storage)", ok,
                       "relative mass error < 1e-3 at the end of the run (fp64; the full suite also reports fp32)",
                       m, [f"{dom.nx}x{dom.ny} @ {dom.dx:g} m synthetic domain, all 2-D edges and 1-D ends closed, "
                           f"100-yr {run.forcing.storm_duration_h:g} h storm x RCP8.5 factor, Horton infiltration, "
                           "bund and park retention"])


# ---------------------------------------------------------------------------
# suite driver
# ---------------------------------------------------------------------------
def _bench_ritter(*a):
    return bench_dambreak(*a, "2", False)


def _bench_stoker(*a):
    return bench_dambreak(*a, "3", True)


# (key, function, runs on the configured device?) — 1-row strips and the small coupling
# case run on the CPU, where tiny tensors are faster than GPU kernel launches
ALL = [("1", bench_lake_at_rest, True), ("2", _bench_ritter, False), ("2b", bench_smooth, False),
       ("3", _bench_stoker, False), ("4", bench_thacker, True), ("5", bench_bump, False),
       ("6", bench_plane, False), ("7", bench_coupling, False), ("8", bench_closed_storm, True)]


def run_suite(cfg: RunConfig, fast: bool = False, only=None, device=None) -> list[BenchResult]:
    from ..viz.provenance import DataProvenance
    dev = device or cfg.resolved_device()
    out = Path(cfg.outdir)
    prov = DataProvenance("SYNTHETIC", cfg.config_hash("solver", "domain", "forcing"), cfg.seed,
                          device=str(dev), precision=cfg.solver.precision, engine="engine",
                          notes=("analytical verification suite",))
    results = []
    for key, fn, on_device in ALL:
        if only and key not in only:
            continue
        t = time.perf_counter()
        try:
            r = fn(dev if on_device else torch.device("cpu"), fast, out, prov, cfg)
        except Exception as e:           # a crashing benchmark is reported as a failure, never skipped
            log.exception("benchmark %s crashed", key)
            r = BenchResult(key, f"benchmark {key}", False, "runs to completion", {}, [f"CRASHED: {e!r}"])
        r.seconds = time.perf_counter() - t
        log.info("benchmark %s %s (%.1fs)", key, "PASS" if r.passed else "FAIL", r.seconds)
        results.append(r)
    if not only:
        write_report(results, out, prov, fast)
    return results


def write_report(results, out: Path, prov, fast: bool) -> Path:
    from ..viz.provenance import markdown_header, write_json
    s = markdown_header(prov, "Benchmark report: analytical verification of the flood engine")
    npass = sum(r.passed for r in results)
    s += f"**Result: {npass}/{len(results)} passed** ({'fast subset, coarse grids' if fast else 'full suite'})\n\n"
    if npass < len(results):
        s += "> **FAILED benchmarks present. The engine must not be used to generate training data.**\n\n"
    s += ("Benchmark 2b (smooth-solution convergence) is an addition to the eight required tests: it is where "
          "the 0.8 / 1.5 order-of-accuracy thresholds are applied, because the dam-break solutions (2, 3) "
          "contain kinks, a dry front or a shock that limit any scheme's L1 order to about 1.\n\n")
    s += "| # | benchmark | result | criterion | time |\n|---|---|---|---|---|\n"
    for r in results:
        s += f"| {r.key} | {r.name} | {'PASS' if r.passed else '**FAIL**'} | {r.criterion} | {r.seconds:.0f} s |\n"
    s += "\n## Details\n\n"
    for r in results:
        s += f"### {r.key}. {r.name}\n\n"
        for n in r.notes:
            s += f"- {n}\n"
        s += "\n| metric | value |\n|---|---|\n"
        for k, v in r.metrics.items():
            s += f"| {k} | {_fmt(v)} |\n"
        for f in r.figures:
            try:
                rel = Path(f).resolve().relative_to(out.resolve()).as_posix()
            except ValueError:
                rel = Path(f).as_posix()
            s += f"\n![{r.key}]({rel})\n"
        s += "\n"
    path = out / "benchmark_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")
    write_json(out / "benchmark_results.json", {"fast": fast, "results": [r.__dict__ for r in results]}, prov)
    return path


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def banner(results) -> str:
    bad = [r.key for r in results if not r.passed]
    line = "=" * 72
    if bad:
        return f"{line}\n BENCHMARKS FAILED: {', '.join(bad)}\n DO NOT USE THE ENGINE FOR DATASET GENERATION\n{line}"
    return f"{line}\n ALL {len(results)} BENCHMARKS PASSED\n{line}"
