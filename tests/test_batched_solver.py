"""Batched dataset generation: does advancing B storms together change the answer?

Two separate questions, deliberately tested apart:

1. Does the batching itself introduce error? Driven through an identical forced
   step sequence, a batched run and a sequential one must agree to floating-point
   round-off. Any larger difference is a bug in the batched indexing, not a
   modelling compromise.
2. What does the shared time step cost? Left to choose their own steps, the
   members of a batch all advance on the smallest step any of them needs, so the
   batched trajectory differs from the sequential one. This bounds that
   difference, and prints the measured figure.
"""
import math

import numpy as np
import pytest
import torch

from hydrointel.config import quick_config
from hydrointel.data import sampler as SMP
from hydrointel.domain.synthetic import generate
from hydrointel.engine import build_engine, build_engine_batch, build_forcing
from hydrointel.forcing import scenarios as SC

CPU = torch.device("cpu")
B = 3
# The requirement was that a shared time step move no peak depth by more than 1 mm.
# It is NOT met. Measured on this 80 m smoke grid: 1.09 mm. Measured at the 20 m
# production resolution over a whole storm (artifacts/batch_throughput.md): 115 and
# 129 mm. That is why `data.batch` defaults to 1 and batching is off in production.
# The assertion below is a regression guard on the documented cost, not a claim that
# the requirement holds.
PEAK_DEPTH_REQUIREMENT_M = 1e-3
PEAK_DEPTH_REGRESSION_M = 2e-3


@pytest.fixture(scope="module")
def setup():
    cfg = quick_config()
    cfg.forcing.storm_duration_h = 0.5
    cfg.forcing.spinup_h = 0.1
    cfg.solver.recession_h = 0.2
    cfg.solver.compile = False
    dom = generate(cfg.domain, cfg.seed)
    area = dom.nx * dom.ny * dom.dx ** 2 / 1e6
    mem = []
    for i in range(B):
        smp = SMP.draw(i, dom, cfg.data, cfg.seed, "train")
        lu = SC.expand_urban(dom.landuse, dom.road, dom.urban_core, dom.dx,
                             SC.ssp_state(smp.ssp).urban_expansion_rate, smp.horizon_year - 2025,
                             protected=dom.sea | dom.channel)
        S, k, n, g, _, _ = SMP.materialise(smp, dom, lu)
        fo = build_forcing(cfg, smp.return_period_yr, smp.rcp, smp.ssp, smp.horizon_year, smp.tide_on,
                           smp.storm_tide_offset_h, area)
        mem.append({"smp": smp, "lu": lu, "fo": fo, "S": S, "k": k, "n": n, "g": g})
    return cfg, dom, mem


def _batched(cfg, dom, mem):
    return build_engine_batch(cfg, dom, [m["fo"] for m in mem], [m["S"] for m in mem], [m["k"] for m in mem],
                              [m["n"] for m in mem], [m["g"] for m in mem], [m["lu"] for m in mem], device=CPU)


def _sequential(cfg, dom, mem):
    return [build_engine(cfg, dom, m["fo"], m["S"], m["k"], m["n"], m["g"], landuse=m["lu"], device=CPU)
            for m in mem]


def test_members_are_actually_different(setup):
    """The comparison would be vacuous if the batch members were the same storm."""
    _, _, mem = setup
    rains = {m["fo"].rain.total_mm for m in mem}
    assert len(rains) == B, f"batch members share a hyetograph: {rains}"
    assert len({tuple(np.round(m["g"], 6)) for m in mem}) > 1 or len({m["smp"].tide_on for m in mem}) > 1


def test_batching_alone_is_exact(setup):
    """Same forced step sequence: batched == sequential to round-off."""
    cfg, dom, mem = setup
    eb, bf = _batched(cfg, dom, mem)
    es = _sequential(cfg, dom, mem)
    t, t_end, n = 0.0, bf.t_end, 0
    while t < t_end - 1e-9 and n < 300:
        dt = min(eb.two.max_dt(), t_end - t)
        n_sub = max(1, math.ceil(dt / eb.one.max_dt() - 1e-9))
        eb.step(t, t_end, dt=dt, n_sub=n_sub)
        for e in es:
            e.step(t, t_end, dt=dt, n_sub=n_sub)
        t += dt
        n += 1
    assert n > 50, "the test did not advance far enough to be meaningful"
    scale = float(eb.two.h.abs().max())
    for b in range(B):
        dh = float((eb.two.h[b] - es[b].two.h).abs().max())
        dA = float((eb.one.A[b] - es[b].one.A).abs().max())
        assert dh <= 1e-12 * max(scale, 1.0), f"member {b}: batched 2-D state differs by {dh:.3e} m"
        assert dA <= 1e-9 * max(float(eb.one.A.abs().max()), 1.0), f"member {b}: batched 1-D area differs by {dA:.3e}"
    # every source term is per member and exact under the same step sequence
    src = eb.mb.sources()
    for term in ("rain", "infiltration", "bnd2d_in", "bnd2d_out"):
        assert np.asarray(src[term]).shape == (B,), f"'{term}' is not accounted per member"
        for b in range(B):
            got, want = float(np.asarray(src[term])[b]), es[b].mb.sources()[term]
            assert got == pytest.approx(want, rel=1e-12, abs=1e-9), f"member {b} '{term}' volume differs"


def test_per_member_mass_balance(setup):
    """Accounting is per member, not pooled: one error per storm, each inside tolerance.

    Source volumes are *not* compared against the sequential runs here. Rain and
    infiltration are integrated as rate(t) x dt, and a batched run partitions the
    window differently (shared step), so a piecewise-constant hyetograph integrates
    to a slightly different total -- measured at 4e-4 relative. The exactness test
    above makes that comparison under an identical step sequence, where it holds.
    """
    cfg, dom, mem = setup
    eb, bf = _batched(cfg, dom, mem)
    es = _sequential(cfg, dom, mem)
    err_b = eb.run(bf.t_end, label="batched")
    err_s = np.array([e.run(m["fo"].t_end, label=f"seq {i}") for i, (e, m) in enumerate(zip(es, mem))])
    assert err_b.shape == (B,), f"batched mass error should be one per member, got {err_b!r}"
    assert np.all(err_b < cfg.solver.mass_tol), f"batched mass errors {err_b}"
    assert np.all(err_s < cfg.solver.mass_tol), f"sequential mass errors {err_s}"
    src = eb.mb.sources()
    assert np.asarray(src["rain"]).shape == (B,)
    assert len(set(np.round(np.asarray(src["rain"]), 3))) == B, "members should not share a rainfall volume"
    for b in range(B):
        rain_b, rain_s = float(np.asarray(src["rain"])[b]), es[b].mb.sources()["rain"]
        assert rain_b == pytest.approx(rain_s, rel=2e-3), f"member {b} rainfall volume differs beyond the step-partition effect"


def test_shared_timestep_cost_is_measured(setup):
    """Each engine choosing its own step: how far apart do the peak depths end up?

    This records the cost of the shared time step. It does not pass the 1 mm
    requirement -- see the note on PEAK_DEPTH_REQUIREMENT_M -- and is not treated as
    if it did; production generation runs unbatched because of it.
    """
    cfg, dom, mem = setup
    eb, bf = _batched(cfg, dom, mem)
    es = _sequential(cfg, dom, mem)
    peaks_b = torch.zeros_like(eb.two.h)
    eb.run(bf.t_end, out_times=bf.times(cfg.solver.output_interval_s),
           on_snapshot=lambda s: peaks_b.copy_(torch.maximum(peaks_b, torch.as_tensor(s.h, dtype=peaks_b.dtype))))
    worst = 0.0
    for b, (e, m) in enumerate(zip(es, mem)):
        peak = torch.zeros_like(e.two.h)
        e.run(m["fo"].t_end, out_times=m["fo"].times(cfg.solver.output_interval_s),
              on_snapshot=lambda s, p=peak: p.copy_(torch.maximum(p, torch.as_tensor(s.h, dtype=p.dtype))))
        d = float((peaks_b[b] - peak).abs().max())
        worst = max(worst, d)
        print(f"member {b}: batched steps {eb.log.steps}, sequential steps {e.log.steps}, "
              f"max |peak depth difference| {d * 1e3:.4f} mm")
    print(f"worst peak-depth difference from the shared time step: {worst * 1e3:.4f} mm "
          f"(requirement {PEAK_DEPTH_REQUIREMENT_M * 1e3:.1f} mm -- "
          f"{'met' if worst < PEAK_DEPTH_REQUIREMENT_M else 'NOT met, which is why data.batch defaults to 1'})")
    assert worst < PEAK_DEPTH_REGRESSION_M, (f"the shared time step moved a peak depth by {worst * 1e3:.3f} mm, "
                                             f"worse than the {PEAK_DEPTH_REGRESSION_M * 1e3:.1f} mm on record")


def test_shared_timestep_agreement_criteria(setup):
    """The shared time step judged by the agreement criteria that replaced the max-over-all-
    cells limits (hydrointel/criteria.py): depth RMSE over wet cells, its 99th percentile, the
    maximum where both runs are deeper than 0.10 m, flooded area, flood volume and per-reach
    peak discharge, at every output time."""
    from hydrointel.criteria import agreement
    cfg, dom, mem = setup
    eb, bf = _batched(cfg, dom, mem)
    es = _sequential(cfg, dom, mem)
    times = bf.times(cfg.solver.output_interval_s / 3)
    hb, qb = [], []
    eb.run(bf.t_end, out_times=times, on_snapshot=lambda s: (hb.append(s.h), qb.append(s.q1)))
    land = ~(dom.sea | dom.channel)
    interior = eb.one.topo.kind <= 1
    reach = eb.one.topo.reach_of[interior]
    for b, (e, m) in enumerate(zip(es, mem)):
        hs, qs = [], []
        e.run(m["fo"].t_end, out_times=times, on_snapshot=lambda s: (hs.append(s.h), qs.append(s.q1)))
        res = agreement(np.stack(hs), np.stack([h[b] for h in hb]), land, dom.dx, cfg.solver.h_dry,
                        np.stack(qs)[:, interior], np.stack([q[b] for q in qb])[:, interior], reach)
        print(f"member {b}: " + ", ".join(f"{k} {c['value']:.3g} ({'ok' if c['passed'] else 'FAIL'})"
                                          for k, c in res["checks"].items()))
        assert res["passed"], f"member {b}: batched run disagrees with the same storm run alone: " + \
            ", ".join(f"{k} {c['value']:.3g} > {c['limit']:g}" for k, c in res["checks"].items() if not c["passed"])


def test_batch_rejects_mismatched_inputs(setup):
    cfg, dom, mem = setup
    with pytest.raises(ValueError, match="disagree in length"):
        build_engine_batch(cfg, dom, [m["fo"] for m in mem], [m["S"] for m in mem[:2]], [m["k"] for m in mem],
                           [m["n"] for m in mem], [m["g"] for m in mem], [m["lu"] for m in mem], device=CPU)
