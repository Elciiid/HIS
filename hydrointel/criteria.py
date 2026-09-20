"""Agreement criteria: when do two engine runs count as the same answer?

Every check of the form "run A and run B must agree" -- float32 against float64,
a batched run against the same storms run alone, a regenerated storm against its
cached copy -- is judged by ``agreement`` against ``AGREEMENT`` below.

Why these numbers (fixed 2026-09-20, before any result was seen; do not adjust
them after seeing one):

The quantity the whole system must resolve is an intervention effect: the change
in peak depth a design makes, roughly 20-60 mm. Noise in the training data must
sit several times below the smallest signal the surrogate is asked to learn,
hence 2 mm on the depth RMSE and 10 mm on its 99th percentile. The per-cell
maximum is restricted to cells where both runs are deeper than 0.10 m and held
to a deliberately loose 50 mm: even away from wet/dry fronts, single cells in a
wetting/drying solver are the noisiest statistic available. Areas and volumes are
held to 0.5%, channel peak discharge per reach to 2%.

What they replace: limits on the maximum over all cells (max |dh| < 5 mm, peak
depth difference < 5 mm). A single cell at a moving wet/dry front differs
chaotically between any two slightly different runs, so such a limit measures the
worst front cell, not the run. Measured on Phase 1: the float32 run and a batched
run each moved individual front cells by 20-50 mm while agreeing on flooded area
to within 0.15% (artifacts/overnight_log.md, entry 7).

Definitions (all on the 20 m engine grid):
  wet             h > h_dry (the engine's own wet threshold) at that output time
  depth RMSE      over every (output time, cell) wet in either run
  p99 |dh|        99th percentile of |h_b - h_a| over the same samples
  restricted max  max |h_b - h_a| over (time, cell) where both runs have h > 0.10 m
  flooded area    land cells whose peak depth >= threshold; relative to run A
  flood volume    the largest total surface-water volume on land over the storm
  peak discharge  max over time and nodes of |Q| on each reach; relative to run A
Land excludes the sea and the 1-D channel footprint (water bodies, not exposure).
"""
from __future__ import annotations

import numpy as np

AGREEMENT = {
    "depth_rmse_wet_m": 0.002,
    "depth_p99_abs_m": 0.010,
    "depth_max_abs_both_deep_m": 0.050,
    "flooded_area_rel": 0.005,           # at each of AREA_THRESHOLDS_M
    "flood_volume_rel": 0.005,
    "reach_peak_discharge_rel": 0.02,    # every reach
}
DEEP_M = 0.10
AREA_THRESHOLDS_M = (0.15, 0.30, 0.50)


def agreement(h_a: np.ndarray, h_b: np.ndarray, land: np.ndarray, dx: float, h_dry: float,
              q_a: np.ndarray | None = None, q_b: np.ndarray | None = None,
              reach_of: np.ndarray | None = None) -> dict:
    """Compare run B with reference run A. ``h_*`` are depth series (nt, ny, nx) at the same
    output times; ``q_*`` channel discharge series (nt, n_nodes) with ``reach_of`` the reach
    index of each node. Returns every metric, its limit and whether it passed."""
    h_a = np.asarray(h_a, np.float64)
    h_b = np.asarray(h_b, np.float64)
    if h_a.shape != h_b.shape:
        raise ValueError(f"depth series shapes differ: {h_a.shape} vs {h_b.shape}")
    d = h_b - h_a
    wet = np.maximum(h_a, h_b) > h_dry
    if not wet.any():
        raise ValueError("no wet cells in either run: the comparison would be vacuous")
    deep = np.minimum(h_a, h_b) > DEEP_M
    ad = np.abs(d[wet])
    m = {"depth_rmse_wet_m": float(np.sqrt(np.mean(d[wet] ** 2))),
         "depth_p99_abs_m": float(np.percentile(ad, 99)),
         "depth_max_abs_both_deep_m": float(np.abs(d[deep]).max()) if deep.any() else 0.0,
         "n_wet_samples": int(wet.sum()), "n_deep_samples": int(deep.sum()),
         "depth_max_abs_any_m": float(np.abs(d).max())}             # reported, not judged
    pk_a, pk_b = h_a.max(0), h_b.max(0)
    area = {}
    for t in AREA_THRESHOLDS_M:
        a, b = float(np.sum(land & (pk_a >= t))), float(np.sum(land & (pk_b >= t)))
        rel = (0.0 if b == 0.0 else float("inf")) if a == 0.0 else abs(b - a) / a
        area[f"{t:g}"] = {"a_ha": a * dx * dx / 1e4, "b_ha": b * dx * dx / 1e4, "rel": rel}
    m["flooded_area"] = area
    va = (h_a * land).sum(axis=(-2, -1)).max() * dx * dx
    vb = (h_b * land).sum(axis=(-2, -1)).max() * dx * dx
    m["flood_volume_rel"] = abs(vb - va) / max(va, 1e-9)
    checks = {k: (m[k], AGREEMENT[k]) for k in ("depth_rmse_wet_m", "depth_p99_abs_m", "depth_max_abs_both_deep_m",
                                                "flood_volume_rel")}
    for t, v in area.items():
        checks[f"flooded_area_rel@{t}"] = (v["rel"], AGREEMENT["flooded_area_rel"])
    if q_a is not None:
        qa, qb = np.abs(np.asarray(q_a, np.float64)), np.abs(np.asarray(q_b, np.float64))
        per = {}
        for r in np.unique(reach_of):
            sel = reach_of == r
            pa, pb = float(qa[:, sel].max()), float(qb[:, sel].max())
            per[int(r)] = {"a_m3s": pa, "b_m3s": pb, "rel": abs(pb - pa) / max(pa, 1e-9)}
        m["reach_peak_discharge"] = per
        checks["reach_peak_discharge_rel (worst reach)"] = (max(v["rel"] for v in per.values()),
                                                            AGREEMENT["reach_peak_discharge_rel"])
    m["checks"] = {k: {"value": float(v), "limit": float(lim), "passed": bool(v <= lim)}
                   for k, (v, lim) in checks.items()}
    m["passed"] = all(c["passed"] for c in m["checks"].values())
    return m
