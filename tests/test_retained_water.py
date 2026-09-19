"""Does reported depth include water held in engineered retention storage?

It must not: the hazard depth is the free water on the ground. Water a retention
store holds is out of the flood, but it must still be counted by the mass balance
and reported as ``stored_volume_m3``.

The engine keeps retained water in its own array (``SWE2D.s_filled``), separate
from the surface depth ``SWE2D.h``; every output (snapshots, FloodResult.depth_series,
depth_max) is built from ``h``. These tests pin that down on a closed flat box where
the answer is known exactly.
"""
from __future__ import annotations

import numpy as np
import torch

from hydrointel.config import SolverConfig
from hydrointel.solver.boundaries import EdgeBC
from hydrointel.solver.coupling import CoupledEngine
from hydrointel.solver.massbalance import MassBalance
from hydrointel.solver.swe2d import SWE2D

RAIN_MS = 50.0 / 3.6e6            # 50 mm/h
DURATION_S = 3600.0               # 50 mm of rain in total
STORE_M = 0.03                    # 30 mm of retention in every cell


def _box(storage):
    cfg = SolverConfig(spatial_order=2, compile=False)
    bcs = {e: EdgeBC("closed") for e in ("west", "east", "south", "north")}
    return SWE2D(np.zeros((8, 8)), 0.03, 10.0, cfg, bcs, torch.device("cpu"), torch.float64, storage=storage)


def _rain(sim, mb=None):
    t = 0.0
    while t < DURATION_S - 1e-9:
        dt = min(sim.max_dt(), 30.0, DURATION_S - t)
        vol = sim.step(dt, t, RAIN_MS)
        if mb is not None:
            mb.add("rain", vol.rain)
        t += dt
    return t


def test_depth_excludes_retained_water():
    sim = _box(np.full((8, 8), STORE_M))
    _rain(sim)
    total = RAIN_MS * DURATION_S                                  # 0.05 m
    # the store fills first and holds exactly its capacity; only the rest is surface water
    assert torch.allclose(sim.s_filled, torch.full_like(sim.s_filled, STORE_M), atol=1e-12)
    assert torch.allclose(sim.h, torch.full_like(sim.h, total - STORE_M), atol=1e-12)


def test_partly_filled_store_leaves_no_surface_water():
    sim = _box(np.full((8, 8), 0.2))                              # bigger than the whole storm
    _rain(sim)
    assert float(sim.h.abs().max()) < 1e-12
    assert torch.allclose(sim.s_filled, torch.full_like(sim.s_filled, RAIN_MS * DURATION_S), atol=1e-12)


def test_retained_water_is_in_the_mass_balance_and_the_stored_volume():
    sim = _box(np.full((8, 8), STORE_M))
    mb = MassBalance(sim.device, 0.0)
    t = _rain(sim, mb)
    rain_v = RAIN_MS * DURATION_S * 64 * sim.area
    assert abs(mb.sources()["rain"] - rain_v) < 1e-9 * rain_v
    assert abs(float(sim.storage_volume()) - STORE_M * 64 * sim.area) < 1e-9
    assert mb.check(float(sim.volume() + sim.storage_volume()), t) < 1e-12
    # counting only the surface would lose exactly the stored volume
    assert mb.check(float(sim.volume()), t) > 0.5


def test_engine_snapshot_reports_surface_depth_only():
    """What simulate() writes to FloodResult.depth_series comes from CoupledEngine.snapshot."""
    sim = _box(np.full((8, 8), STORE_M))
    eng = CoupledEngine(sim, None, sim.cfg, rain=lambda t: RAIN_MS)
    eng.run(DURATION_S, out_times=[DURATION_S], on_snapshot=lambda s: snaps.append(s), progress_s=0)
    snap = snaps[-1]
    assert np.allclose(snap.h, RAIN_MS * DURATION_S - STORE_M, atol=1e-7)
    assert abs(snap.ledger["storage"] - STORE_M * 64 * sim.area) < 1e-9


snaps: list = []
