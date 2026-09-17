"""Mass conservation of the coupled engine and of the individual source terms."""
import numpy as np
import pytest
import torch

from hydrointel.config import SolverConfig, quick_config
from hydrointel.domain import landuse as LU
from hydrointel.domain.synthetic import generate
from hydrointel.engine import build_engine, build_forcing
from hydrointel.solver.boundaries import all_edges
from hydrointel.solver.infiltration import Horton
from hydrointel.solver.massbalance import MassBalance
from hydrointel.solver.swe2d import SWE2D

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def quick():
    cfg = quick_config()
    cfg.forcing.storm_duration_h = 1.0
    cfg.forcing.spinup_h = 0.25
    cfg.solver.recession_h = 0.25
    cfg.solver.compile = False
    return cfg, generate(cfg.domain, cfg.seed)


def test_closed_domain_design_storm_conserves_mass(quick):
    cfg, dom = quick
    fo = build_forcing(cfg, 100, "RCP8.5", "SSP5", 2050, True, 0.0, 52.0)
    storage = dom.base_storage() + 0.3 * (dom.landuse == LU.GRASS)
    eng = build_engine(cfg, dom, fo, storage, np.ones(dom.shape), dom.manning(), np.ones(dom.network.n_reaches),
                       device=CPU, closed=True)
    err = eng.run(fo.t_end)
    s = eng.mb.sources()
    assert s["rain"] > 0 and s["infiltration"] > 0 and float(eng.two.storage_volume()) > 0
    assert s["bnd2d_in"] == s["bnd2d_out"] == s["bnd1d_in"] == s["bnd1d_out"] == 0.0
    assert err < 1e-10, err


def test_open_domain_storm_mass_error_below_tolerance(quick):
    cfg, dom = quick
    fo = build_forcing(cfg, 50, "RCP4.5", "SSP2", 2040, True, 2.0, 52.0)
    eng = build_engine(cfg, dom, fo, dom.base_storage(), np.full(dom.shape, 2.0), dom.manning(),
                       np.full(dom.network.n_reaches, 1.3), device=CPU)
    err = eng.run(fo.t_end)
    s = eng.mb.sources()
    assert s["bnd2d_in"] > 0 and s["bnd2d_out"] > 0 and s["bnd1d_in"] > 0
    assert err < cfg.solver.mass_tol
    assert err < 1e-9


def test_mass_balance_raises_when_violated():
    mb = MassBalance(CPU, 100.0)
    mb.add("rain", torch.tensor(10.0, dtype=torch.float64))
    with pytest.raises(AssertionError, match="mass balance violated"):
        mb.assert_ok(150.0, 0.0, 1e-3)
    assert mb.assert_ok(110.0, 0.0, 1e-3) < 1e-12


def test_infiltration_capped_by_available_water():
    cfg = SolverConfig(compile=False)
    z = np.zeros((6, 6))
    h = Horton.constant(1.0, z.shape, CPU, torch.float64)          # absurd 1 m/s potential rate
    sim = SWE2D(z, 0.03, 10.0, cfg, all_edges("closed"), CPU, horton=h)
    sim.set_state(np.full(z.shape, 0.05))
    vol = sim.step(1.0, 0.0, 0.0)
    assert float(sim.h.min()) >= 0.0
    assert abs(float(vol.infil) - 0.05 * 36 * 100.0) < 1e-9


def test_storage_fills_before_runoff_and_never_changes_bed():
    cfg = SolverConfig(compile=False)
    z = np.zeros((5, 5))
    S = np.full(z.shape, 0.02)
    sim = SWE2D(z, 0.03, 10.0, cfg, all_edges("closed"), CPU, storage=S)
    z0 = sim.z.clone()
    rain = 1e-3                                                     # 1 mm/s for 10 s = 10 mm < 20 mm storage
    for k in range(10):
        sim.step(1.0, float(k), rain)
    assert float(sim.h.max()) == 0.0
    assert torch.allclose(sim.s_filled, torch.full_like(sim.s_filled, 0.01))
    for k in range(20):
        sim.step(1.0, float(k), rain)
    assert float(sim.h.min()) > 0.0                                 # storage full, runoff starts
    assert torch.equal(sim.z, z0)
