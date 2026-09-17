"""Fast subset of the analytical benchmark suite (coarse grids, CPU).

The full suite with convergence studies runs with
    python -m hydrointel.cli benchmark
"""
import numpy as np
import pytest
import torch

from hydrointel.benchmarks import analytical as A
from hydrointel.benchmarks import suite as S
from hydrointel.config import RunConfig
from hydrointel.viz.provenance import DataProvenance

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = RunConfig()
    out = tmp_path_factory.mktemp("bench")
    prov = DataProvenance("SYNTHETIC", cfg.config_hash(), cfg.seed)
    return cfg, out, prov


def test_exact_solutions_are_self_consistent():
    x = np.linspace(0, 100, 1001)
    h, u = A.ritter(x, 5.0, 50.0, 1.0)
    assert h[0] == 1.0 and h[-1] == 0.0 and np.all(np.diff(h) <= 1e-12)
    hs, us = A.star_state(1.0, 0.0, 0.1, 0.0)
    # Rankine-Hugoniot on the right shock: mass flux jump equals speed x depth jump
    s = us * hs / (hs - 0.1)
    assert abs((hs * us - 0.0) - s * (hs - 0.1)) < 1e-10
    hb, q, _ = A.bump_exact(x / 4, "subcritical")
    E = q ** 2 / (2 * 9.81 * hb ** 2) + hb + A.bump_bed(x / 4)
    assert np.ptp(E) < 1e-8


def test_1_lake_at_rest(env):
    r = S.bench_lake_at_rest(CPU, True, *env[1:], env[0])
    assert r.passed, r.metrics


def test_2_ritter(env):
    r = S.bench_dambreak(CPU, True, env[1], env[2], env[0], "2", False)
    assert r.passed, r.metrics


def test_2b_smooth_order(env):
    r = S.bench_smooth(CPU, True, env[1], env[2], env[0])
    assert r.passed, r.metrics


def test_3_stoker(env):
    r = S.bench_dambreak(CPU, True, env[1], env[2], env[0], "3", True)
    assert r.passed, r.metrics


def test_4_thacker_one_period_coarse():
    field = dict(L=4000.0, a=1000.0, h0=2.0, eta=500.0)
    m, _ = S._thacker(CPU, 100, 2, periods=1, **field)
    assert m["relL2_per_period"][0] < 0.05 and m["mass_err"] < 1e-10, m


def test_5_bump_subcritical_coarse():
    x, z, h, he, l1, qerr = S._bump(CPU, "subcritical", 50, 2, 200.0)
    assert l1 < 0.02 and qerr < 0.05, (l1, qerr)


def test_6_tilted_plane():
    tm, qs, qe, te, eq, nr, me = S._plane(CPU, 100, 2)
    assert eq < 0.02 and nr < 0.10 and me < 1e-10, (eq, nr, me)


def test_7_coupling_event(env):
    r = S.bench_coupling(CPU, True, env[1], env[2], env[0])
    assert r.passed, r.metrics
