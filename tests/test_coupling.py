"""1-D <-> 2-D exchange: exact conservation, weir formulas, limiter."""
import numpy as np
import torch

from hydrointel.benchmarks.suite import _overtopping_case
from hydrointel.solver.coupling import exchange

CPU = torch.device("cpu")


def test_exchange_term_conserves_volume_to_machine_precision():
    eng, one, two = _overtopping_case(CPU)
    g = torch.Generator().manual_seed(3)
    for _ in range(100):
        two.set_state(torch.rand(two.z.shape, generator=g, dtype=torch.float64) * 1.5)
        one.A = one.sec.area(torch.rand(one.z.shape, generator=g, dtype=torch.float64) * 4.0)
        v1, v2 = float(one.volume()), float(two.volume())
        V = exchange(one, two, eng.links, 5.0, eng.cfg)
        d1, d2 = float(one.volume()) - v1, float(two.volume()) - v2
        assert abs(d1 + d2) <= 1e-12 * max(float(V.abs().sum()), 1.0)
        assert abs(d1 - float(V.sum())) <= 1e-9 * max(float(V.abs().sum()), 1.0)
        assert float(two.h.min()) >= 0.0 and float(one.A.min()) >= 0.0


def _single_link_state(eng, one, two, eta2, eta1):
    c1 = int(eng.links.cell1d[5])
    z2 = float(eng.links.z2[5])
    zb = float(eng.links.zbank[5])
    two.set_state(torch.zeros_like(two.z))
    h = two.h.reshape(-1).clone()
    h[int(eng.links.uniq_dev[eng.links.inv[5]])] = max(eta2 - z2, 0.0)
    two.set_state(h.reshape(two.z.shape))
    y = torch.clamp(torch.as_tensor(eta1, dtype=torch.float64) - one.z, min=0.0)
    one.A = torch.zeros_like(one.A)
    one.A[c1] = one.sec.area(y)[c1]
    return c1, zb


def test_free_weir_matches_formula_when_unlimited():
    eng, one, two = _overtopping_case(CPU)
    cfg = eng.cfg
    cfg.exchange_relax = 1e9                      # disable limiters for the formula check
    cfg.exchange_limit = 0.5
    c1, zb = _single_link_state(eng, one, two, eta2=0.0, eta1=0.0)
    zb = float(eng.links.zbank[5])
    _single_link_state(eng, one, two, eta2=zb + 0.05, eta1=zb - 1.0)
    dt = 1e-3
    V = exchange(one, two, eng.links, dt, cfg)
    L = float(eng.links.length[5])
    expected = cfg.weir_cw * L * 0.05 ** 1.5 * dt
    assert abs(float(V[5]) - expected) < 1e-9 * expected + 1e-15


def test_villemonte_submergence_reduces_flow():
    eng, one, two = _overtopping_case(CPU)
    cfg = eng.cfg
    cfg.exchange_relax = 1e9
    zb = float(eng.links.zbank[5])
    dt = 1e-3
    _single_link_state(eng, one, two, eta2=zb + 0.20, eta1=zb + 0.10)
    V = float(exchange(one, two, eng.links, dt, cfg)[5])
    L = float(eng.links.length[5])
    free = cfg.weir_cw * L * 0.20 ** 1.5 * dt
    expected = free * (1 - (0.10 / 0.20) ** 1.5) ** 0.385
    assert abs(V - expected) < 1e-9 * expected
    # reversed levels reverse the sign
    _single_link_state(eng, one, two, eta2=zb + 0.10, eta1=zb + 0.20)
    assert float(exchange(one, two, eng.links, dt, cfg)[5]) < 0


def test_limiter_never_overdraws_donor():
    eng, one, two = _overtopping_case(CPU)
    zb = float(eng.links.zbank[5])
    _single_link_state(eng, one, two, eta2=zb + 3.0, eta1=zb - 1.5)
    h_before = two.h.clone()
    exchange(one, two, eng.links, 1e4, eng.cfg)   # absurdly long step
    idx = eng.links.uniq_dev[eng.links.inv[5]]
    assert float(two.h.reshape(-1)[idx]) >= 0.5 * float(h_before.reshape(-1)[idx]) - 1e-12
