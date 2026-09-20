"""The agreement criteria, the effect loss and the two-stage model's mechanics.

These are unit tests on constructed inputs: they check that each criterion measures what
its docstring says, that the effect loss cannot blow up on a pair with no effect, and that
the two-stage model starts at "this design changes nothing" and adds its delta on top.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hydrointel.criteria import AGREEMENT, DEEP_M, agreement
from hydrointel.train_paired import EFFECT_SCALE_M, effect_loss

H_DRY = 1e-3


def _runs(nt=4, ny=8, nx=8, depth=0.5):
    a = np.full((nt, ny, nx), depth)
    return a, a.copy()


def _land(ny=8, nx=8):
    return np.ones((ny, nx), bool)


def test_identical_runs_agree_everywhere():
    a, b = _runs()
    r = agreement(a, b, _land(), 20.0, H_DRY)
    assert r["passed"]
    assert r["depth_rmse_wet_m"] == 0.0 and r["depth_p99_abs_m"] == 0.0
    assert r["flood_volume_rel"] == 0.0
    assert all(v["rel"] == 0.0 for v in r["flooded_area"].values())


def test_rmse_criterion_bites_at_its_limit():
    a, b = _runs()
    b += AGREEMENT["depth_rmse_wet_m"] * 1.5                      # 3 mm everywhere
    r = agreement(a, b, _land(), 20.0, H_DRY)
    assert not r["checks"]["depth_rmse_wet_m"]["passed"]
    assert r["checks"]["depth_rmse_wet_m"]["value"] == pytest.approx(0.003)


def test_a_single_front_cell_does_not_fail_the_run():
    """The point of the restricted maximum: one shallow cell differing by 40 mm is noise,
    not disagreement, and must not fail a run that matches everywhere else.

    The grid here is 64x64x4 (~16k samples). On a smaller grid a single cell would move the
    RMSE itself -- 40 mm over 256 samples is 2.5 mm -- which says nothing about the criteria
    and everything about the test: the limits are written for the 400x325 engine grid over
    ~35 output times, where one cell contributes ~0.02 mm to the RMSE.
    """
    a, b = _runs(ny=64, nx=64, depth=0.5)
    a[0, 0, 0] = b[0, 0, 0] = 0.0
    b[2, 0, 0] = 0.04                                             # a wet/dry front cell appears
    a[2, 0, 0] = 0.0
    r = agreement(a, b, _land(64, 64), 20.0, H_DRY)
    assert r["depth_max_abs_any_m"] == pytest.approx(0.04)        # reported
    assert r["checks"]["depth_max_abs_both_deep_m"]["value"] == 0.0
    assert r["passed"], r["checks"]


def test_deep_cell_disagreement_is_caught():
    a, b = _runs(depth=0.5)
    b[1, 4, 4] += 0.06                                            # both runs deep here
    r = agreement(a, b, _land(), 20.0, H_DRY)
    assert not r["checks"]["depth_max_abs_both_deep_m"]["passed"]
    assert r["checks"]["depth_max_abs_both_deep_m"]["value"] == pytest.approx(0.06)


def test_deep_mask_needs_both_runs_deep():
    a, b = _runs(depth=DEEP_M / 2)                                # shallower than the deep mask
    b += 0.06
    r = agreement(a, b, _land(), 20.0, H_DRY)
    assert r["n_deep_samples"] == 0
    assert r["checks"]["depth_max_abs_both_deep_m"]["value"] == 0.0
    assert not r["checks"]["depth_rmse_wet_m"]["passed"]           # the RMSE still catches it


def test_area_and_volume_criteria():
    a, b = _runs(depth=0.2)                                       # flooded at 0.15 m, not at 0.30
    b[:, 0, :4] = 0.0                                             # 4 of 64 cells go dry: 6.25%
    r = agreement(a, b, _land(), 20.0, H_DRY)
    assert r["flooded_area"]["0.15"]["rel"] == pytest.approx(4 / 64)
    assert not r["checks"]["flooded_area_rel@0.15"]["passed"]
    assert r["checks"]["flood_volume_rel"]["value"] == pytest.approx(4 / 64)


def test_reach_peak_discharge_criterion():
    a, b = _runs()
    q = np.array([[10.0, 20.0], [12.0, 25.0]])
    r = agreement(a, b, _land(), 20.0, H_DRY, q, q * 1.03, np.array([0, 1]))
    assert not r["checks"]["reach_peak_discharge_rel (worst reach)"]["passed"]
    assert r["reach_peak_discharge"][0]["rel"] == pytest.approx(0.03)


def test_vacuous_comparison_fails_loudly():
    a = np.zeros((2, 4, 4))
    with pytest.raises(ValueError, match="no wet cells"):
        agreement(a, a.copy(), _land(4, 4), 20.0, H_DRY)


# --------------------------------------------------------------------------- effect loss
def test_effect_loss_is_one_when_a_real_effect_is_missed():
    H0 = 1.0
    true = torch.full((4, 100), EFFECT_SCALE_M / H0)              # a uniform 2 cm effect
    assert float(effect_loss(torch.zeros_like(true), true, H0)) == pytest.approx(1.0)


def test_effect_loss_is_scale_free_above_the_floor():
    H0 = 1.0
    for amp in (0.05, 0.5):
        true = torch.full((4, 100), amp)
        pred = true * 0.5                                          # half the effect, whatever its size
        assert float(effect_loss(pred, true, H0)) == pytest.approx(0.25)


def test_effect_loss_stays_finite_with_no_effect_and_penalises_invented_change():
    H0 = 1.0
    true = torch.zeros((4, 100))
    assert float(effect_loss(true, true, H0)) == 0.0
    pred = torch.full_like(true, EFFECT_SCALE_M)
    assert float(effect_loss(pred, true, H0)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- two-stage
@pytest.fixture(scope="module")
def two_stage_setup():
    from hydrointel.config import quick_config, RunConfig
    from hydrointel.data.dataset import Scales
    from hydrointel.domain.synthetic import generate
    from hydrointel.model.graph import build_graph, static_node_features
    from hydrointel.model.two_stage import TwoStage
    from hydrointel.solver.swe1d import build_topology
    cfg = quick_config(RunConfig())
    dom = generate(cfg.domain, cfg.seed)
    feats, _ = static_node_features(dom)
    graph = build_graph(dom.dem2d, dom.dx, build_topology(dom.network))
    n_feat = feats.shape[0] + 3 + 8
    scales = Scales(1.0, 8000.0, 3.1, 2580.0, 100.0, 5.0, 3.0, [0.0] * n_feat, [1.0] * n_feat,
                    [0.0] * 5, [1.0] * 5, 100.0, 50.0)
    return TwoStage(cfg.model, n_feat, graph, scales), graph, n_feat


def test_delta_decoders_start_at_zero(two_stage_setup):
    """A fresh two-stage model predicts the baseline exactly: its delta heads are zero, so
    training starts from 'this design changes nothing' rather than from noise."""
    model, graph, _ = two_stage_setup
    assert float(model.stage2.dec2.net[-1].coef.abs().max()) == 0.0
    assert float(model.stage2.dec1.net[-1].coef.abs().max()) == 0.0
    x = torch.randn(graph.n2, model.stage2.cfg.latent)
    tf = torch.zeros(2, graph.n2, 1)
    b = {"series": torch.zeros(3, 8), "cum_rain": torch.zeros(8), "t_end": 1000.0}
    assert float(model.stage2._dec2(x, b, tf).abs().max()) == 0.0


def test_stage2_inputs_carry_the_design_difference_and_stage1_latent(two_stage_setup):
    model, graph, n_feat = two_stage_setup
    d = model.stage2.cfg.latent
    site = {"feat2": torch.zeros(graph.n2, n_feat), "feat1": torch.zeros(graph.n1, 6)}
    base = {"feat2": torch.zeros(graph.n2, n_feat), "feat1": torch.zeros(graph.n1, 6)}
    site["feat2"][:, 10] = 0.7                                     # storage differs from the baseline
    x2s, x1s = torch.randn(graph.n2, d), torch.randn(graph.n1, d)
    out = model.stage2_inputs(site, base, x2s, x1s)
    assert out["feat2"].shape == (graph.n2, n_feat + 3 + d)
    assert out["feat1"].shape == (graph.n1, 6 + d)
    assert float(out["feat2"][:, n_feat + 1].max()) == pytest.approx(0.7)     # the storage difference
    assert torch.equal(out["feat2"][:, n_feat + 3:], x2s)
    assert not out["feat2"].requires_grad


def test_combine_adds_the_delta(two_stage_setup):
    model, _, _ = two_stage_setup
    ob2, od2 = torch.ones(2, 5, 3), torch.full((2, 5, 3), 0.25)
    ob1, od1 = torch.ones(2, 4, 2), torch.zeros(2, 4, 2)
    o2, o1 = model.combine(ob2, ob1, od2, od1)
    assert torch.allclose(o2, torch.full_like(o2, 1.25))
    assert torch.allclose(o1, ob1)
