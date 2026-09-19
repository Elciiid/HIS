"""The Part B contract: shapes, units, provenance, interchangeability, direction."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hydrointel import api
from hydrointel.config import quick_config
from hydrointel.domain import landuse as LU

CPU = torch.device("cpu")
REPO_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "quick"


def short_cfg(outdir):
    cfg = quick_config()
    cfg.outdir = str(outdir)
    cfg.forcing.storm_duration_h = 1.0
    cfg.forcing.spinup_h = 0.25
    cfg.solver.recession_h = 0.5
    cfg.solver.compile = False
    return cfg


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return api.configure(short_cfg(tmp_path_factory.mktemp("api")), CPU)


SCEN = api.Scenario(100, "RCP8.5", "SSP5", 2050, True, 0.0)


@pytest.fixture(scope="module")
def base_result(ctx):
    return api.simulate(api.baseline_site(SCEN), SCEN)


def test_dataclasses_are_frozen(ctx):
    site = api.baseline_site(SCEN)
    with pytest.raises(dataclasses.FrozenInstanceError):
        site.manning = site.manning
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCEN.rcp = "RCP2.6"


def test_simulate_shapes_units_provenance(ctx, base_result):
    dom = ctx.domain
    r = base_result
    nt = len(r.times)
    assert r.depth_max.shape == dom.shape
    assert r.depth_series.shape == (nt, *dom.shape)
    assert r.u.shape == r.v.shape == (nt, *dom.shape)
    assert r.channel_eta.shape[0] == nt and r.channel_q.shape == r.channel_eta.shape
    assert torch.all(r.times[1:] > r.times[:-1]) and float(r.times[0]) == 0.0
    assert float(r.depth_series.min()) >= 0.0
    assert torch.allclose(r.depth_max, r.depth_series.amax(0))
    assert float(r.u.abs().max()) < 20.0 and float(r.v.abs().max()) < 20.0        # m/s, physically bounded
    assert r.infiltrated_volume_m3 > 0 and r.stored_volume_m3 > 0
    assert r.mass_balance_error < 1e-3
    assert r.provenance.source == "SYNTHETIC" and r.provenance.engine == "engine"
    assert r.provenance.config_hash and r.provenance.git_commit
    assert isinstance(r.in_distribution, bool) and isinstance(r.distribution_warnings, list)
    assert not r.in_distribution and r.distribution_warnings      # no dataset envelope in this tmp outdir


def test_baseline_site_matches_landuse(ctx):
    site = api.baseline_site(SCEN)
    lu = ctx.landuse(SCEN.ssp, SCEN.horizon_year)
    assert np.allclose(site.manning.numpy(), LU.lookup(lu, "manning"))
    assert torch.all(site.infil_multiplier == 1) and torch.all(site.channel_gamma == 1)
    # SSP5 by 2050 converts land, so the SSP land surface differs from the base map
    assert (lu != ctx.domain.landuse).any()


def test_invalid_inputs_fail_loudly(ctx):
    site = api.baseline_site(SCEN)
    with pytest.raises(ValueError, match="return_period"):
        api.simulate(site, api.Scenario(20, "RCP8.5", "SSP5", 2050))
    with pytest.raises(ValueError, match="shape"):
        api.simulate(site.replace(manning=site.manning[:-1]), SCEN)
    with pytest.raises(ValueError, match="infil_multiplier"):
        api.simulate(site.replace(infil_multiplier=site.infil_multiplier * 0.5), SCEN)


def test_predict_without_trained_model_raises(ctx):
    with pytest.raises(FileNotFoundError, match="no trained surrogate"):
        api.predict(api.baseline_site(SCEN), SCEN)


def test_storage_never_raises_depth_on_a_plane():
    """Strict directional test: retention upstream on a rained-on plane can only lower
    depths everywhere (no alternative flow paths exist to re-route water)."""
    from hydrointel.benchmarks.suite import run_2d
    from hydrointel.config import SolverConfig
    from hydrointel.solver.boundaries import all_edges
    from hydrointel.solver.swe2d import SWE2D
    n = 60
    x = (np.arange(n) + 0.5) * 5.0
    z = np.tile((300.0 - x) * 0.01, (8, 1))
    out = {}
    for label, S in (("base", np.zeros_like(z)), ("storage", np.where(x < 150, 0.02, 0.0)[None].repeat(8, 0))):
        sim = SWE2D(z, 0.03, 5.0, SolverConfig(compile=False), all_edges("closed", east="transmissive"), CPU, storage=S)
        hmax = torch.zeros_like(sim.h)
        t = 0.0
        while t < 1800.0:
            t, _ = run_2d(sim, t_end=t + 30.0, t0=t, rain=lambda _t: 60.0 / 3.6e6)
            hmax = torch.maximum(hmax, sim.h)
        out[label] = hmax
    assert float((out["storage"] - out["base"]).max()) <= 1e-9
    assert float(out["storage"].sum()) < float(out["base"].sum())


def test_storage_does_not_increase_engine_depth(ctx, base_result):
    """City-scale directional test (the bug that broke the previous system: storage
    modelled as a DEM pit made flooding worse). Storage must not deepen its own
    footprint beyond numerical noise and must reduce total flood volume. Outside the
    footprint, removing water can re-route flow and shift local peaks slightly, so a
    small bounded tolerance applies there."""
    dom = ctx.domain
    dem_before = dom.dem2d.copy()
    site = api.baseline_site(SCEN)
    land = ~(dom.sea | dom.channel)
    X, Y = np.meshgrid(dom.x, dom.y)
    blob = (np.hypot(X - dom.urban_core[0], Y - dom.urban_core[1]) < 1500) & land
    S = site.storage_depth.numpy().copy()
    S[blob] = np.maximum(S[blob], 0.6)
    more = api.simulate(site.replace(storage_depth=torch.as_tensor(S)), SCEN)
    d = (more.depth_max - base_result.depth_max).numpy()
    assert d[blob].max() <= 0.005, f"storage deepened its own footprint by {d[blob].max():.4f} m"
    assert d[blob].mean() < 0.0
    assert d[land].max() <= 0.02, f"storage raised a peak depth by {d[land].max():.4f} m"
    assert np.mean(d[land] > 0.01) <= 0.001
    assert float(more.depth_series.sum()) < float(base_result.depth_series.sum())
    assert more.stored_volume_m3 > base_result.stored_volume_m3
    assert np.array_equal(ctx.domain.dem2d, dem_before)          # storage never touches the DEM


def test_predict_and_simulate_interchangeable_when_model_exists():
    """Runs against the trained quick-mode surrogate in ./artifacts/quick if present."""
    cfg = quick_config()
    cfg.outdir = str(REPO_ARTIFACTS)
    from hydrointel.model.geokan_pino import model_dir
    md = model_dir(cfg)
    tol_path = md / "contract_tolerance.json"
    if not (md / "model.pt").exists() or not tol_path.exists():
        pytest.skip("no evaluated quick-mode surrogate in ./artifacts/quick (run `cli --outdir artifacts/quick all --quick`)")
    api.configure(cfg)
    tol = json.loads(tol_path.read_text())
    scen = api.Scenario(50, "RCP4.5", "SSP2", 2050, True, 0.0)
    site = api.baseline_site(scen)
    p, s = api.predict(site, scen, detail="full"), api.simulate(site, scen)
    for name in ("depth_max", "depth_series", "u", "v", "channel_eta", "channel_q", "times"):
        assert getattr(p, name).shape == getattr(s, name).shape, name
    assert type(p) is type(s)
    land = ~(api.context().domain.sea | api.context().domain.channel)
    rmse = float(np.sqrt(np.mean(((p.depth_max - s.depth_max).numpy()[land]) ** 2)))
    # tolerance established by evaluate.py on held-out runs (x3 margin for a single scenario)
    assert rmse <= 3 * tol["depth_max_rmse_m"], (rmse, tol)
    assert p.provenance.engine == "surrogate" and s.provenance.engine == "engine"


# ---------------------------------------------------------------------------
# batched prediction, output detail, disbenefit (Part B's call shape)
# ---------------------------------------------------------------------------
def _quick_surrogate(tmp_dir):
    """The trained quick-mode surrogate if there is one, else an untrained model of the
    same architecture. These tests check the API's mechanics -- batching, detail modes,
    disbenefit bookkeeping -- which do not depend on the weights, so they must run
    whether or not a trained model happens to be on disk."""
    cfg = quick_config()
    cfg.outdir = str(REPO_ARTIFACTS)
    from hydrointel.model.geokan_pino import model_dir
    if (model_dir(cfg) / "model.pt").exists():
        api.configure(cfg, CPU)
        return cfg
    from hydrointel.benchmarks.predict_speed import _untrained_model
    cfg.outdir = str(tmp_dir)
    ctx = api.configure(cfg, CPU)
    ctx._model = _untrained_model(ctx)
    return cfg


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    _quick_surrogate(tmp_path_factory.mktemp("surrogate"))
    scen = api.Scenario(50, "RCP4.5", "SSP2", 2050, True, 0.0)
    base = api.baseline_site(scen)
    dom = api.context().domain
    land = ~(dom.sea | dom.channel)
    S = base.storage_depth.numpy().copy()
    S[land & (np.arange(S.size).reshape(S.shape) % 7 == 0)] += 0.5
    n = base.manning.numpy() * 1.2
    g = base.channel_gamma.numpy() * 1.5
    sites = [base, base.replace(storage_depth=torch.as_tensor(S)), base.replace(manning=torch.as_tensor(n)),
             base.replace(channel_gamma=torch.as_tensor(g))]
    return scen, sites, api.predict(sites, scen)


def test_predict_accepts_a_population(population):
    scen, sites, many = population
    assert isinstance(many, list) and len(many) == len(sites)
    assert all(isinstance(r, api.FloodSummary) for r in many)
    assert api.predict([], scen) == []
    assert isinstance(api.predict(sites[0], scen), api.FloodSummary)      # single signature still works


def test_batched_and_single_candidates_agree(population):
    """The population call must give each candidate the answer it gets alone."""
    scen, sites, many = population
    for k, (s, m) in enumerate(zip(sites, many)):
        one = api.predict(s, scen)
        # float32 kernels on different batch shapes: round-off, not a modelling difference
        assert torch.allclose(m.depth_max, one.depth_max, atol=1e-4), k
        assert torch.allclose(m.speed_max, one.speed_max, atol=1e-4), k
        assert torch.allclose(m.reach_peak_stage, one.reach_peak_stage, atol=1e-4), k
        assert torch.allclose(m.reach_peak_discharge, one.reach_peak_discharge, rtol=1e-4, atol=1e-3), k
        for name in m.point_level:
            assert torch.allclose(m.point_level[name], one.point_level[name], atol=1e-4), (k, name)
        assert m.mass_balance_error == pytest.approx(one.mass_balance_error, rel=1e-3, abs=1e-6)
        assert m.max_depth_increase_m == pytest.approx(one.max_depth_increase_m, abs=1e-4)


def test_summary_is_the_full_result_reduced(population):
    """Summary mode must carry exactly what full mode would give, minus the histories."""
    scen, sites, many = population
    full = api.predict(sites, scen, detail="full")
    dom = api.context().domain
    from hydrointel.viz.depthmap import monitoring_points
    pts = monitoring_points(dom)
    for s, f in zip(many, full):
        assert isinstance(f, api.FloodResult) and f.depth_series.ndim == 3
        assert not hasattr(s, "depth_series") and not hasattr(s, "u")
        assert torch.allclose(s.depth_max, f.depth_max, atol=1e-4)
        assert torch.allclose(s.speed_max, f.speed_max, atol=1e-4)
        assert torch.allclose(f.depth_max, f.depth_series.amax(0), atol=1e-6)
        spd = torch.sqrt(f.u ** 2 + f.v ** 2).amax(0)
        assert torch.allclose(f.speed_max, spd, atol=1e-4)
        assert set(s.point_level) == {p["name"] for p in pts}
        for p in pts:
            level = torch.as_tensor(dom.dem2d[p["j"], p["i"]], dtype=torch.float32) + f.depth_series[:, p["j"], p["i"]]
            assert torch.allclose(s.point_level[p["name"]], level, atol=1e-4), p["name"]
        assert s.reach_peak_stage.shape == (dom.network.n_reaches,)
        assert s.reach_peak_discharge.shape == (dom.network.n_reaches,)
        assert float(s.reach_peak_stage.max()) == pytest.approx(float(f.channel_eta.max()), abs=1e-4)
        assert s.infiltrated_volume_m3 == pytest.approx(f.infiltrated_volume_m3, rel=1e-4)
        assert s.stored_volume_m3 == pytest.approx(f.stored_volume_m3, rel=1e-4)
        assert s.provenance is not None and s.provenance.engine == "surrogate"


def test_disbenefit_fields_are_consistent(population):
    """max_depth_increase_m and area_worsened_ha against baseline(scenario), both modes."""
    scen, sites, many = population
    full = api.predict(sites, scen, detail="full")
    ref = api.baseline(scen)
    dom = api.context().domain
    land = ~(dom.sea | dom.channel)
    for r in (*many, *full):
        assert r.max_depth_increase_m is not None and r.max_depth_increase_m >= 0.0
        assert r.area_worsened_ha is not None and r.area_worsened_ha >= 0.0
        d = (r.depth_max - ref.depth_max).numpy()
        assert r.max_depth_increase_m == pytest.approx(max(float(d[land].max()), 0.0), abs=1e-4)
        assert r.area_worsened_ha == pytest.approx(
            float(np.sum(d[land] > api.WORSENED_THRESHOLD_M)) * dom.dx ** 2 / 1e4, abs=dom.dx ** 2 / 1e4)
    # the baseline cannot be worse than itself (beyond float32 round-off)
    assert many[0].max_depth_increase_m < 1e-4 and many[0].area_worsened_ha == 0.0


def test_predict_rejects_unknown_detail(population):
    scen, sites, _ = population
    with pytest.raises(ValueError, match="detail"):
        api.predict(sites[:1], scen, detail="everything")
