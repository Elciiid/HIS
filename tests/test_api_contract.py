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
    p, s = api.predict(site, scen), api.simulate(site, scen)
    for name in ("depth_max", "depth_series", "u", "v", "channel_eta", "channel_q", "times"):
        assert getattr(p, name).shape == getattr(s, name).shape, name
    assert type(p) is type(s)
    land = ~(api.context().domain.sea | api.context().domain.channel)
    rmse = float(np.sqrt(np.mean(((p.depth_max - s.depth_max).numpy()[land]) ** 2)))
    # tolerance established by evaluate.py on held-out runs (x3 margin for a single scenario)
    assert rmse <= 3 * tol["depth_max_rmse_m"], (rmse, tol)
    assert p.provenance.engine == "surrogate" and s.provenance.engine == "engine"
