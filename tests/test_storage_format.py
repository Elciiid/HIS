"""Storage format: compression, the capacity guard, and storing finer than the model grid.

A dataset may be stored on a finer grid than the model trains on (``data.store_coarsen``),
so one full-fidelity dataset can train models at several resolutions. That is only sound if
block-averaging a finer record onto the model grid gives exactly what the generator would
have written had it stored at the model grid directly. These tests pin that, the round trip
through a compressed record, and that the capacity guard refuses an oversized dataset before
any storm runs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hydrointel.config import RunConfig, quick_config
from hydrointel.data.generate import (COMPRESSED_SUFFIX, MAX_SNAPSHOTS_PER_STORM, check_capacity, coarsen_mean,
                                      load_record, projected_size, record_path, save_record)


def test_record_path_and_compressed_round_trip(tmp_path):
    rec = {"h": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), "meta": {"x": 1}}
    gz = record_path(tmp_path, 7, compress=True)
    assert gz.name == f"sim_00007{COMPRESSED_SUFFIX}"
    save_record(rec, gz)
    back = load_record(gz)
    assert torch.equal(back["h"], rec["h"]) and back["meta"] == {"x": 1}
    # with no preference, an existing compressed record is found
    assert record_path(tmp_path, 7) == gz
    plain = record_path(tmp_path, 7, compress=False)
    save_record(rec, plain)
    assert torch.equal(load_record(plain)["h"], rec["h"])
    # both formats present: the plain one wins, and both still read
    assert record_path(tmp_path, 7) == plain


def test_compression_saves_space(tmp_path):
    rng = np.random.default_rng(0)
    h = np.abs(rng.standard_normal((35, 162, 200))).astype(np.float32) * (rng.random((35, 162, 200)) > 0.3)
    rec = {"h": torch.as_tensor(h), "forcing": {"horton_rate_mmh": torch.full((35, 162, 200), 12.5)}}
    save_record(rec, record_path(tmp_path, 0, compress=False))
    save_record(rec, record_path(tmp_path, 0, compress=True))
    plain = record_path(tmp_path, 0, compress=False).stat().st_size
    gz = record_path(tmp_path, 0, compress=True).stat().st_size
    assert gz < 0.8 * plain, f"compression saved only {100 * (1 - gz / plain):.0f}%"


def test_capacity_guard_refuses_an_oversized_dataset():
    from hydrointel.domain.synthetic import generate as gen_domain
    cfg = quick_config(RunConfig())
    dom = gen_domain(cfg.domain, cfg.seed)
    small = projected_size(cfg, dom, 10)
    assert small["total_gb"] < 1.0
    with pytest.raises(RuntimeError, match="exceeds the .* GB limit"):
        check_capacity(cfg, dom, 10, limit_gb=small["total_gb"] / 2)
    # and it refuses a snapshot cadence the format does not allow: the production storm is
    # 8.5 h long, so a 60 s output interval would be 511 snapshots
    dense = RunConfig()
    dense.solver.output_interval_s = 60.0
    with pytest.raises(RuntimeError, match="snapshots per storm exceeds"):
        check_capacity(dense, dom, 1)
    assert projected_size(cfg, dom, 10)["snapshots_per_storm"] <= MAX_SNAPSHOTS_PER_STORM


def test_full_resolution_storage_costs_what_the_guard_says():
    """The guard's arithmetic, checked against the alternative it exists to compare."""
    from hydrointel.domain.synthetic import generate as gen_domain
    cfg = RunConfig()
    dom = gen_domain(quick_config(RunConfig()).domain, cfg.seed)
    at2 = projected_size(cfg, dom, 100)
    cfg.data.store_coarsen = 1
    at1 = projected_size(cfg, dom, 100)
    assert at1["store_coarsen"] == 1 and at2["store_coarsen"] == 2
    assert at1["total_gb"] == pytest.approx(4 * at2["total_gb"], rel=0.05)


# --------------------------------------------------------------------------- store vs model grid
def _fake_record(ny, nx, nt=4, seed=0):
    """A record with the fields the builder reads, on an ny x nx grid."""
    from hydrointel.domain import landuse as LU
    rng = np.random.default_rng(seed)
    h = np.abs(rng.standard_normal((nt, ny, nx))).astype(np.float32)
    h[h < 0.3] = 0.0                                              # dry cells, so the weighting matters
    u = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    v = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    lu = np.zeros((ny, nx), dtype=np.int64)
    return {"h": torch.as_tensor(h), "u": torch.as_tensor(u), "v": torch.as_tensor(v),
            "fields": {"manning": torch.as_tensor(rng.random((ny, nx)).astype(np.float32) * 0.05),
                       "storage": torch.as_tensor(rng.random((ny, nx)).astype(np.float32)),
                       "kappa": torch.as_tensor(1 + rng.random((ny, nx)).astype(np.float32)),
                       "landuse_frac": torch.as_tensor(np.eye(LU.N_CLASSES)[lu].transpose(2, 0, 1).astype(np.float32))}}


def _builder(k: int, store_f: int, ny: int, nx: int):
    """A SampleBuilder with only what ``snapshots`` needs, avoiding a domain build."""
    from hydrointel.model.batch import SampleBuilder
    b = SampleBuilder.__new__(SampleBuilder)
    b.k, b.store_f, b.f = k, store_f, k * store_f
    return b


def test_stored_finer_then_averaged_equals_stored_at_the_model_grid():
    """Storing at 20 m and averaging by 2 on load must equal storing at 40 m directly --
    including the depth-weighted velocity average the generator uses."""
    ny, nx, k = 16, 20, 2
    rec = _fake_record(ny, nx, seed=1)
    h, u, v = (rec[x].numpy() for x in ("h", "u", "v"))
    # what the generator writes when it stores at the model grid
    hc = coarsen_mean(h, k)
    safe = np.where(hc > 0, hc, 1.0)
    uc = np.where(hc > 0, coarsen_mean(h * u, k) / safe, 0.0)
    vc = np.where(hc > 0, coarsen_mean(h * v, k) / safe, 0.0)
    got_h, got_u, got_v = _builder(k, 1, ny, nx).snapshots(rec)
    assert np.allclose(got_h, hc) and np.allclose(got_u, uc) and np.allclose(got_v, vc)


def test_snapshots_are_untouched_when_storage_is_the_model_grid():
    rec = _fake_record(8, 8, seed=2)
    h, u, v = _builder(1, 2, 8, 8).snapshots(rec)
    assert np.shares_memory(h, rec["h"].numpy()) or np.allclose(h, rec["h"].numpy())
    assert np.allclose(u, rec["u"].numpy()) and np.allclose(v, rec["v"].numpy())


def test_model_grid_must_be_a_multiple_of_the_storage_grid():
    from hydrointel.data.dataset import Scales
    from hydrointel.model.batch import SampleBuilder
    cfg = quick_config(RunConfig())
    cfg.data.coarsen = 3
    static = {"coarse": {"factor": 2, "features": np.zeros((2, 4, 4), np.float32), "dem2d": np.zeros((4, 4)),
                         "sea": np.zeros((4, 4), bool)}}
    with pytest.raises(ValueError, match="not a whole multiple"):
        SampleBuilder(cfg, object(), static, Scales(1, 1, 1, 1, 1, 0, 1, [0, 0], [1, 1], [0], [1], 1, 1), None, "cpu")
