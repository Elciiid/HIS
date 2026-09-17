"""Same seed -> bit-identical outputs; no salted hash() for seeding anywhere."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from hydrointel.config import RunConfig, derive_seed, quick_config, seed_everything

PKG = Path(__file__).resolve().parents[1] / "hydrointel"


def test_derive_seed_is_stable_across_processes():
    code = "from hydrointel.config import derive_seed; print(derive_seed('sim/00042'))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
            for _ in range(2)}
    assert outs == {str(derive_seed("sim/00042"))}


def test_seed_everything_reproduces_rng_streams():
    def draw():
        seed_everything(123)
        return np.random.rand(5), torch.rand(5)
    a, b = draw(), draw()
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_config_hash_ignores_outdir_and_device_but_not_physics():
    c1, c2 = RunConfig(), RunConfig()
    c2.outdir, c2.device = "elsewhere", "cpu"
    assert c1.config_hash() == c2.config_hash()
    c2.solver.cfl = 0.4
    assert c1.config_hash() != c2.config_hash()
    assert c1.config_hash("domain") == c2.config_hash("domain")


def test_config_roundtrip(tmp_path):
    cfg = quick_config().validate()
    cfg.save(tmp_path / "c.json")
    back = RunConfig.load(tmp_path / "c.json")
    assert back.config_hash() == cfg.config_hash()
    assert isinstance(back.domain.inland_elev_range_m, tuple)


def test_no_builtin_hash_used_for_seeding():
    pat = re.compile(r"(?<![\w.])hash\(")
    offenders = [f"{p.name}:{i + 1}" for p in PKG.rglob("*.py")
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines())
                 if pat.search(line) and not line.lstrip().startswith("#")]
    assert not offenders, f"builtin hash() used in: {offenders}"


def _short_engine_run(device):
    import numpy as np
    from hydrointel.config import quick_config
    from hydrointel.domain.synthetic import generate
    from hydrointel.engine import build_engine, build_forcing
    cfg = quick_config()
    cfg.forcing.storm_duration_h = 0.5
    cfg.forcing.spinup_h = 0.1
    cfg.solver.recession_h = 0.1
    seed_everything(cfg.seed)
    dom = generate(cfg.domain, cfg.seed)
    fo = build_forcing(cfg, 100, "RCP8.5", "SSP5", 2050, True, 0.0, 52.0)
    eng = build_engine(cfg, dom, fo, dom.base_storage(), np.full(dom.shape, 1.5), dom.manning(),
                       np.ones(dom.network.n_reaches), device=device)
    eng.run(fo.t_end)
    return eng.two.h.cpu().numpy(), eng.one.A.cpu().numpy(), dom.dem


def test_engine_bit_identical_on_cpu():
    a, b = _short_engine_run(torch.device("cpu")), _short_engine_run(torch.device("cpu"))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_engine_bit_identical_on_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a, b = _short_engine_run(torch.device("cuda")), _short_engine_run(torch.device("cuda"))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
