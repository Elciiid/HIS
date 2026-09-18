"""Configuration dataclasses, validation, hashing and seeding.

Every tunable number lives here (or in a clearly labelled placeholder block in
the module that owns it). A run is fully described by a ``RunConfig``; its hash
is stamped on every output so any figure can be traced back to its inputs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("hydrointel")

# ---------------------------------------------------------------------------
# PLACEHOLDER_IDF_ROXAS
# Illustrative placeholder coefficients for i = a * T**b / (d + c)**e
# (i in mm/h, T in years, d in minutes). These are NOT PAGASA-published values.
# They were chosen only to give physically sane tropical intensities
# (~160 mm/h at 10 min for T = 100 yr, decaying with duration).
# TODO: replace with PAGASA Roxas/Capiz station IDF
PLACEHOLDER_IDF_ROXAS = {"a": 609.0, "b": 0.18, "c": 12.0, "e": 0.70}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Flood hazard classes for maps: (name, lower bound m, upper bound m, RGBA).
# TODO: adopt the client's official hazard classification
HAZARD_CLASSES = (
    ("None", 0.00, 0.05, (0.0, 0.0, 0.0, 0.0)),
    ("Low", 0.05, 0.15, (0.55, 0.95, 1.00, 0.45)),
    ("Moderate", 0.15, 0.30, (0.25, 0.75, 0.95, 0.70)),
    ("Significant", 0.30, 0.50, (0.10, 0.35, 0.90, 0.80)),
    ("High", 0.50, 1.00, (1.00, 0.60, 0.10, 0.85)),
    ("Severe", 1.00, 1.50, (0.90, 0.10, 0.10, 0.90)),
    ("Critical", 1.50, float("inf"), (0.45, 0.00, 0.05, 0.95)),
)
# ---------------------------------------------------------------------------


@dataclass
class DomainConfig:
    lx_m: float = 8000.0            # E-W extent
    ly_m: float = 6500.0            # N-S extent
    cell_m: float = 20.0            # 10 m for showcase runs
    # coastal profile (north edge is the sea)
    offshore_width_m: float = 300.0
    offshore_elev_m: float = -1.0
    plain_width_m: float = 2000.0
    plain_top_m: float = 2.0
    inland_elev_range_m: tuple[float, float] = (15.0, 25.0)
    # river corridor
    floodplain_width_range_m: tuple[float, float] = (1500.0, 3000.0)
    floodplain_rise_m: float = 6.0          # valley-floor rise from coast to south edge
    channel_width_m: float = 60.0
    channel_incision_range_m: tuple[float, float] = (2.5, 4.0)
    channel_side_slope: float = 1.5         # H:V
    channel_manning: float = 0.030
    river_entry_x_frac: float = 0.72        # enters from the south-east
    river_mouth_x_frac: float = 0.52
    meander_amp_m: float = 450.0
    channel_node_spacing_m: float = 100.0
    # tidal creeks
    n_creeks: int = 2
    creek_width_m: float = 25.0
    creek_incision_range_m: tuple[float, float] = (1.0, 2.0)
    # hills
    n_hills: int = 3
    hill_height_range_m: tuple[float, float] = (20.0, 40.0)
    # urban fabric
    urban_radius_m: float = 1600.0
    street_spacing_m: float = 100.0
    street_width_m: float = 20.0
    block_raise_range_m: tuple[float, float] = (0.2, 0.4)
    building_raise_m: float = 0.3
    # micro-topography
    micro_noise_amp_m: float = 0.10         # std of 1/f noise, within 0.05-0.15
    smoothing_passes: int = 1

    @property
    def nx(self) -> int:
        return int(round(self.lx_m / self.cell_m))

    @property
    def ny(self) -> int:
        return int(round(self.ly_m / self.cell_m))


@dataclass
class SolverConfig:
    g: float = 9.81
    h_dry: float = 1e-3
    kp_eps: float = 1e-12               # Kurganov-Petrova desingularisation
    cfl: float = 0.45
    dt_min: float = 1e-4
    dt_max: float = 5.0
    # MUSCL + SSP-RK2 is the default: compiled it costs ~15% more per step than
    # first order, and first order under-represents slope forcing of thin sheets
    # on steep cells (see benchmark 6).
    spatial_order: int = 2
    flux: str = "hllc"                  # "hllc" | "hll"
    precision: str = "fp64"             # "fp64" | "fp32"
    compile: bool = True                # torch.compile the step on CUDA
    mass_tol: float = 1e-3
    recession_h: float = 2.0            # simulated time after the storm ends
    output_interval_s: float = 900.0
    # 1-D
    slot_width_frac: float = 0.015      # Preissmann slot width / bankfull top width
    cfl_1d: float = 0.45
    # coupling
    weir_cw: float = 1.7                # SI free-weir coefficient
    exchange_limit: float = 0.5         # max fraction of donor volume per step
    exchange_relax: float = 0.5         # fraction of level-equalising volume per step


@dataclass
class CatchmentConfig:
    """Upstream catchment feeding the main river at the south edge.

    All values are placeholders for a Clark (time-area + linear reservoir) routing.
    TODO: replace with the client's hydrological model of the upstream basin.
    """
    area_km2: float = 120.0
    tc_h: float = 3.0                   # time of concentration
    storage_k_h: float = 2.0            # Clark linear-reservoir constant
    runoff_coeff: float = 0.45
    baseflow_m3s: float = 15.0


@dataclass
class TideConfig:
    """Constituent amplitudes (m) and phases (deg) are placeholders giving a mixed
    semidiurnal signal with a ~1.0-1.5 m typical range.
    TODO: replace with harmonic constants from the nearest NAMRIA tide station."""
    msl_offset_m: float = 0.0
    constituents: dict = field(default_factory=lambda: {
        "M2": {"period_h": 12.4206012, "amp_m": 0.36, "phase_deg": 0.0},
        "S2": {"period_h": 12.0, "amp_m": 0.14, "phase_deg": 30.0},
        "K1": {"period_h": 23.9344697, "amp_m": 0.26, "phase_deg": 60.0},
        "O1": {"period_h": 25.8193417, "amp_m": 0.20, "phase_deg": 110.0},
    })


@dataclass
class ForcingConfig:
    idf: dict = field(default_factory=lambda: dict(PLACEHOLDER_IDF_ROXAS))
    storm_duration_h: float = 6.0
    storm_step_min: float = 5.0
    peak_position: float = 0.5
    apply_arf: bool = True
    spinup_h: float = 0.5               # pre-storm period (tide/baseflow only)
    catchment: CatchmentConfig = field(default_factory=CatchmentConfig)
    tide: TideConfig = field(default_factory=TideConfig)


@dataclass
class DataConfig:
    n_sims: int = 400
    # Storms advanced together on one set of kernels. Default 1 -- batching OFF -- on
    # measurement, not on principle. Over a full storm a batch of 8 was 1.22x SLOWER
    # per storm (453 s against 372 s): the shared time step made it run 1.89x more
    # steps than a member needs alone, which more than cancelled the 1.84x it gained
    # per step. It also moved peak depths by ~120 mm. Raise it only for a batch of
    # storms with similar time-step demand; see artifacts/batch_throughput.md.
    batch: int = 1
    test_frac: float = 0.15
    val_frac: float = 0.10
    coarsen: int = 2
    return_periods: tuple[int, ...] = (10, 25, 50, 100)
    horizon_range: tuple[int, int] = (2030, 2100)
    offset_range_h: tuple[float, float] = (-6.0, 6.0)
    tide_on_prob: float = 0.8
    baseline_frac: float = 0.15         # samples with no intervention at all
    storage_range_m: tuple[float, float] = (0.0, 1.5)
    kappa_range: tuple[float, float] = (1.0, 6.0)
    manning_perturb: float = 0.25       # +- fraction
    gamma_range: tuple[float, float] = (0.8, 2.0)


@dataclass
class ModelConfig:
    latent: int = 32
    degree: int = 5
    n_blocks: int = 4
    spectral_modes: int = 12
    forcing_steps: int = 96             # forcing series resampled to this length
    time_fourier: int = 8


@dataclass
class TrainConfig:
    steps: int = 40000
    lr: float = 2e-3
    weight_decay: float = 1e-6
    query_times: int = 6                # time queries per sample per step
    curriculum_frac: float = 0.15
    ramp_frac: float = 0.10
    balance_every: int = 50
    balance_alpha: float = 0.9
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 25
    val_every: int = 500
    ckpt_every: int = 1000


@dataclass
class RunConfig:
    seed: int = 20260917
    device: str = "auto"
    outdir: str = "artifacts"
    quick: bool = False
    domain: DomainConfig = field(default_factory=DomainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- derived -----------------------------------------------------------
    def resolved_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def dtype(self):
        import torch
        return torch.float64 if self.solver.precision == "fp64" else torch.float32

    def to_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))

    def config_hash(self, *sections: str) -> str:
        """Hash of the whole config (or only the named sections), excluding
        fields that do not change results (outdir, device)."""
        d = self.to_dict()
        d.pop("outdir"); d.pop("device")
        # one storm at a time is the reference numerics and predates the batch setting:
        # leave it out so existing datasets and models keep their hashes. A batch > 1
        # shares a time step, changes the results, and so does enter the hash.
        if d["data"].get("batch") == 1:
            d["data"].pop("batch")
        if sections:
            d = {k: d[k] for k in sorted(set(sections) | {"seed", "quick"})}
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def validate(self) -> "RunConfig":
        dm, sv = self.domain, self.solver
        _check(dm.cell_m > 0 and dm.lx_m > 0 and dm.ly_m > 0, "domain extents and cell_m must be positive")
        _check(dm.nx >= 16 and dm.ny >= 16, f"grid too small: {dm.nx}x{dm.ny}")
        _check(0.05 <= dm.micro_noise_amp_m <= 0.15, "micro_noise_amp_m must be within 0.05-0.15 m")
        _check(1 <= dm.n_creeks <= 3, "n_creeks must be 1-3")
        _check(sv.spatial_order in (1, 2), "spatial_order must be 1 or 2")
        _check(sv.flux in ("hllc", "hll"), "flux must be 'hllc' or 'hll'")
        _check(sv.precision in ("fp64", "fp32"), "precision must be fp64 or fp32")
        _check(0 < sv.cfl <= 0.5, "cfl must be in (0, 0.5]")
        _check(0 < sv.exchange_limit <= 0.5, "exchange_limit must be <= 0.5")
        _check(self.data.storage_range_m[0] >= 0, "storage range must be non-negative")
        _check(self.data.kappa_range[0] >= 1.0, "infiltration multiplier must be >= 1")
        _check(all(rp in (10, 25, 50, 100) for rp in self.data.return_periods), "return periods must be 10/25/50/100")
        _check(0 < self.data.test_frac < 0.5 and self.data.test_frac >= 0.15, "test_frac must be >= 0.15")
        return self

    # ---- io ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return _from_dict(cls, d)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RunConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(text) or {}
        else:
            d = json.loads(text)
        return cls.from_dict(d)

    def save(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = self.to_dict()
        out["_config_hash"] = self.config_hash()
        p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def quick_config(cfg: RunConfig | None = None) -> RunConfig:
    """Smoke-test settings: small grid, few sims, short training (CPU < 10 min)."""
    cfg = cfg or RunConfig()
    cfg.quick = True
    cfg.domain.cell_m = 80.0
    cfg.domain.street_spacing_m = 320.0
    cfg.domain.street_width_m = 80.0
    cfg.domain.channel_node_spacing_m = 160.0
    cfg.data.n_sims = 8
    cfg.data.batch = 1
    cfg.data.coarsen = 1
    cfg.forcing.storm_duration_h = 0.5
    cfg.forcing.spinup_h = 0.1
    cfg.solver.recession_h = 0.15
    cfg.solver.output_interval_s = 300.0
    cfg.solver.output_interval_s = 900.0
    cfg.model.latent = 16
    cfg.model.n_blocks = 2
    cfg.model.spectral_modes = 6
    cfg.model.forcing_steps = 48
    cfg.train.steps = 300
    cfg.train.query_times = 3
    cfg.train.val_every = 100
    cfg.train.ckpt_every = 100
    cfg.train.balance_every = 25
    return cfg


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"invalid config: {msg}")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _from_dict(cls, d: dict):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if k not in fields:
            raise ValueError(f"unknown config key '{k}' for {cls.__name__}")
        ftype = fields[k].type
        sub = _DATACLASS_TYPES.get(ftype if isinstance(ftype, str) else getattr(ftype, "__name__", ""))
        if sub is not None and isinstance(v, dict):
            kwargs[k] = _from_dict(sub, v)
        elif isinstance(v, list) and "tuple" in str(ftype):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


_DATACLASS_TYPES = {c.__name__: c for c in (DomainConfig, SolverConfig, CatchmentConfig, TideConfig,
                                            ForcingConfig, DataConfig, ModelConfig, TrainConfig)}


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def derive_seed(key: str) -> int:
    """Stable per-key seed. Never use the salted builtin hash function: it changes per process."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def torch_generator(key: str, device="cpu"):
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(derive_seed(key))
    return g


def numpy_rng(key: str):
    import numpy as np
    return np.random.default_rng(derive_seed(key))
