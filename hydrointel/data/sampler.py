"""Parameter-space sampler for training data.

Interventions enter the physics as four fields, and the dataset must span
them, otherwise every Part B query is an extrapolation:

  storage_depth    S(x,y)  [m]   0 .. 1.5    retention filled before runoff
  infil_multiplier kappa   [-]   1 .. 6      multiplies Horton f0 and fc
  manning          n(x,y)        baseline x (1 +- perturb); coastal planting up to 0.15
  channel_gamma    gamma   [-]   0.8 .. 2.0  per reach conveyance

Patterns are spatially correlated: Gaussian blobs, road-aligned strips,
coastal bands and thresholded smooth noise. A fraction of samples carries no
intervention at all (the baseline). Everything is deterministic from the seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import DataConfig, numpy_rng
from ..domain import landuse as LU
from ..domain.synthetic import Domain, spectral_noise
from ..forcing import scenarios as SC

COASTAL_BAND_M = 800.0
MANGROVE_N_RANGE = (0.06, 0.15)


@dataclass
class Sample:
    index: int
    split: str
    return_period_yr: int
    rcp: str
    ssp: str
    horizon_year: int
    tide_on: bool
    storm_tide_offset_h: float
    baseline: bool
    patterns: list[str]
    storage_delta: np.ndarray = field(repr=False)      # added to the (SSP) baseline storage
    kappa: np.ndarray = field(repr=False)
    manning_factor: np.ndarray = field(repr=False)     # multiplies the (SSP) baseline Manning
    manning_abs: np.ndarray = field(repr=False)        # NaN, or absolute planted value (coastal band)
    gamma: np.ndarray = field(repr=False)

    def meta(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)}
        d["gamma"] = self.gamma.tolist()
        return d


@dataclass
class Envelope:
    storage_range_m: tuple
    kappa_range: tuple
    manning_factor_range: tuple
    manning_planted_range: tuple
    coastal_band_m: float
    gamma_range: tuple
    return_periods: tuple
    rcps: tuple
    ssps: tuple
    horizon_range: tuple
    offset_range_h: tuple
    # observed coverage statistics across the training split
    max_storage_cover: float = 0.0
    max_kappa_cover: float = 0.0
    max_mean_storage_m: float = 0.0
    max_mean_kappa_excess: float = 0.0

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Envelope":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def check(self, storage_abs, storage_base, kappa, manning, manning_base, gamma, dist_coast, active,
              scenario=None) -> list[str]:
        """Return warnings for every way a query leaves the training envelope."""
        w = []
        s = storage_abs[active]
        if s.min() < self.storage_range_m[0] - 1e-9 or s.max() > self.storage_range_m[1] + 1e-9:
            w.append(f"storage_depth outside [{self.storage_range_m[0]}, {self.storage_range_m[1]}] m "
                     f"(query {s.min():.3f}..{s.max():.3f})")
        k = kappa[active]
        if k.min() < self.kappa_range[0] - 1e-9 or k.max() > self.kappa_range[1] + 1e-9:
            w.append(f"infil_multiplier outside {self.kappa_range} (query {k.min():.3f}..{k.max():.3f})")
        ratio = manning[active] / manning_base[active]
        coastal = (dist_coast[active] >= 0) & (dist_coast[active] <= self.coastal_band_m)
        bad = ~((ratio >= self.manning_factor_range[0] - 1e-9) & (ratio <= self.manning_factor_range[1] + 1e-9))
        planted_ok = coastal & (manning[active] >= self.manning_planted_range[0] - 1e-9) & \
            (manning[active] <= self.manning_planted_range[1] + 1e-9)
        if np.any(bad & ~planted_ok):
            w.append(f"manning outside baseline x {self.manning_factor_range} (and not a coastal planting within "
                     f"{self.manning_planted_range}) in {int(np.sum(bad & ~planted_ok))} cells")
        g = np.asarray(gamma)
        if g.min() < self.gamma_range[0] - 1e-9 or g.max() > self.gamma_range[1] + 1e-9:
            w.append(f"channel_gamma outside {self.gamma_range} (query {g.min():.3f}..{g.max():.3f})")
        ds = (storage_abs - storage_base)[active]
        cover_s = float(np.mean(ds > 0.01))
        cover_k = float(np.mean(k > 1.01))
        if cover_s > self.max_storage_cover + 1e-9:
            w.append(f"storage covers {cover_s:.1%} of the land, training maximum was {self.max_storage_cover:.1%}")
        if cover_k > self.max_kappa_cover + 1e-9:
            w.append(f"infiltration boost covers {cover_k:.1%} of the land, training maximum was {self.max_kappa_cover:.1%}")
        if float(np.mean(np.clip(ds, 0, None))) > self.max_mean_storage_m + 1e-9:
            w.append("mean added storage exceeds the training maximum")
        if float(np.mean(k - 1.0)) > self.max_mean_kappa_excess + 1e-9:
            w.append("mean infiltration boost exceeds the training maximum")
        if scenario is not None:
            if scenario.return_period_yr not in self.return_periods:
                w.append(f"return period {scenario.return_period_yr} not in training set {self.return_periods}")
            if not self.horizon_range[0] <= scenario.horizon_year <= self.horizon_range[1]:
                w.append(f"horizon_year {scenario.horizon_year} outside {self.horizon_range}")
            if scenario.tide_on and not self.offset_range_h[0] <= scenario.storm_tide_offset_h <= self.offset_range_h[1]:
                w.append(f"storm_tide_offset_h outside {self.offset_range_h}")
        return w


def envelope_from(cfg: DataConfig) -> Envelope:
    return Envelope(tuple(cfg.storage_range_m), tuple(cfg.kappa_range),
                    (1.0 - cfg.manning_perturb, 1.0 + cfg.manning_perturb), MANGROVE_N_RANGE, COASTAL_BAND_M,
                    tuple(cfg.gamma_range), tuple(cfg.return_periods), tuple(SC.RCP_TABLE), tuple(SC.SSP_TABLE),
                    tuple(cfg.horizon_range), tuple(cfg.offset_range_h))


def _blobs(rng, dom: Domain, weight: np.ndarray, n: int, rmin: float, rmax: float) -> np.ndarray:
    ny, nx = dom.shape
    out = np.zeros((ny, nx))
    p = weight.ravel() / weight.sum()
    X, Y = np.meshgrid(dom.x, dom.y)
    for c in rng.choice(ny * nx, size=n, p=p):
        cy, cx = divmod(int(c), nx)
        r = rng.uniform(rmin, rmax)
        bump = np.exp(-((X - dom.x[cx]) ** 2 + (Y - dom.y[cy]) ** 2) / r ** 2)
        out = np.maximum(out, np.clip(2.0 * bump - 0.4, 0.0, 1.0) * rng.uniform(0.4, 1.0))
    return out


def _road_strips(rng, dom: Domain, width_cells: int) -> np.ndarray:
    m = dom.road.copy()
    for _ in range(width_cells):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    sel = spectral_noise(rng, *dom.shape, dom.dx, beta=2.0, lmin=300.0) > rng.uniform(-0.5, 0.8)
    return (m & sel).astype(float)


def _coastal_band(rng, dom: Domain) -> tuple[np.ndarray, float]:
    w = rng.uniform(100.0, COASTAL_BAND_M)
    return ((dom.dist_coast >= 0) & (dom.dist_coast <= w)).astype(float), w


def _noise_patches(rng, dom: Domain) -> np.ndarray:
    n = spectral_noise(rng, *dom.shape, dom.dx, beta=2.0, lmin=200.0)
    return np.clip((n - rng.uniform(0.3, 1.5)) / 0.8, 0.0, 1.0)


def _pattern(rng, dom, kind: str, land: np.ndarray):
    if kind == "blobs":
        wgt = land * (1.0 + 3.0 * np.isin(dom.landuse, [LU.DENSE_URBAN, LU.RESIDENTIAL, LU.ROAD]))
        return _blobs(rng, dom, wgt + 1e-12, int(rng.integers(1, 9)), 80.0, 600.0)
    if kind == "roads":
        return _road_strips(rng, dom, int(rng.integers(0, 3)))
    if kind == "coastal":
        return _coastal_band(rng, dom)[0]
    return _noise_patches(rng, dom)


def draw(index: int, dom: Domain, cfg: DataConfig, seed: int, split: str) -> Sample:
    rng = numpy_rng(f"sample/{seed}/{index}")
    ny, nx = dom.shape
    land = (~dom.sea & ~dom.channel).astype(float)
    rp = cfg.return_periods[index % len(cfg.return_periods)]
    ssp = str(rng.choice(sorted(SC.SSP_TABLE)))
    rcp = SC.SSP_DEFAULT_RCP[ssp] if rng.uniform() < 0.7 else str(rng.choice(sorted(SC.RCP_TABLE)))
    year = int(rng.integers(cfg.horizon_range[0], cfg.horizon_range[1] + 1))
    tide_on = bool(rng.uniform() < cfg.tide_on_prob)
    offset = float(rng.uniform(*cfg.offset_range_h))
    baseline = bool(rng.uniform() < cfg.baseline_frac)
    dS = np.zeros((ny, nx))
    kap = np.ones((ny, nx))
    nfac = np.ones((ny, nx))
    nabs = np.full((ny, nx), np.nan)
    gamma = np.ones(dom.network.n_reaches)
    used = []
    if not baseline:
        kinds = ["blobs", "roads", "coastal", "noise"]
        if rng.uniform() < 0.8:                                  # storage
            k = str(rng.choice(kinds)); used.append(f"storage:{k}")
            amp = rng.uniform(0.05, cfg.storage_range_m[1])
            dS = _pattern(rng, dom, k, land) * amp * land
        if rng.uniform() < 0.7:                                  # infiltration boost
            k = str(rng.choice(kinds)); used.append(f"kappa:{k}")
            kap = 1.0 + _pattern(rng, dom, k, land) * rng.uniform(0.2, cfg.kappa_range[1] - 1.0) * land
        if rng.uniform() < 0.6:                                  # roughness perturbation (signed, smooth)
            used.append("manning:perturb")
            nfac = 1.0 + cfg.manning_perturb * np.tanh(spectral_noise(rng, ny, nx, dom.dx, beta=2.0, lmin=300.0)) \
                * rng.uniform(0.3, 1.0)
        if rng.uniform() < 0.35:                                 # coastal planting
            used.append("manning:mangrove")
            band, _ = _coastal_band(rng, dom)
            planted = (band > 0) & (land > 0)
            nabs[planted] = rng.uniform(*MANGROVE_N_RANGE)
        if rng.uniform() < 0.7:                                  # conveyance
            used.append("gamma")
            gamma = rng.uniform(*cfg.gamma_range, size=dom.network.n_reaches)
    return Sample(index, split, int(rp), rcp, ssp, year, tide_on, offset, baseline, used,
                  dS, kap, nfac, nabs, gamma)


def assign_splits(n: int, cfg: DataConfig, seed: int, return_periods) -> list[str]:
    """Split by scenario (whole simulations). Test and validation sims are drawn by
    cycling through the return periods so every split covers as many as possible."""
    import math
    rng = numpy_rng(f"split/{seed}")
    rps = [return_periods[i % len(return_periods)] for i in range(n)]
    n_test = max(1, math.ceil(cfg.test_frac * n))
    n_val = max(1, int(round(cfg.val_frac * n)))
    if n_test + n_val >= n:
        raise ValueError(f"n_sims={n} is too small for test/val splits of {n_test}/{n_val}")
    pools = {rp: [i for i in rng.permutation(n) if rps[i] == rp] for rp in return_periods}
    order = []
    while any(pools.values()):
        for rp in return_periods:
            if pools[rp]:
                order.append(int(pools[rp].pop(0)))
    splits = ["train"] * n
    for k, i in enumerate(order[:n_test + n_val]):
        splits[i] = "test" if k < n_test else "val"
    return splits


def materialise(sample: Sample, dom: Domain, lu_ssp: np.ndarray):
    """Absolute site fields for a sample on the SSP land surface."""
    base_S = LU.lookup(lu_ssp, "storage_m")
    base_n = LU.lookup(lu_ssp, "manning")
    storage = np.clip(base_S + sample.storage_delta, 0.0, 1.5)
    manning = np.where(np.isnan(sample.manning_abs), base_n * sample.manning_factor, sample.manning_abs)
    water = dom.sea | dom.channel
    storage[water] = 0.0
    kappa = np.where(water, 1.0, sample.kappa)
    manning = np.where(water, base_n, manning)
    return storage, kappa, manning, sample.gamma.copy(), base_S, base_n
