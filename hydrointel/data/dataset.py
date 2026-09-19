"""Torch dataset over engine simulations, with normalisation statistics.

Characteristic scales (computed on the training split only):
  H0  95th percentile of wet-cell depth (h > h_dry)
  L   domain length (max of the two extents)
  U0  sqrt(g H0),  T0 = L / U0
  Q0  95th percentile of |channel discharge|
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..domain import landuse as LU
from ..forcing import scenarios as SC

RP_LIST = (10, 25, 50, 100)
RCP_LIST = tuple(sorted(SC.RCP_TABLE))
SSP_LIST = tuple(sorted(SC.SSP_TABLE))


@dataclass
class Scales:
    H0: float
    L: float
    U0: float
    T0: float
    Q0: float
    z_mean: float
    z_std: float
    feat_mean: list
    feat_std: list
    n1_mean: list
    n1_std: list
    inflow_scale: float
    rain_scale: float
    V_scale: float = 1.0
    g: float = 9.81

    def save(self, p):
        Path(p).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, p):
        return cls(**json.loads(Path(p).read_text(encoding="utf-8")))


def scalar_features(meta: dict, forcing: dict) -> np.ndarray:
    rp = np.eye(len(RP_LIST))[RP_LIST.index(int(meta["return_period_yr"]))]
    rcp = np.eye(len(RCP_LIST))[RCP_LIST.index(meta["rcp"])]
    ssp = np.eye(len(SSP_LIST))[SSP_LIST.index(meta["ssp"])]
    st = SC.ssp_state(meta["ssp"])
    other = [float(forcing["slr_m"]), (meta["horizon_year"] - 2065) / 35.0, float(meta["tide_on"]),
             meta["storm_tide_offset_h"] / 6.0 if meta["tide_on"] else 0.0,
             float(forcing["rain_factor"]) - 1.0, st.population_factor - 1.0, st.drainage_investment_factor - 1.0]
    return np.concatenate([rp, rcp, ssp, other]).astype(np.float32)


N_SCALAR = len(RP_LIST) + len(RCP_LIST) + len(SSP_LIST) + 7


class SimDataset:
    def __init__(self, root: str | Path, split: str | None = None, cache: int = 24):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        # storms the engine could not simulate are listed in the index but are not data
        self.ids = sorted(int(k) for k, v in self.index.items()
                          if (split is None or v["split"] == split) and v.get("status") != "failed_numerical")
        if not self.ids:
            raise RuntimeError(f"{self.root}: no simulations in split {split!r}")
        self.static = torch.load(self.root / "domain.pt", weights_only=False)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.cache_size = cache

    def __len__(self):
        return len(self.ids)

    def load(self, sim_id: int) -> dict:
        if sim_id in self._cache:
            self._cache.move_to_end(sim_id)
            return self._cache[sim_id]
        rec = torch.load(self.root / f"sim_{sim_id:05d}.pt", weights_only=False, mmap=True)
        self._cache[sim_id] = rec
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rec

    def __getitem__(self, k: int) -> dict:
        return self.load(self.ids[k])


def compute_scales(train: SimDataset, h_dry: float, domain_extent_m: float, g: float = 9.81) -> Scales:
    depths, qs, inflows, rains, vols = [], [], [], [], []
    feats = []
    for sid in train.ids:
        r = train.load(sid)
        h = r["h"].numpy()
        depths.append(h[h > h_dry][::7])
        qs.append(np.abs(r["q1"].numpy()).ravel())
        inflows.append(float(r["forcing"]["inflow_m3s"].max()))
        rains.append(float(r["forcing"]["rain_mmh"].max()))
        feats.append(node_dynamic(r))
        vols.append(float(r["volumes"]["infiltrated_m3"]) + float(r["volumes"]["stored_m3"]))
    d = np.concatenate(depths)
    H0 = float(np.percentile(d, 95)) if d.size else 1.0
    Q0 = float(np.percentile(np.concatenate(qs), 95))
    st = np.asarray(train.static["coarse"]["features"])
    allf = np.concatenate([st, np.stack(feats).mean(0)], axis=0)
    mean = allf.reshape(allf.shape[0], -1).mean(1)
    std = allf.reshape(allf.shape[0], -1).std(1) + 1e-6
    U0 = math.sqrt(g * H0)
    return Scales(H0, domain_extent_m, U0, domain_extent_m / U0, max(Q0, 1.0), float(st[0].mean()), float(st[0].std()),
                  mean.tolist(), std.tolist(), [0.0] * 5, [1.0] * 5, max(inflows), max(rains),
                  max(float(np.median(vols)), 1.0), g)


def node_dynamic(rec: dict) -> np.ndarray:
    """Per-sample 2-D node features: manning, storage, kappa, land-use fractions (C, ny, nx)."""
    f = rec["fields"]
    return np.concatenate([(f["manning"].numpy() * 30.0)[None], f["storage"].numpy()[None],
                           ((f["kappa"].numpy() - 1.0) / 5.0)[None], f["landuse_frac"].numpy()], 0)


DYNAMIC_NAMES = ["manning_x30", "storage_m", "kappa_excess_div5"] + [f"lu_{c.code}" for c in LU.LANDUSE_TABLE]
