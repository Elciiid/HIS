"""Model inputs from a dataset record (training) or from (SiteState, Scenario)
(inference). Both paths go through the same feature code so the surrogate sees
identical inputs in training and in use."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import Scales, node_dynamic, scalar_features
from ..data.generate import coarsen_mean
from ..domain import landuse as LU
from ..domain.channels import apply_gamma
from ..solver.swe1d import Sections, build_topology
from .graph import build_graph


class SampleBuilder:
    """Model inputs on the model grid (``cfg.data.coarsen``), from a dataset stored on the
    storage grid (``domain.pt``'s ``coarse.factor``, which may be finer).

    Block averaging composes, so a dataset stored at full fidelity can train a model at any
    coarser grid: everything the record and the static fields hold is averaged by
    ``k = model factor / storage factor`` on load, and the velocities are averaged weighted
    by depth, exactly as the generator would have written them.
    """

    def __init__(self, cfg, domain, static: dict, scales: Scales, graph, device):
        self.cfg, self.domain, self.static, self.scales, self.graph = cfg, domain, static, scales, graph
        self.device = device
        self.store_f = int(static["coarse"]["factor"])
        self.f = int(cfg.data.coarsen)
        if self.f % self.store_f:
            raise ValueError(f"model grid {self.f}x is not a whole multiple of the stored grid "
                             f"{self.store_f}x: the snapshots cannot be block-averaged onto it")
        self.k = self.f // self.store_f
        st = static["coarse"]["features"]
        st = st.numpy() if torch.is_tensor(st) else st
        st = coarsen_mean(np.asarray(st), self.k)
        self.static_feat = st.reshape(st.shape[0], -1)
        self.mean = np.asarray(scales.feat_mean, np.float32)
        self.std = np.asarray(scales.feat_std, np.float32)
        topo = build_topology(domain.network)
        self.interior = np.nonzero(topo.kind <= 1)[0]
        self.base_topo = topo
        ny, nx = graph.ny, graph.nx
        z = coarsen_mean(np.asarray(static["coarse"]["dem2d"]), self.k)
        self.z2 = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(-1)
        sea = coarsen_mean(np.asarray(static["coarse"]["sea"]).astype(float), self.k) > 0.5
        ch = coarsen_mean(domain.channel.astype(float), self.f) > 0
        mask = ~(sea | ch)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        linked = np.zeros(ny * nx, bool)
        linked[graph.c_node2.cpu().numpy()] = True
        mask = mask.reshape(-1) & ~linked
        self.resid_mask2 = torch.as_tensor(mask, device=device)
        sea_rows = sea.copy()
        sea_rows[:-1] = False                                     # northern boundary row, sea cells
        self.bc_mask2 = torch.as_tensor(sea_rows.reshape(-1), device=device)
        self.cell_c = domain.dx * self.f

    @classmethod
    def from_context(cls, ctx, model):
        from ..api import dataset_dir
        root = dataset_dir(ctx.cfg)
        static = torch.load(root / "domain.pt", weights_only=False)
        return cls(ctx.cfg, ctx.domain, static, model.scales, model.graph, next(model.parameters()).device)

    # ------------------------------------------------------------------
    def _common(self, dyn: np.ndarray, gamma: np.ndarray, series: np.ndarray, rain_mmh: np.ndarray,
                scalars: np.ndarray, t_end: float, ssp_drain: float, horton: np.ndarray | None,
                manning_c: np.ndarray):
        dev, gr, sc = self.device, self.graph, self.scales
        feats = np.concatenate([self.static_feat, dyn.reshape(dyn.shape[0], -1)], 0)
        feats = ((feats - self.mean[:, None]) / self.std[:, None]).T.astype(np.float32)
        T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
        nflat = T(manning_c.reshape(-1) * 30.0)
        eff_gamma = np.clip(gamma * ssp_drain, 0.5, 3.0)
        net = apply_gamma(self.domain.network, eff_gamma)
        topo = build_topology(net)
        it = self.interior
        g_node = eff_gamma[np.maximum(topo.reach_of[it], 0)]
        feat1 = np.concatenate([gr.node1_static.cpu().numpy(), (g_node - 1.0)[:, None]], 1)
        feat1[:, 0] = feat1[:, 0] / 5.0
        e1s = gr.e1_static.cpu().numpy()
        src = gr.e1_src.cpu().numpy()
        e1 = np.concatenate([e1s, (topo.n[it][src] * 30.0)[:, None], (g_node[src] - 1.0)[:, None]], 1)
        yb = np.maximum(topo.zbank - topo.z, 0.2)[it]
        b = {
            "feat2": T(feats), "n_edge2": 0.5 * (nflat[gr.e2_src] + nflat[gr.e2_dst]), "manning2": nflat / 30.0,
            "feat1": T(feat1), "edge1": T(e1),
            "series": T(series), "scalars": T(scalars), "t_end": float(t_end),
            "cum_rain": T(np.concatenate([[0.0], np.cumsum(rain_mmh)[:-1]]) * (t_end / len(rain_mmh)) / 3600.0 / 100.0),
            "rain_mmh": T(rain_mmh), "bed1": T(topo.z[it]), "dx1": T(topo.dx[it]),
            "sec1": Sections(T(topo.b[it]), T(topo.m[it]), T(yb),
                             T(self.cfg.solver.slot_width_frac * (topo.b[it] + 2 * topo.m[it] * yb))),
            "n1": T(topo.n[it]), "z2": self.z2, "resid_mask2": self.resid_mask2, "bc_mask2": self.bc_mask2,
        }
        if horton is not None:
            b["horton_mmh"] = T(horton.reshape(horton.shape[0], -1))
        return b

    def snapshots(self, rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The record's depth and velocity snapshots on the model grid. Velocities are averaged
        weighted by depth (the discharge is what averages, not the speed), which reproduces
        what the generator writes when it stores at the model grid directly."""
        n = lambda a: a.numpy() if torch.is_tensor(a) else np.asarray(a)
        h, u, v = n(rec["h"]), n(rec["u"]), n(rec["v"])
        if self.k == 1:
            return h, u, v
        hc = coarsen_mean(h, self.k)
        safe = np.where(hc > 0, hc, 1.0)
        uc = np.where(hc > 0, coarsen_mean(h * u, self.k) / safe, 0.0)
        vc = np.where(hc > 0, coarsen_mean(h * v, self.k) / safe, 0.0)
        return hc, uc, vc

    def from_record(self, rec: dict) -> dict:
        meta, fo = rec["meta"], rec["forcing"]
        from ..forcing.scenarios import ssp_state
        rain = fo["rain_mmh"].numpy()
        series = np.stack([rain / self.scales.rain_scale, fo["tide_m"].numpy(),
                           fo["inflow_m3s"].numpy() / self.scales.inflow_scale])
        manning_c = coarsen_mean(rec["fields"]["manning"].numpy(), self.k)
        b = self._common(coarsen_mean(node_dynamic(rec), self.k), rec["fields"]["gamma"].numpy(), series, rain,
                         scalar_features(meta, fo), float(fo["t_end"]),
                         ssp_state(meta["ssp"]).drainage_investment_factor,
                         coarsen_mean(fo["horton_rate_mmh"].numpy(), self.k), manning_c)
        dev = self.device
        T = lambda a: torch.as_tensor(a.numpy() if torch.is_tensor(a) else a, dtype=torch.float32, device=dev)
        h, u, v = self.snapshots(rec)
        nt = h.shape[0]
        b["times"] = T(rec["times"])
        b["h"] = T(h).reshape(nt, -1)
        b["u"] = T(u).reshape(nt, -1)
        b["v"] = T(v).reshape(nt, -1)
        b["y1"] = T(rec["eta1"]) - b["bed1"][None]
        b["q1"] = T(rec["q1"])
        led = rec["ledger"]
        # the surface volume the model can see, on the model grid: recomputed from the
        # snapshots rather than read from the ledger, whose extent follows the storage grid
        b["v_target"] = T(h.astype(np.float64).sum(axis=(-2, -1)) * self.cell_c ** 2) + T(led["v1d"])
        b["vol_target"] = torch.stack([T(rec["volumes"]["infiltrated_m3"] * np.ones(1))[0],
                                       T(rec["volumes"]["stored_m3"] * np.ones(1))[0],
                                       T(np.array([float(led["bnd2d_in"][-1] - led["bnd2d_out"][-1]
                                                         + led["bnd1d_in"][-1] - led["bnd1d_out"][-1])]))[0]])
        b["tide_now"] = T(np.array([0.0]))
        return b

    def from_site(self, site, scenario):
        from ..api import context
        from ..data.dataset import scalar_features as sf
        from ..engine import build_forcing
        ctx = context()
        dom, cfg, f = self.domain, self.cfg, self.f
        fo = build_forcing(cfg, scenario.return_period_yr, scenario.rcp, scenario.ssp, scenario.horizon_year,
                           scenario.tide_on, scenario.storm_tide_offset_h, dom.nx * dom.ny * dom.dx ** 2 / 1e6)
        lu = ctx.landuse(scenario.ssp, scenario.horizon_year)
        npf = lambda a: a.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(a) else np.asarray(a, float)
        storage, kappa, manning, gamma = (npf(a) for a in (site.storage_depth, site.infil_multiplier,
                                                           site.manning, site.channel_gamma))
        steps = cfg.model.forcing_steps
        grid = np.linspace(0, fo.t_end, steps)
        rain = fo.rain.series(fo.t_end, steps)
        rec_like = {"fields": {"manning": torch.as_tensor(coarsen_mean(manning, f)),
                               "storage": torch.as_tensor(coarsen_mean(storage, f)),
                               "kappa": torch.as_tensor(coarsen_mean(kappa, f)),
                               "landuse_frac": torch.as_tensor(
                                   coarsen_mean(np.eye(LU.N_CLASSES)[lu].transpose(2, 0, 1), f))}}
        series = np.stack([rain / self.scales.rain_scale, [fo.tide(t) for t in grid],
                           np.array([fo.inflow(t) for t in grid]) / self.scales.inflow_scale])
        meta = {"return_period_yr": scenario.return_period_yr, "rcp": scenario.rcp, "ssp": scenario.ssp,
                "horizon_year": scenario.horizon_year, "tide_on": scenario.tide_on,
                "storm_tide_offset_h": scenario.storm_tide_offset_h}
        forc = {"slr_m": fo.slr_m, "rain_factor": fo.rain_factor}
        b = self._common(node_dynamic(rec_like), gamma, series, rain, sf(meta, forc), fo.t_end,
                         fo.ssp.drainage_investment_factor, None, coarsen_mean(manning, f))
        return b, fo, meta

    # inputs that depend on the candidate site; everything else depends only on the scenario
    PER_CANDIDATE = ("feat2", "n_edge2", "manning2", "feat1", "edge1", "bed1", "dx1", "n1")

    def from_sites(self, sites, scenario):
        """Model inputs for many candidate sites under one scenario, stacked on a
        leading dimension. The forcing (hyetograph, tide, inflow, scenario scalars) is
        built once and shared: it is what makes the candidates comparable."""
        from ..api import context
        from ..data.dataset import scalar_features as sf
        from ..engine import build_forcing
        ctx = context()
        dom, cfg, f = self.domain, self.cfg, self.f
        fo = build_forcing(cfg, scenario.return_period_yr, scenario.rcp, scenario.ssp, scenario.horizon_year,
                           scenario.tide_on, scenario.storm_tide_offset_h, dom.nx * dom.ny * dom.dx ** 2 / 1e6)
        lu = ctx.landuse(scenario.ssp, scenario.horizon_year)
        lu_frac = torch.as_tensor(coarsen_mean(np.eye(LU.N_CLASSES)[lu].transpose(2, 0, 1), f))
        steps = cfg.model.forcing_steps
        grid = np.linspace(0, fo.t_end, steps)
        rain = fo.rain.series(fo.t_end, steps)
        series = np.stack([rain / self.scales.rain_scale, [fo.tide(t) for t in grid],
                           np.array([fo.inflow(t) for t in grid]) / self.scales.inflow_scale])
        meta = {"return_period_yr": scenario.return_period_yr, "rcp": scenario.rcp, "ssp": scenario.ssp,
                "horizon_year": scenario.horizon_year, "tide_on": scenario.tide_on,
                "storm_tide_offset_h": scenario.storm_tide_offset_h}
        scal = sf(meta, {"slr_m": fo.slr_m, "rain_factor": fo.rain_factor})
        npf = lambda a: a.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(a) else np.asarray(a, float)
        per = []
        for site in sites:
            storage, kappa, manning, gamma = (npf(a) for a in (site.storage_depth, site.infil_multiplier,
                                                               site.manning, site.channel_gamma))
            mc = coarsen_mean(manning, f)
            rec_like = {"fields": {"manning": torch.as_tensor(mc), "storage": torch.as_tensor(coarsen_mean(storage, f)),
                                   "kappa": torch.as_tensor(coarsen_mean(kappa, f)), "landuse_frac": lu_frac}}
            per.append(self._common(node_dynamic(rec_like), gamma, series, rain, scal, fo.t_end,
                                    fo.ssp.drainage_investment_factor, None, mc))
        b = dict(per[0])
        for k in self.PER_CANDIDATE:
            b[k] = torch.stack([p[k] for p in per])
        b["sec1"] = Sections(*(torch.stack([getattr(p["sec1"], a) for p in per]) for a in ("b", "m", "yb", "ts")))
        return b, fo, meta

    @classmethod
    def select(cls, b: dict, s: slice) -> dict:
        """The candidates ``s`` of a stacked input, sharing the scenario parts."""
        out = dict(b)
        for k in cls.PER_CANDIDATE:
            out[k] = b[k][s]
        out["sec1"] = Sections(*(getattr(b["sec1"], a)[s] for a in ("b", "m", "yb", "ts")))
        return out

    # ------------------------------------------------------------------
    def upsample(self, a: torch.Tensor) -> torch.Tensor:
        """Bilinear upsampling of (nt, ny_c, nx_c) to the model grid (edge-padded)."""
        dom = self.domain
        if self.f == 1:
            return a
        up = F.interpolate(a[:, None], scale_factor=self.f, mode="bilinear", align_corners=False)[:, 0]
        pad_y, pad_x = dom.ny - up.shape[1], dom.nx - up.shape[2]
        return F.pad(up[:, None], (0, pad_x, 0, pad_y), mode="replicate")[:, 0]

    def volume1d(self, y1: torch.Tensor, b: dict) -> torch.Tensor:
        return (b["sec1"].area(y1) * b["dx1"]).sum(-1)

    def implied_mass_error(self, h: torch.Tensor, y1: torch.Tensor, b: dict, vols: torch.Tensor) -> torch.Tensor:
        """|dV_pred - (rain + boundary - infiltration - storage)| / gross, from the surrogate's own outputs."""
        area = self.cell_c ** 2
        v = h.sum(-1) * area + self.volume1d(y1, b)
        dom = self.domain
        rain_vol = float(b["rain_mmh"].sum()) / 1000.0 / 3600.0 * (b["t_end"] / len(b["rain_mmh"])) \
            * dom.nx * dom.ny * dom.dx ** 2
        net = rain_vol + vols[2] - vols[0] - vols[1]
        gross = rain_vol + vols[2].abs() + vols[0] + vols[1]
        return (v[-1] - v[0] - net).abs() / torch.clamp(gross, min=1.0)


def build_model(cfg, dataset_dir, device):
    from ..api import context
    from .geokan_pino import GeoKANPINO, model_dir
    root = Path(dataset_dir)
    static = torch.load(root / "domain.pt", weights_only=False)
    scales = Scales.load(model_dir(cfg) / "scales.json")
    dom = context().domain
    topo = build_topology(dom.network)
    # the graph lives on the MODEL grid; the dataset may be stored on a finer one
    k = int(cfg.data.coarsen) // int(static["coarse"]["factor"])
    z = coarsen_mean(np.asarray(static["coarse"]["dem2d"]), k)
    graph = build_graph(z, dom.dx * int(cfg.data.coarsen), topo).to(device)
    n_feat2 = len(scales.feat_mean)
    return GeoKANPINO(cfg.model, n_feat2, graph, scales).to(device)
