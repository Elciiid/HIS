"""GeoKAN-PINO: Geometry-aware Kolmogorov-Arnold Physics-Informed Neural Operator.

A time-conditioned operator:
  encode   node features + forcing series + scenario scalars -> latent graph state
  process  Chebyshev-KAN message passing on the 2-D mesh, the 1-D chain and the
           coupling edges (separate edge functions for each), with one FFT
           mixing layer over the 2-D grid for a global receptive field
  decode   latent + query time t -> (h, u, v) on 2-D nodes, (y, Q) on 1-D nodes

Non-dimensionalisation (mandatory for PINN convergence):
  H0 = 95th-percentile wet depth, L = domain length, U0 = sqrt(g H0), T0 = L / U0
  h* = h/H0, u* = u/U0, x* = x/L, t* = t/T0, z* = z/H0.
Substituting into  d(hu)/dt + d(hu^2)/dx + d(huv)/dy + g h d(eta)/dx = -g n^2 u|U| / h^(1/3)
and dividing by U0 H0 / T0 = g H0^2 / L gives
  d(h*u*)/dt* + d(h*u*^2)/dx* + d(h*u*v*)/dy* + h* d(eta*)/dx* = -(g n^2 L / H0^(4/3)) u*|U*| / h*^(1/3)
and continuity becomes  dh*/dt* + div*(h* U*) = (R - I) T0 / H0.
Depth positivity is hard: h* = softplus(raw).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..data.dataset import N_SCALAR, Scales
from .chebykan import ChebyKAN, ChebyMLP
from .graph import Graph

log = logging.getLogger("hydrointel.model")
N_FORCING = 3          # rain, tide, inflow


def interp_series(series: torch.Tensor, t: torch.Tensor, t_end: float) -> torch.Tensor:
    """Linear interpolation of equally spaced samples on [0, t_end] (differentiable in t)."""
    n = series.shape[-1]
    x = torch.clamp(t / t_end, 0.0, 1.0) * (n - 1)
    i0 = torch.clamp(torch.floor(x.detach()).long(), 0, n - 2)
    w = x - i0.to(x.dtype)
    return series[i0] * (1 - w) + series[i0 + 1] * w


class ForcingEncoder(nn.Module):
    def __init__(self, steps: int, d: int, degree: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_FORCING, d, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, padding=2, stride=2), nn.GELU())
        self.fourier_k = 6
        self.head = ChebyMLP([2 * d + 2 * N_FORCING * self.fourier_k + N_SCALAR, d, d], degree, final_norm=True)

    def forward(self, series: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(series[None])[0]                              # (d, T')
        pooled = torch.cat([h.mean(-1), h.amax(-1)])
        spec = torch.fft.rfft(series.float(), dim=-1)[:, 1:self.fourier_k + 1]
        four = torch.cat([spec.real, spec.imag], -1).reshape(-1) / series.shape[-1]
        return self.head(torch.cat([pooled, four, scalars]))


class SpectralMix(nn.Module):
    """FNO-style global mixing: low-mode learnable complex weights on the 2-D grid."""

    def __init__(self, d: int, modes: int):
        super().__init__()
        self.m = modes
        scale = 1.0 / (d * d)
        self.w = nn.Parameter(scale * torch.randn(d, d, 2 * modes, modes, 2))
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()                                             # (..., N, d)
            lead, d = xf.shape[:-2], xf.shape[-1]
            g = xf.transpose(-1, -2).reshape(*lead, d, ny, nx)
            spec = torch.fft.rfft2(g, norm="ortho")
            m = min(self.m, ny // 2, nx // 2 + 1)
            idx_y = torch.cat([torch.arange(m), torch.arange(ny - m, ny)]).to(x.device)
            sub = spec[..., idx_y, :m]                                 # (..., d, 2m, m)
            w = torch.view_as_complex(self.w[:, :, :2 * m, :m].contiguous())
            mixed = torch.einsum("...iyx,oiyx->...oyx", sub, w)
            out = torch.zeros_like(spec)
            out[..., idx_y, :m] = mixed
            back = torch.fft.irfft2(out, s=(ny, nx), norm="ortho").reshape(*lead, d, -1).transpose(-1, -2)
            return xf + self.norm(F.gelu(back + self.lin(xf)))


class Block(nn.Module):
    def __init__(self, d: int, degree: int):
        super().__init__()
        self.e2 = ChebyKAN(2 * d + 5, d, degree)
        self.u2 = ChebyKAN(3 * d, d, degree)
        self.e1 = ChebyKAN(2 * d + 5, d, degree)
        self.u1 = ChebyKAN(4 * d, d, degree)
        self.c21 = ChebyKAN(2 * d + 2, d, degree)
        self.c12 = ChebyKAN(2 * d + 2, d, degree)
        self.n2 = nn.LayerNorm(d)
        self.n1 = nn.LayerNorm(d)

    def forward(self, x2, x1, g, gr: Graph, e2, e1):
        # nodes live on dim -2, so a leading batch of candidates passes straight through
        at = lambda x, idx: x.index_select(-2, idx)
        m2 = self.e2(torch.cat([at(x2, gr.e2_src), at(x2, gr.e2_dst), e2], -1))
        agg2 = m2.new_zeros(*x2.shape[:-1], m2.shape[-1]).index_add_(-2, gr.e2_dst, m2)
        m1 = self.e1(torch.cat([at(x1, gr.e1_src), at(x1, gr.e1_dst), e1], -1))
        agg1 = m1.new_zeros(*x1.shape[:-1], m1.shape[-1]).index_add_(-2, gr.e1_dst, m1)
        a2, a1 = at(x2, gr.c_node2), at(x1, gr.c_node1)
        cs = gr.c_static.expand(*a2.shape[:-1], gr.c_static.shape[-1])
        to1 = self.c21(torch.cat([a2, a1, cs], -1))
        to2 = self.c12(torch.cat([a1, a2, cs], -1))
        agg12 = to1.new_zeros(*x1.shape[:-1], to1.shape[-1]).index_add_(-2, gr.c_node1, to1)
        agg2 = agg2.index_add(-2, gr.c_node2, to2.to(agg2.dtype))
        x2n = self.n2(x2 + self.u2(torch.cat([x2, agg2, g.expand(*x2.shape[:-1], g.shape[-1])], -1)))
        x1n = self.n1(x1 + self.u1(torch.cat([x1, agg1, agg12, g.expand(*x1.shape[:-1], g.shape[-1])], -1)))
        return x2n, x1n


class GeoKANPINO(nn.Module):
    def __init__(self, mcfg: ModelConfig, n_feat2: int, graph: Graph, scales: Scales):
        super().__init__()
        d, deg = mcfg.latent, mcfg.degree
        self.cfg = mcfg
        self.graph = graph
        self.scales = scales
        self.forcing = ForcingEncoder(mcfg.forcing_steps, d, deg)
        self.enc2 = ChebyMLP([n_feat2 + d, d, d], deg, final_norm=True)
        self.enc1 = ChebyMLP([6 + d, d, d], deg, final_norm=True)
        self.blocks = nn.ModuleList(Block(d, deg) for _ in range(mcfg.n_blocks))
        self.spectral = SpectralMix(d, mcfg.spectral_modes)
        self.K = mcfg.time_fourier
        tf = 1 + 2 * self.K + 4
        self.dec2 = ChebyMLP([d + tf, d, d, 3], deg)
        self.dec1 = ChebyMLP([d + tf, d, d, 2], deg)
        self.y1_scale = float(graph.c_static[:, 0].mean())            # mean bankfull depth [m]
        # event totals: infiltration, retained storage, net boundary inflow (non-dim by V_scale)
        self.vol_head = ChebyMLP([3 * d, d, 3], deg)
        self.use_checkpoint = True
        self.decode_chunk_nodes = 150_000      # query-time x node points per inference chunk

    # ------------------------------------------------------------ encode
    def encode(self, b: dict):
        gr = self.graph
        g = self.forcing(b["series"], b["scalars"])            # one scenario: shared by every candidate
        f2, f1, ne = b["feat2"], b["feat1"], b["n_edge2"]
        x2 = self.enc2(torch.cat([f2, g.expand(*f2.shape[:-1], g.shape[-1])], -1))
        x1 = self.enc1(torch.cat([f1, g.expand(*f1.shape[:-1], g.shape[-1])], -1))
        e2 = torch.cat([gr.e2_static.expand(*ne.shape, gr.e2_static.shape[-1]), ne[..., None]], -1)
        e1 = b["edge1"]
        half = len(self.blocks) // 2
        for k, blk in enumerate(self.blocks):
            if k == half:
                x2 = self.spectral(x2, gr.ny, gr.nx)
            if self.use_checkpoint and self.training:
                x2, x1 = checkpoint(blk, x2, x1, g, gr, e2, e1, use_reentrant=False)
            else:
                x2, x1 = blk(x2, x1, g, gr, e2, e1)
        return x2, x1, g

    def volumes(self, x2, x1, g):
        """Predicted event totals [m^3]: (infiltrated, stored, net boundary inflow)."""
        lead = x2.shape[:-2]
        out = self.vol_head(torch.cat([g.expand(*lead, g.shape[-1]), x2.mean(-2), x1.mean(-2)], -1))
        return torch.stack([F.softplus(out[..., 0]), F.softplus(out[..., 1]), out[..., 2]], -1) * self.scales_v()

    def scales_v(self) -> float:
        return float(getattr(self.scales, "V_scale", 1.0))

    # ------------------------------------------------------------ decode
    def time_features(self, tstar: torch.Tensor, b: dict) -> torch.Tensor:
        sc = self.scales
        t = tstar * sc.T0
        te = b["t_end"]
        ang = 2 * math.pi * t / te
        k = torch.arange(1, self.K + 1, device=t.device, dtype=t.dtype)
        four = torch.cat([torch.sin(ang * k), torch.cos(ang * k)], -1)
        rain = interp_series(b["series"][0], t[..., 0], te)[..., None]
        cum = interp_series(b["cum_rain"], t[..., 0], te)[..., None]
        tide = interp_series(b["series"][1], t[..., 0], te)[..., None]
        infl = interp_series(b["series"][2], t[..., 0], te)[..., None]
        return torch.cat([tstar, four, rain, cum, tide, infl], -1)

    @staticmethod
    def _with_time(x, tf):
        """Latent (..., N, d) joined with time features (Q, N, f) -> (..., Q, N, d + f)."""
        lead = x.shape[:-2]
        xq = x.unsqueeze(-3).expand(*lead, tf.shape[0], *x.shape[-2:])
        return torch.cat([xq, tf.expand(*lead, *tf.shape)], -1)

    def _dec2(self, x2, b, tstar):
        raw = self.dec2(self._with_time(x2, self.time_features(tstar, b)))
        return torch.stack([F.softplus(raw[..., 0]), raw[..., 1], raw[..., 2]], -1)

    def _dec1(self, x1, b, tstar):
        raw = self.dec1(self._with_time(x1, self.time_features(tstar, b)))
        return torch.stack([F.softplus(raw[..., 0]), raw[..., 1]], -1)

    def decode(self, x2, x1, b, t_s: torch.Tensor, need_dt: bool = False):
        """t_s: (Q,) seconds. Returns non-dimensional fields (Q, N, C) and, if requested,
        their derivatives with respect to t* (forward-mode AD)."""
        sc = self.scales
        ts2 = (t_s / sc.T0)[:, None, None].expand(-1, x2.shape[-2], 1).contiguous()
        ts1 = (t_s / sc.T0)[:, None, None].expand(-1, x1.shape[-2], 1).contiguous()
        if need_dt:
            o2, d2 = torch.func.jvp(lambda s: self._dec2(x2, b, s), (ts2,), (torch.ones_like(ts2),))
            o1, d1 = torch.func.jvp(lambda s: self._dec1(x1, b, s), (ts1,), (torch.ones_like(ts1),))
            return o2, o1, d2, d1
        if torch.is_grad_enabled():
            return self._dec2(x2, b, ts2), self._dec1(x1, b, ts1), None, None
        # inference: decode a few output times at a time to bound memory
        k = max(1, int(self.decode_chunk_nodes // max(x2.shape[-2], 1)))
        o2 = torch.cat([self._dec2(x2, b, ts2[i:i + k]) for i in range(0, ts2.shape[0], k)], -3)
        o1 = torch.cat([self._dec1(x1, b, ts1[i:i + k]) for i in range(0, ts1.shape[0], k)], -3)
        return o2, o1, None, None

    def dimensional(self, o2, o1):
        sc = self.scales
        h = o2[..., 0] * sc.H0
        u = o2[..., 1] * sc.U0
        v = o2[..., 2] * sc.U0
        y1 = o1[..., 0] * self.y1_scale
        q1 = o1[..., 1] * sc.Q0
        return h, u, v, y1, q1

    # ------------------------------------------------------------ inference
    def _builder(self, ctx):
        """Input builder, reused across calls: Part B calls predict many thousands of times."""
        from .batch import SampleBuilder
        key = id(ctx)
        if getattr(self, "_builder_key", None) != key:
            self._builder_cache = SampleBuilder.from_context(ctx, self)
            self._builder_key = key
        return self._builder_cache

    @torch.no_grad()
    def predict_result(self, site, scenario, ctx, output_interval_s=None):
        """One candidate, full output (kept for callers of the original interface)."""
        return self.predict_many([site], scenario, ctx, output_interval_s, detail="full")[0]

    @torch.no_grad()
    def predict_many(self, sites, scenario, ctx, output_interval_s=None, detail: str = "summary",
                     encode_batch: int | None = None):
        """Evaluate many candidate sites against one scenario.

        The scenario's forcing is encoded once. Candidates are encoded together in
        groups of ``encode_batch`` (the graph encoder needs ~0.5 GB per candidate at
        full resolution, so all of a 64-member population cannot sit on a 6 GB card at
        once), then decoded in chunks of output times.

        ``detail="summary"`` never materialises the (nt, ny, nx) histories: every
        time chunk updates running maxima and point series and is then dropped. It
        saves memory and host transfer, not decoder arithmetic -- a peak depth needs
        the decoder evaluated at every output time either way.
        """
        from ..api import FloodResult, FloodSummary
        from ..solver.swe1d import Sections
        from ..viz.depthmap import monitoring_points
        if detail not in ("full", "summary"):
            raise ValueError(f"detail must be 'full' or 'summary', not {detail!r}")
        self.eval()
        builder = self._builder(ctx)
        dom, gr, sc = builder.domain, self.graph, self.scales
        b, fo, _ = builder.from_sites(sites, scenario)
        dev = b["feat2"].device
        times = fo.times(output_interval_s or ctx.cfg.solver.output_interval_s)
        t = torch.as_tensor(times, dtype=torch.float32, device=dev)
        pts = monitoring_points(dom)
        pj = torch.as_tensor([p["j"] for p in pts], device=dev)
        pi = torch.as_tensor([p["i"] for p in pts], device=dev)
        z_pts = torch.as_tensor(dom.dem2d[pj.cpu().numpy(), pi.cpu().numpy()], dtype=torch.float32, device=dev)
        reach_of = torch.as_tensor(builder.base_topo.reach_of[builder.interior], device=dev)
        n_reach = dom.network.n_reaches
        area_c = builder.cell_c ** 2
        rain_vol = float(b["rain_mmh"].sum()) / 1000.0 / 3600.0 * (b["t_end"] / len(b["rain_mmh"])) \
            * dom.nx * dom.ny * dom.dx ** 2
        eb = max(1, int(encode_batch or getattr(self, "encode_batch", 4)))
        out = []
        for s0 in range(0, len(sites), eb):
            bs = builder.select(b, slice(s0, s0 + eb))
            nb = bs["feat2"].shape[0]
            x2, x1, g = self.encode(bs)
            vols = self.volumes(x2, x1, g)                                     # (nb, 3)
            n2, n1 = x2.shape[-2], x1.shape[-2]
            qk = max(1, int(self.decode_chunk_nodes // (n2 * nb)))
            dmax = smax = None
            p_depth, h_sum, y1s, q1s = [], [], [], []
            if detail == "full":     # filled chunk by chunk: no second copy at the end
                full = [torch.empty((nb, len(times), dom.ny, dom.nx), dtype=torch.float32) for _ in range(3)]
            for q0 in range(0, len(times), qk):
                ts = t[q0:q0 + qk] / sc.T0
                o2 = self._dec2(x2, bs, ts[:, None, None].expand(-1, n2, 1))  # (nb, qc, n2, 3)
                o1 = self._dec1(x1, bs, ts[:, None, None].expand(-1, n1, 1))
                h, u, v, y1, q1 = self.dimensional(o2, o1)
                qc = h.shape[1]
                up = lambda a: builder.upsample(a.reshape(nb * qc, gr.ny, gr.nx)).reshape(nb, qc, dom.ny, dom.nx)
                H, U, V = up(h), up(u), up(v)
                spd = torch.sqrt(U * U + V * V)
                dmax = H.amax(1) if dmax is None else torch.maximum(dmax, H.amax(1))
                smax = spd.amax(1) if smax is None else torch.maximum(smax, spd.amax(1))
                p_depth.append(H[:, :, pj, pi])
                h_sum.append(h.sum(-1))
                y1s.append(y1)
                q1s.append(q1)
                if detail == "full":
                    for acc, a in zip(full, (H, U, V)):
                        acc[:, q0:q0 + qc] = a.float().cpu()
            y1 = torch.cat(y1s, 1)                                             # (nb, nt, n1)
            q1 = torch.cat(q1s, 1)
            eta1 = bs["bed1"][:, None] + y1
            sec = Sections(*(getattr(bs["sec1"], a)[:, None] for a in ("b", "m", "yb", "ts")))
            vol = torch.cat(h_sum, 1) * area_c + (sec.area(y1) * bs["dx1"][:, None]).sum(-1)
            net = rain_vol + vols[:, 2] - vols[:, 0] - vols[:, 1]
            gross = rain_vol + vols[:, 2].abs() + vols[:, 0] + vols[:, 1]
            err = (vol[:, -1] - vol[:, 0] - net).abs() / torch.clamp(gross, min=1.0)
            peak_stage = torch.stack([eta1[..., reach_of == r].amax((-1, -2)) for r in range(n_reach)], -1)
            peak_q = torch.stack([q1[..., reach_of == r].abs().amax((-1, -2)) for r in range(n_reach)], -1)
            p_depth = torch.cat(p_depth, 1)                                    # (nb, nt, P)
            for k in range(nb):
                common = dict(infiltrated_volume_m3=float(vols[k, 0]), stored_volume_m3=float(vols[k, 1]),
                              mass_balance_error=float(err[k]), provenance=None, in_distribution=False)
                ex = {"forcing": fo, "boundary_net_in_m3": float(vols[k, 2]),
                      "note": "volumes are surrogate estimates of the engine ledger; "
                              "their error is reported in validation_report.md"}
                if detail == "full":
                    out.append(FloodResult(
                        depth_max=dmax[k].cpu(), depth_series=full[0][k], u=full[1][k], v=full[2][k],
                        channel_eta=eta1[k].cpu(), channel_q=q1[k].cpu(), times=t.cpu(),
                        speed_max=smax[k].cpu(), extras=ex, **common))
                else:
                    level = (z_pts[None] + p_depth[k]).cpu()
                    out.append(FloodSummary(
                        depth_max=dmax[k].cpu(), speed_max=smax[k].cpu(), times=t.cpu(),
                        point_level={p["name"]: level[:, j] for j, p in enumerate(pts)},
                        reach_peak_stage=peak_stage[k].cpu(), reach_peak_discharge=peak_q[k].cpu(),
                        extras=ex, **common))
            del x2, x1
        return out


def model_dir(cfg) -> Path:
    return Path(cfg.outdir) / "model" / cfg.config_hash("domain", "solver", "forcing", "data", "model", "train")


def load_surrogate(cfg, domain, device) -> GeoKANPINO:
    from .batch import build_model
    d = model_dir(cfg)
    ck = d / "model.pt"
    if not ck.exists():
        raise FileNotFoundError(f"no trained surrogate at {ck}: run `python -m hydrointel.cli train` "
                                "(and `evaluate`) for this configuration first")
    state = torch.load(ck, map_location=device, weights_only=False)
    model = build_model(cfg, state["dataset_dir"], device)
    model.load_state_dict(state["model"])
    model.eval()
    return model
