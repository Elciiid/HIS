"""Two-stage residual surrogate: baseline first, then the change a design makes.

  stage 1   static features + forcing + the scenario's baseline land surface
            -> baseline flood (h, u, v; y, Q)                       [GeoKANPINO]
  stage 2   static features + forcing + the candidate SiteState
            + the difference between the candidate and the baseline site
            + stage 1's latent state (its representation of the baseline flood)
            -> delta (h, u, v; y, Q) and the candidate's event totals  [DeltaGeoKAN]
  result    baseline + delta (depth clamped at zero)

Why: Part B asks what a design changes, typically 2-6 cm on a flood that is
metres deep in places. A single network that rebuilds the whole field for every
candidate has to recover a 2 cm difference from two independently predicted 1 m
fields; Phase 1's surrogate kept 5% of the engine's effect magnitude
(artifacts/diagnostics/a1.json). Here the effect is the second stage's whole
output. In Part B stage 1 runs once per scenario and is cached; only stage 2
runs per candidate.

Stage 1's latent is passed to stage 2 detached: stage 1 is trained on baseline
records only, and stage 2 cannot pull it away from them.
"""
from __future__ import annotations

import torch
from torch import nn

from .chebykan import ChebyMLP
from .geokan_pino import GeoKANPINO

N_DELTA = 3            # manning, storage, kappa: the per-cell fields a design changes
N_STATIC = 9           # static node features ahead of them in feat2 (graph.static_node_features)


class DeltaGeoKAN(GeoKANPINO):
    """GeoKANPINO whose outputs are changes, not states: no positivity on h, and extra
    node inputs (the design's differences and stage 1's latent)."""

    def __init__(self, mcfg, n_feat2: int, graph, scales, n_extra1: int):
        super().__init__(mcfg, n_feat2, graph, scales)
        d, deg = mcfg.latent, mcfg.degree
        self.enc1 = ChebyMLP([6 + n_extra1 + d, d, d], deg, final_norm=True)
        # start at "this design changes nothing": the output layers of both delta decoders
        # begin at zero, so training moves away from no-change rather than from noise
        for dec in (self.dec2, self.dec1):
            torch.nn.init.zeros_(dec.net[-1].coef)

    def _dec2(self, x2, b, tstar):
        return self.dec2(self._with_time(x2, self.time_features(tstar, b)))

    def _dec1(self, x1, b, tstar):
        return self.dec1(self._with_time(x1, self.time_features(tstar, b)))


class TwoStage(nn.Module):
    def __init__(self, mcfg, n_feat2: int, graph, scales):
        super().__init__()
        d = mcfg.latent
        self.stage1 = GeoKANPINO(mcfg, n_feat2, graph, scales)
        self.stage2 = DeltaGeoKAN(mcfg, n_feat2 + N_DELTA + d, graph, scales, n_extra1=d)
        self.graph, self.scales = graph, scales
        self.y1_scale = self.stage1.y1_scale

    @staticmethod
    def stage2_inputs(b_site: dict, b_base: dict, x2s: torch.Tensor, x1s: torch.Tensor) -> dict:
        """Stage 2's input: the candidate's batch with the design's differences from the
        baseline and stage 1's (detached) latent appended to the node features."""
        f2, f0 = b_site["feat2"], b_base["feat2"]
        sl = slice(N_STATIC, N_STATIC + N_DELTA)
        delta = f2[..., sl] - f0[..., sl]
        x2s, x1s = x2s.detach(), x1s.detach()
        lead = f2.shape[:-2]
        out = dict(b_site)
        out["feat2"] = torch.cat([f2, delta, x2s.expand(*lead, *x2s.shape[-2:])], -1)
        f1 = b_site["feat1"]
        out["feat1"] = torch.cat([f1, x1s.expand(*f1.shape[:-2], *x1s.shape[-2:])], -1)
        return out

    @staticmethod
    def combine(ob2, ob1, od2, od1):
        """Modified state = baseline + delta, non-dimensional (Q, N, C)."""
        return ob2 + od2, ob1 + od1

    # ------------------------------------------------------------ inference
    # GeoKANPINO's batched inference, with the two-stage hooks below: stage 1 is encoded
    # once per call on baseline(scenario), stage 2 once per group of candidates.
    predict_many = GeoKANPINO.predict_many
    predict_result = GeoKANPINO.predict_result
    _builder = GeoKANPINO._builder
    decode_chunk_nodes = 150_000

    def dimensional(self, o2, o1):
        return self.stage1.dimensional(o2, o1)

    def _prepare_call(self, sites, scenario, builder) -> None:
        from ..api import baseline_site
        b0, _, _ = builder.from_sites([baseline_site(scenario)], scenario)
        x2s, x1s, _ = self.stage1.encode(b0)
        self._call = (b0, x2s, x1s)

    def _release_call(self) -> None:
        self._call = None

    def _group_state(self, bs):
        b0, x2s, x1s = self._call
        b2 = self.stage2_inputs(bs, b0, x2s, x1s)
        x2d, x1d, gd = self.stage2.encode(b2)
        return (x2d, x1d, b2), self.stage2.volumes(x2d, x1d, gd)

    def _group_decode(self, state, bs, ts):
        x2d, x1d, b2 = state
        b0, x2s, x1s = self._call
        tq = lambda x: ts[:, None, None].expand(-1, x.shape[-2], 1)
        o2 = self.stage1._dec2(x2s, b0, tq(x2s)) + self.stage2._dec2(x2d, b2, tq(x2d))
        o1 = self.stage1._dec1(x1s, b0, tq(x1s)) + self.stage2._dec1(x1d, b2, tq(x1d))
        # depth cannot go negative, whatever the delta says
        return (torch.cat([o2[..., :1].clamp(min=0.0), o2[..., 1:]], -1),
                torch.cat([o1[..., :1].clamp(min=0.0), o1[..., 1:]], -1))
