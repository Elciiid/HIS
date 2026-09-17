"""Infiltration models: Horton decay and constant rate, with an intervention
multiplier kappa(x,y) on both f0 and fc."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..domain import landuse as LU

MMH_TO_MS = 1.0 / 3.6e6


@dataclass
class Horton:
    f0: torch.Tensor      # m/s
    fc: torch.Tensor      # m/s
    k: torch.Tensor       # 1/s
    t0: float = 0.0       # decay clock starts here (rain onset)

    @classmethod
    def from_landuse(cls, landuse: np.ndarray, kappa, device, dtype, t0: float = 0.0) -> "Horton":
        kap = torch.as_tensor(np.asarray(kappa, dtype=np.float64) if not torch.is_tensor(kappa) else kappa,
                              dtype=dtype, device=device)
        t = lambda a: torch.as_tensor(LU.lookup(landuse, a), dtype=dtype, device=device)
        return cls(t("f0_mmh") * MMH_TO_MS * kap, t("fc_mmh") * MMH_TO_MS * kap, t("k_per_h") / 3600.0, t0)

    @classmethod
    def constant(cls, rate_ms, shape, device, dtype) -> "Horton":
        r = torch.full(shape, float(rate_ms), device=device, dtype=dtype)
        return cls(r, r.clone(), torch.zeros(shape, device=device, dtype=dtype))

    def rate(self, t: float) -> torch.Tensor:
        """f(t) = fc + (f0 - fc) exp(-k t)  [m/s]"""
        tt = torch.clamp(torch.as_tensor(t) - self.t0, min=0.0)
        return self.fc + (self.f0 - self.fc) * torch.exp(-self.k * tt)
