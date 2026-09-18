"""Volume accounting for the coupled engine.

relative_error = |dV - sum(sources)| / max(sum|sources|, eps)
dV counts 2-D surface water, 1-D channel water and sub-grid storage. eps is
1e-9 of the initial stored volume (plus 1 m^3) so that runs with no sources
(e.g. lake at rest) are judged against round-off, not divided by zero.
Accumulators stay on the device as float64 tensors to avoid a sync per step.

A batched run is accounted per member, never pooled: every quantity carries the
batch dimension through to the error, so one failing storm out of sixteen is
still a failure and is named as such.
"""
from __future__ import annotations

import numpy as np
import torch

TERMS = ("rain", "infiltration", "bnd2d_in", "bnd2d_out", "bnd1d_in", "bnd1d_out", "clip")


class MassBalance:
    def __init__(self, device, v0):
        self.device = device
        self.v0 = np.asarray(_np(v0), dtype=np.float64)
        self.batch = self.v0.shape[0] if self.v0.ndim else None
        # one accumulator per (term, device) so adding never forces a device copy
        self.acc: dict[tuple[str, str], torch.Tensor] = {}
        self.series: list[dict] = []

    def add(self, name: str, value) -> None:
        if name not in TERMS:
            raise KeyError(name)
        if not torch.is_tensor(value):
            value = torch.tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)
        value = value.to(torch.float64)
        if self.batch is not None and value.dim() == 0:
            value = value.expand(self.batch)
        if value.shape != self.v0.shape:
            raise ValueError(f"mass term '{name}' has shape {tuple(value.shape)}, expected {self.v0.shape}")
        key = (name, str(value.device))
        if key not in self.acc:
            self.acc[key] = torch.zeros_like(value)
        self.acc[key] += value

    def sources(self) -> dict:
        """Accumulated volume per term: a float unbatched, an array per member when batched."""
        zero = np.zeros(self.v0.shape)
        out = {k: zero.copy() for k in TERMS}
        for (k, _), v in self.acc.items():
            out[k] = out[k] + _np(v)
        return {k: (v if self.batch is not None else float(v)) for k, v in out.items()}

    def check(self, volume_now, t: float, record: bool = True):
        s = self.sources()
        net = s["rain"] - s["infiltration"] + s["bnd2d_in"] - s["bnd2d_out"] + s["bnd1d_in"] - s["bnd1d_out"]
        gross = s["rain"] + s["infiltration"] + s["bnd2d_in"] + s["bnd2d_out"] + s["bnd1d_in"] + s["bnd1d_out"]
        dv = np.asarray(_np(volume_now), dtype=np.float64) - self.v0
        # "clip" is volume created by clamping negative depths to zero; it is an error term, not a source
        err = np.abs(dv - net) / np.maximum(gross, 1e-9 * np.abs(self.v0) + 1.0)
        if record:
            f = (lambda x: x) if self.batch is not None else float
            self.series.append({"t": t, "volume": f(_np(volume_now)), "net_sources": f(net), "gross": f(gross),
                                "clip": f(s["clip"]), "rel_error": f(err)})
        return err if self.batch is not None else float(err)

    def assert_ok(self, volume_now, t: float, tol: float, label: str = ""):
        err = self.check(volume_now, t)
        bad = np.nonzero(~(np.asarray(err) < tol))[0] if self.batch is not None else ([] if err < tol else [0])
        if len(bad):
            where = f" in {label}" if label else ""
            if self.batch is not None:
                detail = ", ".join(f"member {int(i)}: {np.asarray(err)[int(i)]:.3e}" for i in bad[:8])
                raise AssertionError(f"mass balance violated{where} for {len(bad)}/{self.batch} batch members "
                                     f"(tolerance {tol:.1e}) -- {detail}")
            raise AssertionError(f"mass balance violated{where}: relative error {float(err):.3e} >= {tol:.1e} "
                                 f"(sources {self.sources()})")
        return err


def _np(v):
    if torch.is_tensor(v):
        return v.detach().to(torch.float64).cpu().numpy()
    return v
