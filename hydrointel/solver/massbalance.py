"""Volume accounting for the coupled engine.

relative_error = |dV - sum(sources)| / max(sum|sources|, eps)
dV counts 2-D surface water, 1-D channel water and sub-grid storage. eps is
1e-9 of the initial stored volume (plus 1 m^3) so that runs with no sources
(e.g. lake at rest) are judged against round-off, not divided by zero.
Accumulators stay on the device as float64 tensors to avoid a sync per step.
"""
from __future__ import annotations

import torch

TERMS = ("rain", "infiltration", "bnd2d_in", "bnd2d_out", "bnd1d_in", "bnd1d_out", "clip")


class MassBalance:
    def __init__(self, device, v0: float):
        self.device = device
        self.v0 = float(v0)
        # one accumulator per (term, device) so adding never forces a device copy
        self.acc: dict[tuple[str, str], torch.Tensor] = {}
        self.series: list[dict] = []

    def add(self, name: str, value) -> None:
        if name not in TERMS:
            raise KeyError(name)
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), dtype=torch.float64)
        key = (name, str(value.device))
        if key not in self.acc:
            self.acc[key] = torch.zeros((), dtype=torch.float64, device=value.device)
        self.acc[key] += value.to(torch.float64)

    def sources(self) -> dict[str, float]:
        out = {k: 0.0 for k in TERMS}
        for (k, _), v in self.acc.items():
            out[k] += float(v)
        return out

    def check(self, volume_now: float, t: float, record: bool = True) -> float:
        s = self.sources()
        net = s["rain"] - s["infiltration"] + s["bnd2d_in"] - s["bnd2d_out"] + s["bnd1d_in"] - s["bnd1d_out"]
        gross = s["rain"] + s["infiltration"] + s["bnd2d_in"] + s["bnd2d_out"] + s["bnd1d_in"] + s["bnd1d_out"]
        dv = volume_now - self.v0
        # "clip" is volume created by clamping negative depths to zero; it is an error term, not a source
        err = abs(dv - net) / max(gross, 1e-9 * abs(self.v0) + 1.0)
        if record:
            self.series.append({"t": t, "volume": volume_now, "net_sources": net, "gross": gross,
                                "clip": s["clip"], "rel_error": err})
        return err

    def assert_ok(self, volume_now: float, t: float, tol: float, label: str = "") -> float:
        err = self.check(volume_now, t)
        if not err < tol:
            raise AssertionError(f"mass balance violated{(' in ' + label) if label else ''}: "
                                 f"relative error {err:.3e} >= {tol:.1e} (sources {self.sources()})")
        return err
