"""Upstream inflow hydrograph: Clark unit-hydrograph routing of the design rainfall
over the contributing catchment (time-area histogram + linear reservoir).

The time-area curve is the standard dimensionless HEC-style form
  A(t)/A = 1.414 (t/tc)^1.5            for t <= tc/2
  1 - A(t)/A = 1.414 (1 - t/tc)^1.5    for t >  tc/2
and the reservoir is routed with the Muskingum-form Clark coefficients
  O_k = c I_k + (1 - c) O_{k-1},  c = 2 dt / (2 K + dt).
Runoff generation uses a constant runoff coefficient (placeholder).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CatchmentConfig
from .hyetograph import MM_H_TO_M_S, Hyetograph


def time_area_fraction(t_over_tc: np.ndarray) -> np.ndarray:
    s = np.clip(t_over_tc, 0.0, 1.0)
    return np.where(s <= 0.5, 1.414 * s ** 1.5, 1.0 - 1.414 * (1.0 - s) ** 1.5)


@dataclass
class InflowHydrograph:
    step_s: float
    q_m3s: np.ndarray
    t0: float = 0.0

    def __call__(self, t: float) -> float:
        x = (t - self.t0) / self.step_s
        if x <= 0:
            return float(self.q_m3s[0])
        if x >= len(self.q_m3s) - 1:
            return float(self.q_m3s[-1])
        k = int(x)
        w = x - k
        return float((1 - w) * self.q_m3s[k] + w * self.q_m3s[k + 1])

    @property
    def peak(self) -> float:
        return float(self.q_m3s.max())

    def volume_m3(self) -> float:
        return float(np.trapezoid(self.q_m3s, dx=self.step_s))


def clark_hydrograph(rain: Hyetograph, cc: CatchmentConfig, t_end: float, arf: float = 1.0,
                     step_s: float = 60.0) -> InflowHydrograph:
    n = int(np.ceil(t_end / step_s)) + 1
    t = np.arange(n) * step_s
    excess = np.array([rain.rate_ms(tt) for tt in t]) * arf * cc.runoff_coeff        # m/s
    tc = cc.tc_h * 3600.0
    m = int(np.ceil(tc / step_s))
    ta = np.diff(time_area_fraction(np.arange(m + 1) * step_s / tc))                  # fraction per step
    inflow = np.convolve(excess, ta)[:n] * cc.area_km2 * 1e6                         # m^3/s translated
    K = cc.storage_k_h * 3600.0
    c = 2 * step_s / (2 * K + step_s)
    out = np.zeros(n)
    for k in range(1, n):
        out[k] = c * 0.5 * (inflow[k] + inflow[k - 1]) + (1 - c) * out[k - 1]
    return InflowHydrograph(step_s, out + cc.baseflow_m3s, 0.0)
