"""Design-storm hyetographs (alternating-block method) and gauge series."""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import idf as IDF

MM_H_TO_M_S = 1.0 / 3.6e6
SQ_MI_PER_KM2 = 0.386102


def areal_reduction_factor(area_km2: float, duration_h: float) -> float:
    """Leclerc & Schaake (1972) exponential fit to the US Weather Bureau TP-29
    depth-area curves: ARF = 1 - exp(-1.1 t^0.25) + exp(-1.1 t^0.25 - 0.01 A),
    t in hours, A in square miles."""
    A = area_km2 * SQ_MI_PER_KM2
    k = 1.1 * duration_h ** 0.25
    return float(1.0 - np.exp(-k) + np.exp(-k - 0.01 * A))


@dataclass
class Hyetograph:
    """Piecewise-constant rainfall intensity starting at ``t_start`` (s)."""
    step_s: float
    intensity_mmh: np.ndarray
    t_start: float = 0.0
    label: str = ""

    @property
    def duration_s(self) -> float:
        return self.step_s * len(self.intensity_mmh)

    @property
    def total_mm(self) -> float:
        return float(self.intensity_mmh.sum() * self.step_s / 3600.0)

    def rate_ms(self, t: float) -> float:
        k = int(np.floor((t - self.t_start) / self.step_s + 1e-9))
        if k < 0 or k >= len(self.intensity_mmh):
            return 0.0
        return float(self.intensity_mmh[k]) * MM_H_TO_M_S

    def scaled(self, factor: float) -> "Hyetograph":
        return Hyetograph(self.step_s, self.intensity_mmh * factor, self.t_start, f"{self.label} x{factor:.3f}")

    def series(self, t_end: float, n: int) -> np.ndarray:
        """Intensity (mm/h) resampled as block means onto n equal steps of [0, t_end]."""
        fine = (np.arange(20 * n) + 0.5) * t_end / (20 * n)
        vals = np.array([self.rate_ms(t) for t in fine]) / MM_H_TO_M_S
        return vals.reshape(n, 20).mean(axis=1)


def alternating_block(T: int, duration_h: float, step_min: float, coeffs_or_idf, peak_position: float = 0.5,
                      t_start: float = 0.0) -> Hyetograph:
    n = int(round(duration_h * 60 / step_min))
    d = step_min * np.arange(1, n + 1)
    if isinstance(coeffs_or_idf, dict):
        cum = IDF.depth_mm(T, d, coeffs_or_idf)
    else:
        cum = coeffs_or_idf.intensity_mmh(T, d) * d / 60.0
    blocks = np.diff(np.concatenate([[0.0], cum]))
    if np.any(blocks < -1e-9):
        raise ValueError("IDF depth is not monotone in duration; cannot build alternating blocks")
    order = np.argsort(-blocks)
    out = np.zeros(n)
    centre = int(np.clip(round(peak_position * (n - 1)), 0, n - 1))
    pos = [centre]
    left, right = centre - 1, centre + 1
    for k in range(1, n):
        take_right = (k % 2 == 1)
        if (take_right and right < n) or left < 0:
            pos.append(right); right += 1
        else:
            pos.append(left); left -= 1
    for rank, p in zip(order, pos):
        out[p] = blocks[rank]
    return Hyetograph(step_min * 60.0, out / (step_min / 60.0), t_start, f"{T}-yr {duration_h:g} h alternating block")


def gauge_series(path: str | Path, t_start: float = 0.0) -> Hyetograph:
    """Rain gauge CSV (datetime, intensity mm/h) at a constant step."""
    times, vals = [], []
    with Path(path).open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                times.append(_dt.datetime.fromisoformat(r["datetime"]))
                vals.append(float(r["intensity"]))
            except (KeyError, ValueError) as e:
                raise ValueError(f"{path}: row {i + 2}: expected columns datetime (ISO 8601), intensity (mm/h) — {e}") from e
    if len(times) < 2:
        raise ValueError(f"{path}: rain gauge series needs at least two rows")
    steps = np.diff([t.timestamp() for t in times])
    if np.any(steps <= 0) or np.ptp(steps) > 1e-6:
        raise ValueError(f"{path}: rain gauge timestamps must be strictly increasing at a constant step")
    if min(vals) < 0:
        raise ValueError(f"{path}: negative rainfall intensity")
    return Hyetograph(float(steps[0]), np.array(vals), t_start, f"gauge {Path(path).name}")
