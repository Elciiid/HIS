"""Tidal boundary: harmonic constituents + sea-level rise, with storm/tide phasing."""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import TideConfig


@dataclass
class Tide:
    cfg: TideConfig
    slr_m: float = 0.0
    on: bool = True
    shift_s: float = 0.0        # time shift applied to the astronomical signal

    def astronomical(self, t):
        t = np.asarray(t, float) + self.shift_s
        eta = np.zeros_like(t)
        for c in self.cfg.constituents.values():
            w = 2 * np.pi / (c["period_h"] * 3600.0)
            eta = eta + c["amp_m"] * np.cos(w * t - np.radians(c["phase_deg"]))
        return eta

    def __call__(self, t: float) -> float:
        base = self.cfg.msl_offset_m + self.slr_m
        return float(base + (self.astronomical(t) if self.on else 0.0))

    def align(self, storm_peak_s: float, offset_h: float) -> "Tide":
        """Shift the signal so that the highest (spring) high water of the first 30
        days of the constituent record falls ``offset_h`` hours after the storm
        peak (negative = before the peak)."""
        probe = Tide(self.cfg, 0.0, True, 0.0)
        tt = np.arange(0.0, 30 * 24 * 3600.0, 300.0)
        hw = tt[np.argmax(probe.astronomical(tt))]                  # a spring high water in the first 30 days
        target = storm_peak_s + offset_h * 3600.0
        return Tide(self.cfg, self.slr_m, self.on, hw - target)

    def range_m(self, days: float = 30.0) -> float:
        tt = np.arange(0.0, days * 86400.0, 600.0)
        e = Tide(self.cfg, 0.0, True, self.shift_s).astronomical(tt)
        return float(e.max() - e.min())


@dataclass
class GaugeTide:
    """Observed tide gauge series (manifest layer ``tide_gauge``), m above MSL."""
    t_s: np.ndarray
    eta: np.ndarray
    slr_m: float = 0.0

    @classmethod
    def from_csv(cls, path: str | Path, slr_m: float = 0.0) -> "GaugeTide":
        ts, es = [], []
        with Path(path).open(encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                try:
                    ts.append(_dt.datetime.fromisoformat(r["datetime"]).timestamp())
                    es.append(float(r["elevation"]))
                except (KeyError, ValueError) as e:
                    raise ValueError(f"{path}: row {i + 2}: expected datetime (ISO 8601), elevation (m MSL) — {e}") from e
        t = np.array(ts)
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError(f"{path}: tide gauge timestamps must be strictly increasing")
        return cls(t - t[0], np.array(es), slr_m)

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.t_s, self.eta) + self.slr_m)
