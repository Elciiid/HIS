"""Intensity-duration-frequency curves.

    i = a * T**b / (d + c)**e      [mm/h], T in years, d in minutes

The default coefficients are PLACEHOLDER_IDF_ROXAS (config.py): illustrative
values, not PAGASA-published. Replace them before any real use.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RETURN_PERIODS = (10, 25, 50, 100)


def intensity_mmh(T: float, d_min, coeffs: dict) -> np.ndarray:
    d = np.asarray(d_min, dtype=float)
    if np.any(d <= 0):
        raise ValueError("IDF duration must be positive (minutes)")
    if T <= 0:
        raise ValueError("return period must be positive")
    return coeffs["a"] * T ** coeffs["b"] / (d + coeffs["c"]) ** coeffs["e"]


def depth_mm(T: float, d_min, coeffs: dict) -> np.ndarray:
    return intensity_mmh(T, d_min, coeffs) * np.asarray(d_min, float) / 60.0


class TabulatedIDF:
    """IDF supplied as a table (manifest layer ``idf_curve``); log-log interpolation in duration."""

    def __init__(self, table: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.table = table

    @classmethod
    def from_csv(cls, path: str | Path) -> "TabulatedIDF":
        rows: dict[int, list[tuple[float, float]]] = {}
        with Path(path).open(encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                try:
                    T, d, it = int(float(r["return_period_yr"])), float(r["duration_min"]), float(r["intensity"])
                except (KeyError, ValueError) as e:
                    raise ValueError(f"{path}: row {i + 2}: expected return_period_yr, duration_min, "
                                     f"intensity (mm/h) — {e}") from e
                rows.setdefault(T, []).append((d, it))
        return cls({T: (np.array(sorted(v))[:, 0], np.array(sorted(v))[:, 1]) for T, v in rows.items()})

    def intensity_mmh(self, T, d_min):
        if T not in self.table:
            raise ValueError(f"IDF table has no {T}-yr curve (have {sorted(self.table)})")
        d, i = self.table[T]
        return np.exp(np.interp(np.log(np.asarray(d_min, float)), np.log(d), np.log(i)))
