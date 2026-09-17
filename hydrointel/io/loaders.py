"""Raster and table readers with explicit unit, CRS and nodata checks.

GeoTIFF: rasterio if installed, otherwise Pillow + the GeoTIFF tags
(ModelPixelScale 33550, ModelTiepoint 33922, GDAL_NODATA 42113). Only north-up
rasters without rotation are accepted. ESRI ASCII grids (.asc) are read in
pure Python. Every failure names the file, the field and what was expected.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("hydrointel.io")


class InputDataError(ValueError):
    """A supplied input file is present but unusable."""


@dataclass
class Raster:
    data: np.ndarray          # (rows, cols), row 0 = NORTH as stored in the file
    xmin: float
    ymax: float
    cell: float
    nodata: float | None
    path: Path

    @property
    def shape(self):
        return self.data.shape

    @property
    def bounds(self):
        r, c = self.data.shape
        return (self.xmin, self.ymax - r * self.cell, self.xmin + c * self.cell, self.ymax)

    def south_up(self) -> np.ndarray:
        """Array flipped to this package's convention (row 0 = south)."""
        return self.data[::-1].copy()


def read_raster(path: str | Path, declared_nodata=None) -> Raster:
    path = Path(path)
    if not path.exists():
        raise InputDataError(f"{path}: file listed in manifest does not exist")
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        r = _read_geotiff(path)
    elif suffix == ".asc":
        r = _read_ascii_grid(path)
    else:
        raise InputDataError(f"{path}: unsupported raster format '{suffix}' (expected .tif/.tiff or .asc)")
    if declared_nodata is not None:
        if r.nodata is not None and not np.isclose(r.nodata, float(declared_nodata)):
            raise InputDataError(f"{path}: manifest nodata={declared_nodata} but the file declares {r.nodata}")
        r.nodata = float(declared_nodata)
    if not np.isfinite(r.cell) or r.cell <= 0:
        raise InputDataError(f"{path}: invalid pixel size {r.cell}")
    return r


def _read_geotiff(path: Path) -> Raster:
    try:
        import rasterio
        with rasterio.open(path) as ds:
            if ds.count != 1:
                raise InputDataError(f"{path}: expected a single-band raster, found {ds.count} bands")
            t = ds.transform
            if abs(t.b) > 0 or abs(t.d) > 0 or not np.isclose(t.a, -t.e):
                raise InputDataError(f"{path}: rotated or non-square pixels are not supported")
            return Raster(ds.read(1).astype(np.float64), t.c, t.f, t.a, ds.nodata, path)
    except ImportError:
        pass
    from PIL import Image
    try:
        img = Image.open(path)
    except Exception as e:
        raise InputDataError(f"{path}: cannot open TIFF ({e}); install rasterio for compressed/tiled GeoTIFFs") from e
    tags = img.tag_v2
    if 33550 not in tags or 33922 not in tags:
        raise InputDataError(f"{path}: missing GeoTIFF georeferencing tags (ModelPixelScale/ModelTiepoint)")
    sx, sy = tags[33550][0], tags[33550][1]
    tp = tags[33922]
    if not np.isclose(sx, sy):
        raise InputDataError(f"{path}: non-square pixels ({sx} x {sy}) are not supported")
    if tp[0] != 0 or tp[1] != 0:
        raise InputDataError(f"{path}: tiepoint must reference pixel (0, 0)")
    nod = tags.get(42113)
    nodata = float(str(nod).strip("\x00 ")) if nod not in (None, "") else None
    return Raster(np.array(img, dtype=np.float64), float(tp[3]), float(tp[4]), float(sx), nodata, path)


def _read_ascii_grid(path: Path) -> Raster:
    head = {}
    with path.open(encoding="utf-8") as f:
        for _ in range(6):
            pos = f.tell()
            line = f.readline()
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() in ("ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
                                                         "yllcenter", "cellsize", "nodata_value"):
                head[parts[0].lower()] = float(parts[1])
            else:
                f.seek(pos)
                break
        data = np.loadtxt(f, dtype=np.float64, ndmin=2)
    for k in ("ncols", "nrows", "cellsize"):
        if k not in head:
            raise InputDataError(f"{path}: ASCII grid header missing '{k}'")
    if data.shape != (int(head["nrows"]), int(head["ncols"])):
        raise InputDataError(f"{path}: header says {int(head['nrows'])}x{int(head['ncols'])}, data is {data.shape}")
    c = head["cellsize"]
    xll = head.get("xllcorner", head.get("xllcenter", 0.0) - c / 2)
    yll = head.get("yllcorner", head.get("yllcenter", 0.0) - c / 2)
    return Raster(data, xll, yll + data.shape[0] * c, c, head.get("nodata_value"), path)


def resample(r: Raster, bounds, cell: float, ny: int, nx: int, method: str) -> np.ndarray:
    """Resample onto the model grid (row 0 = south). ``method`` is 'bilinear' or 'nearest'.
    Nodata stays NaN so callers can decide what to do with it."""
    src = r.data.copy()
    if r.nodata is not None:
        src[np.isclose(src, r.nodata)] = np.nan
    xs = bounds[0] + (np.arange(nx) + 0.5) * cell
    ys = bounds[1] + (np.arange(ny) + 0.5) * cell
    col = (xs - r.xmin) / r.cell - 0.5
    row = (r.ymax - ys) / r.cell - 0.5
    R, C = src.shape
    if col.min() < -0.5 or col.max() > C - 0.5 or row.min() < -0.5 or row.max() > R - 0.5:
        raise InputDataError(f"{r.path}: raster extent {r.bounds} does not cover the model bounds {tuple(bounds)}")
    if method == "nearest":
        ci = np.clip(np.round(col).astype(int), 0, C - 1)
        ri = np.clip(np.round(row).astype(int), 0, R - 1)
        out = src[ri[:, None], ci[None, :]]
    elif method == "bilinear":
        c0 = np.clip(np.floor(col).astype(int), 0, C - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, R - 2)
        wc = np.clip(col - c0, 0, 1)[None, :]
        wr = np.clip(row - r0, 0, 1)[:, None]
        a = src[r0[:, None], c0[None, :]]
        b = src[r0[:, None], c0[None, :] + 1]
        c = src[r0[:, None] + 1, c0[None, :]]
        d = src[r0[:, None] + 1, c0[None, :] + 1]
        out = (1 - wr) * ((1 - wc) * a + wc * b) + wr * ((1 - wc) * c + wc * d)
    else:
        raise ValueError(f"unknown resampling method '{method}'")
    log.info("resampled %s (%dx%d @ %g m) -> model grid %dx%d @ %g m using %s",
             r.path.name, R, C, r.cell, ny, nx, cell, method)
    return out


def read_table(path: str | Path, schema: list[str]) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise InputDataError(f"{path}: file listed in manifest does not exist")
    with path.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in schema if c not in (rdr.fieldnames or [])]
        if missing:
            raise InputDataError(f"{path}: missing column(s) {missing}; expected schema {schema}")
        rows = list(rdr)
    if not rows:
        raise InputDataError(f"{path}: table is empty")
    return rows
