"""Real-data input contract (``artifacts/inputs/manifest.json``).

If the manifest is absent the system runs on synthetic terrain and says so
(one WARNING, plus the SYNTHETIC stamp on every output). If it is present, it
must be complete and consistent; any problem raises ``InputDataError`` naming
the file, the field and the expected units. Nothing silently falls back to
synthetic data.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..domain import landuse as LU
from ..domain.channels import build_from_rows
from ..domain.synthetic import Domain, polyline_distance
from .loaders import InputDataError, read_raster, read_table, resample

log = logging.getLogger("hydrointel.io")
_WARNED = False

ALLOWED_UNITS = {
    "dem": {"m"},
    "landuse": {"class"},
    "manning": {"s/m^(1/3)"},
    "channel_net": {"m"},
    "rain_gauge": {"mm/h"},
    "tide_gauge": {"m_MSL"},
    "idf_curve": {"mm/h"},
    "obs_flood": {"m"},
}
REQUIRED_LAYERS = ("dem", "landuse", "channel_net", "rain_gauge", "tide_gauge", "idf_curve")
SCHEMAS = {
    "channel_net": ["reach_id", "node_id", "x", "y", "bed_elev", "width", "side_slope", "manning"],
    "rain_gauge": ["datetime", "intensity"],
    "tide_gauge": ["datetime", "elevation"],
    "idf_curve": ["return_period_yr", "duration_min", "intensity"],
    "obs_flood": ["event_id", "x", "y", "max_depth", "source"],
}
# Projected, metre-based CRSs accepted without pyproj: WGS84/UTM (326xx/327xx),
# PRS92 / Philippines zones (3121-3125), Luzon 1911 / Philippines zones (25391-25395).
_KNOWN_METRIC = re.compile(r"^EPSG:(326\d\d|327\d\d|312[1-5]|2539[1-5])$")

# landuse legend keywords -> class code
_LEGEND_KEYS = [
    (LU.WATER, ("water", "channel", "river", "sea")),
    (LU.ROAD, ("paved", "road", "street")),
    (LU.DENSE_URBAN, ("dense", "commercial", "industrial")),
    (LU.RESIDENTIAL, ("residential", "subdivision", "settlement")),
    (LU.BARE, ("bare", "cleared", "barren")),
    (LU.GRASS, ("grass", "park", "open space", "lawn")),
    (LU.CROPLAND, ("crop", "rice", "paddy", "agri")),
    (LU.MANGROVE, ("mangrove", "coastal veg", "nipa")),
]


@dataclass
class RealInputs:
    manifest_path: Path
    dataset_id: str
    crs: str
    rain_gauge: Path
    tide_gauge: Path
    idf_curve: Path
    obs_flood: Path | None
    manning: np.ndarray | None
    notes: list[str] = field(default_factory=list)


def find_manifest(outdir: str | Path) -> Path | None:
    global _WARNED
    p = Path(outdir) / "inputs" / "manifest.json"
    if p.exists():
        return p
    if not _WARNED:
        log.warning("no %s found: running on SYNTHETIC terrain. Results are not predictions for any real place.", p)
        _WARNED = True
    return None


def _crs_ok(crs: str, path: Path) -> None:
    try:
        from pyproj import CRS
        c = CRS.from_user_input(crs)
        if not c.is_projected:
            raise InputDataError(f"{path}: field 'crs' = {crs} is not a projected CRS (need metres)")
        units = {ax.unit_name for ax in c.axis_info}
        if not units <= {"metre", "meter"}:
            raise InputDataError(f"{path}: field 'crs' = {crs} has axis units {units}, expected metres")
        return
    except ImportError:
        pass
    if not _KNOWN_METRIC.match(str(crs)):
        raise InputDataError(f"{path}: field 'crs' = {crs} cannot be verified as projected/metric without pyproj; "
                             "use a UTM or PRS92 EPSG code, or install pyproj")


def _legend_map(legend: dict, path: Path) -> dict[int, int]:
    out = {}
    for code, name in legend.items():
        n = str(name).lower()
        hit = [c for c, keys in _LEGEND_KEYS if any(k in n for k in keys)]
        if len(hit) != 1:
            raise InputDataError(f"{path}: landuse legend entry {code}: '{name}' does not map to exactly one "
                                 f"model class ({[c.name for c in LU.LANDUSE_TABLE]})")
        try:
            out[int(code)] = hit[0]
        except ValueError as e:
            raise InputDataError(f"{path}: landuse legend key '{code}' must be an integer raster code") from e
    return out


def load(manifest_path: str | Path, domain_cfg) -> tuple[Domain, RealInputs]:
    mp = Path(manifest_path)
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputDataError(f"{mp}: not valid JSON ({e})") from e
    for k in ("dataset_id", "crs", "cell_size_m", "bounds_m", "vertical_datum", "layers"):
        if k not in m:
            raise InputDataError(f"{mp}: missing required field '{k}'")
    _crs_ok(m["crs"], mp)
    if m["vertical_datum"] != "MSL":
        raise InputDataError(f"{mp}: field 'vertical_datum' = {m['vertical_datum']!r}; the tidal boundary and "
                             "sea-level-rise tables are referenced to MSL, so elevations must be too")
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in m["bounds_m"])
    except (TypeError, ValueError) as e:
        raise InputDataError(f"{mp}: field 'bounds_m' must be [xmin, ymin, xmax, ymax] in metres") from e
    cell = float(m["cell_size_m"])
    if cell <= 0 or xmax <= xmin or ymax <= ymin:
        raise InputDataError(f"{mp}: bounds_m / cell_size_m are inconsistent")
    nx, ny = (xmax - xmin) / cell, (ymax - ymin) / cell
    if abs(nx - round(nx)) > 1e-6 or abs(ny - round(ny)) > 1e-6:
        raise InputDataError(f"{mp}: bounds_m extent is not a whole number of {cell} m cells")
    if not np.isclose(cell, domain_cfg.cell_m):
        raise InputDataError(f"{mp}: cell_size_m = {cell} but the run config uses domain.cell_m = "
                             f"{domain_cfg.cell_m}; set the config to match the delivered grid")
    nx, ny = int(round(nx)), int(round(ny))
    sea_edge = m.get("sea_edge", "north")
    if sea_edge != "north":
        raise InputDataError(f"{mp}: field 'sea_edge' = {sea_edge!r}; this version supports a northern sea edge only")
    layers = m["layers"]
    for name in REQUIRED_LAYERS:
        if name not in layers:
            raise InputDataError(f"{mp}: required layer '{name}' is missing")
    for name, spec in layers.items():
        if name not in ALLOWED_UNITS:
            raise InputDataError(f"{mp}: unknown layer '{name}' (known: {sorted(ALLOWED_UNITS)})")
        if spec.get("units") not in ALLOWED_UNITS[name]:
            raise InputDataError(f"{mp}: layer '{name}' field 'units' = {spec.get('units')!r}; expected one of "
                                 f"{sorted(ALLOWED_UNITS[name])}")
        if "file" not in spec:
            raise InputDataError(f"{mp}: layer '{name}' has no 'file'")
        if name in SCHEMAS and spec.get("schema") not in (None, SCHEMAS[name]):
            raise InputDataError(f"{mp}: layer '{name}' field 'schema' = {spec.get('schema')}; expected {SCHEMAS[name]}")
    base = mp.parent
    bounds = (xmin, ymin, xmax, ymax)
    notes = []

    # DEM
    dem_r = read_raster(base / layers["dem"]["file"], layers["dem"].get("nodata"))
    dem = resample(dem_r, bounds, cell, ny, nx, "bilinear")
    notes.append(f"dem resampled bilinear from {dem_r.cell:g} m")
    if np.isnan(dem).any():
        raise InputDataError(f"{dem_r.path}: {int(np.isnan(dem).sum())} model cells fall on DEM nodata; the engine "
                             "cannot simulate unknown terrain — fill the DEM (units m, MSL) before delivery")
    if dem.min() < -50 or dem.max() > 3000:
        raise InputDataError(f"{dem_r.path}: elevations {dem.min():.1f}..{dem.max():.1f} look wrong for metres above MSL")
    # land use
    lu_spec = layers["landuse"]
    if "legend" not in lu_spec:
        raise InputDataError(f"{mp}: layer 'landuse' needs a 'legend' mapping raster codes to class names")
    lmap = _legend_map(lu_spec["legend"], mp)
    lu_r = read_raster(base / lu_spec["file"], lu_spec.get("nodata"))
    raw = resample(lu_r, bounds, cell, ny, nx, "nearest")
    notes.append(f"landuse resampled nearest from {lu_r.cell:g} m")
    if np.isnan(raw).any():
        raise InputDataError(f"{lu_r.path}: {int(np.isnan(raw).sum())} model cells are landuse nodata")
    codes = np.unique(raw.astype(np.int64))
    unknown = [c for c in codes if c not in lmap]
    if unknown:
        raise InputDataError(f"{lu_r.path}: raster codes {unknown} are not in the manifest legend")
    lu = np.vectorize(lmap.get)(raw.astype(np.int64)).astype(np.int8)
    # optional manning
    manning = None
    if "manning" in layers:
        mr = read_raster(base / layers["manning"]["file"], layers["manning"].get("nodata"))
        manning = resample(mr, bounds, cell, ny, nx, "nearest")
        if np.isnan(manning).any() or manning.min() <= 0 or manning.max() > 0.5:
            raise InputDataError(f"{mr.path}: Manning n must be finite and within (0, 0.5] s/m^(1/3)")
        notes.append("manning raster supplied (overrides land-use table)")
    # sea: water class at or below MSL, connected to the north edge
    sea = _connected_from_north((lu == LU.WATER) & (dem <= 0.0))
    if not sea[-1].any():
        raise InputDataError(f"{mp}: no open-water cells at or below MSL on the northern edge; the tidal boundary "
                             "needs sea along the north edge")
    # channel network
    rows = read_table(base / layers["channel_net"]["file"], SCHEMAS["channel_net"])
    to_local = lambda xw, yw: (xw - xmin, yw - ymin)
    for r in rows:
        try:
            r["x"], r["y"] = to_local(float(r["x"]), float(r["y"]))
        except ValueError as e:
            raise InputDataError(f"{layers['channel_net']['file']}: non-numeric x/y (m, {m['crs']})") from e
    dem_south_up = dem

    def z_bank(x, y):
        i, j = int(x // cell), int(y // cell)
        if not (0 <= i < nx and 0 <= j < ny):
            raise InputDataError(f"{layers['channel_net']['file']}: node at ({x + xmin:.1f}, {y + ymin:.1f}) lies "
                                 "outside bounds_m")
        win = dem_south_up[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3]
        return float(np.max(win))

    def is_sea(x, y):
        i, j = min(int(x // cell), nx - 1), min(int(y // cell), ny - 1)
        return bool(sea[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2].any()) or j >= ny - 1

    net = build_from_rows(rows, z_bank, bounds, is_sea, str(layers["channel_net"]["file"]))
    for r in net.reaches:
        log.info("channel reach %s: upstream end %s, downstream end %s", r.reach_id, r.up[0], r.down[0])
    x = (np.arange(nx) + 0.5) * cell
    y = (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(x, y)
    d_ch = np.full((ny, nx), np.inf)
    footprint = np.zeros((ny, nx), bool)
    dem2d = dem.copy()
    for rch in net.reaches:
        d, pos = polyline_distance(rch.x, rch.y, X, Y) if rch.n > 1 else (None, None)
        half = 0.5 * np.interp(pos, np.arange(rch.n), rch.width + 2 * rch.side_slope * (rch.z_bank - rch.bed))
        sel = (d <= np.maximum(half, 0.55 * cell)) & ~sea
        footprint |= sel
        dem2d[sel] = np.interp(pos[sel], np.arange(rch.n), rch.z_bank)
        d_ch = np.minimum(d_ch, d)
    dist_coast = np.where(sea, -1.0, 1.0) * np.abs(Y - Y[sea].max())
    road = lu == LU.ROAD
    built = np.isin(lu, [LU.DENSE_URBAN, LU.RESIDENTIAL])
    urban = np.argwhere(lu == LU.DENSE_URBAN)
    core = (float(urban[:, 1].mean() * cell), float(urban[:, 0].mean() * cell)) if len(urban) else (nx * cell / 2, ny * cell / 2)
    obs = layers.get("obs_flood")
    real = RealInputs(mp, m["dataset_id"], m["crs"], base / layers["rain_gauge"]["file"],
                      base / layers["tide_gauge"]["file"], base / layers["idf_curve"]["file"],
                      (base / obs["file"]) if obs else None, manning, notes)
    for p in (real.rain_gauge, real.tide_gauge, real.idf_curve) + ((real.obs_flood,) if real.obs_flood else ()):
        if not p.exists():
            raise InputDataError(f"{p}: file listed in manifest does not exist")
    dom = Domain(domain_cfg, cell, x, y, dem, dem2d, lu, sea, footprint, road, built, d_ch, dist_coast, net, core,
                 source=m["dataset_id"], report={"grid": f"{nx} x {ny} @ {cell:g} m", "crs": m["crs"],
                                                 "bounds_m": bounds, "notes": notes})
    log.info("loaded real dataset %s (%s)", m["dataset_id"], dom.report["grid"])
    return dom, real


def _connected_from_north(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    out[-1] = mask[-1]
    for _ in range(mask.shape[0] * 2):
        grown = out.copy()
        grown[:-1] |= out[1:]
        grown[1:] |= out[:-1]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        grown &= mask
        if (grown == out).all():
            break
        out = grown
    return out
