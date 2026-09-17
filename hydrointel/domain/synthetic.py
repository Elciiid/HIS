"""Synthetic Roxas-like domain: DEM, land use, channel network.

Grid convention used everywhere in this package: arrays are (ny, nx), row 0 is
the SOUTH edge, column 0 the WEST edge, cell centres at ((i+.5)dx, (j+.5)dx).
The north edge is the sea. Everything is deterministic from the run seed.

This terrain is procedurally generated. It resembles the setting of Roxas City
(coastal plain, river mouth, city core) only in broad structure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import DomainConfig, numpy_rng
from . import landuse as LU
from .channels import ChannelNetwork, Junction, Reach

log = logging.getLogger("hydrointel.domain")


@dataclass
class Domain:
    cfg: DomainConfig
    dx: float
    x: np.ndarray                 # (nx,) cell-centre x
    y: np.ndarray                 # (ny,) cell-centre y
    dem: np.ndarray               # terrain incl. carved channels (display, pure 2-D runs)
    dem2d: np.ndarray             # channel footprint raised to bank crest (coupled runs)
    landuse: np.ndarray           # int8 class raster
    sea: np.ndarray               # bool
    channel: np.ndarray           # bool, 1-D footprint (main river + creeks, onshore)
    road: np.ndarray              # bool
    building: np.ndarray          # bool
    dist_channel: np.ndarray      # m
    dist_coast: np.ndarray        # m, negative offshore
    network: ChannelNetwork
    urban_core: tuple[float, float]
    source: str = "SYNTHETIC"
    report: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape

    @property
    def ny(self) -> int:
        return self.dem.shape[0]

    @property
    def nx(self) -> int:
        return self.dem.shape[1]

    def manning(self) -> np.ndarray:
        return LU.lookup(self.landuse, "manning")

    def base_storage(self) -> np.ndarray:
        return LU.lookup(self.landuse, "storage_m")

    def exposure(self) -> np.ndarray:
        return LU.lookup(self.landuse, "exposure")

    def cell_index(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.clip(np.floor(x / self.dx), 0, self.nx - 1))
        j = int(np.clip(np.floor(y / self.dx), 0, self.ny - 1))
        return j, i


# ---------------------------------------------------------------------------
# noise helpers (pure numpy)
# ---------------------------------------------------------------------------
def spectral_noise(rng, ny, nx, dx, beta=1.0, lmin=0.0, lmax=np.inf):
    """Zero-mean unit-std noise with amplitude spectrum ~ 1/k**beta, band-limited
    to wavelengths in [lmin, lmax]."""
    white = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny, d=dx)[:, None]
    kx = np.fft.rfftfreq(nx, d=dx)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    with np.errstate(divide="ignore"):
        amp = np.where(k > 0, k ** -beta, 0.0)
        wl = np.where(k > 0, 1.0 / np.maximum(k, 1e-30), np.inf)
    amp = amp * (wl >= lmin) * (wl <= lmax)
    f = np.fft.irfft2(np.fft.rfft2(white) * amp, s=(ny, nx))
    f -= f.mean()
    s = f.std()
    return f / s if s > 0 else f


def smooth_series(rng, n, ds, wavelength):
    """Band-limited random walk sampled at n points spaced ds (unit std)."""
    w = rng.standard_normal(n)
    k = np.fft.rfftfreq(n, d=ds)
    spec = np.fft.rfft(w) * np.exp(-(k * wavelength) ** 2)
    spec[0] = 0.0
    s = np.fft.irfft(spec, n=n)
    return s / (s.std() + 1e-30)


def box_smooth(a, passes=1):
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[1 + dj:p.shape[0] - 1 + dj, 1 + di:p.shape[1] - 1 + di]
                for dj in (-1, 0, 1) for di in (-1, 0, 1)) / 9.0
    return a


def polyline_distance(px, py, X, Y, chunk=200_000):
    """Distance from grid points (X, Y) to a polyline; also returns the fractional
    vertex position of the nearest point."""
    ax, ay = px[:-1], py[:-1]
    bx, by = px[1:], py[1:]
    vx, vy = bx - ax, by - ay
    vv = np.maximum(vx * vx + vy * vy, 1e-12)
    Xf, Yf = X.ravel(), Y.ravel()
    dist = np.empty(Xf.size)
    pos = np.empty(Xf.size)
    step = max(1, chunk // len(ax))
    for s in range(0, Xf.size, step):
        xs, ys = Xf[s:s + step, None], Yf[s:s + step, None]
        t = np.clip(((xs - ax) * vx + (ys - ay) * vy) / vv, 0.0, 1.0)
        d2 = (xs - ax - t * vx) ** 2 + (ys - ay - t * vy) ** 2
        k = np.argmin(d2, axis=1)
        r = np.arange(len(k))
        dist[s:s + step] = np.sqrt(d2[r, k])
        pos[s:s + step] = k + t[r, k]
    return dist.reshape(X.shape), pos.reshape(X.shape)


def fill_single_cell_pits(z, protect, eps=0.0, max_iter=200):
    """Raise cells lower than all 8 neighbours (outside ``protect``) to just above
    their lowest neighbour. Returns (z, n_filled, n_remaining)."""
    z = z.copy()
    total = 0
    for _ in range(max_iter):
        p = np.pad(z, 1, mode="edge")
        nb = np.stack([p[1 + dj:p.shape[0] - 1 + dj, 1 + di:p.shape[1] - 1 + di]
                       for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)])
        nmin = nb.min(axis=0)
        pit = (z < nmin) & ~protect
        pit[0, :] = pit[-1, :] = pit[:, 0] = pit[:, -1] = False   # edges drain through boundaries
        n = int(pit.sum())
        if n == 0:
            return z, total, 0
        z[pit] = nmin[pit] + eps
        total += n
    return z, total, n


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def generate(cfg: DomainConfig, seed: int) -> Domain:
    rng = numpy_rng(f"domain/{seed}")
    dx = cfg.cell_m
    nx, ny = cfg.nx, cfg.ny
    lx, ly = nx * dx, ny * dx
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)

    # --- coastline and inland distance -------------------------------------
    xs = np.linspace(0, lx, 256)
    coast_wiggle = 60.0 * smooth_series(rng, 256, lx / 255, 1200.0)
    y_coast_line = ly - cfg.offshore_width_m + coast_wiggle
    y_coast = np.interp(x, xs, y_coast_line)
    S = y_coast[None, :] - Y                       # inland distance, <0 offshore
    s_max = float(y_coast.mean())
    sea = S < 0

    # --- coastal gradient ---------------------------------------------------
    z_inland = np.interp(x, xs, rng.uniform(*cfg.inland_elev_range_m) +
                         0.25 * np.diff(cfg.inland_elev_range_m)[0] * smooth_series(rng, 256, lx / 255, 2500.0))
    z_inland = np.clip(z_inland, *cfg.inland_elev_range_m)[None, :]
    wp = cfg.plain_width_m
    frac = np.clip((S - wp) / (s_max - wp), 0.0, None)
    base = np.where(S < wp, cfg.plain_top_m * np.clip(S, 0, None) / wp,
                    cfg.plain_top_m + (z_inland - cfg.plain_top_m) * frac ** 1.3)
    base = np.where(sea, cfg.offshore_elev_m + 0.002 * S, base)
    pert = spectral_noise(rng, ny, nx, dx, beta=2.0, lmin=800.0)
    base = base + np.where(sea, 0.0, 1.2 * pert * np.clip(S / s_max, 0, 1))

    # --- river centreline: band-limited walk, parametrised by y (monotone to sea)
    n_pts = max(64, int(ly / 50))
    yc = np.linspace(0.0, ly, n_pts)
    s_frac = np.clip(yc / s_max, 0, 1)
    x0, x1 = cfg.river_entry_x_frac * lx, cfg.river_mouth_x_frac * lx
    meander = cfg.meander_amp_m * smooth_series(rng, n_pts, ly / (n_pts - 1), 900.0)
    taper = np.sin(np.pi * np.clip(yc / ly, 0, 1)) ** 0.5
    xc = x0 + (x1 - x0) * s_frac + meander * taper
    xc = np.clip(xc, 0.15 * lx, 0.85 * lx)
    d_riv, pos_riv = polyline_distance(xc, yc, X, Y)

    # floodplain width and floor
    fp_w = np.interp(yc, yc, np.mean(cfg.floodplain_width_range_m) +
                     0.5 * np.diff(cfg.floodplain_width_range_m)[0] *
                     np.tanh(smooth_series(rng, n_pts, ly / (n_pts - 1), 2000.0)))
    fp_half = 0.5 * np.interp(Y, yc, fp_w)
    z_fp = 0.3 + cfg.floodplain_rise_m * np.clip(S / s_max, 0, 1)
    w_fp = 1.0 / (1.0 + (d_riv / fp_half) ** 8)
    land = ~sea
    z = np.where(land, w_fp * np.minimum(z_fp, base) + (1 - w_fp) * base, base)

    # hills in the southern third, outside the corridor
    hill_field = np.zeros_like(z)
    placed = 0
    for _ in range(200):
        if placed >= cfg.n_hills:
            break
        hx, hy = rng.uniform(0.08 * lx, 0.92 * lx), rng.uniform(0.02 * ly, ly / 3)
        k = int(np.clip(hy / ly * (n_pts - 1), 0, n_pts - 1))
        if abs(hx - xc[k]) < fp_w[k] / 2 + 600:
            continue
        r = rng.uniform(600, 1200)
        jh, ih = min(int(hy / dx), ny - 1), min(int(hx / dx), nx - 1)
        relief = max(rng.uniform(*cfg.hill_height_range_m) - z[jh, ih], 5.0)   # crest elevation in range
        hill_field = np.maximum(hill_field, relief * np.exp(-((X - hx) ** 2 + (Y - hy) ** 2) / r ** 2))
        placed += 1
    z = z + hill_field * (1 - w_fp) * land

    z = box_smooth(z, cfg.smoothing_passes)
    micro = spectral_noise(rng, ny, nx, dx, beta=1.0) * cfg.micro_noise_amp_m
    z = z + micro

    # --- main channel profile (1-D design values, not the noisy DEM) --------
    node_ds = cfg.channel_node_spacing_m
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))])
    y_coast_at = lambda xx: float(np.interp(xx, x, y_coast))
    # stop the 1-D river at the coastline
    on_land = np.array([yy < y_coast_at(xx) for xx, yy in zip(xc, yc)])
    arc_end = arc[np.argmin(on_land)] if not on_land.all() else arc[-1]
    n_nodes = max(8, int(arc_end / node_ds))
    a_nodes = np.linspace(0.5 * arc_end / n_nodes, arc_end - 0.5 * arc_end / n_nodes, n_nodes)
    rx, ry = np.interp(a_nodes, arc, xc), np.interp(a_nodes, arc, yc)
    r_s = np.array([y_coast_at(a) for a in rx]) - ry
    incision = np.interp(a_nodes, np.linspace(0, arc_end, 32),
                         np.mean(cfg.channel_incision_range_m) + 0.5 * np.diff(cfg.channel_incision_range_m)[0]
                         * np.tanh(smooth_series(rng, 32, arc_end / 31, 3000.0)))
    r_bank_design = 0.3 + cfg.floodplain_rise_m * np.clip(r_s / s_max, 0, 1)
    r_bed = np.minimum.accumulate(r_bank_design - incision)
    half_top = 0.5 * cfg.channel_width_m + cfg.channel_side_slope * incision

    z_pre_channel = z.copy()
    # carve the main channel into the display DEM
    pos_n = np.interp(pos_riv, np.arange(n_pts), arc)                  # arc length of nearest point
    bed_g = np.interp(pos_n, a_nodes, r_bed)
    half_b = max(0.5 * cfg.channel_width_m, 0.55 * dx)
    depth_g = np.interp(pos_n, a_nodes, r_bank_design) - bed_g
    ht_g = half_b + cfg.channel_side_slope * depth_g
    prof = np.clip((ht_g - d_riv) / np.maximum(ht_g - half_b, 1e-6), 0, 1)
    carve = bed_g + (1 - prof) * depth_g
    in_ch = d_riv <= ht_g
    z = np.where(in_ch, np.minimum(z, carve), z)
    main_fp = in_ch & (pos_n <= arc_end) & land

    # --- tidal creeks: branch from the lower reach toward the sea -----------
    creek_fp = np.zeros_like(main_fp)
    creek_lines = []
    lower = np.where((r_s > 400) & (r_s < 1900))[0]
    if len(lower) < cfg.n_creeks:
        lower = np.arange(max(0, n_nodes - 3 * cfg.n_creeks), n_nodes - 1)
    picks = _spaced_picks(rng, lower, cfg.n_creeks, min_gap=max(4, len(lower) // (2 * cfg.n_creeks)))
    for c, kb in enumerate(picks):
        side = -1.0 if c % 2 == 0 else 1.0
        bx_, by_ = rx[kb], ry[kb]
        ex = np.clip(bx_ + side * rng.uniform(900, 1800), 0.05 * lx, 0.95 * lx)
        ey = y_coast_at(ex) + 150.0
        t = np.linspace(0, 1, 60)
        wig = rng.uniform(-300, 300) * np.sin(np.pi * t) + rng.uniform(-80, 80) * np.sin(3 * np.pi * t)
        cxl = bx_ + (ex - bx_) * t + wig
        cyl = by_ + (ey - by_) * t ** 0.8
        d_c, pos_c = polyline_distance(cxl, cyl, X, Y)
        cinc = rng.uniform(*cfg.creek_incision_range_m)
        c_half = max(0.5 * cfg.creek_width_m, 0.55 * dx)
        c_top = c_half + cfg.channel_side_slope * cinc
        # creek bed follows surface along the path, forced non-increasing toward sea
        surf = _sample(z_pre_channel, dx, cxl, cyl)
        cbed = np.minimum.accumulate(surf - cinc)
        bed_c = np.interp(pos_c, np.arange(60), cbed)
        sel = d_c <= c_top
        zc = bed_c + np.clip((d_c - c_half) / (c_top - c_half + 1e-6), 0, 1) * cinc
        z = np.where(sel & (d_riv > half_b), np.minimum(z, zc), z)
        onland = np.array([yy < y_coast_at(xx) for xx, yy in zip(cxl, cyl)])
        end_c = int(np.argmin(onland)) if not onland.all() else len(onland)
        creek_fp |= sel & land & (pos_c <= end_c)
        creek_lines.append((cxl, cyl, cbed, cinc, kb, onland))

    # --- urban fabric -------------------------------------------------------
    k_core = int(np.argmin(np.abs(r_s - 1000.0)))
    core = (float(rx[k_core] - 700.0), float(ry[k_core]))
    R = np.hypot(X - core[0], Y - core[1])
    rho = np.exp(-(R / cfg.urban_radius_m) ** 2)
    sp, sw = cfg.street_spacing_m, max(cfg.street_width_m, dx)
    street = ((np.mod(X, sp) < sw) | (np.mod(Y, sp) < sw))
    edge_noise = spectral_noise(rng, ny, nx, dx, beta=1.2, lmin=120.0)
    urban_ext = (rho + 0.12 * edge_noise > 0.3) & land & ~main_fp & ~creek_fp & (S > 60)
    road = street & urban_ext
    bi, bj = np.floor(X / sp).astype(np.int64), np.floor(Y / sp).astype(np.int64)
    nbx, nby = int(bi.max()) + 1, int(bj.max()) + 1
    u_block = rng.uniform(size=(nby, nbx))
    raise_block = rng.uniform(*cfg.block_raise_range_m, size=(nby, nbx))
    built = (u_block[bj, bi] < rho * 1.25) & urban_ext & ~street
    lot = urban_ext & ~street
    # streets are the low drainage paths; blocks sit above them, buildings higher
    z = z + np.where(lot, raise_block[bj, bi], 0.0) + np.where(built, cfg.building_raise_m, 0.0)

    # --- 1-D network assembly ------------------------------------------------
    channel_fp = main_fp | creek_fp
    dem_display = z.copy()
    rbank = _bank_crest(dem_display, dx, rx, ry, half_top, channel_fp, r_bed)
    net = _assemble_network(cfg, rx, ry, r_bed, rbank, incision, picks, creek_lines,
                                        z, dx, channel_fp, y_coast_at)

    # pits (outside channels and sea)
    z_filled, n_fill, n_left = fill_single_cell_pits(z, protect=channel_fp | sea)
    if n_left:
        raise RuntimeError(f"synthetic DEM still has {n_left} single-cell pits after filling")
    dem_display = z_filled

    # coupled DEM: 1-D footprint raised to the bank crest of the nearest node
    dem2d = dem_display.copy()
    fp_idx = np.nonzero(channel_fp)
    all_x = np.concatenate([r.x for r in net.reaches])
    all_y = np.concatenate([r.y for r in net.reaches])
    all_b = np.concatenate([r.z_bank for r in net.reaches])
    px, py = X[fp_idx], Y[fp_idx]
    near = np.argmin((px[:, None] - all_x[None]) ** 2 + (py[:, None] - all_y[None]) ** 2, axis=1)
    dem2d[fp_idx] = all_b[near]

    # --- land use ------------------------------------------------------------
    lu = _landuse(rng, ny, nx, dx, sea, channel_fp, road, built, lot, rho, w_fp, hill_field, S,
                  creek_lines, X, Y)
    d_ch = np.minimum.reduce([d_riv] + [polyline_distance(cl[0], cl[1], X, Y)[0] for cl in creek_lines]) \
        if creek_lines else d_riv

    report = {
        "grid": f"{nx} x {ny} @ {dx:g} m",
        "pits_filled": n_fill,
        "elev_min_m": float(dem_display.min()), "elev_max_m": float(dem_display.max()),
        "n_channel_nodes": int(sum(r.n for r in net.reaches)),
        "n_reaches": net.n_reaches, "n_junctions": len(net.junctions),
        "urban_core_xy": core,
        "sea_fraction": float(sea.mean()),
        "landuse_fractions": {LU.CLASS_NAMES[c]: float((lu == c).mean()) for c in range(LU.N_CLASSES)},
        "multi_cell_depressions": "not filled (only single-cell pits are removed)",
    }
    log.info("synthetic domain %s, %d pits filled", report["grid"], n_fill)
    return Domain(cfg, dx, x, y, dem_display, dem2d, lu, sea, channel_fp, road, built,
                  d_ch, S, net, core, "SYNTHETIC", report)


def _spaced_picks(rng, candidates, k, min_gap):
    for _ in range(500):
        p = np.sort(rng.choice(candidates, size=k, replace=False))
        if k == 1 or np.diff(p).min() >= min_gap:
            return p
    raise RuntimeError("cannot place tidal-creek junctions with the requested spacing; "
                       "reduce n_creeks or channel_node_spacing_m")


def _sample(z, dx, px, py):
    ny, nx = z.shape
    i = np.clip((px / dx).astype(int), 0, nx - 1)
    j = np.clip((py / dx).astype(int), 0, ny - 1)
    return z[j, i]


def _bank_crest(z, dx, rx, ry, half_top, footprint, bed):
    """Lower of the two bank crests, sampled just outside the channel footprint."""
    tx, ty = np.gradient(rx), np.gradient(ry)
    nrm = np.hypot(tx, ty) + 1e-12
    nx_, ny_ = -ty / nrm, tx / nrm
    ny, nxg = z.shape
    out = np.empty(len(rx))
    for k in range(len(rx)):
        vals = []
        for sgn in (-1, 1):
            for off in np.arange(half_top[k] + 0.5 * dx, half_top[k] + 4 * dx, 0.5 * dx):
                px, py = rx[k] + sgn * off * nx_[k], ry[k] + sgn * off * ny_[k]
                i, j = int(px // dx), int(py // dx)
                if 0 <= i < nxg and 0 <= j < ny and not footprint[j, i]:
                    vals.append(z[j, i]); break
        out[k] = min(vals) if vals else bed[k] + 2.0
    # smooth bank line lightly; enforce minimum freeboard above bed
    out = np.convolve(np.pad(out, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return np.maximum(out, bed + 0.5)


def _assemble_network(cfg, rx, ry, r_bed, rbank, incision, picks, creek_lines, z, dx, footprint, y_coast_at):
    net = ChannelNetwork()
    n = len(rx)
    w, m, nm = cfg.channel_width_m, cfg.channel_side_slope, cfg.channel_manning
    ds = cfg.channel_node_spacing_m
    # junction nodes on the main river
    for kb in picks:
        net.junctions.append(Junction(float(rx[kb]), float(ry[kb]), float(r_bed[kb]), w, m, nm,
                                      float(rbank[kb]), length=ds))
    bounds = [0] + [int(k) for k in picks] + [n]
    for seg in range(len(bounds) - 1):
        a = bounds[seg] + (1 if seg > 0 else 0)
        b = bounds[seg + 1]
        idx = np.arange(a, b)
        if len(idx) < 2:
            raise RuntimeError("junctions too close together on the main river; adjust channel_node_spacing_m")
        up = ("inflow",) if seg == 0 else ("junction", seg - 1)
        down = ("junction", seg) if seg < len(picks) else ("stage",)
        rid = f"main_{seg}"
        net.reaches.append(Reach(rid, rx[idx], ry[idx], r_bed[idx], np.full(len(idx), w), np.full(len(idx), m),
                                 np.full(len(idx), nm), rbank[idx], up, down))
        if seg == 0:
            net.inflow_reach = rid
    for c, (cxl, cyl, cbed, cinc, kb, onland) in enumerate(creek_lines):
        end = int(np.argmin(onland)) if not onland.all() else len(cxl)
        arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cxl[:end]), np.diff(cyl[:end])))])
        nn = max(3, int(arc[-1] / ds))
        a_n = np.linspace(arc[-1] / nn * 0.75, arc[-1] - 0.5 * arc[-1] / nn, nn)
        px, py = np.interp(a_n, arc, cxl[:end]), np.interp(a_n, arc, cyl[:end])
        pb = np.interp(a_n, arc, cbed[:end])
        pb = np.minimum.accumulate(pb)
        cw = cfg.creek_width_m
        top = 0.5 * cw + m * cinc
        bank = _bank_crest(z, dx, px, py, np.full(nn, top), footprint, pb)
        net.reaches.append(Reach(f"creek_{c}", px, py, pb, np.full(nn, cw), np.full(nn, m), np.full(nn, nm),
                                 bank, ("junction", c), ("stage",)))
    net.validate()
    return net


def _landuse(rng, ny, nx, dx, sea, channel, road, built, lot, rho, w_fp, hills, S, creek_lines, X, Y):
    lu = np.full((ny, nx), LU.GRASS, dtype=np.int8)
    n1 = spectral_noise(rng, ny, nx, dx, beta=1.5, lmin=150.0)
    n2 = spectral_noise(rng, ny, nx, dx, beta=1.5, lmin=150.0)
    rural = np.where(n1 > -0.1, LU.CROPLAND, LU.GRASS)
    rural = np.where(n2 > 1.1, LU.BARE, rural)
    rural = np.where((n1 < -1.2) & (n2 < 0), LU.RESIDENTIAL, rural)     # scattered villages
    lu[:] = rural
    lu[(w_fp > 0.5) & (n2 < 1.3)] = LU.CROPLAND                          # ricefields on the floodplain
    lu[(hills > 5.0)] = np.where(n2[hills > 5.0] > 0.6, LU.BARE, LU.GRASS)
    near_creek = np.zeros_like(sea)
    for cxl, cyl, *_ in creek_lines:
        d, _ = polyline_distance(cxl, cyl, X, Y)
        near_creek |= d < 90
    mang = ((S >= 0) & (S < 250 + 60 * n1)) | (near_creek & (S < 900))
    lu[mang] = LU.MANGROVE
    lu[lot] = np.where(rng.uniform(size=int(lot.sum())) < 0.5, LU.GRASS, LU.BARE)
    lu[built] = np.where((rho + 0.15 * n1)[built] > 0.55, LU.DENSE_URBAN, LU.RESIDENTIAL)
    lu[road] = LU.ROAD
    lu[channel | sea] = LU.WATER
    return lu
