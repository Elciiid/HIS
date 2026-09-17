"""Loss terms for GeoKAN-PINO.

All 2-D residuals use the starred (non-dimensional) variables defined in
geokan_pino.py, time derivatives from forward-mode AD with respect to t*, and
spatial derivatives from central differences on the coarse grid. They are
evaluated only on wet interior cells away from the sea boundary and from cells
that exchange water with the channel (where the lateral source is not known).

For wet cells the retention store is already full (it fills before any surface
water remains), so the continuity source is rainfall minus the potential
infiltration rate.

1-D residuals are evaluated at chain faces where both neighbours are below the
bank crest (no lateral exchange) and scaled by reference magnitudes:
continuity by Q0 / dx_ref, momentum by g A_ref S_ref with S_ref = 1e-3.
"""
from __future__ import annotations

import torch

MMH = 1.0 / 3.6e6


def _ddx(a, d):
    out = torch.zeros_like(a)
    out[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (2 * d)
    return out


def _ddy(a, d):
    out = torch.zeros_like(a)
    out[..., 1:-1, :] = (a[..., 2:, :] - a[..., :-2, :]) / (2 * d)
    return out


def data_loss(o2, o1, b, qi, sc, y1_scale):
    """Depth-weighted MSE (weight 1 + h/0.3) plus velocity and channel terms."""
    h_t = b["h"][qi] / sc.H0
    w = 1.0 + b["h"][qi] / 0.3
    lh = (w * (o2[..., 0] - h_t) ** 2).mean()
    wet = (b["h"][qi] > 1e-3).float()
    lu = (wet * ((o2[..., 1] - b["u"][qi] / sc.U0) ** 2 + (o2[..., 2] - b["v"][qi] / sc.U0) ** 2)).sum() \
        / wet.sum().clamp(min=1.0)
    ly = ((o1[..., 0] - b["y1"][qi] / y1_scale) ** 2).mean()
    lq = ((o1[..., 1] - b["q1"][qi] / sc.Q0) ** 2).mean()
    return lh + lu + ly + lq, {"h": lh, "uv": lu, "y1": ly, "q1": lq}


def unweighted_metrics(o2, b, qi, sc):
    h = o2[..., 0] * sc.H0
    ht = b["h"][qi]
    wet = ht > 0.05
    rmse_all = ((h - ht) ** 2).mean().sqrt()
    rmse_wet = ((h - ht)[wet] ** 2).mean().sqrt() if wet.any() else torch.zeros(())
    return {"rmse_h_all": rmse_all, "rmse_h_wet": rmse_wet}


def residuals_2d(o2, d2, b, qi, sc, gr, cell):
    Q = o2.shape[0]
    ny, nx = gr.ny, gr.nx
    shp = (Q, ny, nx)
    h, u, v = (o2[..., k].reshape(shp) for k in range(3))
    dh, du, dv = (d2[..., k].reshape(shp) for k in range(3))
    dhu = h * du + u * dh
    dhv = h * dv + v * dh
    ds = cell / sc.L
    z = (b["z2"] / sc.H0).reshape(1, ny, nx)
    eta = h + z
    t = b["times"][qi]
    te = b["t_end"]
    n_r = b["rain_mmh"].shape[0]
    k = torch.clamp((t / te * n_r).long(), 0, n_r - 1)
    rain = b["rain_mmh"][k] * MMH                                   # (Q,)
    infil = b["horton_mmh"][qi].reshape(shp) * MMH
    src = (rain[:, None, None] - infil) * sc.T0 / sc.H0
    cont = dh + _ddx(h * u, ds) + _ddy(h * v, ds) - src
    n2 = b["manning2"].reshape(1, ny, nx)
    speed = torch.sqrt(u * u + v * v + 1e-12)
    hf = torch.clamp(h, min=1e-3 / sc.H0) ** (1.0 / 3.0)
    kf = sc.g * n2 * n2 * sc.L / sc.H0 ** (4.0 / 3.0)
    momx = dhu + _ddx(h * u * u, ds) + _ddy(h * u * v, ds) + h * _ddx(eta, ds) + kf * u * speed / hf
    momy = dhv + _ddx(h * u * v, ds) + _ddy(h * v * v, ds) + h * _ddy(eta, ds) + kf * v * speed / hf
    wet = (b["h"][qi].reshape(shp) > 1e-3) & b["resid_mask2"].reshape(1, ny, nx)
    nw = wet.sum().clamp(min=1)
    ms = lambda r: (r[wet] ** 2).sum() / nw
    return {"continuity": ms(cont), "momentum_x": ms(momx), "momentum_y": ms(momy)}


def residuals_1d(o1, d1, b, sc, gr, y1_scale):
    even = torch.arange(0, gr.e1_src.shape[0], 2, device=o1.device)
    L, R = gr.e1_src[even], gr.e1_dst[even]
    dist = gr.e1_static[even, 2] * 100.0
    y = o1[..., 0] * y1_scale
    q = o1[..., 1] * sc.Q0
    dy = d1[..., 0] * y1_scale / sc.T0
    dq = d1[..., 1] * sc.Q0 / sc.T0
    sec = b["sec1"]
    A = sec.area(y)
    T = sec.top_width(y)
    eta = b["bed1"][None] + y
    aL, aR = A[:, L].clamp(min=1e-3), A[:, R].clamp(min=1e-3)
    abar = 0.5 * (aL + aR)
    qbar = 0.5 * (q[:, L] + q[:, R])
    pbar = 0.5 * (sec.perimeter(y)[:, L] + sec.perimeter(y)[:, R])
    rbar = (abar / pbar).clamp(min=1e-3)
    nbar = 0.5 * (b["n1"][L] + b["n1"][R])
    cont = 0.5 * (T[:, L] * dy[:, L] + T[:, R] * dy[:, R]) + (q[:, R] - q[:, L]) / dist
    mom = (0.5 * (dq[:, L] + dq[:, R]) + (q[:, R] ** 2 / aR - q[:, L] ** 2 / aL) / dist
           + sc.g * abar * (eta[:, R] - eta[:, L]) / dist + sc.g * nbar ** 2 * qbar * qbar.abs() / (abar * rbar ** (4 / 3)))
    yb = sec.yb
    below = (y[:, L] < yb[L]) & (y[:, R] < yb[R]) & (y[:, L] > 1e-2) & (y[:, R] > 1e-2)
    below = below.detach()
    nb = below.sum().clamp(min=1)
    a_ref = sec.bank_area().mean()
    c_scale = sc.Q0 / dist.mean()
    m_scale = sc.g * a_ref * 1e-3
    return {"stvenant": (((cont / c_scale) ** 2)[below].sum() + ((mom / m_scale) ** 2)[below].sum()) / nb}


def boundary_loss(o2, b, qi, sc, tide_fn):
    """Predicted free surface on the northern sea row must match the tide."""
    m = b["bc_mask2"]
    if not bool(m.any()):
        return torch.zeros((), device=o2.device)
    t = b["times"][qi]
    tide = tide_fn(t) / sc.H0                                       # (Q,)
    eta = o2[..., 0][:, m] + (b["z2"][m] / sc.H0)[None]
    return ((eta - tide[:, None]) ** 2).mean()


def global_mass_loss(o2, o1, b, qi, sc, cell, volume1d, y1_scale):
    v2 = o2[..., 0].sum(-1) * sc.H0 * cell * cell
    v1 = volume1d(o1[..., 0] * y1_scale, b)
    return (((v2 + v1) - b["v_target"][qi]) / sc.V_scale) ** 2


@torch.no_grad()
def physics_floor(b, sc, gr, cell):
    """Residual of the ENGINE's own coarse output, with time derivatives from central
    differences between snapshots. The surrogate cannot be expected to go below this
    level: it reflects coarse resolution and output spacing, not model error."""
    h = b["h"] / sc.H0
    u = b["u"] / sc.U0
    v = b["v"] / sc.U0
    t = b["times"] / sc.T0
    k = torch.arange(1, h.shape[0] - 1, device=h.device)
    dts = (t[k + 1] - t[k - 1])[:, None]
    dh = (h[k + 1] - h[k - 1]) / dts
    du = (u[k + 1] - u[k - 1]) / dts
    dv = (v[k + 1] - v[k - 1]) / dts
    o2 = torch.stack([h[k], u[k], v[k]], -1)
    d2 = torch.stack([dh, du, dv], -1)
    return {kk: float(vv) for kk, vv in residuals_2d(o2, d2, b, k, sc, gr, cell).items()}
