"""Exact solutions used by the verification suite (numpy, float64)."""
from __future__ import annotations

import numpy as np

G = 9.81


# ---------------------------------------------------------------------------
# Dam breaks: exact Riemann solver for the 1-D shallow-water equations
# (two-rarefaction / shock branches, dry-bed case = Ritter)
# ---------------------------------------------------------------------------
def ritter(x, t, x0, h0, g=G):
    """Ritter dry-bed dam break. Returns (h, u)."""
    c0 = np.sqrt(g * h0)
    xi = (np.asarray(x, float) - x0) / t
    h = np.where(xi <= -c0, h0, np.where(xi >= 2 * c0, 0.0, (2 * c0 - xi) ** 2 / (9 * g)))
    u = np.where((xi > -c0) & (xi < 2 * c0), 2.0 / 3.0 * (xi + c0), 0.0)
    return h, u


def _f(h, hk, g):
    ck = np.sqrt(g * hk)
    if h <= hk:
        return 2.0 * (np.sqrt(g * h) - ck)
    return (h - hk) * np.sqrt(0.5 * g * (h + hk) / (h * hk))


def star_state(hL, uL, hR, uR, g=G):
    """Star-region depth and velocity for wet left/right states (bisection)."""
    if hL <= 0 or hR <= 0:
        raise ValueError("star_state needs wet states; use ritter() for a dry bed")
    fun = lambda h: _f(h, hL, g) + _f(h, hR, g) + uR - uL
    lo, hi = 1e-12, 10.0 * max(hL, hR) + (uL - uR) ** 2 / g
    if fun(lo) > 0:
        raise ValueError("dry region generated (vacuum); not supported here")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fun(mid) > 0:
            hi = mid
        else:
            lo = mid
    hs = 0.5 * (lo + hi)
    us = 0.5 * (uL + uR) + 0.5 * (_f(hs, hR, g) - _f(hs, hL, g))
    return hs, us


def riemann(x, t, x0, hL, uL, hR, uR, g=G):
    """Exact solution of the wet-wet Riemann problem (e.g. Stoker). Returns (h, u)."""
    hs, us = star_state(hL, uL, hR, uR, g)
    cL, cR, cs = np.sqrt(g * hL), np.sqrt(g * hR), np.sqrt(g * hs)
    S = (np.asarray(x, float) - x0) / t
    h = np.empty_like(S)
    u = np.empty_like(S)
    left = S <= us
    # left wave
    if hs > hL:
        sl = uL - cL * np.sqrt(0.5 * (hs + hL) * hs) / hL
        m = left & (S <= sl); h[m], u[m] = hL, uL
        m = left & (S > sl); h[m], u[m] = hs, us
    else:
        shl, stl = uL - cL, us - cs
        m = left & (S <= shl); h[m], u[m] = hL, uL
        m = left & (S >= stl); h[m], u[m] = hs, us
        m = left & (S > shl) & (S < stl)
        c = (uL + 2 * cL - S[m]) / 3.0
        h[m], u[m] = c * c / g, (uL + 2 * cL + 2 * S[m]) / 3.0
    right = ~left
    if hs > hR:
        sr = uR + cR * np.sqrt(0.5 * (hs + hR) * hs) / hR
        m = right & (S >= sr); h[m], u[m] = hR, uR
        m = right & (S < sr); h[m], u[m] = hs, us
    else:
        shr, str_ = uR + cR, us + cs
        m = right & (S >= shr); h[m], u[m] = hR, uR
        m = right & (S <= str_); h[m], u[m] = hs, us
        m = right & (S > str_) & (S < shr)
        c = (-uR + 2 * cR + S[m]) / 3.0
        h[m], u[m] = c * c / g, (uR - 2 * cR + 2 * S[m]) / 3.0
    return h, u


# ---------------------------------------------------------------------------
# Thacker (1981) planar surface oscillating in a paraboloid (frictionless)
# ---------------------------------------------------------------------------
def thacker_planar(X, Y, t, a=1.0, h0=0.1, eta=0.5, L=4.0, g=G):
    """Bed z = -h0 (1 - r^2/a^2) about the domain centre. Returns (h, u, v, z, period)."""
    om = np.sqrt(2 * g * h0) / a
    xc, yc = X - L / 2, Y - L / 2
    z = -h0 * (1 - (xc ** 2 + yc ** 2) / a ** 2)
    surf = eta * h0 / a ** 2 * (2 * xc * np.cos(om * t) + 2 * yc * np.sin(om * t) - eta)
    h = np.maximum(surf - z, 0.0)
    wet = h > 0
    u = np.where(wet, -eta * om * np.sin(om * t), 0.0)
    v = np.where(wet, eta * om * np.cos(om * t), 0.0)
    return h, u, v, z, 2 * np.pi / om


# ---------------------------------------------------------------------------
# Steady flow over a bump (frictionless, 25 m channel)
# ---------------------------------------------------------------------------
def bump_bed(x):
    x = np.asarray(x, float)
    return np.where((x > 8) & (x < 12), 0.2 - 0.05 * (x - 10) ** 2, 0.0)


def _energy_root(q, E, z, branch, g=G):
    """Depth with specific energy E - z on the sub- or supercritical branch."""
    hc = (q * q / g) ** (1 / 3)
    spec = lambda h: q * q / (2 * g * h * h) + h + z
    if branch == "sub":
        lo, hi = np.full_like(z, hc), np.full_like(z, E + 1.0)
    else:
        lo, hi = np.full_like(z, 1e-6), np.full_like(z, hc)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        above = spec(mid) > E
        if branch == "sub":
            hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
        else:
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def bump_exact(x, case, g=G):
    """case in {'subcritical', 'transcritical', 'shock'}. Returns (h, q, info)."""
    x = np.asarray(x, float)
    z = bump_bed(x)
    zmax = 0.2
    if case == "subcritical":
        q, hout = 4.42, 2.0
        E = q * q / (2 * g * hout ** 2) + hout
        return _energy_root(q, E, z, "sub"), q, {"q": q, "h_out": hout}
    q = 1.53 if case == "transcritical" else 0.18
    hc = (q * q / g) ** (1 / 3)
    E1 = 1.5 * hc + zmax
    h = np.where(x <= 10, _energy_root(q, E1, z, "sub"), _energy_root(q, E1, z, "super"))
    info = {"q": q}
    if case == "shock":
        hout = 0.33
        E2 = q * q / (2 * g * hout ** 2) + hout
        h2 = _energy_root(q, E2, z, "sub")
        fr1 = q * q / (g * h ** 3)
        conj = 0.5 * h * (np.sqrt(1 + 8 * fr1) - 1)
        xs_grid = np.linspace(10.0001, 25, 200001)
        zg = bump_bed(xs_grid)
        h1g = _energy_root(q, E1, zg, "super")
        h2g = _energy_root(q, E2, zg, "sub")
        cg = 0.5 * h1g * (np.sqrt(1 + 8 * q * q / (g * h1g ** 3)) - 1)
        k = np.argmin(np.abs(cg - h2g))
        xs = xs_grid[k]
        h = np.where(x < xs, h, h2)
        info.update({"h_out": hout, "x_shock": float(xs)})
        del conj
    return h, q, info


# ---------------------------------------------------------------------------
# Kinematic-wave runoff from a tilted plane under constant rainfall
# ---------------------------------------------------------------------------
def kinematic_plane(t, length, slope, n, rain_ms, duration):
    """Outlet discharge per unit width q(t) [m^2/s] for t <= duration
    (rising limb and equilibrium). Manning: q = alpha h^m, m = 5/3."""
    alpha, m = np.sqrt(slope) / n, 5.0 / 3.0
    t = np.asarray(t, float)
    te = (length / (alpha * rain_ms ** (m - 1))) ** (1 / m)
    q = np.where(t < te, alpha * (rain_ms * t) ** m, rain_ms * length)
    q = np.where(t <= duration, q, np.nan)
    return q, te
