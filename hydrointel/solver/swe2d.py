"""2-D finite-volume shallow-water solver (PyTorch, fully vectorised).

State (h, hu, hv) on a Cartesian grid, rows south->north, columns west->east.

Numerics (each item is load-bearing):
* Well-balanced hydrostatic reconstruction (Audusse et al.): at each face
  z* = max(zL, zR), h*_{L,R} = max(0, eta_{L,R} - z*), flux evaluated on the
  starred states, plus the pressure correction g/2 (h^2 - h*^2) on each side,
  plus (2nd order) the centred bed source.
* HLLC flux (HLL optional) with two-rarefaction / dry-bed wave-speed estimates.
* Kurganov-Petrova velocity desingularisation, u = 0 below h_dry.
* Semi-implicit Manning friction after the flux update.
* Adaptive time step on wet cells. For the unsplit update we use
  dt = CFL / max((|u|+c)/dx + (|v|+c)/dy), which is the positivity-preserving
  form for four-face cells and never exceeds the per-direction minimum.
* Optional MUSCL (minmod on h, eta, u, v; first order next to dry cells) with
  SSP-RK2.
* Sources: rainfall, Horton infiltration capped by available water, and a
  sub-grid retention store S that fills before a cell contributes runoff.
  Storage never alters the bed elevation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..config import SolverConfig
from .boundaries import EDGES, EdgeBC, edge_values, pad_axis
from .infiltration import Horton

SQRT2 = math.sqrt(2.0)


def compile_if_cuda(fn, device, cfg):
    """torch.compile on CUDA (Inductor/Triton); eager elsewhere (no C++ toolchain assumed)."""
    if getattr(cfg, "compile", True) and torch.device(device).type == "cuda":
        import torch._dynamo.config as dc
        import torch._inductor.config as ic
        dc.recompile_limit = max(getattr(dc, "recompile_limit", 8), 256)
        dc.cache_size_limit = max(getattr(dc, "cache_size_limit", 8), 256)
        if hasattr(ic, "deterministic"):
            ic.deterministic = True
        return torch.compile(fn, dynamic=False)
    return fn


def minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))


def riemann_flux(hL, uL, vL, hR, uR, vR, g: float, scheme: str = "hllc"):
    """Numerical flux across a face (normal velocity u, tangential v).
    Returns (F_h, F_hu, F_hv, max |wave speed|)."""
    tiny = 1e-14
    cL, cR = torch.sqrt(g * hL), torch.sqrt(g * hR)
    dryL, dryR = hL <= tiny, hR <= tiny
    u_s = 0.5 * (uL + uR) + cL - cR
    c_s = 0.5 * (cL + cR) + 0.25 * (uL - uR)
    SL = torch.minimum(uL - cL, u_s - c_s)
    SR = torch.maximum(uR + cR, u_s + c_s)
    SL = torch.where(dryL, uR - 2.0 * cR, SL)
    SR = torch.where(dryL, uR + cR, SR)
    SL = torch.where(dryR, uL - cL, SL)
    SR = torch.where(dryR, uL + 2.0 * cL, SR)
    both_dry = dryL & dryR

    qL, qR = hL * uL, hR * uR
    FL1, FR1 = qL, qR
    FL2, FR2 = qL * uL + 0.5 * g * hL * hL, qR * uR + 0.5 * g * hR * hR
    den = SR - SL
    den = torch.where(den.abs() < tiny, torch.ones_like(den), den)
    H1 = (SR * FL1 - SL * FR1 + SL * SR * (hR - hL)) / den
    H2 = (SR * FL2 - SL * FR2 + SL * SR * (qR - qL)) / den
    left, right = SL >= 0, SR <= 0
    F1 = torch.where(left, FL1, torch.where(right, FR1, H1))
    F2 = torch.where(left, FL2, torch.where(right, FR2, H2))
    if scheme == "hllc":
        num = SL * hR * (uR - SR) - SR * hL * (uL - SL)
        dd = hR * (uR - SR) - hL * (uL - SL)
        SM = torch.where(dd.abs() < tiny, 0.5 * (uL + uR), num / torch.where(dd.abs() < tiny, torch.ones_like(dd), dd))
        F3 = F1 * torch.where(SM >= 0, vL, vR)
    else:
        FL3, FR3 = qL * vL, qR * vR
        H3 = (SR * FL3 - SL * FR3 + SL * SR * (hR * vR - hL * vL)) / den
        F3 = torch.where(left, FL3, torch.where(right, FR3, H3))
    zero = torch.zeros_like(F1)
    F1 = torch.where(both_dry, zero, F1)
    F2 = torch.where(both_dry, zero, F2)
    F3 = torch.where(both_dry, zero, F3)
    smax = torch.where(both_dry, zero, torch.maximum(SL.abs(), SR.abs()))
    return F1, F2, F3, smax


@dataclass
class StepVolumes:
    bnd_in: torch.Tensor      # volume entering through boundary faces (m^3, >= 0)
    bnd_out: torch.Tensor     # volume leaving through boundary faces (m^3, >= 0)
    rain: torch.Tensor
    infil: torch.Tensor
    clip: torch.Tensor


class SWE2D:
    def __init__(self, z, manning, dx: float, cfg: SolverConfig, bcs: dict[str, EdgeBC],
                 device=None, dtype=torch.float64, storage=None, horton: Horton | None = None,
                 rain_weight=None, dy: float | None = None):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=self.device).clone() if not torch.is_tensor(a) \
            else a.to(device=self.device, dtype=dtype).clone()
        self.z = t(z)
        self.ny, self.nx = self.z.shape
        self.n = t(manning) if (torch.is_tensor(manning) or hasattr(manning, "shape")) \
            else torch.full_like(self.z, float(manning))
        self.dx, self.dy = float(dx), float(dy if dy is not None else dx)
        self.area = self.dx * self.dy
        self.bcs = bcs
        self.g = cfg.g
        self.order = cfg.spatial_order
        self.ng = 1 if self.order == 1 else 2
        self.storage = t(storage) if storage is not None else None
        self.horton = horton
        self.rain_weight = t(rain_weight) if rain_weight is not None else None
        zeros = torch.zeros_like(self.z)
        self.h, self.hu, self.hv = zeros.clone(), zeros.clone(), zeros.clone()
        self.s_filled = zeros.clone()
        self.infil_cum = zeros.clone()
        self.n_dt_clamped = 0
        self._core = compile_if_cuda(self._step_core, self.device, cfg)
        self._rate = compile_if_cuda(self._max_rate, self.device, cfg)

    # ---------------------------------------------------------------- state
    def set_state(self, h, hu=None, hv=None):
        self.h = torch.as_tensor(h, dtype=self.dtype, device=self.device).clone()
        self.hu = torch.zeros_like(self.h) if hu is None else torch.as_tensor(hu, dtype=self.dtype, device=self.device).clone()
        self.hv = torch.zeros_like(self.h) if hv is None else torch.as_tensor(hv, dtype=self.dtype, device=self.device).clone()

    def set_level(self, eta: float, mask=None):
        h = torch.clamp(eta - self.z, min=0.0)
        if mask is not None:
            h = torch.where(torch.as_tensor(mask, device=self.device), h, torch.zeros_like(h))
        self.set_state(h)

    def velocities(self, h, hu, hv):
        h4 = h ** 4
        den = torch.sqrt(h4 + torch.clamp(h4, min=self.cfg.kp_eps))
        wet = h >= self.cfg.h_dry
        u = torch.where(wet, SQRT2 * h * hu / den, torch.zeros_like(h))
        v = torch.where(wet, SQRT2 * h * hv / den, torch.zeros_like(h))
        return u, v

    def volume(self) -> torch.Tensor:
        return self.h.to(torch.float64).sum() * self.area

    def storage_volume(self) -> torch.Tensor:
        return self.s_filled.to(torch.float64).sum() * self.area

    # ------------------------------------------------------------- time step
    def _max_rate(self, h, hu, hv):
        u, v = self.velocities(h, hu, hv)
        c = torch.sqrt(self.g * h)
        wet = h > self.cfg.h_dry
        rate = torch.where(wet, (u.abs() + c) / self.dx + (v.abs() + c) / self.dy, torch.zeros_like(c))
        return rate.max()

    def max_dt(self) -> float:
        m = float(self._rate(self.h, self.hu, self.hv))
        dt = self.cfg.cfl / m if m > 0 else self.cfg.dt_max
        if dt < self.cfg.dt_min or dt > self.cfg.dt_max:
            if dt < self.cfg.dt_min:
                self.n_dt_clamped += 1
            dt = min(max(dt, self.cfg.dt_min), self.cfg.dt_max)
        return dt

    # ------------------------------------------------------------ fluxes
    def _sweep(self, h, un, ut, z):
        """Face fluxes along the last dim of padded arrays.
        Returns (F1, F2, F3, corrL, corrR, Sc) with F* on the n+1 faces and Sc the
        centred 2nd-order bed source on the n cells (None for 1st order)."""
        g, ng = self.g, self.ng
        n = h.shape[-1] - 2 * ng
        sl = slice(ng - 1, ng + n + 1)
        H, U, V, Z = h[..., sl], un[..., sl], ut[..., sl], z[..., sl]
        Sc = None
        if self.order == 2:
            E = H + Z
            Epad = h + z

            def slope(q):
                d = q[..., 1:] - q[..., :-1]
                return minmod(d[..., :-1], d[..., 1:])
            hmin = torch.minimum(torch.minimum(h[..., :-2], h[..., 1:-1]), h[..., 2:])
            ok = (hmin >= self.cfg.h_dry).to(h.dtype)
            sH, sE, sU, sV = (slope(q) * ok for q in (h, Epad, un, ut))
            Hm, Hp = H - 0.5 * sH, H + 0.5 * sH
            Em, Ep = E - 0.5 * sE, E + 0.5 * sE
            Um, Up = U - 0.5 * sU, U + 0.5 * sU
            Vm, Vp = V - 0.5 * sV, V + 0.5 * sV
            Zm, Zp = Em - Hm, Ep - Hp
            Sc = -0.5 * g * (Hm[..., 1:-1] + Hp[..., 1:-1]) * (Zp[..., 1:-1] - Zm[..., 1:-1])
        else:
            Hm = Hp = H
            Um = Up = U
            Vm = Vp = V
            Zm = Zp = Z
        hL, uL, vL, zL = Hp[..., :-1], Up[..., :-1], Vp[..., :-1], Zp[..., :-1]
        hR, uR, vR, zR = Hm[..., 1:], Um[..., 1:], Vm[..., 1:], Zm[..., 1:]
        zf = torch.maximum(zL, zR)
        hLs = torch.clamp(hL + zL - zf, min=0.0)
        hRs = torch.clamp(hR + zR - zf, min=0.0)
        F1, F2, F3, _ = riemann_flux(hLs, uL, vL, hRs, uR, vR, g, self.cfg.flux)
        corrL = 0.5 * g * (hL * hL - hLs * hLs)
        corrR = 0.5 * g * (hR * hR - hRs * hRs)
        return F1, F2, F3, corrL, corrR, Sc

    @staticmethod
    def _donor_scale(F1, theta):
        """Per-face scale factor = theta of the upwind (donor) cell; ghosts are unlimited."""
        one = torch.ones_like(theta[..., :1])
        tp = torch.cat([one, theta, one], dim=-1)
        return torch.where(F1 > 0, tp[..., :-1], tp[..., 1:])

    @staticmethod
    def _assemble(F1, F2, F3, corrL, corrR, Sc, dn):
        d1 = -(F1[..., 1:] - F1[..., :-1]) / dn
        d2 = -((F2[..., 1:] + corrL[..., 1:]) - (F2[..., :-1] + corrR[..., :-1])) / dn
        if Sc is not None:
            d2 = d2 + Sc / dn
        d3 = -(F3[..., 1:] - F3[..., :-1]) / dn
        return d1, d2, d3

    def rhs(self, h, hu, hv, vals, dt):
        u, v = self.velocities(h, hu, hv)
        hx, ux, vx, zx = pad_axis(self.bcs, h, u, v, self.z, vals, self.ng, "x", self.g, self.cfg.h_dry)
        X = self._sweep(hx, ux, vx, zx)
        hy, uy, vy, zy = pad_axis(self.bcs, h, u, v, self.z, vals, self.ng, "y", self.g, self.cfg.h_dry)
        Y = self._sweep(hy.T, vy.T, uy.T, zy.T)
        # conservative positivity limiter: a cell may not export more than it holds this step
        Fx, Gy = X[0], Y[0]
        out = ((torch.relu(Fx[..., 1:]) + torch.relu(-Fx[..., :-1])) / self.dx
               + (torch.relu(Gy[..., 1:]) + torch.relu(-Gy[..., :-1])).T / self.dy)
        need = out * dt
        theta = torch.where(need > h, h / torch.where(need > 0, need, torch.ones_like(need)), torch.ones_like(h))
        sx = self._donor_scale(Fx, theta)
        sy = self._donor_scale(Gy, theta.T)
        a1, a2, a3 = self._assemble(Fx * sx, X[1], X[2] * sx, X[3], X[4], X[5], self.dx)
        b1, b2, b3 = self._assemble(Gy * sy, Y[1], Y[2] * sy, Y[3], Y[4], Y[5], self.dy)
        dh = a1 + b1.T
        dhu = a2 + b3.T
        dhv = a3 + b2.T
        Fx, Gy = Fx * sx, Gy * sy
        # gross boundary inflow / outflow rates (m^3/s), face by face
        w, e, so, no = Fx[:, 0], Fx[:, -1], Gy[:, 0], Gy[:, -1]
        qin = (torch.relu(w).sum() + torch.relu(-e).sum()) * self.dy + (torch.relu(so).sum() + torch.relu(-no).sum()) * self.dx
        qout = (torch.relu(-w).sum() + torch.relu(e).sum()) * self.dy + (torch.relu(-so).sum() + torch.relu(no).sum()) * self.dx
        return dh, dhu, dhv, torch.stack([qin, qout])

    # ------------------------------------------------------------ one step
    def step(self, dt: float, t: float, rain_rate: float = 0.0) -> StepVolumes:
        sc = torch.tensor([dt, t, rain_rate] + edge_values(self.bcs, t) + edge_values(self.bcs, t + dt),
                          dtype=self.dtype, device=self.device)
        out = self._core(self.h, self.hu, self.hv, self.s_filled, self.infil_cum, sc)
        self.h, self.hu, self.hv, self.s_filled, self.infil_cum = out[:5]
        return StepVolumes(*out[5:])

    def _step_core(self, h0, hu0, hv0, s_filled, infil_cum, sc):
        """Pure tensor update (compiled on CUDA). sc = [dt, t, rain, bc(t) x4, bc(t+dt) x4]."""
        dt, t, rain_rate = sc[0], sc[1], sc[2]
        vals0 = dict(zip(EDGES, sc[3:7]))
        vals1 = dict(zip(EDGES, sc[7:11]))
        f64 = torch.float64
        d = self.rhs(h0, hu0, hv0, vals0, dt)
        h1, hu1, hv1 = h0 + dt * d[0], hu0 + dt * d[1], hv0 + dt * d[2]
        qin = d[3]
        neg = torch.clamp(-h1, min=0.0)
        clip = neg.to(f64).sum() * self.area
        h1 = h1 + neg
        if self.order == 2:
            d2 = self.rhs(h1, hu1, hv1, vals1, dt)
            h1 = 0.5 * (h0 + h1 + dt * d2[0])
            hu1 = 0.5 * (hu0 + hu1 + dt * d2[1])
            hv1 = 0.5 * (hv0 + hv1 + dt * d2[2])
            qin = 0.5 * (qin + d2[3])
            neg = torch.clamp(-h1, min=0.0)
            clip = clip + neg.to(f64).sum() * self.area
            h1 = h1 + neg
        bnd = qin.to(f64) * dt.to(f64)          # [in, out]

        # --- rainfall, infiltration, retention storage ---------------------
        if self.rain_weight is None:
            h1 = h1 + rain_rate * dt
            rain_v = (rain_rate * dt).to(f64) * (self.nx * self.ny * self.area)
        else:
            add = rain_rate * dt * self.rain_weight
            h1 = h1 + add
            rain_v = add.to(f64).sum() * self.area
        inf_v = torch.zeros((), dtype=f64, device=h1.device)
        if self.horton is not None:
            pot = self.horton.rate(t) * dt
            take_surf = torch.minimum(pot, h1)
            safe = torch.where(h1 > 0, h1, torch.ones_like(h1))
            scale = torch.where(h1 > 0, (h1 - take_surf) / safe, torch.zeros_like(h1))
            h1 = h1 - take_surf
            hu1, hv1 = hu1 * scale, hv1 * scale
            taken = take_surf
            if self.storage is not None:
                take_store = torch.minimum(pot - take_surf, s_filled)
                s_filled = s_filled - take_store
                taken = taken + take_store
            infil_cum = infil_cum + taken
            inf_v = taken.to(f64).sum() * self.area
        if self.storage is not None:
            room = torch.clamp(self.storage - s_filled, min=0.0)
            fill = torch.minimum(room, h1)
            safe = torch.where(h1 > 0, h1, torch.ones_like(h1))
            scale = torch.where(h1 > 0, (h1 - fill) / safe, torch.zeros_like(h1))
            h1 = h1 - fill
            hu1, hv1 = hu1 * scale, hv1 * scale
            s_filled = s_filled + fill

        # --- semi-implicit friction ---------------------------------------
        u, v = self.velocities(h1, hu1, hv1)
        speed = torch.sqrt(u * u + v * v)
        hf = torch.clamp(h1, min=self.cfg.h_dry)
        denom = 1.0 + dt * self.g * self.n * self.n * speed / hf ** (4.0 / 3.0)
        u, v = u / denom, v / denom
        wet = h1 >= self.cfg.h_dry
        hu_new = torch.where(wet, h1 * u, torch.zeros_like(h1))
        hv_new = torch.where(wet, h1 * v, torch.zeros_like(h1))
        return h1, hu_new, hv_new, s_filled, infil_cum, bnd[0], bnd[1], rain_v, inf_v, clip
