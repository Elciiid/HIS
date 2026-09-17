"""1-D St. Venant solver on a channel network (finite volume, PyTorch).

State per cell: wetted area A [m^2] and discharge Q [m^3/s].

    dA/dt + dQ/dx = q_lat
    dQ/dt + d(Q^2/A + g I1)/dx = g A (S0 - Sf) + g I2

Discretisation
* Cells are the network nodes; junction nodes are level-pool cells shared by
  the reaches that meet there (mass conserved, one water level, no momentum).
  Boundaries (inflow hydrograph, tidal stage, wall) are ghost cells.
* Hydrostatic reconstruction generalised to channels of varying section: at a
  face, z* = max(zL, zR), y*_{L,R} = max(0, eta_{L,R} - z*), and the face is
  given one trapezoidal section (mean of its neighbours). The HLL flux is
  evaluated on the starred states and each cell receives the pressure term
  g I1_face(y*) of its own side back, so that
      dQ_i/dt = -[ (F_{i+1/2} - p(y*_{L,i+1/2})) - (F_{i-1/2} - p(y*_{R,i-1/2})) ] / dx_i
  which is exactly balanced for still water and consistent with
  -g A d(eta)/dx for any section variation (d I1/dy = A).
* Trapezoidal sections with a Preissmann slot above the bank crest
  (width = slot_width_frac x bankfull top width) so surcharged reaches need no
  special-casing.
* Semi-implicit Manning friction, CFL time step on wet cells, and the same
  conservative outflow limiter as the 2-D solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..config import SolverConfig
from ..domain.channels import ChannelNetwork
from .swe2d import compile_if_cuda


# ---------------------------------------------------------------------------
# section geometry (vectorised)
# ---------------------------------------------------------------------------
@dataclass
class Sections:
    b: torch.Tensor       # bottom width
    m: torch.Tensor       # side slope H:V
    yb: torch.Tensor      # bankfull depth
    ts: torch.Tensor      # slot width

    def bank_area(self):
        return self.yb * (self.b + self.m * self.yb)

    def area(self, y):
        yt = torch.minimum(y, self.yb)
        at = yt * (self.b + self.m * yt)
        return at + self.ts * torch.clamp(y - self.yb, min=0.0)

    def top_width(self, y):
        return torch.where(y <= self.yb, self.b + 2.0 * self.m * y, self.ts)

    def i1(self, y):
        """First moment of area about the free surface: int_0^y (y - s) w(s) ds."""
        yt = torch.minimum(y, self.yb)
        above = torch.clamp(y - self.yb, min=0.0)
        i_t = self.b * yt * yt / 2.0 + self.m * yt ** 3 / 3.0
        at = yt * (self.b + self.m * yt)
        return i_t + at * above + self.ts * above * above / 2.0

    def perimeter(self, y):
        yt = torch.minimum(y, self.yb)
        return self.b + 2.0 * yt * torch.sqrt(1.0 + self.m * self.m)

    def depth(self, A):
        A = torch.clamp(A, min=0.0)
        ab = self.bank_area()
        y_t = 2.0 * A / (self.b + torch.sqrt(self.b * self.b + 4.0 * self.m * A))
        return torch.where(A <= ab, y_t, self.yb + (A - ab) / self.ts)


def hll_1d(AL, QL, pL, cL, AR, QR, pR, cR, uL, uR):
    tiny = 1e-14
    dryL, dryR = AL <= tiny, AR <= tiny
    SL = torch.minimum(uL - cL, uR - cR)
    SR = torch.maximum(uL + cL, uR + cR)
    SL = torch.where(dryL, uR - 2.0 * cR, SL)
    SR = torch.where(dryL, uR + cR, SR)
    SL = torch.where(dryR, uL - cL, SL)
    SR = torch.where(dryR, uL + 2.0 * cL, SR)
    FL1, FR1 = QL, QR
    FL2, FR2 = QL * uL + pL, QR * uR + pR
    den = SR - SL
    den = torch.where(den.abs() < tiny, torch.ones_like(den), den)
    H1 = (SR * FL1 - SL * FR1 + SL * SR * (AR - AL)) / den
    H2 = (SR * FL2 - SL * FR2 + SL * SR * (QR - QL)) / den
    left, right = SL >= 0, SR <= 0
    F1 = torch.where(left, FL1, torch.where(right, FR1, H1))
    F2 = torch.where(left, FL2, torch.where(right, FR2, H2))
    both = dryL & dryR
    z = torch.zeros_like(F1)
    return torch.where(both, z, F1), torch.where(both, z, F2)


# ---------------------------------------------------------------------------
# network topology -> flat FV arrays
# ---------------------------------------------------------------------------
@dataclass
class Topology:
    n_cells: int
    kind: np.ndarray            # 0 reach, 1 junction, 2 ghost-inflow, 3 ghost-stage, 4 ghost-closed
    reach_of: np.ndarray        # reach index per cell (-1 for junction/ghost)
    ghost_src: np.ndarray       # interior neighbour for ghost cells (-1 otherwise)
    ghost_sign: np.ndarray      # +1 if the ghost is downstream of its neighbour, -1 upstream
    face_l: np.ndarray
    face_r: np.ndarray
    z: np.ndarray
    b: np.ndarray
    m: np.ndarray
    zbank: np.ndarray
    n: np.ndarray
    dx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    chain: np.ndarray           # chainage within own reach


def build_topology(net: ChannelNetwork) -> Topology:
    cols = {k: [] for k in ("kind", "reach", "gsrc", "gsign", "z", "b", "m", "zbank", "n", "dx", "x", "y", "chain")}
    fl, fr = [], []

    def add(kind, reach, z, b, m, zb, n, dx, x, y, ch=0.0, gsrc=-1, gsign=0):
        for k, v in zip(cols, (kind, reach, gsrc, gsign, z, b, m, zb, n, dx, x, y, ch)):
            cols[k].append(v)
        return len(cols["kind"]) - 1

    jcell = []
    for j in net.junctions:
        jcell.append(add(1, -1, j.bed, j.width, j.side_slope, j.z_bank, j.manning, j.length, j.x, j.y))
    kinds = {"inflow": 2, "stage": 3, "closed": 4}
    for ri, r in enumerate(net.reaches):
        ds, ch = r.spacing, r.chainage
        ids = [add(0, ri, r.bed[k], r.width[k], r.side_slope[k], r.z_bank[k], r.manning[k], ds[k], r.x[k], r.y[k], ch[k])
               for k in range(r.n)]
        for a, b in zip(ids[:-1], ids[1:]):
            fl.append(a); fr.append(b)
        for end, first, sign in ((r.up, ids[0], -1), (r.down, ids[-1], +1)):
            if end[0] == "junction":
                other = jcell[end[1]]
            else:
                k = 0 if sign < 0 else r.n - 1
                other = add(kinds[end[0]], ri, r.bed[k], r.width[k], r.side_slope[k], r.z_bank[k], r.manning[k],
                            ds[k], r.x[k], r.y[k], ch[k], gsrc=first, gsign=sign)
            if sign < 0:
                fl.append(other); fr.append(first)
            else:
                fl.append(first); fr.append(other)
    a = {k: np.array(v) for k, v in cols.items()}
    return Topology(len(a["kind"]), a["kind"].astype(np.int64), a["reach"].astype(np.int64), a["gsrc"].astype(np.int64),
                    a["gsign"].astype(np.int64), np.array(fl), np.array(fr), a["z"], a["b"], a["m"], a["zbank"],
                    a["n"], a["dx"], a["x"], a["y"], a["chain"])


@dataclass
class Step1DVolumes:
    bnd_in: torch.Tensor      # gross volume entering through boundary ghosts (m^3)
    bnd_out: torch.Tensor     # gross volume leaving (m^3)
    clip: torch.Tensor


class SWE1D:
    def __init__(self, net: ChannelNetwork, cfg: SolverConfig, device=None, dtype=torch.float64,
                 inflow=None, stage=None):
        """``inflow(t)`` -> m^3/s at every inflow end; ``stage(t)`` -> water level at stage ends."""
        net.validate()
        self.net, self.cfg = net, cfg
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.topo = tp = build_topology(net)
        T = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=self.device)
        self.z, self.nman, self.dx = T(tp.z), T(tp.n), T(tp.dx)
        yb = np.maximum(tp.zbank - tp.z, 0.2)
        ts = cfg.slot_width_frac * (tp.b + 2 * tp.m * yb)
        self.sec = Sections(T(tp.b), T(tp.m), T(yb), T(ts))
        fl, fr = tp.face_l, tp.face_r
        zf = np.maximum(tp.z[fl], tp.z[fr])
        bank_f = 0.5 * (tp.zbank[fl] + tp.zbank[fr])
        ybf = np.maximum(bank_f - zf, 0.1)
        bf, mf = 0.5 * (tp.b[fl] + tp.b[fr]), 0.5 * (tp.m[fl] + tp.m[fr])
        self.fsec = Sections(T(bf), T(mf), T(ybf), T(cfg.slot_width_frac * (bf + 2 * mf * ybf)))
        self.fl, self.fr = T(fl, torch.long), T(fr, torch.long)
        self.is_ghost = T(tp.kind >= 2, torch.bool)
        self.is_junc = T(tp.kind == 1, torch.bool)
        self.interior = ~self.is_ghost
        self.g_idx = T(np.nonzero(tp.kind >= 2)[0], torch.long)
        self.g_src = T(tp.ghost_src[tp.kind >= 2], torch.long)
        self.g_sign = T(tp.ghost_sign[tp.kind >= 2].astype(float))
        gk = tp.kind[tp.kind >= 2]
        self.g_inflow = T(gk == 2, torch.bool)
        self.g_stage = T(gk == 3, torch.bool)
        self.g_closed = T(gk == 4, torch.bool)
        self.a_dry = self.sec.area(torch.full_like(self.z, cfg.h_dry))
        self.inflow, self.stage = inflow, stage
        self.A = torch.zeros_like(self.z)
        self.Q = torch.zeros_like(self.z)
        self._core = compile_if_cuda(self._step_core, self.device, cfg)

    # ---------------------------------------------------------------- state
    def depth(self, A=None):
        return self.sec.depth(self.A if A is None else A)

    def level(self):
        return self.z + self.depth()

    def set_level(self, eta):
        y = torch.clamp(torch.as_tensor(eta, dtype=self.dtype, device=self.device) - self.z, min=0.0)
        self.A = self.sec.area(y)
        self.Q = torch.zeros_like(self.A)

    def volume(self) -> torch.Tensor:
        return (self.A * self.dx)[self.interior].to(torch.float64).sum()

    def velocity(self, A, Q, sec: Sections | None = None):
        y = (sec or self.sec).depth(A)
        return torch.where(y >= self.cfg.h_dry, Q / torch.clamp(A, min=1e-12), torch.zeros_like(Q))

    def max_dt(self) -> float:
        A, Q = self.A, self.Q
        y = self.sec.depth(A)
        wet = (y > self.cfg.h_dry) & self.interior
        if not bool(wet.any()):
            return self.cfg.dt_max
        c = torch.sqrt(self.cfg.g * A / self.sec.top_width(y))
        u = self.velocity(A, Q)
        rate = torch.where(wet, (u.abs() + c) / self.dx, torch.zeros_like(c))
        return min(self.cfg.cfl_1d / float(rate.max()), self.cfg.dt_max)

    # --------------------------------------------------------------- ghosts
    def _fill_ghosts(self, A, Q, q_in, eta_sea):
        g = self.cfg.g
        As, Qs = A[self.g_src], Q[self.g_src]
        sec_g = Sections(self.sec.b[self.g_idx], self.sec.m[self.g_idx], self.sec.yb[self.g_idx], self.sec.ts[self.g_idx])
        ys = sec_g.depth(As)
        zg = self.z[self.g_idx]
        # outward-normal velocity of the interior neighbour
        un = self.g_sign * self.velocity(As, Qs, sec_g)
        wet_i = ys >= self.cfg.h_dry
        # stage: Riemann invariant u_n + 2c carried out of the domain
        yst = torch.clamp(eta_sea - zg, min=0.0)
        Ast = sec_g.area(yst)
        ci = torch.sqrt(g * As / sec_g.top_width(ys))
        cg = torch.sqrt(g * Ast / sec_g.top_width(yst))
        un_st = torch.where(wet_i, un + 2.0 * (ci - cg), torch.zeros_like(un))
        super_out = wet_i & (un > ci)
        un_st = torch.where(super_out, un, un_st)
        Ast = torch.where(super_out, As, Ast)
        un_st = torch.where(sec_g.depth(Ast) >= self.cfg.h_dry, un_st, torch.zeros_like(un_st))
        Q_st = self.g_sign * Ast * un_st
        # inflow: prescribed Q (into the domain = opposite to the outward normal)
        y_c = torch.clamp((q_in * q_in / (g * sec_g.b * sec_g.b)) ** (1.0 / 3.0), min=self.cfg.h_dry)
        A_in = torch.where(wet_i, As, sec_g.area(y_c))
        Q_in = -self.g_sign * q_in
        Ag = torch.where(self.g_stage, Ast, torch.where(self.g_inflow, A_in, As))
        Qg = torch.where(self.g_stage, Q_st, torch.where(self.g_inflow, Q_in, -Qs))
        A = A.index_put((self.g_idx,), Ag)
        Q = Q.index_put((self.g_idx,), Qg)
        return A, Q

    # ----------------------------------------------------------------- step
    def step(self, dt: float, t: float) -> Step1DVolumes:
        q_in = float(self.inflow(t)) if self.inflow is not None else 0.0
        eta = float(self.stage(t)) if self.stage is not None else 0.0
        sc = torch.tensor([dt, q_in, eta], dtype=self.dtype, device=self.device)
        A, Q, b_in, b_out, clip = self._core(self.A, self.Q, sc)
        self.A, self.Q = A, Q
        return Step1DVolumes(b_in, b_out, clip)

    def _step_core(self, A, Q, sc):
        dt, q_in, eta = sc[0], sc[1], sc[2]
        g, f64 = self.cfg.g, torch.float64
        A, Q = self._fill_ghosts(A, Q, q_in, eta)
        y = self.sec.depth(A)
        u = self.velocity(A, Q)
        fl, fr, fs = self.fl, self.fr, self.fsec
        zL, zR = self.z[fl], self.z[fr]
        zf = torch.maximum(zL, zR)
        yL = torch.clamp(zL + y[fl] - zf, min=0.0)
        yR = torch.clamp(zR + y[fr] - zf, min=0.0)
        AL, AR = fs.area(yL), fs.area(yR)
        uL, uR = u[fl], u[fr]
        pL, pR = g * fs.i1(yL), g * fs.i1(yR)
        cL = torch.sqrt(g * AL / fs.top_width(yL))
        cR = torch.sqrt(g * AR / fs.top_width(yR))
        F1, F2 = hll_1d(AL, AL * uL, pL, cL, AR, AR * uR, pR, cR, uL, uR)
        # conservative outflow limiter (donor = upwind cell; ghosts unlimited)
        vol = A * self.dx
        out = torch.zeros_like(A).index_add(0, fl, torch.relu(F1)).index_add(0, fr, torch.relu(-F1))
        need = out * dt
        theta = torch.where(need > vol, vol / torch.where(need > 0, need, torch.ones_like(need)), torch.ones_like(A))
        theta = torch.where(self.is_ghost, torch.ones_like(theta), theta)
        s = torch.where(F1 > 0, theta[fl], theta[fr])
        F1 = F1 * s
        dA = torch.zeros_like(A).index_add(0, fl, -F1).index_add(0, fr, F1)
        dQ = torch.zeros_like(A).index_add(0, fl, -(F2 - pL)).index_add(0, fr, F2 - pR)
        A1 = A + dt * dA / self.dx
        Q1 = Q + dt * dQ / self.dx
        neg = torch.clamp(-A1, min=0.0)
        clip = (neg * self.dx)[self.interior].to(f64).sum()
        A1 = A1 + neg
        # boundary exchange: faces touching a ghost cell
        gl, gr = self.is_ghost[fl], self.is_ghost[fr]
        zero = torch.zeros_like(F1)
        b_in = torch.where(gl, torch.relu(F1), zero).sum() + torch.where(gr, torch.relu(-F1), zero).sum()
        b_out = torch.where(gl, torch.relu(-F1), zero).sum() + torch.where(gr, torch.relu(F1), zero).sum()
        b_in, b_out = b_in.to(f64) * dt.to(f64), b_out.to(f64) * dt.to(f64)
        # semi-implicit friction
        y1 = self.sec.depth(A1)
        Ac = torch.clamp(A1, min=1e-12)
        R = Ac / self.sec.perimeter(y1)
        denom = 1.0 + dt * g * self.nman ** 2 * Q1.abs() / (Ac * torch.clamp(R, min=self.cfg.h_dry) ** (4.0 / 3.0))
        Q1 = Q1 / denom
        Q1 = torch.where((y1 >= self.cfg.h_dry) & ~self.is_junc, Q1, torch.zeros_like(Q1))
        # ghosts keep their previous values (they are refilled next step)
        A1 = torch.where(self.is_ghost, A, A1)
        Q1 = torch.where(self.is_ghost, Q, Q1)
        return A1, Q1, b_in, b_out, clip
