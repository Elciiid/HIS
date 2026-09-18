"""Boundary conditions for the 2-D solver, implemented as ghost cells.

Each edge (west/east/south/north) carries an ``EdgeBC``:
  closed        reflective wall: h, z copied, normal velocity reversed
  transmissive  zero-gradient: everything copied
  stage         imposed free-surface level eta(t); ghost depth max(0, eta - z);
                normal velocity from the outgoing Riemann invariant u_n + 2c
                (subcritical in- and outflow), zero-gradient if supercritical out
  discharge     imposed inflow per unit width q(t) [m^2/s] (positive = into domain)

``value`` may be a float or a callable of time returning a float. For a batched
run it may also return one value per batch member (array or 1-D tensor); the
solver reshapes it to broadcast over the grid.

Every routine here addresses the grid through the last two dimensions, so the
same code serves a single (ny, nx) state and a batched (B, ny, nx) one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

EDGES = ("west", "east", "south", "north")


@dataclass
class EdgeBC:
    kind: str = "transmissive"
    value: float | Callable[[float], float] | None = None

    def __post_init__(self):
        if self.kind not in ("closed", "transmissive", "stage", "discharge"):
            raise ValueError(f"unknown boundary kind '{self.kind}'")
        if self.kind in ("stage", "discharge") and self.value is None:
            raise ValueError(f"boundary kind '{self.kind}' needs a value")

    def at(self, t: float):
        """Boundary value at time t: a float, or one value per batch member."""
        v = self.value(t) if callable(self.value) else self.value
        return v if (torch.is_tensor(v) or hasattr(v, "shape")) else float(v)


def edge_values(bcs: dict[str, EdgeBC], t: float) -> list:
    return [bcs[e].at(t) if bcs[e].kind in ("stage", "discharge") else 0.0 for e in EDGES]


def all_edges(kind: str = "closed", **overrides) -> dict[str, EdgeBC]:
    bcs = {e: EdgeBC(kind) for e in EDGES}
    for k, v in overrides.items():
        bcs[k] = v if isinstance(v, EdgeBC) else EdgeBC(v)
    return bcs


def edge_ghost(bc: EdgeBC, h, un, ut, z, val, g: float, h_dry: float):
    """Ghost state for one edge. ``un`` is the velocity along the OUTWARD normal;
    ``val`` is the boundary value (0-d tensor) at the evaluation time."""
    if bc.kind == "closed":
        return h, -un, ut, z
    if bc.kind == "transmissive":
        return h, un, ut, z
    if bc.kind == "stage":
        hg = torch.clamp(val - z, min=0.0)
        ci, cg = torch.sqrt(g * h), torch.sqrt(g * hg)
        wet_i = h >= h_dry
        un_g = torch.where(wet_i, un + 2.0 * (ci - cg), torch.zeros_like(un))
        supercrit_out = wet_i & (un > ci)
        un_g = torch.where(supercrit_out, un, un_g)
        hg = torch.where(supercrit_out, h, hg)
        un_g = torch.where(hg >= h_dry, un_g, torch.zeros_like(un_g))
        ut_g = torch.where(un_g > 0, ut, torch.zeros_like(ut))
        return hg, un_g, ut_g, z
    # discharge
    hc = torch.clamp((val * val / g) ** (1.0 / 3.0), min=h_dry)
    hg = torch.where(h >= h_dry, h, hc.expand_as(h))
    un_g = -val / hg
    return hg, un_g, torch.zeros_like(ut), z


def pad_axis(bcs: dict[str, EdgeBC], h, u, v, z, vals, ng: int, axis: str, g: float, h_dry: float):
    """Pad (h, u, v, z) with ``ng`` ghost layers along x (axis='x') or y (axis='y').
    ``vals`` maps edge name -> 0-d tensor boundary value.

    Closed and transmissive edges use mirrored ghost layers (ghost k mirrors interior
    layer k), so a second-order reconstruction sees mirror-image states at a wall and
    the wall mass flux is exactly zero. Stage and discharge ghosts repeat one state."""
    if axis == "x":
        lo, hi = "west", "east"
        dim = -1
        un_c, ut_c = u, v
    else:
        lo, hi = "south", "north"
        dim = -2
        un_c, ut_c = v, u
    n = h.shape[dim]

    def widen(a):
        """Repeat a one-cell-thick ghost slab out to ``ng`` layers, at whatever rank."""
        shp = list(a.shape)
        shp[dim] = ng
        return a.expand(*shp)

    def mirror(a, side):
        # plain slicing + flip (index tensors trip an Inductor scheduler bug)
        if n < ng:
            return widen(a.narrow(dim, 0 if side == lo else n - 1, 1))
        if side == lo:
            return a.narrow(dim, 0, ng).flip(dim)          # outermost ghost = layer ng-1
        return a.narrow(dim, n - ng, ng).flip(dim)         # adjacent ghost = layer n-1

    def first(a, side):
        return a.narrow(dim, 0 if side == lo else n - 1, 1)

    parts = {}
    for side, sgn in ((lo, -1.0), (hi, 1.0)):
        bc = bcs[side]
        take = mirror if bc.kind in ("closed", "transmissive") else first
        hh, zz = take(h, side), take(z, side)
        un, ut = sgn * take(un_c, side), take(ut_c, side)
        hg, ung, utg, zg = edge_ghost(bc, hh, un, ut, zz, vals[side], g, h_dry)
        if take is first:
            hg, ung, utg, zg = (widen(x) for x in (hg, ung, utg, zg))
        un_back = sgn * ung
        comps = (hg, un_back, utg, zg) if axis == "x" else (hg, utg, un_back, zg)
        parts[side] = comps
    out = [torch.cat(_align((parts[lo][k], a, parts[hi][k]), dim), dim=dim) for k, a in enumerate((h, u, v, z))]
    return out


def _align(tensors, dim: int):
    """Broadcast ``tensors`` against each other on every dimension but ``dim``, so they
    can be concatenated. A per-member boundary value may batch a field the interior
    does not (the bed is shared by every member; the tidal stage is not)."""
    flat = torch.broadcast_shapes(*(t.shape[:dim] + t.shape[dim:][1:] for t in tensors))
    return [t.expand(*flat[:len(flat) + dim + 1], t.shape[dim], *flat[len(flat) + dim + 1:]) for t in tensors]
