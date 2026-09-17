"""1-D channel network: topology and trapezoidal cross-sections.

A network is a set of reaches (ordered upstream -> downstream) joined at
junction nodes. A reach end is one of:
  ("inflow",)       upstream hydrograph boundary
  ("stage",)        tidal/stage boundary (sea)
  ("closed",)       wall
  ("junction", j)   shared junction node j (equal level, mass conserved)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Reach:
    reach_id: str
    x: np.ndarray
    y: np.ndarray
    bed: np.ndarray           # m
    width: np.ndarray         # bottom width, m
    side_slope: np.ndarray    # H:V
    manning: np.ndarray
    z_bank: np.ndarray        # bank crest, m
    up: tuple = ("inflow",)
    down: tuple = ("stage",)

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def spacing(self) -> np.ndarray:
        """Cell length per node (half distance to each neighbour; end nodes use one side)."""
        d = np.hypot(np.diff(self.x), np.diff(self.y))
        if self.n == 1:
            return np.array([1.0])
        ds = np.empty(self.n)
        ds[0], ds[-1] = d[0], d[-1]
        ds[1:-1] = 0.5 * (d[:-1] + d[1:])
        return ds

    @property
    def chainage(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])


@dataclass
class Junction:
    x: float
    y: float
    bed: float
    width: float
    side_slope: float
    manning: float
    z_bank: float
    length: float             # effective storage length


@dataclass
class ChannelNetwork:
    reaches: list[Reach] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    inflow_reach: str | None = None      # reach receiving the upstream catchment hydrograph

    @property
    def n_reaches(self) -> int:
        return len(self.reaches)

    def reach_index(self, rid: str) -> int:
        return [r.reach_id for r in self.reaches].index(rid)

    def validate(self) -> None:
        for r in self.reaches:
            for name in ("bed", "width", "side_slope", "manning", "z_bank"):
                a = getattr(r, name)
                if a.shape != r.x.shape or not np.all(np.isfinite(a)):
                    raise ValueError(f"reach {r.reach_id}: field '{name}' has bad shape or non-finite values")
            if np.any(r.width <= 0) or np.any(r.side_slope < 0) or np.any(r.manning <= 0):
                raise ValueError(f"reach {r.reach_id}: width/manning must be > 0 and side_slope >= 0")
            if np.any(r.z_bank - r.bed < 0.2):
                raise ValueError(f"reach {r.reach_id}: bank crest must be at least 0.2 m above bed")
            for end in (r.up, r.down):
                if end[0] == "junction" and not 0 <= end[1] < len(self.junctions):
                    raise ValueError(f"reach {r.reach_id}: junction index {end[1]} out of range")
                if end[0] not in ("inflow", "stage", "closed", "junction"):
                    raise ValueError(f"reach {r.reach_id}: unknown boundary kind {end[0]}")

    # ---- CSV io (manifest schema) -----------------------------------------
    SCHEMA = ["reach_id", "node_id", "x", "y", "bed_elev", "width", "side_slope", "manning"]

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.SCHEMA)
            for r in self.reaches:
                pts = list(range(r.n))
                if r.up[0] == "junction":
                    j = self.junctions[r.up[1]]
                    w.writerow([r.reach_id, f"J{r.up[1]}", j.x, j.y, j.bed, j.width, j.side_slope, j.manning])
                for i in pts:
                    w.writerow([r.reach_id, f"{r.reach_id}_{i}", r.x[i], r.y[i], r.bed[i], r.width[i],
                                r.side_slope[i], r.manning[i]])
                if r.down[0] == "junction":
                    j = self.junctions[r.down[1]]
                    w.writerow([r.reach_id, f"J{r.down[1]}", j.x, j.y, j.bed, j.width, j.side_slope, j.manning])


def trapezoid_top_width(width, side_slope, depth):
    return width + 2.0 * side_slope * depth


def build_from_rows(rows: list[dict], z_bank_fn, domain_bounds, sea_fn, file_label: str) -> ChannelNetwork:
    """Build a network from manifest CSV rows. Node ids starting with 'J' shared
    between reaches are junctions. A reach end without a junction is a stage
    boundary if ``sea_fn(x, y)`` is true there, otherwise an inflow boundary
    (upstream end) or a closed end (downstream end)."""
    by_reach: dict[str, list[dict]] = {}
    for r in rows:
        by_reach.setdefault(r["reach_id"], []).append(r)
    jnodes: dict[str, int] = {}
    net = ChannelNetwork()
    for rid, rr in by_reach.items():
        ids = [r["node_id"] for r in rr]
        up, down = ("inflow",), ("closed",)
        if ids[0].startswith("J"):
            up = ("junction", _junction(net, jnodes, rr[0], z_bank_fn))
            rr = rr[1:]
        if rr and rr[-1]["node_id"].startswith("J"):
            down = ("junction", _junction(net, jnodes, rr[-1], z_bank_fn))
            rr = rr[:-1]
        if len(rr) < 2:
            raise ValueError(f"{file_label}: reach '{rid}' needs at least 2 non-junction nodes")
        x = np.array([float(r["x"]) for r in rr]); y = np.array([float(r["y"]) for r in rr])
        if up[0] == "inflow" and sea_fn(x[0], y[0]):
            up = ("stage",)
        if down[0] == "closed" and sea_fn(x[-1], y[-1]):
            down = ("stage",)
        bed = np.array([float(r["bed_elev"]) for r in rr])
        net.reaches.append(Reach(
            rid, x, y, bed,
            np.array([float(r["width"]) for r in rr]),
            np.array([float(r["side_slope"]) for r in rr]),
            np.array([float(r["manning"]) for r in rr]),
            np.array([z_bank_fn(xi, yi) for xi, yi in zip(x, y)]), up, down))
        if up == ("inflow",) and net.inflow_reach is None:
            net.inflow_reach = rid
    for r in net.reaches:
        r.z_bank = np.maximum(r.z_bank, r.bed + 0.2)
    net.validate()
    return net


def _junction(net: ChannelNetwork, jnodes: dict, row: dict, z_bank_fn) -> int:
    key = row["node_id"]
    if key not in jnodes:
        x, y = float(row["x"]), float(row["y"])
        bed = float(row["bed_elev"])
        net.junctions.append(Junction(x, y, bed, float(row["width"]), float(row["side_slope"]),
                                      float(row["manning"]), max(z_bank_fn(x, y), bed + 0.2),
                                      length=2.0 * float(row["width"])))
        jnodes[key] = len(net.junctions) - 1
    return jnodes[key]


def apply_gamma(net: ChannelNetwork, gamma) -> ChannelNetwork:
    """Scale conveyance per reach: bottom width x sqrt(gamma) and bankfull depth
    x sqrt(gamma) (bed lowered/raised about the bank crest), so the bankfull area of
    a rectangular section scales by gamma. Dredging therefore changes the 1-D bed,
    never the 2-D DEM. Junctions take the gamma of the reach that ends at them."""
    import copy
    g = np.asarray(gamma, dtype=float).reshape(-1)
    if g.size != net.n_reaches:
        raise ValueError(f"channel_gamma has {g.size} entries, network has {net.n_reaches} reaches")
    if np.any(g <= 0):
        raise ValueError("channel_gamma must be positive")
    out = copy.deepcopy(net)
    for r, gi in zip(out.reaches, g):
        s = np.sqrt(gi)
        r.width = r.width * s
        r.bed = r.z_bank - (r.z_bank - r.bed) * s
        if r.down[0] == "junction":
            j = out.junctions[r.down[1]]
            j.width *= s
            j.bed = j.z_bank - (j.z_bank - j.bed) * s
    out.validate()
    return out
