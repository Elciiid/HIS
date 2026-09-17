"""Provenance records and stamping for every figure, CSV and report."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .. import SYNTHETIC_WATERMARK

_REPO = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "no-commit"
        dirty = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "git-unavailable"


@dataclass(frozen=True)
class DataProvenance:
    source: str                     # "SYNTHETIC" or a real dataset id
    config_hash: str
    seed: int
    git_commit: str = field(default_factory=git_commit)
    timestamp: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    device: str = ""
    precision: str = ""
    engine: str = ""                # "engine" | "surrogate"
    notes: tuple[str, ...] = ()

    @property
    def synthetic(self) -> bool:
        return self.source == "SYNTHETIC"

    def line(self) -> str:
        parts = [f"data={self.source}", f"cfg={self.config_hash}", f"git={self.git_commit}",
                 f"seed={self.seed}", self.timestamp]
        if self.engine:
            parts.insert(1, f"model={self.engine}")
        if self.precision:
            parts.append(self.precision)
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "DataProvenance":
        return dataclasses.replace(self, **kw)


def stamp(fig, prov: DataProvenance, caption: str | None = None) -> None:
    """Write the provenance line on the canvas; watermark synthetic runs."""
    eng = fig.get_layout_engine()
    if eng is not None and hasattr(eng, "set"):
        try:
            eng.set(rect=(0.0, 0.02, 1.0, 0.965 if prov.synthetic else 0.99))   # keep margins for the stamps
        except (TypeError, ValueError):
            pass
    fig.text(0.005, 0.003, prov.line(), fontsize=6.5, color="#333333", ha="left", va="bottom",
             family="monospace", alpha=0.9)
    if caption:
        fig.text(0.995, 0.003, caption, fontsize=6.5, color="#333333", ha="right", va="bottom")
    if prov.synthetic:
        fig.text(0.5, 0.5, SYNTHETIC_WATERMARK, fontsize=22, color="#b00020", alpha=0.18,
                 ha="center", va="center", rotation=22, weight="bold", zorder=1000)
        fig.text(0.5, 0.995, SYNTHETIC_WATERMARK, fontsize=9, color="#b00020", ha="center", va="top",
                 weight="bold")


def savefig(fig, path: str | Path, prov: DataProvenance, caption: str | None = None, dpi: int = 130) -> Path:
    import matplotlib.pyplot as plt
    stamp(fig, prov, caption)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def header_lines(prov: DataProvenance, comment: str = "#") -> str:
    lines = [f"{comment} provenance: {prov.line()}"]
    if prov.synthetic:
        lines.append(f"{comment} {SYNTHETIC_WATERMARK}")
    for n in prov.notes:
        lines.append(f"{comment} note: {n}")
    return "\n".join(lines) + "\n"


def write_csv(path: str | Path, header: list[str], rows, prov: DataProvenance) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(header_lines(prov))
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(_fmt(v) for v in r) + "\n")
    return path


def markdown_header(prov: DataProvenance, title: str) -> str:
    s = f"# {title}\n\n"
    if prov.synthetic:
        s += f"> **{SYNTHETIC_WATERMARK}**\n\n"
    s += "| provenance | |\n|---|---|\n"
    for k, v in prov.to_dict().items():
        if k == "notes":
            v = "; ".join(v) if v else "-"
        s += f"| {k} | `{v}` |\n"
    return s + "\n"


def write_json(path: str | Path, obj: dict, prov: DataProvenance) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"provenance": prov.to_dict(), **obj}
    path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    return path


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
