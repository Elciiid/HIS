"""Training and validation figures (loss terms and adaptive weights on log axes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def training_figures(cfg, curves: list[dict], overfit: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .provenance import DataProvenance, savefig
    from .. import api
    ctx = api.context()
    prov = ctx.provenance("surrogate", ("domain", "solver", "forcing", "data", "model", "train"))
    tr = [c for c in curves if "total" in c]
    va = [c for c in curves if "val_rmse_h_all" in c]
    steps = np.array([c["step"] for c in tr])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax[0, 0].semilogy(steps, [c["total"] for c in tr], label="total (weighted)")
    ax[0, 0].semilogy(steps, [c["loss_data"] for c in tr], label="data")
    ramp_on = [c["step"] for c in tr if c["ramp"] > 0]
    if ramp_on:
        ax[0, 0].axvline(ramp_on[0], color="k", ls=":", label="physics ramp starts")
    ax[0, 0].set_title("loss"); ax[0, 0].legend()
    keys = sorted({k[5:] for c in tr for k in c if k.startswith("loss_") and k != "loss_data"})
    for k in keys:
        x = [c["step"] for c in tr if f"loss_{k}" in c]
        y = [c[f"loss_{k}"] for c in tr if f"loss_{k}" in c]
        ax[0, 1].semilogy(x, np.maximum(y, 1e-12), label=k)
    ax[0, 1].set_title("individual terms (unweighted)"); ax[0, 1].legend(fontsize=8)
    lk = sorted({k[7:] for c in tr for k in c if k.startswith("lambda_")})
    for k in lk:
        ax[1, 0].semilogy(steps, [c.get(f"lambda_{k}", np.nan) for c in tr], label=k)
    ax[1, 0].set_title("adaptive weights λ (gradient-norm balancing)"); ax[1, 0].legend(fontsize=8)
    if va:
        ax[1, 1].plot([c["step"] for c in va], [c["val_rmse_h_all"] for c in va], "o-", label="val depth RMSE, all cells")
        ax[1, 1].plot([c["step"] for c in va], [c["val_rmse_h_wet"] for c in va], "s-", label="val depth RMSE, wet cells")
    ax[1, 1].set_title("validation (unweighted, m)"); ax[1, 1].legend()
    for a in ax.ravel():
        a.set_xlabel("step"); a.grid(alpha=0.3, which="both")
    savefig(fig, Path(cfg.outdir) / "figures" / "training_curves.png", prov)
    if overfit:
        fig, a = plt.subplots(figsize=(7, 4), constrained_layout=True)
        a.semilogy(overfit["history"])
        a.set_title(f"overfit sanity check on {overfit['n_samples']} simulations "
                    f"(x{overfit['reduction']:.3g}, {'passed' if overfit['passed'] else 'FAILED'})")
        a.set_xlabel("step"); a.set_ylabel("data loss")
        savefig(fig, Path(cfg.outdir) / "figures" / "overfit_check.png", prov)
