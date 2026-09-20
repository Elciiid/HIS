"""Adaptive loss weighting by gradient-norm balancing.

Every ``every`` steps, for each non-data term i:
    lambda_i <- alpha * lambda_i + (1 - alpha) * ||grad L_data|| / (||grad L_i|| + eps)
The first update, when the physics terms switch on, sets lambda_i to the target
directly (starting the EMA from 1 would let an untrained residual swamp the data
term for hundreds of steps).
All lambdas are logged so a reviewer can see them evolve.
"""
from __future__ import annotations

import math

import torch


class GradNormBalancer:
    def __init__(self, names, alpha: float = 0.9, every: int = 50, eps: float = 1e-12,
                 lo: float = 1e-4, hi: float = 1e4, floors: dict | None = None):
        """``floors`` gives a term its own lower bound (x lambda_data = 1): a term the
        balancing must not be allowed to suppress, as the effect loss must not be."""
        self.names = list(names)
        self.lam = {n: 1.0 for n in self.names}
        self.alpha, self.every, self.eps, self.lo, self.hi = alpha, every, eps, lo, hi
        self.floors = dict(floors or {})
        self.initialised = False

    def due(self, step: int) -> bool:
        return step % self.every == 0 or not self.initialised

    @staticmethod
    def _norm(loss, params):
        """Gradient norm of one loss term. Returns 0.0 when it is not finite: under mixed
        precision a term's gradients can overflow, and a non-finite norm carries no
        information about how to weight it -- the weight must then stay as it was, not become
        inf and take the whole loss with it."""
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq = sum((g.detach().float() ** 2).sum() for g in grads if g is not None)
        if not torch.is_tensor(sq):
            return 0.0
        n = float(torch.sqrt(sq))
        return n if math.isfinite(n) else 0.0

    def update(self, data_loss, terms: dict, params) -> None:
        params = [p for p in params if p.requires_grad]
        gd = self._norm(data_loss, params)
        if gd <= 0.0:                      # no usable reference this step: keep every weight
            self.initialised = True
            return
        for n in self.names:
            if n not in terms or not terms[n].requires_grad:
                continue
            gi = self._norm(terms[n], params)
            if gi <= 0.0:
                continue
            target = gd / (gi + self.eps)
            # the first update (when the physics terms switch on) starts from the target itself
            new = target if not self.initialised else self.alpha * self.lam[n] + (1 - self.alpha) * target
            self.lam[n] = float(min(max(new, self.lo, self.floors.get(n, 0.0)), self.hi))
        self.initialised = True

    def state_dict(self):
        return {"lam": dict(self.lam), "initialised": self.initialised}

    def load_state_dict(self, d):
        self.lam.update(d["lam"])
        self.initialised = d.get("initialised", True)
