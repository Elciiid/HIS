"""Chebyshev Kolmogorov-Arnold layer.

    phi(x)_j = sum_i sum_{k=0..d} C[i, j, k] * T_k(tanh(x_i))

tanh maps each input into [-1, 1], where Chebyshev polynomials are well
conditioned. T_k follows the recurrence T_0 = 1, T_1 = z, T_{k+1} = 2 z T_k - T_{k-1}
(a loop over the degree only, never over the batch).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def chebyshev_basis(z: torch.Tensor, degree: int) -> torch.Tensor:
    """T_0..T_d evaluated at z; returns z.shape + (degree + 1,)."""
    T = [torch.ones_like(z)]
    if degree >= 1:
        T.append(z)
    for _ in range(2, degree + 1):
        T.append(2.0 * z * T[-1] - T[-2])
    return torch.stack(T, dim=-1)


class ChebyKAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, degree: int = 5):
        super().__init__()
        self.in_dim, self.out_dim, self.degree = in_dim, out_dim, degree
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1) / math.sqrt(in_dim * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_basis(torch.tanh(x), self.degree)            # (..., in, d+1)
        return torch.einsum("...ik,ijk->...j", basis, self.coef)

    def extra_repr(self) -> str:
        return f"in={self.in_dim}, out={self.out_dim}, degree={self.degree}"


class ChebyMLP(nn.Module):
    """Stacked ChebyKAN layers with LayerNorm in between."""

    def __init__(self, dims: list[int], degree: int = 5, final_norm: bool = False):
        super().__init__()
        layers = []
        for k, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(ChebyKAN(a, b, degree))
            if k < len(dims) - 2 or final_norm:
                layers.append(nn.LayerNorm(b))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
