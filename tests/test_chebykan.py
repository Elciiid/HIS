import numpy as np
import torch
from numpy.polynomial import chebyshev as C

from hydrointel.model.chebykan import ChebyKAN, ChebyMLP, chebyshev_basis


def test_recurrence_matches_numpy():
    z = torch.linspace(-1, 1, 101, dtype=torch.float64)
    B = chebyshev_basis(z, 7).numpy()
    for k in range(8):
        ref = C.chebval(z.numpy(), np.eye(8)[k])
        np.testing.assert_allclose(B[:, k], ref, atol=1e-13)


def test_fits_smooth_function():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 256).unsqueeze(1)
    y = torch.sin(3 * x) * torch.exp(-x ** 2)
    model = ChebyMLP([1, 16, 1], degree=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for _ in range(3000):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    assert loss.item() < 1e-3, loss.item()


def test_gradient_flows_to_inputs_and_coefficients():
    layer = ChebyKAN(4, 3, degree=5)
    x = torch.randn(10, 4, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert layer.coef.grad is not None and layer.coef.grad.abs().sum() > 0


def test_init_scale():
    layer = ChebyKAN(20, 8, degree=5)
    std = layer.coef.detach().std().item()
    assert abs(std - (1 / (20 * 6)) ** 0.5) < 0.03
