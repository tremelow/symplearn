import numpy as np

import torch
from torch import nn

from symplearn.dynamics import AbstractDiffEq, AbstractDegenLagrangian
from symplearn.networks import UniformNormalizer1d, create_mlp

DIMENSION = 2


class LotkaVolterra(AbstractDegenLagrangian):
    def __init__(self, *, a=1.0, b=1.0, c=2.0, d=1.0):
        super().__init__(DIMENSION)
        self.a, self.b, self.c, self.d = a, b, c, d

    def hamiltonian(self, x, y, t):
        hx = self.d * x - self.c * torch.log(x)
        hy = self.b * y - self.a * torch.log(y)
        return torch.sum(hx + hy, -1)

    def oneform(self, x, y):
        return -torch.log(y) / x

    def euler_lagrange_maps(self, x, y, t):
        dx_q = torch.log(y) / x**2
        dy_q = -1.0 / (x * y)
        dx_h = self.d - self.c / x
        dy_h = self.b - self.a / y
        return (dx_q.unsqueeze(-1), dy_q.unsqueeze(-1)), (dx_h, dy_h)

    def vector_field(self, z, t):
        x, y = torch.tensor_split(z, 2, dim=-1)
        dt_x = x * (self.a - self.b * y)
        dt_y = y * (self.d * x - self.c)
        return torch.cat((dt_x, dt_y), dim=-1)


class NeuralSympLV(AbstractDegenLagrangian, nn.Module):
    def __init__(self):
        AbstractDegenLagrangian.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.oneform_nn = create_mlp(DIMENSION, DIMENSION // 2, [35] * 3)
        self.hamiltonian_nn = create_mlp(DIMENSION, 1, [35] * 3)

    def oneform(self, x, y):
        z = self.norm(torch.cat((x, y), -1))
        return self.oneform_nn(z)

    def hamiltonian(self, x, y, t):
        z = self.norm(torch.cat((x, y), -1))
        return self.hamiltonian_nn(z).sum(-1)


class NeuralBaseLV(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.vector_field_nn = create_mlp(DIMENSION, DIMENSION, [60] * 3)

    def vector_field(self, z, t):
        z = self.norm(z)
        return self.vector_field_nn(z)

    def forward(self, z, t):
        return self.vector_field(z, t)


class NeuralStepLV(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.vector_field_nn = create_mlp(DIMENSION, DIMENSION, [20] * 2)

    def forward(self, z, t, dt):
        dt_z = self.vector_field(z, t)
        return z + dt * dt_z

    def vector_field(self, z, t):
        z = self.norm(z)
        return self.vector_field_nn(z)


def test():
    model = LotkaVolterra()
    rng = torch.Generator().manual_seed(42)
    z = torch.randn(model.dim, generator=rng)
    x, y = torch.tensor_split(z, 2, dim=-1)

    implem_maps = model.euler_lagrange_maps(x, y, None)
    parent_maps = AbstractDegenLagrangian.euler_lagrange_maps(model, x, y, None)
    for i in range(2):
        for j in range(2):
            assert implem_maps[i][j].shape == parent_maps[i][j].shape
            assert torch.allclose(implem_maps[i][j], parent_maps[i][j])

    implem_vf = model.vector_field(z, None)
    parent_vf = AbstractDegenLagrangian.vector_field(model, z, None)
    assert implem_vf.shape == parent_vf.shape
    assert torch.allclose(implem_vf, parent_vf)
