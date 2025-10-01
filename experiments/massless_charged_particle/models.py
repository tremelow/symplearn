import torch
from torch import nn

from symplearn.dynamics import AbstractDiffEq, AbstractDegenLagrangian
from symplearn.networks import UniformNormalizer1d, create_mlp

DIMENSION = 2


class MasslessChargedParticle(AbstractDegenLagrangian):
    def __init__(self, *, a0=1.0, e0=1.0):
        super().__init__(DIMENSION)
        self.a0 = a0
        self.e0 = e0

    def hamiltonian(self, x, y, t):
        return self.e0 * (2.0 - torch.cos(x) - torch.sin(y)).sum(-1)

    def oneform(self, x, y):
        return -self.a0 * y * (1.0 + 2.0 * x**2 + (2.0 / 3.0) * y**2)

    def euler_lagrange_maps(self, x, y, t):
        dx_q = -4.0 * self.a0 * y * x
        dy_q = -self.a0 * (1.0 + 2.0 * x**2 + 2.0 * y**2)
        dx_h = self.e0 * torch.sin(x)
        dy_h = -self.e0 * torch.cos(y)
        return (dx_q.unsqueeze(-1), dy_q.unsqueeze(-1)), (dx_h, dy_h)

    def vector_field(self, z, t):
        x, y = torch.tensor_split(z, 2, dim=-1)
        b = self.a0 * (1.0 + 2.0 * x**2 + 2.0 * y**2)
        dt_x = self.e0 * torch.cos(y) / b
        dt_y = self.e0 * torch.sin(x) / b
        return torch.cat((dt_x, dt_y), dim=-1)


class NeuralSympMCP(AbstractDegenLagrangian, nn.Module):
    def __init__(self):
        AbstractDegenLagrangian.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.oneform_nn = create_mlp(DIMENSION, DIMENSION // 2, [50] * 2)
        self.hamiltonian_nn = create_mlp(DIMENSION, 1, [50] * 2)

    def oneform(self, x, y):
        z = self.norm(torch.cat((x, y), -1))
        return self.oneform_nn(z)

    def hamiltonian(self, x, y, t):
        z = self.norm(torch.cat((x, y), -1))
        return self.hamiltonian_nn(z).sum(-1)


class NeuralBaseMCP(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.vector_field_nn = create_mlp(DIMENSION, DIMENSION, [50] * 3)

    def vector_field(self, z, t):
        z = self.norm(z)
        return self.vector_field_nn(z)

    def forward(self, z, t):
        return self.vector_field(z, t)


class NeuralStepMCP(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm = UniformNormalizer1d(DIMENSION)
        self.vector_field_nn = create_mlp(DIMENSION, DIMENSION, [30] * 2)

    def forward(self, z, t, dt):
        dt_z = self.vector_field(z, t)
        return z + dt * dt_z

    def vector_field(self, z, t):
        z = self.norm(z)
        return self.vector_field_nn(z)


def test():
    model = MasslessChargedParticle()
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
