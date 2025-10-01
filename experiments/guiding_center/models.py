import numpy as np

import torch
from torch import nn

from symplearn.dynamics import AbstractDiffEq, AbstractDegenLagrangian
from symplearn.networks import UniformNormalizer1d, PeriodicLayer, create_mlp

DIMENSION = 4


class GuidingCenter(AbstractDegenLagrangian):
    def __init__(self, *, b0=1.0, r0=1.0, q0=2.0, mu=2.25e-6, quad_order=20):
        super().__init__(DIMENSION)
        self.mu = mu
        self.b0 = b0
        self.r0 = r0
        self.q0 = q0

        nodes, weights = np.polynomial.legendre.leggauss(quad_order)
        # integrate on [0,1] instead of [-1,1]
        quad_nodes = 0.5 * (nodes + 1.0)
        quad_weights = 0.5 * weights
        self.quad_nodes = torch.tensor(quad_nodes)

        # integrate t * f(t) dt and t**2 * f(t) dt
        self.a_th_weights = torch.tensor(quad_weights * quad_nodes)
        self.a_thth_weights = self.a_th_weights * self.quad_nodes

    def to_cartesian(self, z):
        th, phi, r, u = z.tensor_split(4, axis=-1)
        R = self.r0 + r * torch.cos(th)
        x1, x2 = R * torch.cos(phi), R * torch.sin(phi)
        x3 = r * torch.sin(th)
        return torch.cat((x1, x2, x3, u), dim=-1)

    def a_th_integral(self, rho):
        # rho**2 * (rho - log(1 + rho)) / rho**2
        rho_t = self.quad_nodes * rho.unsqueeze(-1)
        return torch.sum(self.a_th_weights / (1.0 + rho_t), -1)

    def a_thth_integral(self, rho):
        # -d `a_th_integral` / d rho
        rho_t = self.quad_nodes * rho.unsqueeze(-1)
        return torch.sum(self.a_thth_weights / (1.0 + rho_t).square(), -1)

    def hamiltonian(self, x, y, t):
        (th, _), (r, u) = x.tensor_split(2, axis=-1), y.tensor_split(2, axis=-1)
        s = torch.sqrt(1.0 + r.square() / (self.q0 * self.r0) ** 2)
        rho = r * torch.cos(th) / self.r0

        mag_field = self.b0 * s / (1.0 + rho)

        return (0.5 * u.square() + self.mu * mag_field).sum(-1)

    def oneform(self, x, y):
        (th, _), (r, u) = x.tensor_split(2, axis=-1), y.tensor_split(2, axis=-1)
        rho = r * torch.cos(th) / self.r0

        b0_r2 = self.b0 * r.square()

        # self.b0 * r**2 * (rho - log(1 + rho)) / rho**2
        a_th = b0_r2 * self.a_th_integral(rho)
        a_phi = -0.5 * b0_r2 / self.q0 + u * self.r0 * (1.0 + rho)
        return torch.cat((a_th, a_phi), axis=-1)

    def euler_lagrange_maps(self, x, y, t):
        (th, _), (r, u) = x.tensor_split(2, axis=-1), y.tensor_split(2, axis=-1)
        rho = r * torch.cos(th) / self.r0
        Z = r * torch.sin(th) / self.r0
        rr = torch.square(r / (self.q0 * self.r0))

        inv_rho_p1 = 1.0 / (rho + 1)
        sqrt_rr = torch.sqrt(1.0 + rr)

        q_th_th = self.b0 * r.square() * Z * self.a_thth_integral(rho)
        q_th_r = self.b0 * r * inv_rho_p1
        q_th_u = torch.zeros_like(q_th_r)

        q_phi_th = -u * self.r0 * Z
        q_phi_r = -self.b0 * r / self.q0 + u * torch.cos(th)
        q_phi_u = self.r0 * (1.0 + rho)

        tmp_h = self.mu * self.b0 * sqrt_rr * inv_rho_p1
        h_th = tmp_h * Z * inv_rho_p1
        h_phi = torch.zeros_like(h_th)
        h_r = tmp_h * (rr / (1.0 + rr) - rho * inv_rho_p1) / r
        h_u = u

        dth_q = torch.stack((q_th_th, q_phi_th), -2)
        dphi_q = torch.zeros_like(dth_q)
        dr_q = torch.stack((q_th_r, q_phi_r), -2)
        du_q = torch.stack((q_th_u, q_phi_u), -2)
        dx_q, dy_q = torch.cat((dth_q, dphi_q), -1), torch.cat((dr_q, du_q), -1)

        dx_h, dy_h = torch.cat((h_th, h_phi), -1), torch.cat((h_r, h_u), -1)

        return (dx_q, dy_q), (dx_h, dy_h)

    def vector_field(self, z, t):
        th, phi, r, u = z.tensor_split(4, axis=-1)
        rho = r * torch.cos(th) / self.r0
        Z = r * torch.sin(th) / self.r0
        rr = torch.square(r / (self.q0 * self.r0))

        inv_rho_p1 = 1.0 / (1.0 + rho)
        sqrt_rr = torch.sqrt(1.0 + rr)

        a_th_r = self.b0 * r * inv_rho_p1

        a_phi_th = -u * self.r0 * Z
        a_phi_r = -self.b0 * r / self.q0 + u * torch.cos(th)
        a_phi_u = self.r0 * (1.0 + rho)

        tmp_h = self.mu * self.b0 * sqrt_rr * inv_rho_p1
        h_th = tmp_h * Z * inv_rho_p1
        h_phi = torch.zeros_like(h_th)
        h_r = tmp_h * (rr / (1.0 + rr) - rho * inv_rho_p1) / r
        h_u = u

        dt_phi = h_u / a_phi_u
        dt_th = (h_r - a_phi_r * dt_phi) / a_th_r
        dt_r = (a_phi_th * dt_phi - h_th) / a_th_r
        dt_u = -(a_phi_th * dt_th + a_phi_r * dt_r + h_phi) / a_phi_u

        return torch.cat((dt_th, dt_phi, dt_r, dt_u), dim=-1)


class NeuralSympGC(AbstractDegenLagrangian, nn.Module):
    def __init__(self):
        AbstractDegenLagrangian.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.norm_in = UniformNormalizer1d(DIMENSION // 2)
        self.norm_out = UniformNormalizer1d(1)
        self.repeats = 6
        self.periodic = PeriodicLayer(1, num_repeat=6)
        in_dim = self.periodic.out_dim + 2 * self.repeats
        self.oneform_nn = create_mlp(in_dim, DIMENSION // 2, [48] * 3)
        self.hamiltonian_nn = create_mlp(in_dim, 1, [48] * 3)

    def set_norm(self, range_z, range_dt_z):
        z_min, z_max = range_z
        _, y_min = torch.tensor_split(z_min, 2, dim=-1)
        _, y_max = torch.tensor_split(z_max, 2, dim=-1)
        self.norm_in.set(y_min, y_max)

        dt_z_min, dt_z_max = range_dt_z
        norm_dt_z_min = dt_z_min.square().sum(-1, keepdims=True).sqrt()
        norm_dt_z_max = dt_z_max.square().sum(-1, keepdims=True).sqrt()
        self.norm_out.set_from_out(norm_dt_z_min, norm_dt_z_max)

    def preprocess(self, x, y):
        th, _ = torch.tensor_split(x, 2, dim=-1)
        x = self.periodic(th)
        y = torch.repeat_interleave(self.norm_in(y), self.repeats, -1)
        return torch.cat((x, y), -1)

    def oneform(self, x, y):
        inputs = self.preprocess(x, y)
        return self.oneform_nn(inputs)

    def hamiltonian(self, x, y, t):
        inputs = self.preprocess(x, y)
        return self.norm_out(self.hamiltonian_nn(inputs)).sum(-1)

    def lagrangian_maps(self, x, y, t):
        inputs = self.preprocess(x, y)
        oneform = self.oneform_nn(inputs)
        hamiltonian = self.norm_out(self.hamiltonian_nn(inputs)).sum(-1)
        return oneform, hamiltonian


class NeuralBaseGC(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        # pre-process
        self.norm_in = UniformNormalizer1d(DIMENSION // 2)
        self.repeats = 6
        self.periodic = PeriodicLayer(1, num_repeat=6)
        in_dim = self.periodic.out_dim + 2 * self.repeats
        # post-process
        self.norm_out = UniformNormalizer1d(DIMENSION)
        # NN
        self.vector_field_nn = create_mlp(in_dim, DIMENSION, [70] * 3)

    def set_norm(self, range_z, range_dt_z):
        z_min, z_max = range_z
        _, y_min = torch.tensor_split(z_min, 2, dim=-1)
        _, y_max = torch.tensor_split(z_max, 2, dim=-1)
        self.norm_in.set(y_min, y_max)

        dt_z_min, dt_z_max = range_dt_z
        self.norm_out.set_from_out(dt_z_min, dt_z_max)

    def vector_field(self, z, t):
        x, y = z.tensor_split(2, dim=-1)
        th, _ = torch.tensor_split(x, 2, dim=-1)
        x = self.periodic(th)
        y = torch.repeat_interleave(self.norm_in(y), self.repeats, -1)
        vf = self.vector_field_nn(torch.cat((x, y), -1))
        return self.norm_out(vf)

    def forward(self, z, t):
        return self.vector_field(z, t)


def test():
    model = GuidingCenter()
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
