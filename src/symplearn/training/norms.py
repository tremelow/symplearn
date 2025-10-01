import numpy as np

import torch
from torch import nn


class EuclideanSquaredNorm:
    def __init__(self, preproc=nn.Identity()):
        self.preproc = preproc

    def __call__(self, u: torch.Tensor):
        v = self.preproc(u)
        return v.square().mean(-1, keepdim=True)


class EuclideanLogCond:
    def __init__(self, preproc=nn.Identity()):
        self.preproc = preproc

    def __call__(self, J: torch.Tensor):
        D = self.preproc(J)
        return torch.log10(torch.linalg.cond(D)).mean()


class EuclideanLogCondWithDet:
    def __init__(self, preproc=nn.Identity()):
        self.preproc = preproc

    def __call__(self, J: torch.Tensor):
        D = self.preproc(J)
        cond = torch.linalg.cond(D)
        det = torch.linalg.det(D)
        # print(det.mean())
        return torch.log10(cond * (1.0 + (det - 1.0).square())).mean()


class GramPreprocess:
    def __init__(self, u_ref: torch.Tensor, eps: float = 1e-8):
        with torch.no_grad():
            mag_dims = tuple(range(len(u_ref.shape)))[:-1]
            gram_mat = torch.einsum("...i,...j->...ij", u_ref, u_ref).mean(mag_dims)
            U, S, Vh = np.linalg.svd(gram_mat.cpu().numpy(), hermitian=True)
            sqrt_S = np.sqrt(S + eps)
            self.sqrt_M = torch.tensor(U @ (np.diag(sqrt_S) @ Vh))

            sqrt_inv_S = np.sqrt(1.0 / (S + eps))
            self.sqrt_inv_M = torch.tensor(U @ (np.diag(sqrt_inv_S) @ Vh))

    def vect(self, u: torch.Tensor):
        return u @ self.sqrt_inv_M

    def mat(self, J: torch.Tensor):
        return J @ self.sqrt_M


class ScaledMSNorm:
    def __init__(
        self,
        squared_norm: callable = EuclideanSquaredNorm(),
        abs_weight: float = 1.0,
        rel_weight: float = 1.0,
        rel_eps: float = 1e-8,
    ):
        self.abs_weight = abs_weight
        self.rel_weight = rel_weight
        self.rel_eps = rel_eps
        self.norm_fun = squared_norm

    def norm(self, u: torch.Tensor, scaler: torch.Tensor):
        u_abs_norm = self.norm_fun(u)
        scaler_norm = self.norm_fun(scaler)
        u_rel_norm = u_abs_norm / (self.rel_eps + scaler_norm)
        return u_abs_norm, u_rel_norm

    def __call__(self, u: torch.Tensor, scaler: torch.Tensor):
        u_abs_norm, u_rel_norm = self.norm(u, scaler)
        abs_loss = u_abs_norm.mean()
        rel_loss = u_rel_norm.mean()
        loss = self.abs_weight * abs_loss + self.rel_weight * rel_loss
        return loss, {"abs": abs_loss.item(), "rel": rel_loss.item()}


class GramMSNorm(ScaledMSNorm):
    def __init__(
        self,
        squared_norm: callable = EuclideanSquaredNorm(),
        abs_weight: float = 1.0,
        rel_weight: float = 1.0,
        rel_eps: float = 1e-8,
        gram_eps: float = 1e-8,
    ):
        super().__init__(squared_norm, abs_weight, rel_weight, rel_eps)
        self.gram_eps = gram_eps

    def build_gram_mat(self, scaler: torch.Tensor):
        mag_dims = tuple(range(len(scaler.shape)))[:-1]
        gram_mat = torch.einsum("...i,...j->...ij", scaler, scaler).mean(mag_dims)
        U, S, Vh = np.linalg.svd(gram_mat.cpu().numpy(), hermitian=True)
        sqrt_inv_S = 1.0 / (np.sqrt(S) + self.gram_eps)
        sqrt_inv_M = torch.tensor(U @ (np.diag(sqrt_inv_S) @ Vh))
        return sqrt_inv_M

    def norm(self, u: torch.Tensor, scaler: torch.Tensor):
        sqrt_inv_M = self.build_gram_mat(scaler)
        u_abs_norm = self.norm_fun(u @ sqrt_inv_M)
        scaler_norm = self.norm_fun(scaler @ sqrt_inv_M)
        u_rel_norm = u_abs_norm / (self.rel_eps + scaler_norm)

        return u_abs_norm, u_rel_norm


class DiagMSNorm(ScaledMSNorm):
    def __init__(
        self,
        squared_norm: callable = EuclideanSquaredNorm(),
        abs_weight: float = 1.0,
        rel_weight: float = 1.0,
        rel_eps: float = 1e-8,
        inv_eps: float = 1e-8,
    ):
        super().__init__(squared_norm, abs_weight, rel_weight, rel_eps)
        self.inv_eps = inv_eps

    def norm(self, u: torch.Tensor, scaler: torch.Tensor):
        lead_dims = tuple(range(len(scaler.shape)))[:-1]
        scaler_amp = scaler.square().mean(lead_dims).sqrt()
        inv_scaler_amp = 1.0 / (self.inv_eps + scaler_amp)
        u_abs_norm = self.norm_fun(u * inv_scaler_amp)
        scaler_norm = self.norm_fun(scaler * inv_scaler_amp)
        u_rel_norm = u_abs_norm / (self.rel_eps + scaler_norm)

        return u_abs_norm, u_rel_norm
