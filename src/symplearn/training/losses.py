import torch
from torch.func import vmap, jacrev

from ..dynamics import AbstractDiffEq, AbstractDegenLagrangian
from ..numerics import EulerDVISimulation

from .norms import ScaledMSNorm, GramMSNorm, EuclideanLogCond


class VectorFieldLoss:
    """
    VectorFieldLoss is a class designed to compute the loss for training a neural network
    that models a vector field. It supports both regularized and non-regularized loss
    computation based on the provided regularization weight.

    Methods
    -------
    loss_no_regul(model, z, dt_z)
        Computes the loss without regularization.
    aux_error_reg(z, x_t, t)
        Computes auxiliary errors for regularization.
    error_reg(z, t, dt_z)
        Computes the dominant term in the error of the scheme.
    loss_regul(model, z_batch, dz_dt_batch, test=False)
        Computes the loss with regularization.

    Notes
    -----
    - If `reg_weight` is None, the class computes the loss without regularization.
    - If `reg_weight` is provided, the class computes the loss with regularization,
      which includes additional terms to account for the error in the scheme.
    """

    def __init__(
        self,
        model: AbstractDiffEq | AbstractDegenLagrangian,
        reg_weight: float | None = 1e-8,
        loss_fn: ScaledMSNorm = GramMSNorm(),
    ):

        self.model = model
        self.reg_weight = reg_weight
        self.loss_fn = loss_fn
        self._jac_aux_error_reg = jacrev(self._aux_error_reg, has_aux=True)

    def loss_no_regul(self, z, t, dt_z):
        dt_z_pred = vmap(self.model.vector_field)(z, t)
        loss, tracker = self.loss_fn(dt_z_pred - dt_z, dt_z)
        tracker = {"tot": loss.item(), **tracker}
        return loss, tracker

    def _aux_error_reg(self, z, x_t, t):
        x, y = torch.tensor_split(z, 2, -1)
        (dx_q, dy_q), (dx_h, dy_h) = self.model.euler_lagrange_maps(x, y, t)

        dx_lag = dx_q.transpose(-1, -2) @ x_t - dx_h
        dy_lag = dy_q.transpose(-1, -2) @ x_t - dy_h

        dt_x = torch.linalg.solve(dy_q.T, dy_h)
        dt_y = torch.linalg.solve(dy_q, (dx_q.T - dx_q) @ dt_x - dx_h)
        dt_z = torch.cat((dt_x, dt_y), -1)
        return (dx_lag, dy_lag), (dx_q, dy_q, dt_z)

    def _error_reg(self, z, t, dt_z):
        """
        Compute the dominant term in the error of the scheme.
        """

        dt_x, _ = torch.tensor_split(dt_z, 2, -1)
        dd_lag, aux = self._jac_aux_error_reg(z, dt_x, t)
        dxz_lag, dyz_lag = dd_lag
        dx_q, dy_q, dt_z_pred = aux

        err_x_rhs = dyz_lag @ dt_z
        err_reg_x = torch.linalg.solve(dy_q.T, err_x_rhs)

        err_y_rhs = -((dx_q + dx_q.T) @ err_reg_x) + dxz_lag @ dt_z
        err_reg_y = torch.linalg.solve(dy_q, err_y_rhs)

        err_vf = dt_z - dt_z_pred
        err_reg = torch.cat((err_reg_x, err_reg_y), -1)

        return err_vf, err_reg

    def _error_reg_with_det(self, z, t, dt_z):
        """
        Compute the dominant term in the error of the scheme.
        """

        dt_x, _ = torch.tensor_split(dt_z, 2, -1)
        dd_lag, aux = self._jac_aux_error_reg(z, dt_x, t)
        dxz_lag, dyz_lag = dd_lag
        dx_q, dy_q, dt_z_pred = aux

        err_x_rhs = dyz_lag @ dt_z
        err_reg_x = torch.linalg.solve(dy_q.T, err_x_rhs)

        err_y_rhs = -((dx_q + dx_q.T) @ err_reg_x) + dxz_lag @ dt_z
        err_reg_y = torch.linalg.solve(dy_q, err_y_rhs)

        err_vf = dt_z - dt_z_pred
        err_reg = torch.cat((err_reg_x, err_reg_y), -1)

        det_w = torch.linalg.det(dy_q.T @ dy_q)
        reg_det_w = torch.log1p((det_w - 1.0).square())
        # reg_det_w = torch.slogdet(dy_q)[1].square()

        return err_vf, err_reg, reg_det_w

    def loss_regul(self, z, t, dt_z):
        # err_vf, err_reg, reg_det_w = vmap(self._error_reg_with_det)(z, t, dt_z)
        err_vf, err_reg = vmap(self._error_reg)(z, t, dt_z)
        dz_loss, dz_tracker = self.loss_fn(err_vf, dt_z)
        reg_loss, reg_tracker = self.loss_fn(err_reg, dt_z)

        loss = dz_loss + self.reg_weight * reg_loss
        # reg_det_w = reg_det_w.mean()
        # loss = dz_loss + self.reg_weight * (reg_loss + reg_det_w)

        tracker = {"tot": loss.item()}
        for key in dz_tracker:
            tracker[key + "_dz"] = dz_tracker[key]
        for key in reg_tracker:
            tracker[key + "_reg"] = reg_tracker[key]
        # tracker["reg_det"] = reg_det_w.item()

        return loss, tracker

    def __call__(self, z, t, dt_z):
        if self.reg_weight is None:
            return self.loss_no_regul(z, t, dt_z)
        else:
            return self.loss_regul(z, t, dt_z)


class EulerDVINewtonLoss:
    def __init__(
        self,
        model: AbstractDegenLagrangian,
        dt: float | torch.Tensor,
        reg_weight: float = 1e-8,
        norm_fn: ScaledMSNorm = GramMSNorm(),
        reg_fn: callable = EuclideanLogCond(),
    ):
        self.model = model
        self.dvi = EulerDVISimulation(model, dt)

        self.reg_weight = reg_weight
        self.norm_fn = norm_fn
        self.reg_fn = reg_fn
        self.dvi_with_jacobian = jacrev(self.compute_dvi, argnums=2, has_aux=True)

    def compute_dvi(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        z_next: torch.Tensor,
        t_cur: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The scheme used for learning.

        Parameters
        ----------
        z_prev : torch.Tensor
            The state of the system at the previous time step. Of shape `[dim]`.
        z_cur : torch.Tensor
            The state of the system at the current time step. Of shape `[dim]`.
        z_next : torch.Tensor
            The state of the system at the next time step. Of shape `[dim]`.
        t_cur : float or torch.Tensor
            The current time. Scalar.

        Returns
        -------
        torch.Tensor
            The residual of the scheme, combining contributions from the
            current and next states. Of shape `[dim]`.
        """
        x_prev, _ = torch.tensor_split(z_prev, 2, -1)
        x_cur, y_cur = torch.tensor_split(z_cur, 2, -1)
        x_next, y_next = torch.tensor_split(z_next, 2, -1)

        dx_lag0 = self.dvi.d_lag_cur(x_prev, x_cur, y_cur, t_cur)

        state_next = (x_next, y_next, t_cur + self.dvi.dt)
        dx_lag1, dy_lag1 = self.dvi.d_lag_next(x_cur, *state_next)

        sch = torch.cat((dx_lag0 + dx_lag1, dy_lag1), -1)
        return sch, sch

    def __call__(self, z0, z1, z2, t1):
        jac_dvi, dvi = vmap(self.dvi_with_jacobian)(z0, z1, z2, t1)

        dz_newton = torch.linalg.solve(jac_dvi, dvi)
        dz_data = z2 - z1
        err, err_tracker = self.norm_fn(dz_newton, dz_data)
        reg = self.reg_fn(jac_dvi)
        loss = err + self.reg_weight * reg
        tracker = {"tot": loss.item(), **err_tracker, "reg": reg.item()}

        return loss, tracker


class ImexEulerLoss:
    def __init__(
        self,
        model: AbstractDegenLagrangian,
        dt: float | torch.Tensor,
        reg_weight: float = 1e-8,
        norm_fn: ScaledMSNorm = GramMSNorm(),
        reg_fn: callable = EuclideanLogCond(),
    ):
        self.model = model
        self.dt = dt

        self.reg_weight = reg_weight
        self.norm_fn = norm_fn
        self.reg_fn = reg_fn

    def compute_scheme(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        z_next: torch.Tensor,
        t_cur: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x_prev, _ = torch.tensor_split(z_prev, 2, -1)
        x_cur, y_cur = torch.tensor_split(z_cur, 2, -1)
        x_next, y_next = torch.tensor_split(z_next, 2, -1)

        dx_ex, dy_ex = x_cur - x_prev, y_next - y_cur

        (dx_q, dy_q), (dx_h, dy_h) = self.model.euler_lagrange_maps(x_cur, y_cur, t_cur)
        dx_sch = torch.linalg.solve(dy_q.T, self.dt * dy_h)

        dx_ex_next = x_next - x_cur
        dy_rhs = dx_q.T @ dx_ex - dx_q @ dx_ex_next - self.dt * dx_h
        dy_sch = torch.linalg.solve(dy_q, dy_rhs)

        dz_sch = torch.cat((dx_sch, dy_sch), -1)
        dz_ex = torch.cat((dx_ex, dy_ex), -1)
        dx_jac = torch.cat((-dy_q.T, torch.zeros_like(dy_q)), -1)
        dy_jac = torch.cat((torch.zeros_like(dy_q), dy_q), -1)
        jac = torch.cat((dx_jac, dy_jac), -2)
        return dz_ex - dz_sch, dz_ex, jac

    def __call__(self, z0, z1, z2, t1):
        sch_err, dz_ex, jac_sch = vmap(self.compute_scheme)(z0, z1, z2, t1)

        err, err_tracker = self.norm_fn(sch_err, dz_ex)
        reg = self.reg_fn(jac_sch)
        loss = err + self.reg_weight * reg
        tracker = {"tot": loss.item(), **err_tracker, "reg": reg.item()}

        return loss, tracker


class NaiveStepperLoss:
    def __init__(
        self,
        model: AbstractDiffEq,
        dt: float | torch.Tensor,
        reg_weight: float | None = 1e-8,
        norm_fn: ScaledMSNorm = ScaledMSNorm(),
    ):
        self.model = model
        self.dt = dt
        self.reg_weight = reg_weight
        self.norm_fn = norm_fn

    def __call__(self, z0, z1, z2, t1):
        z10 = self.model(z0, t1 - self.dt, self.dt)
        err10, err10_tracker = self.norm_fn(z10 - z1, z1 - z0)

        z21 = self.model(z1, t1, self.dt)
        err21, err21_tracker = self.norm_fn(z21 - z2, z2 - z1)

        z20 = self.model(z10, t1, self.dt)
        err20, err20_tracker = self.norm_fn(z20 - z2, z2 - z0)

        loss = err10 + err21 + err20
        tracker = {"tot": loss.item()}
        for key in err10_tracker:
            tracker[key + "10"] = err10_tracker[key]
        for key in err20_tracker:
            tracker[key + "20"] = err20_tracker[key]
        for key in err21_tracker:
            tracker[key + "21"] = err21_tracker[key]

        return loss, tracker
