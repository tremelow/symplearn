import warnings

import torch
from torch.func import jacrev, vmap


def double_first_arg(fun, has_aux=False):
    if has_aux:

        def doubled_fun(*args, **kwargs):
            res = fun(*args, **kwargs)
            return res[0], (res[0], res[1])

    else:

        def doubled_fun(*args, **kwargs):
            res = fun(*args, **kwargs)
            return res, (res, torch.empty(()))

    return doubled_fun


class NewtonRaphsonSolver:
    def __init__(
        self,
        max_iters=10,
        abs_tol=1e-10,
        rel_tol=1e-8,
        rel_eps=1e-6,
    ):
        self.max_iters = max_iters
        self.atol = abs_tol
        self.rtol = rel_tol
        self.eps = rel_eps

    def set_fun(self, fun: callable, has_aux: bool = False, batched: bool = False):
        """
        Set the function to be solved.

        Parameters
        ----------
        fun : callable
            The function to be solved.
        """
        self.jacres_fun = jacrev(double_first_arg(fun, has_aux=has_aux), has_aux=True)
        if batched:
            self.jacres_fun = vmap(self.jacres_fun)
        self.has_aux = has_aux

    def solve(
        self,
        z_init: torch.Tensor,
        *args: tuple,
        step_size: float = 1.0,
        z_shift: float | torch.Tensor = 0.0,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the equation using Newton's method.

        Parameters
        ----------
        scheme : callable
            The scheme function to be used.
        z_init : torch.Tensor
            The initial guess for the solution.
        *args : tuple
            Additional arguments to be passed to the scheme function.
        step_size : float, optional
            The step size for the update, by default 1.0
        z_shift : float | torch.Tensor, optional
            The shift to be applied to the solution when computing the relative error, by default 0.0
        **kwargs : dict
            Additional keyword arguments to be passed to the scheme function.
            Cannot be used when `batched` is set to True.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The updated solution and the lagrangian value.
        """
        z = z_init
        jac, (res, aux) = self.jacres_fun(z, *args, **kwargs)
        sq_err = res.square().mean(-1).max().sqrt()
        abs_err, rel_err = sq_err, sq_err
        for _ in range(self.max_iters):
            jac, res, aux = jac.detach(), res.detach(), aux.detach()
            dz = torch.linalg.solve(jac, res)
            z = z - step_size * dz

            sq_err = dz.square()
            rel_sq_err = sq_err / (self.eps + (z - z_shift).square())
            abs_norm, rel_norm = sq_err.sum(-1), rel_sq_err.sum(-1)
            abs_err, rel_err = abs_norm.max().sqrt(), rel_norm.max().sqrt()
            if abs_err < self.atol and rel_err < self.rtol:
                break

            jac, (res, aux) = self.jacres_fun(z, *args, **kwargs)
        else:
            warnings.warn(
                f"Newton iterations did not converge, abs_err: {abs_err.max().item()}, rel_err: {rel_err.max().item()}",
                RuntimeWarning,
            )

        if self.has_aux:
            return z, aux
        return z


def test():
    def rosenbrock(z):
        a, b = 1.0, 100.0
        x, y = z
        return (a - x) ** 2 + b * (y - x**2) ** 2, z


    solver = NewtonRaphsonSolver()
    solver.set_fun(torch.func.grad(rosenbrock, has_aux=True), batched=True, has_aux=True)
    assert torch.allclose(solver.solve(torch.randn(5, 2)), torch.ones(2))
