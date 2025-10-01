import warnings

import numpy as np
from scipy.integrate import solve_ivp

import torch
from torch.func import vmap, grad, jacrev, jacfwd

from ..dynamics import AbstractDiffEq, AbstractDegenLagrangian
from .solver import NewtonRaphsonSolver


class AbstractUniformTimeStepSimulation:
    """
    Abstract class for time-stepping methods.
    """

    def __init__(self, pb: AbstractDiffEq, dt: float | torch.Tensor):
        self.pb = pb
        self.dt = dt

    def step(self, z_cur, t_cur):
        raise NotImplementedError

    def simulate(self, z0: torch.Tensor, nt: int, t0: float | torch.Tensor = 0.0):
        """
        Simulate the time evolution of a system using the specified time-stepping method.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial state of the system as a tensor. Of shape `[b, dim]`.
        nt : int
            Number of timesteps to simulate.
        t0 : float | torch.Tensor
            Initial time of the simulation. Scalar, default is 0.0.

        Returns
        -------
        t_sim : torch.Tensor
            Tensor containing the simulation time points. Of shape `[nt + 1]`.
        z_sim : torch.Tensor
            Tensor containing the simulated states at each time point. Of shape `[b, nt + 1, dim]`.
        """
        vmap_step = vmap(self.step)

        z_sim = torch.empty(nt + 1, *z0.shape)
        z_sim[0] = z0
        t_sim = t0 + self.dt * torch.arange(nt + 1)[:, None].expand(*z_sim.shape[:-1])

        for n in range(nt):
            z_sim[n + 1] = vmap_step(z_sim[n], t_sim[n]).detach()

        return t_sim.transpose(0, 1), z_sim.transpose(0, 1)


class QuasiExactSimulation(AbstractUniformTimeStepSimulation):
    """
    Quasi-exact simulation of a system using `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    pb : AbstractDiffEq
        The problem definition containing the vector field.
    dt : float | torch.Tensor
        The time step size for the simulation.
    """

    def step(self, zk: torch.Tensor, tk: float | torch.Tensor):
        raise NotImplementedError("QuasiExactSimulation does not support step method.")

    def simulate(
        self,
        z0: torch.Tensor,
        nt: int,
        t0: float | torch.Tensor = 0.0,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        method: str = "DOP853",
        **kwargs
    ):
        """
        Simulate the time evolution of a system using `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial state of the system as a tensor. Of shape `[b, dim]`.
        nt : int
            Number of timesteps to simulate.
        t0 : float | torch.Tensor, default=0.0
            Initial time of the simulation. Scalar.
        rtol : float, default=1e-10
            The relative tolerance for the solver.
        atol : float, default=1e-12
            The absolute tolerance for the solver.
        method : str, default='DOP853'
            The integration method to use.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to `solve_ivp`.

        Returns
        -------
        t_sim : torch.Tensor
            Tensor containing the simulation time points. Of shape `[nt + 1]`.
        z_sim : torch.Tensor
            Tensor containing the simulated states at each time point. Of shape `[b, nt + 1, dim]`.
        """
        t_eval = t0 + self.dt * np.arange(nt + 1)
        u0 = z0.flatten().detach().cpu().numpy()
        vmap_vf = vmap(self.pb.vector_field)

        def vf(t: float, z_flat: np.ndarray):
            z_cur = torch.from_numpy(z_flat).reshape(*z0.shape)
            t_cur = torch.full(z0.shape[:-1], t)
            dz_cur = vmap_vf(z_cur, t_cur)
            return dz_cur.flatten().detach().cpu().numpy()

        solver_args = kwargs | dict(t_eval=t_eval, rtol=rtol, atol=atol, method=method)
        sol = solve_ivp(vf, (t0, t_eval[-1]), u0, **solver_args)

        y = sol.y.reshape(*tuple(z0.shape), -1)  # (n_traj, z_dim, nt+1)
        z = np.transpose(y, (0, 2, 1))  # (n_traj, nt+1, z_dim)
        z_sim = torch.tensor(z)
        t_sim = torch.tensor(sol.t)[None, :].expand(*z_sim.shape[:-1])
        if len(sol.t) <= nt:
            warnings.warn(
                f"Simulation time points are less than expected, the simulation probably crashed early.",
                RuntimeWarning,
            )
        return t_sim, z_sim


class ExplicitEulerSimulation(AbstractUniformTimeStepSimulation):
    """
    Explicit Euler time-stepping method for simulating the time evolution of a system.

    Parameters
    ----------
    pb : AbstractDiffEq
        The problem definition containing the vector field.
    dt : float | torch.Tensor
        The time step size for the simulation.
    """

    def step(self, zk: torch.Tensor, tk: float | torch.Tensor):
        return zk + self.dt * self.pb.vector_field(zk, tk)


def step_euler(
    pb: AbstractDiffEq,
    zk: torch.Tensor,
    tk: float | torch.Tensor,
    dt: float | torch.Tensor,
):
    return zk + dt * pb.vector_field(zk, tk)


class RK4Simulation(AbstractUniformTimeStepSimulation):
    """
    Runge-Kutta 4th order time-stepping method for simulating the time evolution of a system,
    with a fixed time-step.

    Parameters
    ----------
    pb : AbstractDiffEq
        The problem definition containing the vector field.
    dt : float | torch.Tensor
        The time step size for the simulation.
    """

    def step(self, zk: torch.Tensor, tk: float | torch.Tensor):
        k1 = self.dt * self.pb.vector_field(zk, tk)
        k2 = self.dt * self.pb.vector_field(zk + 0.5 * k1, tk + 0.5 * self.dt)
        k3 = self.dt * self.pb.vector_field(zk + 0.5 * k2, tk + 0.5 * self.dt)
        k4 = self.dt * self.pb.vector_field(zk + k3, tk + self.dt)
        return zk + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def step_rk4(
    pb: AbstractDiffEq,
    zk: torch.Tensor,
    tk: float | torch.Tensor,
    dt: float | torch.Tensor,
):
    k1 = dt * pb.vector_field(zk, tk)
    k2 = dt * pb.vector_field(zk + 0.5 * k1, tk + 0.5 * dt)
    k3 = dt * pb.vector_field(zk + 0.5 * k2, tk + 0.5 * dt)
    k4 = dt * pb.vector_field(zk + k3, tk + dt)
    return zk + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


class EulerDVISimulation(AbstractUniformTimeStepSimulation):
    """
    A class implementing the first-order (Euler) Degenerate Variational Integrator
    (DVI) for simulating the time evolution of a degenerate Lagrangian system.

    Attributes
    ----------
    pb : AbstractDegenLagrangian
        The problem definition containing the Lagrangian and other related functions.
    dt : float or torch.Tensor
        The time step size for the simulation.
    d_lag_cur : callable
        The gradient of the discrete Lagrangian with respect to the current state.
    d_lag_next : callable
        The gradient of the discrete Lagrangian with respect to the next state.

    Methods
    -------
    __init__(pb, dt)
        Initialize the EulerDVISimulation instance with the problem definition and time step size.
    backstep_x(x0, y0, t0)
        Compute the previous position using the backward explicit Euler scheme.
    discrete_lag(x0, x1, y1, t1)
        Compute the discrete Lagrangian for the given states and time.
    scheme(z_next, x_cur, y_cur, t_cur, dx_lag0=None)
        Compute the residual of the scheme and the derivative of the Lagrangian
        with respect to the next state.
    scheme_z(z_prev, z_cur, z_next, t_cur)
        Compute the residual of the scheme for learning purposes.
    simulate(z0, t0, nt, init_step=None, solver=NewtonRaphsonSolver())
        Simulate the time evolution of the system using a specified solver.

    Notes
    -----
    - The class assumes the system state `z` can be split into two components `x` and `y`
      along the last dimension.
    - The solver used in the `simulate` method must support the `set_fun` method with
      `has_aux=True` and be callable with the required arguments.
    """

    def __init__(
        self,
        pb: AbstractDegenLagrangian,
        dt: float | torch.Tensor,
    ):
        self.pb = pb
        self.dt = dt
        # D_2 L(x_prev, x_cur, y_cur)
        self.d_lag_cur = grad(self.discrete_lag, argnums=1)
        # D_1 L(x_cur, x_next, y_next), D_3 L(x_cur, x_next, y_next)
        self.d_lag_next = grad(self.discrete_lag, argnums=(0, 2))

    def backstep_x(
        self, x0: torch.Tensor, y0: torch.Tensor, t0: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the previous position `x` with the backward explicit Euler scheme.

        Parameters
        ----------
        x0 : torch.Tensor
            The current "position" component of the state. Of shape `[dim/2]`.
        y0 : torch.Tensor
            The current "momentum" component of the state. Of shape `[dim/2]`.
        t0 : float or torch.Tensor
            The current time. Scalar.

        Returns
        -------
        torch.Tensor
            The previous position component of the state. Of shape `[dim/2]`.
        """
        dy_q = jacrev(self.pb.oneform, 1)(x0, y0)
        dy_h = grad(self.pb.hamiltonian, 1)(x0, y0, t0)
        dx = torch.linalg.solve(dy_q.T, dy_h)
        return x0 - self.dt * dx

    def discrete_lag(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        t1: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the discrete Lagrangian for the given states and time.

        Parameters
        ----------
        x0 : torch.Tensor
            The position component of the state at the previous time step. Of shape `[dim/2]`.
        x1 : torch.Tensor
            The position component of the state at the current time step. Of shape `[dim/2]`.
        y1 : torch.Tensor
            The momentum component of the state at the current time step. Of shape `[dim/2]`.
        t1 : float or torch.Tensor
            The current time. Scalar.

        Returns
        -------
        torch.Tensor
            The computed discrete Lagrangian. Scalar.
        """
        x_t = (x1 - x0) / self.dt
        return self.pb.lagrangian(x1, y1, x_t, t1)

    def scheme_with_lag(
        self,
        z_next: torch.Tensor,
        x_cur: torch.Tensor,
        t_next: float | torch.Tensor,
        dx_lag0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The scheme used for simulation.

        Parameters
        ----------
        z_next : torch.Tensor
            The state of the system at the next time step. Of shape `[dim]`.
        x_cur : torch.Tensor
            The position component of the current state. Of shape `[dim/2]`.
        t_cur : float or torch.Tensor
            The current time. Scalar.
        dx_lag0 : torch.Tensor or None
            The derivative of the discrete Lagrangian with respect to the current state,
            evaluated at the previous time step. If None, it will be computed.

        Returns
        -------
        res : torch.Tensor
            The residual of the scheme, combining contributions from the
            current and next states. Of shape `[dim]`.
        dx_lag1 : torch.Tensor
            The derivative of the Lagrangian with respect to the next state.
            Of shape `[dim]`.
        """
        x_next, y_next = torch.tensor_split(z_next, 2, -1)
        state_next = (x_next, y_next, t_next)
        dx_lag1, dy_lag1 = self.d_lag_next(x_cur, *state_next)
        res = torch.cat((dx_lag0 + dx_lag1, dy_lag1), -1)
        return res

    def scheme(
        self,
        z_next: torch.Tensor,
        x_cur: torch.Tensor,
        y_cur: torch.Tensor,
        t_cur: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The scheme used for simulation.

        Parameters
        ----------
        z_next : torch.Tensor
            The state of the system at the next time step. Of shape `[dim]`.
        x_cur : torch.Tensor
            The position component of the current state. Of shape `[dim/2]`.
        y_cur : torch.Tensor
            The velocity component of the current state. Of shape `[dim/2]`.
        t_cur : float or torch.Tensor
            The current time. Scalar.

        Returns
        -------
        res : torch.Tensor
            The residual of the scheme, combining contributions from the
            current and next states. Of shape `[dim]`.
        """
        x_prev = self.backstep_x(x_cur, y_cur, t_cur)
        dx_lag0 = self.d_lag_cur(x_prev, x_cur, y_cur, t_cur)
        return self.scheme_with_lag(z_next, x_cur, t_cur + self.dt, dx_lag0)

    def simulate(
        self,
        z0: torch.Tensor,
        nt: int,
        t0: float | torch.Tensor = 0.0,
        init_step=None,
        solver=NewtonRaphsonSolver(),
    ):
        """
        Simulate the time evolution of a system using a specified solver.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial state of the system as a tensor. Of shape `[b, dim]`.
        t0 : float or torch.Tensor
            Initial time of the simulation. Scalar.
        nt : int
            Number of timesteps to simulate.
        init_step : callable, optional
            Function to compute the initial guess for the solver. If None,
            defaults to using the explicit Euler method (`step_euler`).
        solver : callable, optional
            Solver to compute the next state. Defaults to `NewtonRaphsonSolver()`.

        Returns
        -------
        t_sim : torch.Tensor
            Tensor containing the simulation time points. Of shape `[nt + 1]`.
        z_sim : torch.Tensor
            Tensor containing the simulated states at each time point. Of shape `[b, nt + 1, dim]`.

        Notes
        -----
        - The function assumes the system state `z` can be split into two components
          `x` and `y` along the last dimension.
        - The solver must support the `set_fun` method with `has_aux=True` and
          be callable with the required arguments.
        """
        if init_step is None:
            init_step = vmap(ExplicitEulerSimulation(self.pb, self.dt).step)
        solver.set_fun(self.scheme_with_lag, batched=True)
        get_lag = vmap(self.d_lag_cur)

        # saves
        z_sim = torch.empty(nt + 1, *z0.shape)
        z_sim[0] = z0
        t_sim = t0 + self.dt * torch.arange(nt + 1)[:, None].expand(*z_sim.shape[:-1])

        # initialise values
        z_cur, t_cur = z_sim[0], t_sim[0]
        x_cur, y_cur = torch.tensor_split(z_cur, 2, -1)
        x_prev = vmap(self.backstep_x)(x_cur, y_cur, t_cur)

        for n in range(1, nt + 1):
            dx_lag0 = get_lag(x_prev, x_cur, y_cur, t_cur)
            z_next = init_step(z_cur, t_cur)
            t_next = t_sim[n]
            z_next = solver.solve(z_next, x_cur, t_next, dx_lag0, z_shift=z_cur).detach()
            z_sim[n] = z_next

            x_prev = x_cur
            z_cur = z_next
            x_cur, y_cur = torch.tensor_split(z_cur, 2, -1)
            t_cur = t_next

        return t_sim.transpose(0, 1), z_sim.transpose(0, 1)
