import numpy as np
from scipy.integrate import solve_ivp

import torch
from torch import nn
from torch.func import jacrev, grad


class AbstractDiffEq:
    """
    Abstract base class for defining a differential equation.

    Attributes
    ----------
    dim : int
        The dimension of the differential equation.

    Methods
    -------
    vector_field(t, z)
        Computes the vector field of the differential equation.
    np_vector_field(t, z)
        Computes the vector field of the differential equation using NumPy (`t` is a dummy argument).
    generate_trajectories(z0, tf, nt, rtol=1e-10, atol=1e-10, method='DOP853', **kwargs)
        Generates trajectories from initial conditions using `solve_ivp`.
    """

    def __init__(self, dim) -> None:
        assert dim > 0 and isinstance(
            dim, int
        ), "Dimension should be a positive integer."
        self.dim = dim

    def vector_field(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """
        Compute the vector field for the given state and time. Must be implemented
        in a subclass.

        Parameters
        ----------
        z : torch.Tensor
            The system's state, of shape (dim,).
        t : float | torch.Tensor
            The current time, of shape ().

        Returns
        -------
        torch.Tensor
            The computed vector field as a tensor.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError


class NeuralDiffEq(AbstractDiffEq, nn.Module):
    """
    A neural network-based implementation of the AbstractDiffEq class.
    This class combines the functionality of a neural network with the structure
    of a differential equation.

    Parameters
    ----------
    dim : int
        The dimension of the system.
    vector_field_nn : nn.Module
        A neural network module representing the vector field.
    """

    def __init__(self, dim, vector_field_nn):
        AbstractDiffEq.__init__(self, dim)
        nn.Module.__init__(self)
        self.vector_field_nn = vector_field_nn

    def vector_field(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """
        Computes the vector field using the neural network module.

        Parameters
        ----------
        z : torch.Tensor
            The system's state.
        t : float | torch.Tensor
            The current time.

        Returns
        -------
        torch.Tensor
            The computed vector field.
        """

        return self.vector_field_nn(z)


class AbstractDegenLagrangian(AbstractDiffEq):
    """
    AbstractDegenLagrangian is an abstract class for defining Lagrangians of the form:
        L(x, y, x_t, t) = q(x, y) x_t - H(x, y, t)
    where:
    - `q(x, y)` is a one-form (vector-like quantity).
    - `H(x, y, t)` is the Hamiltonian (scalar quantity).

    The canonical of this specific ansatz has `x` as the position and `y` as the momentum of the system.

    This class provides methods to define and compute the Lagrangian, its associated maps, and the corresponding vector field for the system.

    Methods
    -------
    AbstractDegenLagrangian(dim)
        Initializes the system with a given dimension. Ensures the dimension is even (required for Hamiltonian systems).
    oneform(x, y)
        Abstract method to compute the one-form `q(x, y)`. Must be implemented by subclasses.
    hamiltonian(x, y, t)
        Abstract method to compute the Hamiltonian `H(x, y, t)`. Must be implemented by subclasses.
    lagrangian_maps(x, y, t)
        Computes the one-form `q(x, y)` and Hamiltonian `H(x, y, t)`.
    lagrangian(x, y, x_t, t)
        Computes the Lagrangian `L(x, y, x_t, t)` using the one-form and Hamiltonian.
    euler_lagrange_maps(x, y, t)
        Computes the derivatives of the one-form and Hamiltonian with respect to `x` and `y`.
    vector_field(z, t)
        Computes the vector field of the system by solving the Euler-Lagrange equations.

    Attributes
    ----------
        dim : int
            The dimension of the system. Must be even.
    """

    def __init__(self, dim):
        assert (
            dim % 2 == 0
        ), "The dimension of the system is odd. It cannot be Hamiltonian."
        super().__init__(dim)

    def oneform(self, x: torch.Tensor, y: torch.Tensor):
        """
        Computes a one-form operation on the input vector `y`. Must be implemented in a subclass.

        Parameters
        ----------
        x : torch.Tensor
            Input variable representing the point at which the one-form is evaluated.
        y : torch.Tensor
            Input vector to which the one-form is applied.

        Returns
        -------
        torch.Tensor
            A vector resulting from the one-form operation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def hamiltonian(self, x: torch.Tensor, y: torch.Tensor, t: float | torch.Tensor):
        """
        Computes the Hamiltonian of the system. Must be implemented in a subclass.

        Parameters
        ----------
        x : torch.Tensor
            The generalized coordinates of the system.
        y : torch.Tensor
            The generalized momenta of the system.
        t : float | torch.Tensor
            The time variable.

        Returns
        -------
        torch.Tensor
            The scalar value of the Hamiltonian.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError

    def lagrangian_maps(
        self, x: torch.Tensor, y: torch.Tensor, t: float | torch.Tensor
    ):
        """
        Computes the Lagrangian maps for the given inputs.

        Parameters
        ----------
        x : torch.Tensor
            The first input tensor representing the state variable.
        y : torch.Tensor
            The second input tensor representing the state variable.
        t : float or torch.Tensor
            The time variable, which can be a scalar or a tensor.

        Returns
        -------
        q : torch.Tensor
            The one-form computed from the inputs.
        h : torch.Tensor
            The Hamiltonian computed from the inputs and time.
        """
        return self.oneform(x, y), self.hamiltonian(x, y, t)

    def lagrangian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_t: torch.Tensor,
        t: float | torch.Tensor,
    ):
        """
        Computes the Lagrangian of the system,
            L(x, y, x_t, t) = q(x, y) x_t - H(x, y, t).
        where `q(x, y)` is the one-form and `H(x, y, t)` is the Hamiltonian.

        Parameters
        ----------
        x : torch.Tensor
            The first part of the coordinates of the system. The position in the
            canonical case.
        y : torch.Tensor
            The second part of the coordinates, w.r.t. which the Jacobian of the
            one-form is invertible. The momentum in the canonical case.
        x_t : torch.Tensor
            The time derivative of `x`.
        t : float or torch.Tensor
            The time variable.

        Returns
        -------
        torch.Tensor
            The computed Lagrangian of the system.
        """
        q, h = self.lagrangian_maps(x, y, t)
        return torch.sum(q * x_t, -1) - h

    def euler_lagrange_maps(self, x, y, t):
        """
        Computes the derivatives of the one-form and Hamiltonian with respect to
        the input variables `x` and `y`, and returns them as tuples of partial
        derivatives.

        Parameters
        ----------
        x : array-like
            The first part of the coordinates of the system. The position in the
            canonical case.
        y : array-like
            The second part of the coordinates, w.r.t. which the Jacobian of the
            one-form is invertible. The momentum in the canonical case.
        t : float or array-like
            The time variable.

        Returns
        -------
        dq : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the Jacobian of the one-form with respect to `x` and `y`.
        dh : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the gradient of the Hamiltonian with respect to `x` and `y`.
        """
        dx_q, dy_q = jacrev(self.oneform, (0, 1))(x, y)
        dx_h, dy_h = grad(self.hamiltonian, (0, 1))(x, y, t)
        return (dx_q, dy_q), (dx_h, dy_h)

    def vector_field(self, z, t):
        """
        Computes the vector field for the given state and time, solving a linear
        system from the derivation of the Lagrangian (the Euler-Lagrange equations),
        which can be expressed as
        1. (Dy q).T @ dt_x - grad_y H = 0
        2. (Dx q).T @ dt_x - grad_x H = (Dx q) @ dt_x + (Dy q) @ dt_y

        Parameters
        ----------
        z : torch.Tensor
            The state tensor, which is expected to have two components
            (position and momentum) concatenated along the last dimension.
        t : torch.Tensor or float
            The current time.

        Returns
        -------
        torch.Tensor
            The time derivatives of the state tensor, concatenated along the
            last dimension. The result has the same shape as the input `z`.

        Notes
        -----
        The method uses `torch.linalg.solve` to solve linear systems for the time derivatives.
        """
        x, y = torch.tensor_split(z, 2, -1)
        (dx_q, dy_q), (dx_h, dy_h) = self.euler_lagrange_maps(x, y, t)

        dt_x = torch.linalg.solve(dy_q.T, dy_h)

        y_rhs = (dx_q.T - dx_q) @ dt_x
        dt_y = torch.linalg.solve(dy_q, y_rhs - dx_h)

        return torch.cat((dt_x, dt_y), -1)


class AbstractCanonicalHamiltonian(AbstractDegenLagrangian):
    """
    AbstractCanonicalHamiltonian is an abstract class for defining canonical Hamiltonian problems of the form:
        dq / dt = H_p(q, p, t)
        dp / dt = -H_q(q, p, t)
    where:
    - `q` are the positions,
    - `p` are the momenta,
    - `H(q, p, t)` is the Hamiltonian (scalar quantity), with differentials `H_q` and `H_p`.

    See `AbstractDegenLagrangian` for methods and attributes.
    """

    def oneform(self, q: torch.Tensor, p: torch.Tensor):
        return p

    def vector_field(self, z, t):
        q, p = torch.tensor_split(z, 2, -1)
        dq_h, dp_h = grad(self.hamiltonian, (0, 1))(q, p, t)
        dt_q, dt_p = dp_h, -dq_h
        return torch.cat((dt_q, dt_p), -1)


class NeuralDegenLagrangian(AbstractDegenLagrangian, nn.Module):
    """
    A neural network-based implementation of the AbstractDegenLagrangian class.
    This class combines the functionality of a neural network with the structure
    of a Lagrangian system.

    Parameters
    ----------
    dim : int
        The dimension of the system.
    oneform : nn.Module
        A neural network module representing the one-form.
    hamiltonian : nn.Module
        A neural network module representing the Hamiltonian.
    """

    def __init__(self, dim, oneform_nn, hamiltonian_nn):
        AbstractDegenLagrangian.__init__(self, dim)
        nn.Module.__init__(self)
        self.oneform_nn = oneform_nn
        self.hamiltonian_nn = hamiltonian_nn

    def oneform(self, x, y):
        """
        Computes the one-form using the neural network module.

        Parameters
        ----------
        x : torch.Tensor
            The first part of the coordinates of the system.
        y : torch.Tensor
            The second part of the coordinates of the system.

        Returns
        -------
        torch.Tensor
            The computed one-form.
        """
        return self.oneform_nn(x, y)

    def hamiltonian(self, x, y, t):
        """
        Computes the Hamiltonian using the neural network module.

        Parameters
        ----------
        x : torch.Tensor
            The first part of the coordinates of the system.
        y : torch.Tensor
            The second part of the coordinates of the system.
        t : float | torch.Tensor
            The time variable.

        Returns
        -------
        torch.Tensor
            The computed Hamiltonian.
        """
        return self.hamiltonian_nn(x, y, t)
