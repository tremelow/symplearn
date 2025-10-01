import torch
from torch import nn


class STanh(nn.Module):
    """
    Scaled Tanh activation function.
    This activation function is defined as:
        f(x) = (1 + scaling * x) * tanh(x)
    where scaling is a learnable parameter.
    """

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        scaling_val = 1.0  # 1.0 / np.tanh(1.0) - 1.0
        self.scaling = nn.Parameter(scaling_val * torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (1.0 + self.scaling * inputs) * torch.tanh(inputs)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


def str_to_activation(name: str) -> nn.Module:
    """
    Converts a string representation of an activation function to the corresponding PyTorch activation function.

    Parameters
    ----------
    name : str
           The name of the activation function.

    Returns
    -------
    nn.Module
        The PyTorch activation function.

    Raises
    ------
    KeyError
        If the given name does not match any supported activation function.
    """
    key = name.lower()
    nonlinearities = {
        "elu:": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "stanh": STanh,
        "stan": STanh,
    }

    if key not in nonlinearities.keys():
        raise KeyError(f"Could not match '{name}' to an activation function.")
    return nonlinearities[key]


def create_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int] = [],
    activation: str = "stanh",
    final_bias=False,
    **kwargs,
) -> nn.Sequential:
    """
    Creates a multi-layer perceptron (MLP) neural network model.

    Parameters
    ----------
    in_dim : int
        The size of the input layer.
    out_dim : int
        The size of the output layer.
    hidden_dims : list[int], optional
        A list of integers specifying the sizes of the hidden layers. Default is an empty list.
    activation : str, optional
        The activation function to use. Default is "stanh".
    final_bias : bool, optional
        Whether to include a bias term in the final layer. Default is False.
    **kwargs: dict, optional
        Additional keyword arguments for the activation function.

    Returns
    -------
    nn.Sequential
        The MLP as a sequential model.
    """

    in_dims = [in_dim] + hidden_dims[:-1]
    out_dims = hidden_dims
    activ = str_to_activation(activation)
    layers = []
    for d0, d1 in zip(in_dims, out_dims):
        layers.append(nn.Linear(d0, d1))

        if activ == STanh:
            layers.append(activ(d1, **kwargs))

        else:
            layers.append(activ(**kwargs))

    layers.append(nn.Linear(in_dims[-1], out_dim, bias=final_bias))
    return nn.Sequential(*layers)
