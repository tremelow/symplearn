import numpy as np

import torch
from torch import nn


class UniformNormalizer1d(nn.Module):
    """
    Scales and shifts the inputs so that every coordinate is in [0,1].

    Assumes that the input tensor is 1D.
    """
    # TODO? Add a "reverse" mode for it to be an "denormalizer"
    # could be useful for multi-scale vector fields

    def __init__(self, dim):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.scaling = nn.Parameter(torch.ones(dim), requires_grad=False)

    def set(self, in_min, in_max):
        """
        Set the minimum and maximum values which will go to 0 and 1, respectively.
        """
        self.shift.set_(-in_min)
        self.scaling.set_(1.0 / (in_max - in_min))

    def set_from_out(self, out_min, out_max):
        """
        Set the parameters so that the output will go from [0, 1] to [out_min, out_max].
        """
        self.scaling.set_(out_max - out_min)
        self.shift.set_(out_min / self.scaling)

    def forward(self, inputs):
        return (inputs + self.shift) * self.scaling
    
    def extra_repr(self):
        return f"shift={self.shift}, scaling={self.scaling}"


class PeriodicLayer(nn.Module):
    """
    Pre-processing layer which makes the input 2pi-periodic. For each index i,
    the input `x[i]` is mapped to `[cos(2π*xi + φi1), ..., cos(2π*xi + φir)]`.
    Every mapping is concatenated into a single one-dimensional vector.

    Source: https://ieeexplore.ieee.org/document/10227556
    """

    def __init__(self, in_dim, num_repeat=4):
        super().__init__()
        self.out_dim = in_dim * num_repeat
        self.num_repeat = num_repeat
        self.phi = nn.Parameter(torch.rand(self.out_dim) * 2.0 * np.pi)

    def forward(self, inputs):
        rep_in = torch.repeat_interleave(inputs, self.num_repeat, -1)
        cosines = torch.cos(rep_in + self.phi)
        return 0.5 * (1.0 + cosines)


class HalfPeriodicLayer(nn.Module):
    """
    Pre-processing layer which repeats and makes periodic inputwise, for each index i, maps `xi -> [cos(2π*xi/fi + φ1), ..., cos(2π*xi/ωi + φr)]`.
    If `fi == 0`, then this index is simply repeated and no shift or cosine is applied.

    If `num_repeat[i] == 0` for some index `i`, then the index is discarded.

    Source: https://ieeexplore.ieee.org/document/10227556
    """

    def __init__(self, periods, num_repeat):
        super().__init__()
        periods = np.array(periods)
        num_repeat = np.array(num_repeat)
        is_periodic = periods != 0
        self.num_periodic = is_periodic.sum().item()
        self.num_repeat = num_repeat

        self.perm = nn.Parameter(
            torch.tensor(np.argsort(True ^ is_periodic)), requires_grad=False
        )
        # self.pulsations = torch.tensor(2.0 * np.pi / periods[is_periodic])

        pulsations = (
            torch.ones(r, 1) * (2.0 * np.pi / T)
            # torch.ones(r) * (2.0 * np.pi / T)
            for (r, T) in zip(num_repeat[is_periodic], periods[is_periodic])
        )
        self.pulsations = torch.block_diag(*pulsations)

        repeats = [torch.ones(r, 1) for r in num_repeat[True ^ is_periodic]]
        # repeats = [torch.ones(r) for r in num_repeat[True ^ is_periodic]]
        self.repeats = torch.block_diag(*repeats)

        self.phi = nn.Parameter(torch.rand(sum(num_repeat[is_periodic])) * 2.0 * np.pi)
        self.out_dim = sum(num_repeat)

    def forward(self, inputs):
        # unchanged_dims = tuple(1 for _ in inputs.shape[:-1])
        z = inputs[..., self.perm]  # reorder inputs
        x, y = torch.tensor_split(
            z, (self.num_periodic,), -1
        )  # split between periodic and non-periodic variables
        puls_x = torch.einsum("ij,...j->...i", self.pulsations, x)
        # puls_x = x @ self.pulsations
        cos_x = torch.cos(puls_x + self.phi)
        repeat_y = torch.einsum("ij,...j->...i", self.repeats, y)
        # repeat_y = y @ self.repeats
        return torch.cat((0.5 * (1.0 + cos_x), repeat_y), dim=-1)
        # return torch.cat((cos_x, repeat_y), dim=-1)
