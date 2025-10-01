import numpy as np

import torch
from torch import nn

from symplearn.dynamics import (
    AbstractDiffEq,
    AbstractDegenLagrangian,
    AbstractCanonicalHamiltonian,
)
from symplearn.networks import create_mlp

from models import DIMENSION, LotkaVolterra


class NeuralSympLV(AbstractDegenLagrangian, nn.Module):
    def __init__(self):
        AbstractDegenLagrangian.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.oneform_nn = create_mlp(DIMENSION, DIMENSION // 2, [30] * 2)
        self.hamiltonian_nn = create_mlp(DIMENSION, 1, [30] * 2)

    def oneform(self, x, y):
        z = torch.cat((x, y), -1)
        return self.oneform_nn(z)

    def hamiltonian(self, x, y, t):
        z = torch.cat((x, y), -1)
        return self.hamiltonian_nn(z).sum(-1)


class NeuralHamLV(AbstractCanonicalHamiltonian, nn.Module):
    def __init__(self):
        AbstractCanonicalHamiltonian.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.hamiltonian_nn = create_mlp(DIMENSION, 1, [40] * 2)

    def hamiltonian(self, x, y, t):
        z = torch.cat((x, y), -1)
        return self.hamiltonian_nn(z).sum(-1)


class NeuralBaseLV(AbstractDiffEq, nn.Module):
    def __init__(self):
        AbstractDiffEq.__init__(self, DIMENSION)
        nn.Module.__init__(self)
        self.vector_field_nn = create_mlp(DIMENSION, DIMENSION, [40] * 2)

    def vector_field(self, z, t):
        return self.vector_field_nn(z)
