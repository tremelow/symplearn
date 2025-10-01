import os

import numpy as np
import matplotlib.pyplot as plt

import torch

from symplearn.datasets import VectorFieldDataset, SnapshotDataset

from models import MasslessChargedParticle, NeuralBaseMCP, NeuralSympMCP


def get_reduced_val_data():
    model = MasslessChargedParticle()
    val_data = VectorFieldDataset("val")

    z_all = val_data.z
    x_all, y_all = z_all.tensor_split(2, -1)
    h_all = model.hamiltonian(x_all, y_all, None)

    idx_sort = torch.argsort(h_all)
    n = len(z_all)
    idx = -(n // np.array([2, 3, 6]))
    # idx = -(n // np.arange(2, 7))
    z_sorted = z_all[idx_sort]
    z0 = z_sorted[idx]

    t0 = torch.zeros_like(z0)[:, 0]

    data_dvi = SnapshotDataset("val")
    dt = data_dvi.dt
    return t0, z0, dt


def get_extreme_val_data():
    model = MasslessChargedParticle()
    val_data = VectorFieldDataset("val")

    z_all = val_data.z
    x_all, y_all = z_all.tensor_split(2, -1)
    h_all = model.hamiltonian(x_all, y_all, None)

    idx_sort = torch.argsort(h_all)
    n = len(z_all)
    idx = [-(n // 4), -(n // 5), -10, -4, -1]
    z_sorted = z_all[idx_sort]
    z0 = z_sorted[idx]

    t0 = torch.zeros_like(z0)[:, 0]

    data_dvi = SnapshotDataset("val")
    dt = data_dvi.dt
    return t0, z0, dt



def load_models():
    model = MasslessChargedParticle()

    model_vf_base = NeuralBaseMCP()
    model_vf_base.load_state_dict(
        torch.load(os.path.join("nn", "baseline.pt"), weights_only=True)
    )
    model_vf_base.eval()

    model_vf_reg = NeuralSympMCP()
    model_vf_reg.load_state_dict(
        torch.load(os.path.join("nn", "vf_reg.pt"), weights_only=True)
    )
    model_vf_reg.eval()

    model_vf_no_reg = NeuralSympMCP()
    model_vf_no_reg.load_state_dict(
        torch.load(os.path.join("nn", "vf_no_reg.pt"), weights_only=True)
    )
    model_vf_no_reg.eval()

    model_dvi = NeuralSympMCP()
    model_dvi.load_state_dict(
        torch.load(os.path.join("nn", "dvi.pt"), weights_only=True)
    )
    model_dvi.eval()

    return {
        "ref": model,
        "vf_base": model_vf_base,
        "vf_reg": model_vf_reg,
        "vf_no_reg": model_vf_no_reg,
        "dvi": model_dvi,
    }

def init_env():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")

    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9

    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE, title_fontsize=BIGGER_SIZE)  # legend fontsize

    pb = MasslessChargedParticle()
    models = load_models()
    return pb, models
