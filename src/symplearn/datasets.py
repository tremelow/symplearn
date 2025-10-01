import os

import torch
from torch import nn

from scipy.stats.qmc import LatinHypercube
from scipy.integrate import solve_ivp

from .dynamics import AbstractDegenLagrangian
from torch.utils.data import Dataset, BatchSampler, SequentialSampler


class SnapshotDataset(Dataset):
    def __init__(self, split, dir="data"):
        path = os.path.join(dir, f"snapshot-{split}.pt")
        data = torch.load(path, weights_only=True)
        self.z = data["z"]
        self.t = data["t"]
        self.num_traj, nt = self.t.shape
        self.num_steps = nt - 2
        self.dt = data["dt"]

    def __len__(self):
        return self.num_traj * self.num_steps

    def __getitem__(self, idx):
        i, j = divmod(idx, self.num_steps)
        z0, z1, z2 = self.z[i, j : j + 3]
        return z0, z1, z2, self.t[i, j+1]

    def bounds(self):
        min_z = self.z.min(0).values.min(0).values
        max_z = self.z.max(0).values.max(0).values
        min_t = self.t.min(0).values.min(0).values
        max_t = self.t.max(0).values.max(0).values
        return (min_z, max_z), (min_t, max_t)

    def velocity_bounds(self):
        dt_z = torch.diff(self.z, dim=1) / self.dt
        min_dt_z = dt_z.min(0).values.min(0).values
        max_dt_z = dt_z.max(0).values.max(0).values
        return min_dt_z, max_dt_z


class VectorFieldDataset(Dataset):
    def __init__(self, split, dir="data"):
        path = os.path.join(dir, f"vf-{split}.pt")
        data = torch.load(path, weights_only=True)
        self.z = data["z"]
        self.t = data["t"]
        self.dz_dt = data["dz_dt"]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.t[idx], self.dz_dt[idx]

    def bounds(self):
        range_z = self.z.min(0).values, self.z.max(0).values
        range_t = self.t.min(0).values, self.t.max(0).values
        range_dt_z = self.dz_dt.min(0).values, self.dz_dt.max(0).values
        return range_z, range_t, range_dt_z


def generate_datasets(
    model: AbstractDegenLagrangian,
    z_init: torch.Tensor,
    dt: float,
    *,
    nt=2,
    offset_init=False,
    seed=None,
    batch_sim=2000,
    sim_params={"atol": 1e-12, "rtol": 1e-12},
):
    """
    Generate datasets for training and evaluation of a model based on its vector field.
    This function generates trajectories of a dynamical system modeled by the given
    `AbstractDegenLagrangian` instance. It supports splitting the dataset into three
    parts with different time-step offsets and returns two datasets: one for snapshots
    and another for vector field evaluations.

    Arguments
    ---------
    model : AbstractDegenLagrangian
        The model representing the dynamical system.
    z_init : torch.Tensor
        Initial conditions for the trajectories.
    dt : float
        Time step for the simulation.
    nt : int, optional
        Number of time steps for the trajectories. Defaults to 2.
    offset_init : bool, optional
        Whether to offset the initial conditions using
        Latin Hypercube sampling. Defaults to False.
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    init_params : dict, optional
        Parameters for generating initial conditions.
        Defaults to {"device": "cpu"}.
    batch_sim : int, optional
        Batch size for trajectory simulation. If set to 0,
        all trajectories are simulated at once. Defaults to 2000.
    sim_params : dict, optional
        Parameters for the simulation solver, such as
        tolerances. Defaults to {"atol": 1e-12, "rtol": 1e-12}.

    Returns
    -------
    SnapshotDataset
        A dataset containing snapshots of the system's state at
        different time steps.
    VectorFieldDataset
        A dataset containing the system's state and the
        corresponding vector field evaluations.
    """

    n_init = len(z_init)
    tf = nt * dt

    if offset_init:
        tf_init = tf * LatinHypercube(d=1, seed=seed).random(n_init)[:, 0]
        seed = 242210 if seed is None else seed
        for n, (z0, tf0) in enumerate(zip(z_init, tf_init)):
            z0 = z0.cpu().detach().numpy()
            z0_offset = solve_ivp(model.vector_field, (0.0, tf0), z0).y[:, -1]
            z_init[n, :] = torch.tensor(z0_offset)

    if batch_sim > 0:
        z_traj = torch.empty(n_init, nt + 1, model.z_dim)
        for idx in BatchSampler(
            SequentialSampler(torch.arange(n_init)), batch_sim, False
        ):
            _, batch_z_traj = model.generate_trajectories(
                z_init[idx], nt * dt, nt, **sim_params
            )
            z_traj[idx] = batch_z_traj
    else:
        _, z_traj = model.generate_trajectories(z_init, tf, nt, **sim_params)

    # split dataset in 3 parts, each with a different offset of 0, 1 or 2 time-steps
    z_p0, z_p1, z_p2 = z_traj.tensor_split(3)

    def split_three(z, i):
        z0 = z[:, i:-2:3, :].flatten(end_dim=-2)
        z1 = z[:, (i + 1) : -1 : 3, :].flatten(end_dim=-2)
        z2 = z[:, (i + 2) :: 3, :].flatten(end_dim=-2)
        return z0, z1, z2

    z0_p0, z1_p0, z2_p0 = split_three(z_p0, 0)
    z0_p1, z1_p1, z2_p1 = split_three(z_p1, 1)
    z0_p2, z1_p2, z2_p2 = split_three(z_p2, 2)

    z0 = torch.cat((z0_p0, z0_p1, z0_p2), 0)
    z1 = torch.cat((z1_p0, z1_p1, z1_p2), 0)
    z2 = torch.cat((z2_p0, z2_p1, z2_p2), 0)

    #         z0, z1, z2 = z_traj[:, :-2:3, :], z_traj[:, 1:-1:3, :], z_traj[:, 2::3, :]

    #         actual_nt = int(z2.shape[1])
    #         z0 = z0[:, :actual_nt, :]
    #         z1 = z1[:, :actual_nt, :]

    #         z0, z1, z2 = z0.flatten(end_dim=-2), z1.flatten(end_dim=-2), z2.flatten(end_dim=-2)
    n_data = len(z2)
    dz0_dt = model.vector_field(z0)

    return SnapshotDataset(z0, z1, z2, dt), VectorFieldDataset(z0, dz0_dt)
