import os

import numpy as np
from scipy.stats import qmc

import torch

from symplearn.numerics import QuasiExactSimulation

from models import MasslessChargedParticle

DEFAULT_SEED = 242210
TIMESTEP = 0.5


def bounding_box(h_min, h_max):
    assert h_min < h_max, "h_min must be less than h_max"
    assert h_min >= 0.0, "h_min must be non negative"
    assert h_max <= 4.0, "h_max must be less than or equal to 4.0"

    r_max = np.arccos(1.0 - h_max)  # y = pi/2
    # x = y - pi/2
    if h_max > 2.0:
        r_max = max(np.pi, np.sqrt(2.0) * np.arccos(1.0 - 0.5 * h_max))
    r_min = np.sqrt(2.0) * np.arccos(1.0 - 0.5 * h_min)
    return [[-np.pi, r_min**2], [np.pi, r_max**2]]


def generate_initial_conditions(
    model: MasslessChargedParticle,
    n_init: int,
    seed=DEFAULT_SEED,
    h_min=0.0,
    h_max=1.5,
):
    """
    Generate initial conditions for a massless charged particle model.
    This function generates `n` initial conditions for a massless charged particle
    using Latin Hypercube sampling within a specified energy range. The initial
    conditions are filtered to ensure they satisfy the given energy constraints.

    Parameters
    ----------
    model : MasslessChargedParticle
        The massless charged particle model for which the initial conditions
        are generated. The model must have an `e0` attribute and a `hamiltonian`
        method.
    n_init : int
        The number of initial conditions to generate. Defaults to 5000.
    seed : int, optional
        Seed for the random number generator used in Latin Hypercube sampling.
        Defaults to `DEFAULT_SEED`.
    h_min : float, optional
        The minimum energy value for the initial conditions. Must be non-negative
        and less than `h_max`. Defaults to 0.0.
    h_max : float, optional
        The maximum energy value for the initial conditions. Must be greater than
        `h_min` and less than or equal to 4.0. Defaults to 1.5.

    Returns
    -------
    x0 : torch.Tensor
        Tensor containing the x-coordinates of the initial conditions.
    y0 : torch.Tensor
        Tensor containing the y-coordinates of the initial conditions.

    Raises
    ------
    AssertionError
        If `h_min` is not less than `h_max`.
        If `h_min` is negative.
        If `h_max` is greater than 4.0.

    Notes
    -----
    The function uses a bounding box in polar coordinates to generate candidate
    initial conditions, which are then filtered based on the energy constraints
    defined by the model's Hamiltonian. If the number of valid initial conditions
    is insufficient, additional samples are generated until the required number
    is met.
    """

    rng = qmc.LatinHypercube(d=2, seed=seed)
    box = bounding_box(h_min, h_max)

    # generate data in the bounding box, and filter depending on energy
    for i in range(10):
        n = (i + 1) * n_init
        unif_u0 = qmc.scale(rng.random(n=n), *box)
        all_th0, all_r0 = torch.from_numpy(unif_u0.T[..., None])
        all_r0 = torch.sqrt(all_r0)

        all_x0 = all_r0 * torch.cos(all_th0)
        all_y0 = all_r0 * torch.sin(all_th0) + 0.5 * np.pi

        all_h0 = model.hamiltonian(all_x0, all_y0, None)
        mask = (h_min <= all_h0) & (all_h0 <= h_max)
        if mask.sum() >= n_init:
            x0, y0 = all_x0[mask][:n_init], all_y0[mask][:n_init]
            t0 = torch.zeros(x0.shape[0])
            return t0, torch.cat((x0, y0), dim=-1)

    # if we are here, we didn't find enough initial conditions
    raise RuntimeError("Unable to generate enough initial conditions")


def generate_snapshots(z0, t0, dt, batch_size=500):
    simul = QuasiExactSimulation(model, dt)
    t, z = torch.empty(0), torch.empty(0, 3, model.dim)
    for z0_batch in z0.split(batch_size):
        t_batch, z_batch = simul.simulate(z0_batch, 2)
        t = torch.cat((t, t_batch), dim=0)
        z = torch.cat((z, z_batch), dim=0)

    return t, z


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    os.makedirs("data", exist_ok=True)
    model = MasslessChargedParticle()

    n_init = 15_000
    # 80% train, 15% test, 5% validation
    train_idx = np.arange(int(0.8 * n_init))
    test_idx = np.arange(int(0.8 * n_init), int(0.95 * n_init))
    val_idx = np.arange(int(0.95 * n_init), n_init)
    splits = {"train": train_idx, "test": test_idx, "val": val_idx}

    t0, z0 = generate_initial_conditions(model, n_init)
    dt = TIMESTEP
    t, z = generate_snapshots(z0, t0, dt)

    for key in splits:
        z_set = z[splits[key]]
        t_set = t[splits[key]]
        snapshot = {"z": z_set, "t": t_set, "dt": dt}
        torch.save(snapshot, os.path.join("data", f"snapshot-{key}.pt"))

    for key in splits:
        z0_set, z1_set, z2_set = z[splits[key]].tensor_split(3, dim=0)
        t0_set, t1_set, t2_set = t[splits[key]].tensor_split(3, dim=0)
        z0_set, t0_set = z0_set[:, :-2], t0_set[:, :-2]
        z1_set, t1_set = z1_set[:, 1:-1], t1_set[:, 1:-1]
        z2_set, t2_set = z2_set[:, 2:], t2_set[:, 2:]

        z_set = torch.cat((z0_set, z1_set, z2_set), dim=0).reshape(-1, model.dim)
        t_set = torch.cat((t0_set, t1_set, t2_set), dim=0).reshape(-1)
        z_set = z_set.reshape(-1, model.dim)
        t_set = t_set.reshape(-1)
        dt_z_set = model.vector_field(z_set, t_set)

        vf = {"z": z_set, "t": t_set, "dz_dt": dt_z_set}
        torch.save(vf, os.path.join("data", f"vf-{key}.pt"))
