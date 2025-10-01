import os

import numpy as np
from scipy.stats import qmc
from scipy.optimize import root

import torch

from symplearn.numerics import QuasiExactSimulation

from models import LotkaVolterra

DEFAULT_SEED = 20230926
TIMESTEP = 0.1


def bounding_box(model, h_max):
    a, b, c, d = model.a, model.b, model.c, model.d
    x_eq, y_eq = c / d, a / b

    # find the extremal values x such that H(x,y_eq) = h_max
    ham = lambda x, y: d * x - c * np.log(x) + b * y - a * np.log(y)
    g_x = lambda x: ham(x, y_eq) - h_max
    x_min_init = np.exp(-(h_max + b * y_eq - a * np.log(y_eq)) / c) # x ≈ 0
    x_min = root(g_x, x_min_init).x[0] * (1.0 - 1e-6)  # safety factor
    x_max_init = (h_max + c * np.log(x_eq) - b * y_eq + a * np.log(y_eq)) / d
    x_max = root(g_x, x_max_init).x[0] * (1.0 + 1e-6)

    # same for y, H(x_eq, y) = h_max
    g_y = lambda y: ham(x_eq, y) - h_max
    y_min_init = np.exp(-(h_max + d * x_eq - c * np.log(x_eq)) / b) # y ≈ 0
    y_min = root(g_y, y_min_init).x[0] * (1.0 - 1e-6)  # safety factor
    y_max_init = (h_max + a * np.log(y_eq) - d * x_eq + c * np.log(x_eq)) / b
    y_max = root(g_y, y_max_init).x[0] * (1.0 + 1e-6)

    return [x_min, y_min], [x_max, y_max]


def generate_initial_conditions(
    model: LotkaVolterra,
    n_init: int,
    seed=DEFAULT_SEED,
    h_min=0.0,
    h_max=4.4,
):
    """
    Generate initial conditions for a massless charged particle model.
    This function generates `n` initial conditions for a massless charged particle
    using Latin Hypercube sampling within a specified energy range. The initial
    conditions are filtered to ensure they satisfy the given energy constraints.

    Parameters
    ----------
    model : LotkaVolterra
        The massless charged particle model for which the initial conditions
        are generated. The model must have an `e0` attribute and a `hamiltonian`
        method.
    n_init : int
        The number of initial conditions to generate. Defaults to 5000.
    seed : int, optional
        Seed for the random number generator used in Latin Hypercube sampling.
        Defaults to `DEFAULT_SEED`.
    h_min : float, optional
        The minimum energy value for the initial conditions. Must be less than 
        `h_max` and might not be reached. Defaults to 0.0.
    h_max : float, optional
        The maximum energy value for the initial conditions. Must be greater than
        `h_min` and less than or equal to 4.0. Defaults to 0.45.

    Returns
    -------
    t : torch.Tensor
        Tensor containing initial times (all zeros).
    z : torch.Tensor
        Tensor containing the initial conditions.


    Notes
    -----
    The function uses a bounding box to generate candidate
    initial conditions, which are then filtered based on the energy constraints
    defined by the model's Hamiltonian. If the number of valid initial conditions
    is insufficient, additional samples are generated until the required number
    is met.
    """

    rng = qmc.LatinHypercube(d=2, seed=seed)
    z_min, z_max = bounding_box(model, h_max)

    # generate data in the bounding box, and filter depending on energy
    for i in range(10):
        n = (i + 1) * n_init
        unif_u0 = qmc.scale(rng.random(n=n), z_min, z_max)
        all_x0, all_y0 = torch.from_numpy(unif_u0.T[..., None])

        all_h0 = model.hamiltonian(all_x0, all_y0, None)
        mask = (h_min <= all_h0) & (all_h0 <= h_max)
        if mask.sum() >= n_init:
            x0, y0 = all_x0[mask][:n_init], all_y0[mask][:n_init]
            t0 = torch.zeros(x0.shape[0])
            return t0, torch.cat((x0, y0), dim=-1)

    # if we are here, we didn't find enough initial conditions
    raise RuntimeError("Unable to generate enough initial conditions")


def generate_snapshots(z0, t0, dt, nt=5, batch_size=500):
    simul = QuasiExactSimulation(model, dt)
    t, z = torch.empty(0, nt + 1), torch.empty(0, nt + 1, model.dim)
    for z0_batch in z0.split(batch_size):
        t_batch, z_batch = simul.simulate(z0_batch, nt)
        t = torch.cat((t, t_batch), dim=0)
        z = torch.cat((z, z_batch), dim=0)

    return t, z


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    os.makedirs("data", exist_ok=True)
    model = LotkaVolterra()

    n_init = 2500
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
