import os

import numpy as np
from scipy.stats import qmc

import torch

torch.set_default_dtype(torch.float64)

from symplearn.numerics import QuasiExactSimulation

from models import GuidingCenter

DEFAULT_SEED = 242210
TIMESTEP = 37974.6 / 20  # 20 points per period


def generate_initial_conditions(
    model,
    n_data,
    *,
    seed=242210,
    th_range=[-0.1 * np.pi, 0.1 * np.pi],
    phi_range=[0.0, 2 * np.pi],
    r_range=[0.03, 0.055],
    u_min=[-9e-4, -3e-4],
) -> torch.Tensor:
    r_sq_range = np.array(r_range) ** 2
    z_range = np.array([th_range, phi_range, r_sq_range, u_min]).T
    sampler = qmc.LatinHypercube(d=model.dim, seed=seed)
    z0 = qmc.scale(sampler.random(n=n_data), *z_range)
    z0[:, 2] = np.sqrt(z0[:, 2])  # from r^2 to r
    return torch.zeros(n_data), torch.tensor(z0)


# def generate_initial_conditions_split(
#     model,
#     n_data,
#     *,
#     seed=242210,
#     th_range1=[-0.1 * np.pi, 0.1 * np.pi],
#     r_range1=[0.03, 0.05],
#     u_range1=[-7.5e-4, -2.5e-4],
#     th_range2=[0.9 * np.pi, 1.1 * np.pi],
#     r_range2=[0.055, 0.075],
#     u_range2=[-4e-4, -1e-4],
#     ratio1=0.995,
# ) -> torch.Tensor:
#     n1 = int(np.ceil(ratio1 * n_data))
#     n2 = n_data - n1
#     phi_range = [0.0, 2 * np.pi]
#     r_range1, r_range2 = np.array(r_range1) ** 2, np.array(r_range2) ** 2
#     # first half
#     a_range1 = np.array([th_range1, phi_range, r_range1, u_range1]).T
#     # a_min1, a_max1 = [th_min1, phi_min, r_min1**2, u_min1], [th_max1, phi_max, r_max1**2, u_max1]
#     sampler = qmc.LatinHypercube(d=model.dim, seed=seed)
#     z1 = qmc.scale(sampler.random(n=n1), *a_range1)

#     # second half
#     a_range2 = np.array([th_range2, phi_range, r_range2, u_range2]).T
#     # a_min2, a_max2 = [th_min2, phi_min, r_min2**2, u_min2], [th_max2, phi_max, r_max2**2, u_max2]
#     sampler = qmc.LatinHypercube(d=model.dim, seed=seed)
#     z2 = qmc.scale(sampler.random(n=n2), *a_range2)

#     z0 = np.concatenate((z1, z2), 0)
#     z0[:, 2] = np.sqrt(z0[:, 2])  # from r^2 to r

#     np.random.default_rng(seed=seed).shuffle(z0)

#     return torch.zeros(n_data), torch.tensor(z0)


def generate_snapshots(model, z0, t0, dt, nt=60, batch_size=500):
    simul = QuasiExactSimulation(model, dt)
    t, z = torch.empty(0, nt + 1), torch.empty(0, nt + 1, model.dim)
    for z0_batch, t0_batch in zip(z0.split(batch_size), t0.split(batch_size)):
        t_batch, z_batch = simul.simulate(z0_batch, nt)
        t_batch = t_batch + t0_batch.unsqueeze(-1)
        t = torch.cat((t, t_batch), dim=0)
        z = torch.cat((z, z_batch), dim=0)

    return t, z


if __name__ == "__main__":
    n_init = 600
    os.makedirs("data", exist_ok=True)
    # 80% train, 15% test, 5% validation
    train_idx = np.arange(int(0.8 * n_init))
    test_idx = np.arange(int(0.8 * n_init), int(0.95 * n_init))
    val_idx = np.arange(int(0.95 * n_init), n_init)
    splits = {"train": train_idx, "test": test_idx, "val": val_idx}

    model = GuidingCenter()

    t0, z0 = generate_initial_conditions(model, n_init)
    dt = TIMESTEP
    t, z = generate_snapshots(model, z0, t0, dt)

    for key in splits:
        z_set = z[splits[key]]
        t_set = t[splits[key]]
        snapshot = {"z": z_set, "t": t_set, "dt": dt}
        torch.save(snapshot, os.path.join("data", f"snapshot-{key}.pt"))

    for key in splits:
        z0_set, z1_set, z2_set = z[splits[key]].tensor_split(3, dim=0)
        t0_set, t1_set, t2_set = t[splits[key]].tensor_split(3, dim=0)
        z0_set, t0_set = z0_set[:, :-2:3], t0_set[:, :-2:3]
        z1_set, t1_set = z1_set[:, 1:-1:3], t1_set[:, 1:-1:3]
        z2_set, t2_set = z2_set[:, 2::3], t2_set[:, 2::3]

        z_set = torch.cat((z0_set, z1_set, z2_set), dim=0).reshape(-1, model.dim)
        t_set = torch.cat((t0_set, t1_set, t2_set), dim=0).reshape(-1)
        dt_z_set = model.vector_field(z_set, t_set)

        vf = {"z": z_set, "t": t_set, "dz_dt": dt_z_set}
        torch.save(vf, os.path.join("data", f"vf-{key}.pt"))
