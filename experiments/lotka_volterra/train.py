import os

import torch

from symplearn.datasets import VectorFieldDataset, SnapshotDataset
from symplearn.training.norms import GramMSNorm
from symplearn.training import (
    VectorFieldLoss,
    EulerDVINewtonLoss,
    NaiveStepperLoss,
    Trainer,
)

from models import NeuralSympLV, NeuralBaseLV, NeuralStepLV


DEFAULT_SEED = 42


def default_train(model, trainer: Trainer, name: str):
    prefix = os.path.join("nn", name)
    traces = []

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    trace = trainer.train(20, opt, prefix)
    traces.append(trace)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trace = trainer.train(500, opt, prefix)
    traces.append(trace)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    trace = trainer.train(500, opt, prefix)
    traces.append(trace)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    trace = trainer.train(500, opt, prefix)
    traces.append(trace)

    train_keys = traces[0]["train"].keys()
    test_keys = traces[0]["train"].keys()
    trace = {
        "train": {
            key: sum([t["train"][key] for t in traces], start=[]) for key in train_keys
        },
        "test": {
            key: sum([t["test"][key] for t in traces], start=[]) for key in test_keys
        },
        "time": [t["time"] for t in traces],
    }
    torch.save(trace, prefix + "_trace.pt")


def train_vf(reg_weight):
    if reg_weight is None:
        name = "baseline"
        model = NeuralBaseLV()
    else:
        if reg_weight == 0:
            name = "vf_no_reg"
        else:
            name = "vf_reg"
        model = NeuralSympLV()

    train_data = VectorFieldDataset("train")
    bounds_z, _, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = VectorFieldDataset("test")

    r = 1e-2
    norm_fn = GramMSNorm(abs_weight=r / (1.0 + r), rel_weight=1.0 / (1.0 + r))
    loss_fn = VectorFieldLoss(model, reg_weight=reg_weight, loss_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, name)


def train_dvi():
    model = NeuralSympLV()

    train_data = SnapshotDataset("train")
    bounds_z, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = SnapshotDataset("test")

    norm_dt_z = torch.diff(train_data.z, dim=1).square().mean(-1)
    r = (torch.quantile(norm_dt_z, 1e-2) / torch.quantile(norm_dt_z, 0.99)).item()
    norm_fn = GramMSNorm(abs_weight=r / (1.0 + r), rel_weight=1.0 / (1.0 + r))
    loss_fn = EulerDVINewtonLoss(model, train_data.dt, norm_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, "dvi")


def train_stepper():
    model = NeuralStepLV()

    train_data = SnapshotDataset("train")
    bounds_z, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = SnapshotDataset("test")

    norm_dt_z = torch.diff(train_data.z, dim=1).square().mean(-1)
    r = torch.quantile(norm_dt_z, 1e-2) / torch.quantile(norm_dt_z, 0.99)
    norm_fn = GramMSNorm(abs_weight=r / (1.0 + r), rel_weight=1.0 / (1.0 + r))
    loss_fn = NaiveStepperLoss(model, train_data.dt, norm_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, "step")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # perform every training sequentially, resetting the random seed 
    # every time for reproducibility

    torch.manual_seed(DEFAULT_SEED)
    print("VF learning with regularization")
    train_vf(1e-6)
    print()

    torch.manual_seed(DEFAULT_SEED)
    print("VF learning without regularization")
    train_vf(0.0)
    print()

    torch.manual_seed(DEFAULT_SEED)
    print("VF learning without structure")
    train_vf(None)
    print()

    torch.manual_seed(DEFAULT_SEED)
    print("DVI learning")
    train_dvi()
    print()

    # torch.manual_seed(DEFAULT_SEED)
    # print("Naive stepper learning")
    # train_stepper()
    # print()
