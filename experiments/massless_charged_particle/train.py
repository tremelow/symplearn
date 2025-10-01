import os

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from symplearn.datasets import VectorFieldDataset, SnapshotDataset
from symplearn.training.norms import GramMSNorm
from symplearn.training import (
    VectorFieldLoss,
    EulerDVINewtonLoss,
    NaiveStepperLoss,
    Trainer,
)

import models

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
        model = models.NeuralBaseMCP()
    else:
        if reg_weight == 0:
            name = "vf_no_reg"
        else:
            name = "vf_reg"
        model = models.NeuralSympMCP()

    train_data = VectorFieldDataset("train")
    bounds_z, _, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = VectorFieldDataset("test")

    norm_fn = GramMSNorm(abs_weight=1e-1)
    loss_fn = VectorFieldLoss(model, reg_weight=reg_weight, loss_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, name)


def train_dvi():
    model = models.NeuralSympMCP()

    train_data = SnapshotDataset("train")
    bounds_z, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = SnapshotDataset("test")

    norm_fn = GramMSNorm(abs_weight=1e-1)
    loss_fn = EulerDVINewtonLoss(model, train_data.dt, norm_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, "dvi")


def train_stepper():
    model = models.NeuralStepMCP()

    train_data = SnapshotDataset("train")
    bounds_z, _ = train_data.bounds()
    model.norm.set(*bounds_z)
    test_data = SnapshotDataset("test")

    norm_fn = GramMSNorm(abs_weight=1e-1)
    loss_fn = NaiveStepperLoss(model, train_data.dt, norm_fn=norm_fn)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    default_train(model, trainer, "step")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # perform every training sequentially, resetting the random seed
    # every time for reproducibility

    # torch.manual_seed(DEFAULT_SEED)
    # print("VF learning with regularization")
    # train_vf(1e-4)
    # print()

    # torch.manual_seed(DEFAULT_SEED)
    # print("VF learning without regularization")
    # train_vf(0.0)
    # print()

    torch.manual_seed(DEFAULT_SEED)
    print("VF learning without structure")
    train_vf(None)
    print()

    # torch.manual_seed(DEFAULT_SEED)
    # print("DVI learning")
    # train_dvi()
    # print()

    # torch.manual_seed(DEFAULT_SEED)
    # print("Naive stepper learning")
    # train_stepper()
    # print()
