import os

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from symplearn.datasets import VectorFieldDataset, SnapshotDataset
from symplearn.training.norms import GramMSNorm, DiagMSNorm, ScaledMSNorm, EuclideanLogCondWithDet
from symplearn.training import (
    VectorFieldLoss,
    EulerDVINewtonLoss,
    NaiveStepperLoss,
    ImexEulerLoss,
    Trainer,
)

import models

DEFAULT_SEED = 42


def assemble_traces(traces):
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
    return trace


def trainings(
    model,
    trainer: Trainer,
    model_save_path: str,
    num_epochs: list[int] = [20, 500, 500],
    learning_rates: list[float] = [1e-2, 1e-3, 1e-4],
):
    traces = []
    for n, lr in zip(num_epochs, learning_rates):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        trace = trainer.train(n, opt, model_save_path)
        traces.append(trace)

    return assemble_traces(traces)


def train_vf(reg_weight, gram=True):
    training_params = {}
    loss_params = dict(reg_weight=reg_weight, loss_fn=GramMSNorm(gram_eps=1e-12))

    train_data = VectorFieldDataset("train")
    bounds_z, _, bounds_dt_z = train_data.bounds()
    test_data = VectorFieldDataset("test")
    
    if reg_weight is None and gram:
        model = models.NeuralBaseGC()
        name = "baseline"
    else:
        model = models.NeuralSympGC()
        if not gram:
            name = "vf_no_gram"
            loss_params["loss_fn"] = ScaledMSNorm()
        else:
            name = "vf_no_reg" if reg_weight == 0.0 else "vf_reg"

    model_prefix = os.path.join("nn", name)
    model.set_norm(bounds_z, bounds_dt_z)

    loss_fn = VectorFieldLoss(model, **loss_params)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    trace = trainings(model, trainer, model_prefix, **training_params)
    torch.save(trace, model_prefix + "_trace.pt")


def train_dvi():
    model = models.NeuralSympGC()
    model_prefix = os.path.join("nn", "dvi")

    train_data = SnapshotDataset("train")
    bounds_z, _ = train_data.bounds()
    bounds_dt_z = train_data.velocity_bounds()
    model.set_norm(bounds_z, bounds_dt_z)
    test_data = SnapshotDataset("test")

    imex_train = dict(num_epochs=[20, 500], learning_rates=[1e-2, 1e-3])
    loss_fn = ImexEulerLoss(model, train_data.dt)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    trace_imex = trainings(model, trainer, model_prefix, **imex_train)

    dvi_train = dict(num_epochs=[500], learning_rates=[1e-4])
    loss_fn = EulerDVINewtonLoss(model, train_data.dt)
    trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
    trace_dvi = trainings(model, trainer, model_prefix, **dvi_train)

    trace = assemble_traces([trace_imex, trace_dvi])
    torch.save(trace, model_prefix + "_trace.pt")


# def train_stepper():
#     model = models.NeuralStepGC()
#     model_prefix = os.path.join("nn", "step")

#     train_data = SnapshotDataset("train")
#     bounds_z, _ = train_data.bounds()
#     model.set_norm(*bounds_z)
#     test_data = SnapshotDataset("test")

#     norm_fn = GramMSNorm(abs_weight=1.0)
#     loss_fn = NaiveStepperLoss(model, train_data.dt, norm_fn=norm_fn)
#     trainer = Trainer(train_data, test_data, loss_fn, batch_size=500)
#     trace = trainings(model, trainer, model_prefix)
#     torch.save(trace, model_prefix + "_trace.pt")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # perform every training sequentially, resetting the random seed
    # every time for reproducibility

    torch.manual_seed(DEFAULT_SEED)
    print("VF learning with regularization")
    train_vf(1.0)
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
    print("VF learning with structure but without Gram norm")
    train_vf(None, gram=False)
    print()

    torch.manual_seed(DEFAULT_SEED)
    print("DVI learning")
    train_dvi()
    print()

    # torch.manual_seed(DEFAULT_SEED)
    # print("Naive stepper learning")
    # train_stepper()
    # print()
