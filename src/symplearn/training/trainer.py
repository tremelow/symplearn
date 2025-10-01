import time

import torch
from torch.utils.data import Dataset, DataLoader


class Trainer:

    def __init__(
        self, train_data: Dataset, test_data: Dataset, loss_fun, batch_size=100, rng=None
    ):
        if rng is None:
            rng = torch.Generator().manual_seed(42)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=rng
        )
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.loss_fun = loss_fun

    def epoch_loss(self, dataloader, optimizer=None, scheduler=None):
        avg_loss, avg_vals = 0.0, {}
        for batch_data in dataloader:
            batch_loss, batch_trace = self.loss_fun(*batch_data)
            if optimizer is not None:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            weight = len(batch_data[0]) / len(dataloader.dataset)
            avg_loss += weight * batch_loss.item()
            for key in batch_trace:
                avg_key = avg_vals.get(key, 0.0)
                avg_vals[key] = avg_key + weight * batch_trace[key]

        if scheduler is not None:
            scheduler.step()

        return avg_loss, avg_vals

    def train(
        self,
        n_epochs,
        optimizer,
        save_prefix,
        scheduler=None,
        show_every=50,
    ):
        if show_every is None:
            show_every = n_epochs + 1
        time0 = time.time()

        model = self.loss_fun.model
        torch.save(model.state_dict(), save_prefix + ".pt")

        min_loss, init_trace = self.epoch_loss(self.test_loader)
        trace = {
            "train": {key: [] for key in init_trace},
            "test": {key: [] for key in init_trace},
        }
        for epoch in range(n_epochs):
            model.train()
            _, train_vals = self.epoch_loss(
                self.train_loader, optimizer=optimizer, scheduler=scheduler
            )
            for key in train_vals:
                trace["train"][key].append(train_vals[key])

            model.eval()
            test_loss, test_vals = self.epoch_loss(self.test_loader)

            for key in test_vals:
                trace["test"][key].append(test_vals[key])

            print_epoch = False
            time_epoch = time.time() - time0
            print_out = f"Epoch {epoch+1:4d} / {n_epochs} ({time_epoch:.2f}s elapsed)"

            if test_loss < min_loss:
                print_out = print_out + " -- SAVED"
                print_epoch = True

                min_loss = test_loss
                torch.save(model.state_dict(), save_prefix + ".pt")

            if print_epoch or (epoch + 1) % show_every == 0:
                print(print_out)
                train_out = ", ".join([f"{train_vals[key]:.2e} ({key})" for key in train_vals])
                test_out = ", ".join([f"{test_vals[key]:.2e} ({key})" for key in test_vals])
                print(f"train : {train_out}")
                print(f"test :  {test_out}")
                print()

        model.load_state_dict(torch.load(save_prefix + ".pt", weights_only=True))

        training_duration = time.time() - time0
        print("Elapsed time:", training_duration, "s")
        trace["time"] = training_duration
        return trace
