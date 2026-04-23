import csv
import math
import os
import time

import torch

from .parallel import model_state_dict


def append_csv_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class Trainer:
    def __init__(
        self,
        net,
        num_epochs,
        device,
        batch_size=10000,
        lr=0.001,
        name="",
        K_min=1,
        K_max=55,
        all_moves=None,
        inverse_moves=None,
        V0=None,
        train_walkers_num=0,
        val_walkers_num=0,
    ):
        self.net = net.to(device)
        self.lr = lr
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epoch = 0
        self.id = int(time.time())
        self.log_dir = "logs"
        self.weights_dir = "weights"
        self.name = name
        self.K_min = K_min
        self.K_max = K_max
        default_train_walkers = max(1_000_000 // self.K_max, 1)
        self.walkers_num = default_train_walkers if int(train_walkers_num) <= 0 else int(train_walkers_num)
        self.val_walkers_num = max(int(val_walkers_num), 0)
        self.all_moves = all_moves
        self.n_gens = all_moves.size(0)
        self.inverse_moves = inverse_moves
        self.V0 = V0
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.best_weights_file = f"{self.weights_dir}/{self.name}_{self.id}_best.pth"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def do_random_step(self, states, last_moves):
        possible_moves = torch.ones((states.size(0), self.n_gens), dtype=torch.bool, device=self.device)
        valid_mask = last_moves >= 0
        if valid_mask.any():
            rows = torch.arange(states.size(0), device=self.device)[valid_mask]
            possible_moves[rows, self.inverse_moves[last_moves[valid_mask]]] = False
        next_moves = torch.multinomial(possible_moves.float(), 1).squeeze(-1)
        new_states = torch.gather(states, 1, self.all_moves[next_moves])
        return new_states, next_moves

    def generate_random_walks(self, walkers_num):
        total = walkers_num * (self.K_max - self.K_min + 1)
        depths = torch.arange(self.K_min, self.K_max + 1, device=self.device).repeat_interleave(walkers_num)
        states = self.V0.repeat(total, 1)
        last_moves = torch.full((total,), -1, dtype=torch.int64, device=self.device)

        for step in range(self.K_max):
            cutoff = 0 if step < self.K_min else walkers_num * (step - self.K_min + 1)
            if cutoff >= total:
                break
            states[cutoff:], last_moves[cutoff:] = self.do_random_step(states[cutoff:], last_moves[cutoff:])

        perm = torch.randperm(total, device=self.device)
        return states[perm], depths[perm]

    def _run_epoch(self, X, Y, training=True):
        self.net.train(training)
        total_loss = 0.0
        total_items = 0

        context = torch.enable_grad if training else torch.no_grad
        with context():
            for start in range(0, X.size(0), self.batch_size):
                data = X[start : start + self.batch_size]
                target = Y[start : start + self.batch_size]
                output = self.net(data)
                loss = self.criterion(output, target)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                batch_items = data.size(0)
                total_loss += float(loss.item()) * batch_items
                total_items += batch_items

        return total_loss / max(total_items, 1)

    def run(self):
        X_val = None
        Y_val = None
        val_gen_time = 0.0
        if self.val_walkers_num > 0:
            val_gen_started = time.time()
            X_val, Y_val = self.generate_random_walks(self.val_walkers_num)
            val_gen_time = time.time() - val_gen_started

        log_file = f"{self.log_dir}/train_{self.name}_{self.id}.csv"
        for _ in range(self.num_epochs):
            self.epoch += 1

            data_gen_started = time.time()
            X, Y = self.generate_random_walks(self.walkers_num)
            data_gen_time = time.time() - data_gen_started

            epoch_started = time.time()
            train_loss = self._run_epoch(X, Y.float(), training=True)
            epoch_time = time.time() - epoch_started

            val_loss = None
            if X_val is not None and Y_val is not None:
                val_loss = self._run_epoch(X_val, Y_val.float(), training=False)

            row = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "vertices_seen": X.size(0),
                "data_gen_time": data_gen_time,
                "train_epoch_time": epoch_time,
            }
            if val_loss is not None:
                row["val_loss"] = val_loss
                row["val_vertices_seen"] = X_val.size(0)
                row["val_gen_time"] = val_gen_time
                row["best_val_loss"] = min(self.best_val_loss, val_loss)
            append_csv_row(log_file, row)

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = self.epoch
                torch.save(model_state_dict(self.net), self.best_weights_file)

            if (self.epoch & (self.epoch - 1)) == 0:
                weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}.pth"
                torch.save(model_state_dict(self.net), weights_file)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if val_loss is None:
                    print(f"[{timestamp}] Saved weights at epoch {self.epoch:5d}. Train Loss: {train_loss:.2f}")
                else:
                    print(
                        f"[{timestamp}] Saved weights at epoch {self.epoch:5d}. "
                        f"Train Loss: {train_loss:.2f}. Val Loss: {val_loss:.2f}. Best Val: {self.best_val_loss:.2f}"
                    )

        if (self.epoch & (self.epoch - 1)) != 0:
            final_weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}.pth"
            torch.save(model_state_dict(self.net), final_weights_file)
        else:
            final_weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}.pth"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if X_val is None or Y_val is None:
            print(f"[{timestamp}] Finished. Saved final weights at epoch {self.epoch}. Train Loss: {train_loss:.2f}.")
        else:
            print(
                f"[{timestamp}] Finished. Saved final weights at epoch {self.epoch}. "
                f"Train Loss: {train_loss:.2f}. Val Loss: {val_loss:.2f}. "
                f"Best Val: {self.best_val_loss:.2f} at epoch {self.best_epoch}."
            )

        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_weights_file": final_weights_file,
            "best_val_loss": None if math.isinf(self.best_val_loss) else self.best_val_loss,
            "best_epoch": self.best_epoch,
            "best_weights_file": self.best_weights_file if self.best_epoch > 0 else None,
            "train_walkers_num": self.walkers_num,
            "val_walkers_num": self.val_walkers_num,
        }
