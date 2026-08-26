import os
from datetime import datetime

import torch
import wandb
from torch.optim import Optimizer

from causalpfn.models import InContextModel

from .base import Callback


class Checkpoint(Callback):
    """Store the top k models based on training loss"""

    def __init__(
        self,
        checkpoint_root: str,
        callback_name: str | None,
        frequency: int = 1,
        checkpoint_dir_name: str | None = None,
        top_k: int = 2,
    ):
        super().__init__(callback_name=callback_name)
        self.frequency = frequency
        self.checkpoint_root = checkpoint_root

        if wandb.run is not None:
            self.default_name = f"wandb-{wandb.run.id}"
        else:
            self.default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.checkpoint_dir_name = checkpoint_dir_name or self.default_name
        self.top_scores = [float("inf")] * top_k
        self.top_epochs = [-1] * top_k
        self.top_k = top_k

    def load_state(self, checkpoint: dict) -> None:
        """Restore callback bookkeeping, while accepting legacy checkpoints."""
        self.default_name = checkpoint.get("run_name", self.default_name)
        self.checkpoint_dir_name = checkpoint.get("checkpoint_dir_name", self.default_name)

        scores = list(checkpoint.get("top_scores", []))[: self.top_k]
        epochs = list(checkpoint.get("top_epochs", []))[: self.top_k]
        self.top_scores = scores + [float("inf")] * (self.top_k - len(scores))
        self.top_epochs = epochs + [-1] * (self.top_k - len(epochs))

    @torch.no_grad()
    def __call__(
        self,
        callback_idx: int,
        model: InContextModel,
        optimizer: Optimizer,
        epoch: int,
        train_loss: float,
        device: str,
        rank: int,
        wandb_enabled: bool,
        lr_scheduler=None,
    ):
        if (epoch + 1) % self.frequency == 0:
            checkpoint_dir = os.path.join(self.checkpoint_root, self.checkpoint_dir_name)
            if rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)

            save_top_checkpoint = False
            if self.top_k > 0 and train_loss < self.top_scores[-1]:
                removal_idx = self.top_epochs[-1]
                if removal_idx != -1 and rank == 0:
                    previous_checkpoint = os.path.join(checkpoint_dir, f"epoch_{removal_idx:04d}.pt")
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)
                self.top_epochs[-1] = epoch
                self.top_scores[-1] = train_loss
                everything = list(zip(self.top_scores, self.top_epochs))
                everything_sorted = sorted(everything, key=lambda x: x[0])
                for i, (score, e) in enumerate(everything_sorted):
                    self.top_scores[i] = score
                    self.top_epochs[i] = e
                save_top_checkpoint = True

            training_state = {
                "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
                "model_config": model.model_config,
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "epoch": epoch,
                "train_loss": train_loss,
                "run_name": self.default_name,
                "checkpoint_dir_name": self.checkpoint_dir_name,
                "top_scores": self.top_scores,
                "top_epochs": self.top_epochs,
            }

            if rank == 0:
                if save_top_checkpoint:
                    torch.save(training_state, os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt"))
                torch.save(training_state, os.path.join(checkpoint_dir, "latest.pt"))
