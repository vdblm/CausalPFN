from abc import ABC, abstractmethod

from torch.nn import Module
from torch.optim import Optimizer


class Callback(ABC):
    """The base callback class that is used in training"""

    def __init__(self, callback_name: str | None):
        self.callback_name = callback_name or self.__class__.__name__

    @abstractmethod
    def __call__(
        self,
        callback_idx: int,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        train_loss: float,
        device: str,
        rank: int,
        wandb_enabled: bool,
        lr_scheduler=None,
    ):
        """
        The method that is called during training after each epoch.

        Args:
            callback_idx (int): The index of the callback.
            model (Module): The model being trained.
            optimizer (Optimizer): The optimizer used for training.
            epoch (int): The current epoch number.
            train_loss (float): The training loss for the current epoch.
            device (str): The device on which the model is trained.
            rank (int): The rank of the process in distributed training.
            wandb_enabled (bool): Whether Weights & Biases logging is enabled
                                (guaranteed to be disabled for rank != 0)
            lr_scheduler: Optional learning-rate scheduler whose state may be checkpointed.
        """
        pass
