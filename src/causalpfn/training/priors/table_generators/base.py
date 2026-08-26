from abc import ABC

import torch


class TableGenerator(ABC):
    """
    Abstract base class for table generation models.

    This class defines the interface for generating synthetic or real-world tabular data, from either
    joint distributions like P(X1, X2, ...) or conditional distributions like P(Y1, Y2, ... | X1, X2, ...).
    Subclasses must implement the core sampling methods, i.e., `_sample_table` to generate tables unconditionally
    and/or `_sample_conditional_table` to generate tables conditionally based on input data.

    Attributes:
        name (str): Human-readable name identifier for the generator.
        device (str): Device string (e.g., 'cpu', 'cuda:0') where tensors will be placed.
    """

    def __init__(self, name: str, device: str, *args, **kwargs):
        self.name = name
        self.device = device

    def __str__(self):
        return self.name

    def _sample_table(self, n_samples: int, n_columns: int, *args, **kwargs) -> torch.Tensor:
        """
        Internal method to sample a table unconditionally.

        This method must be implemented by subclasses to define the core table generation logic.

        Args:
            n_samples (int): Number of rows to generate.
            n_columns (int): Number of columns in the generated table.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Generated table tensor of shape (n_samples, n_columns).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method to sample a table.")

    def _sample_conditional_table(self, input: torch.Tensor, n_columns: int, *args, **kwargs) -> torch.Tensor:
        """
        Internal method to sample a table conditionally based on input data.

        This method must be implemented by subclasses to define conditional table generation logic.

        Args:
            input (torch.Tensor): Conditioning input tensor with shape (n_samples, n_input_features).
            n_columns (int): Number of columns in the generated table.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Generated table tensor conditioned on input of shape (n_samples, n_columns).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method to sample a conditional table.")

    def sample_table(self, n_samples: int, n_columns: int, *args, **kwargs) -> torch.Tensor:
        return self._sample_table(n_samples=n_samples, n_columns=n_columns, *args, **kwargs).to(self.device)

    def sample_conditional_table(self, input: torch.Tensor, n_columns: int, *args, **kwargs) -> torch.Tensor:
        return self._sample_conditional_table(input=input, n_columns=n_columns, *args, **kwargs).to(self.device)
