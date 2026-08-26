from typing import Callable

import torch

from causalpfn.training.priors.utils import num2cat

from .base import TableGenerator
from .utils import PFN_MLP


class SyntheticTableGenerator(TableGenerator):

    def __init__(
        self,
        device: str,
        n_layer_dist: Callable[[], int],
        n_hidden_dist: Callable[[], int],
        dense_prob_dist: Callable[[], float],
        init_std_dist: Callable[[], float],
        noise_std_dist: Callable[[], float],
        n_block_max_dist: Callable[[], int] | None,
        categorical_columns_prob: float,
        categorical_columns_ordered_prob: float,
        pre_sample_prob: float = 0.0,
        independent_prob: float = 0.0,  # if True, then P(Y|X) = P(Y) for conditional tables
        linear_probability: float = 0.0,
        linear_weights_std: float = 1.0,
    ):
        self.n_layer_dist = n_layer_dist
        self.n_hidden_dist = n_hidden_dist
        self.dense_prob_dist = dense_prob_dist
        self.init_std_dist = init_std_dist
        self.noise_std_dist = noise_std_dist
        self.n_block_max_dist = n_block_max_dist
        self.independent_prob = independent_prob
        self.pre_sample_prob = pre_sample_prob
        self.categorical_columns_prob = categorical_columns_prob
        self.categorical_columns_ordered_prob = categorical_columns_ordered_prob
        self.linear_probability = linear_probability
        self.linear_weights_std = linear_weights_std

        super().__init__(device=device, name="Synthetic PFN MLP")

    def _add_categorical_covariates(self, table: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.categorical_columns_prob:
            for col in range(table.shape[-1]):
                if torch.rand(1) < 0.5:
                    max_categories = max(round(torch.distributions.Gamma(1, 0.1).sample().item()), 2)
                    table[..., col] = num2cat(
                        table[..., col],
                        max_categories,
                        ordered_p=self.categorical_columns_ordered_prob,
                    )

        return table

    def _sample_table(self, n_samples: int, n_columns: int) -> torch.Tensor:
        model = PFN_MLP(
            input_dim=self.n_hidden_dist(),
            n_columns=n_columns,
            n_layers=self.n_layer_dist(),
            mlp_hidden_dim=self.n_hidden_dist(),
            dense_prob=self.dense_prob_dist(),
            init_std=self.init_std_dist(),
            noise_std=self.noise_std_dist(),
            n_block_max=self.n_block_max_dist() if self.n_block_max_dist is not None else None,
            device=self.device,
            pre_sample=torch.rand(1) < self.pre_sample_prob,
        )
        table = model.forward(inp=None, n_samples=n_samples)
        return self._add_categorical_covariates(table)

    def _sample_nonlinear_conditional_table(self, input: torch.Tensor, n_columns: int) -> torch.Tensor:
        model = PFN_MLP(
            input_dim=input.shape[-1],
            n_columns=n_columns,
            n_layers=self.n_layer_dist(),
            mlp_hidden_dim=self.n_hidden_dist(),
            dense_prob=self.dense_prob_dist(),
            init_std=self.init_std_dist(),
            noise_std=self.noise_std_dist(),
            n_block_max=self.n_block_max_dist() if self.n_block_max_dist is not None else None,
            device=self.device,
            pre_sample=torch.rand(1) < self.pre_sample_prob,
        )
        table = model.forward(inp=input)
        return self._add_categorical_covariates(table)

    def _sample_linear_conditional_table(self, input: torch.Tensor, n_columns: int) -> torch.Tensor:
        weights = torch.randn(input.shape[-1], n_columns, device=self.device) * self.linear_weights_std
        bias = torch.randn(n_columns, device=self.device) * self.linear_weights_std
        return torch.matmul(input, weights) + bias

    def _sample_conditional_table(self, input: torch.Tensor, n_columns: int) -> torch.Tensor:
        if torch.rand(1) < self.independent_prob:
            return self._sample_table(n_samples=input.shape[0], n_columns=n_columns)
        if torch.rand(1) < self.linear_probability:
            return self._sample_linear_conditional_table(input, n_columns)
        return self._sample_nonlinear_conditional_table(input, n_columns)
