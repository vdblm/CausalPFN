import math

import torch
from torch import nn
from torch.distributions import Bernoulli as TorchBernoulli

from causalpfn.training.priors.utils import PriorGenerationError


class StepActivation(nn.Module):
    def forward(self, input):
        return torch.sign(input)


class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class CosineActivation(nn.Module):
    def forward(self, input):
        return torch.cos(input)


DEFAULT_ACT_FUNCS = [
    CosineActivation,
    SineActivation,
    StepActivation,
    nn.ELU,
    nn.GELU,
    nn.Hardsigmoid,
    nn.Identity,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.Sigmoid,
    nn.SiLU,
    nn.Softplus,
    nn.Tanh,
]


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = False, device: str = "cpu"
    ):
        """
        Linear layer with an applied mask over the weights.

        Parameters:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Mask to apply over weights. Zero values in the mask will zero out the corresponding weights.
            bias: If set to False, the layer will not learn an additive bias.
        """
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device)
        if mask.size() != (in_features, out_features):
            raise PriorGenerationError(
                f"Mask size {mask.size()} does not match weight size {(in_features, out_features)}"
            )
        self.register_buffer("mask", mask.to(torch.bool))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MaskedLinear layer."""
        return torch.nn.functional.linear(input, self.weight * self.mask.T, self.bias)


class MaskedBlockLinear(MaskedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = False,
        init_std: float = 1.0,
        n_block_max: int | None = None,
        device: str = "cpu",
    ):
        """
        Block linear layer with an applied mask over the weights.

        Parameters:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Mask to apply over weights.
            bias: If set to False, the layer will not learn an additive bias.
            init_std: Standard deviation for the initialization.
            n_block_max: Maximum number of blocks to use.
        """
        self.init_std = init_std
        self.n_block_max = n_block_max
        super().__init__(in_features=in_features, out_features=out_features, mask=mask, bias=bias, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        # Blockwise dropout
        torch.nn.init.zeros_(self.weight)
        if self.n_block_max is None:
            n_block_max = math.ceil(math.sqrt(min(self.weight.shape[0], self.weight.shape[1])))
        else:
            n_block_max = self.n_block_max

        n_blocks = torch.randint(1, n_block_max + 1, ()).item()
        n_blocks = min(n_blocks, self.weight.shape[0], self.weight.shape[1])
        w, h = self.weight.shape[0] // n_blocks, self.weight.shape[1] // n_blocks
        keep_prob = (
            n_blocks * w * h
        ) / self.weight.numel()  # Keeps the same standard deviation for the sume of the weights
        for block in range(0, n_blocks):
            torch.nn.init.normal_(
                self.weight[
                    w * block : w * (block + 1),
                    h * block : h * (block + 1),
                ],
                std=self.init_std / keep_prob**0.5,
            )


class PFN_MLP(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_columns: int,
        n_layers: int,
        mlp_hidden_dim: int,
        dense_prob: float,
        init_std: float,
        noise_std: float,
        n_block_max: int | None,
        device: str,
        pre_sample: bool,
        name: str = "PFN MLP",
    ):
        super().__init__()

        if n_layers <= 1:
            raise PriorGenerationError(f"n_layers must be greater than 1, got {n_layers}")

        if input_dim <= 0:
            raise PriorGenerationError(f"input_dim must be greater than 0, got {input_dim}")

        self.n_columns = n_columns

        mlp_hidden_dim = max(mlp_hidden_dim, 2 * n_columns)

        self.input_dim = input_dim
        self.device = device
        self.name = name
        self.pre_sample = pre_sample

        def rand_layer(out_dim):
            mask = TorchBernoulli(torch.ones((mlp_hidden_dim, out_dim)) * dense_prob).sample().to(device)
            modules = [
                DEFAULT_ACT_FUNCS[torch.randint(0, len(DEFAULT_ACT_FUNCS), (1,))[0]](),
                MaskedBlockLinear(
                    mlp_hidden_dim,
                    out_dim,
                    mask=mask,
                    bias=False,
                    device=device,
                    init_std=init_std,
                    n_block_max=n_block_max,
                ),
                GaussianNoise(noise_std),
            ]
            return nn.Sequential(*modules)

        mask = TorchBernoulli(torch.ones((input_dim, mlp_hidden_dim)) * dense_prob).sample().to(device)
        self.layers = [
            MaskedBlockLinear(
                input_dim,
                mlp_hidden_dim,
                mask=mask,
                bias=False,
                device=device,
                init_std=init_std,
                n_block_max=n_block_max,
            )
        ]
        self.layers += [rand_layer(mlp_hidden_dim) for _ in range(n_layers - 1)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inp: torch.Tensor | None, n_samples: int | None = None) -> torch.Tensor:
        if inp is not None and self.input_dim != inp.shape[-1]:
            raise PriorGenerationError(f"Input shape {inp.shape} does not match {self.input_dim}")
        if inp is None and n_samples is None:
            raise PriorGenerationError("Either x or n_samples must be provided")

        if inp is None:
            if self.pre_sample:
                x_mean = torch.normal(0, 1, size=(self.input_dim,), device=self.device)
                x_std = (torch.normal(0, 1, size=(self.input_dim,), device=self.device) * x_mean).abs()
                inp = torch.normal(
                    mean=x_mean.unsqueeze(0).expand(n_samples, -1), std=x_std.unsqueeze(0).expand(n_samples, -1)
                )
            else:
                inp = torch.normal(0.0, 1.0, size=(n_samples, self.input_dim), device=self.device)

        inp = inp.to(self.device)
        outputs = [inp]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]
        outputs_flat: torch.Tensor = torch.cat(outputs, -1)

        rand_indices = torch.randperm(outputs_flat.shape[-1], device=self.device)[: self.n_columns]
        table = outputs_flat[:, rand_indices]

        if torch.isnan(table).any():
            raise PriorGenerationError(f"NaN in {self.name}")

        # random covariates rotation
        if table.shape[-1] > 0:
            table_shift = torch.randint(0, table.shape[-1], (1,), device=self.device).item()
            table = torch.cat((table[..., table_shift:], table[..., :table_shift]), dim=-1)

        return table
