import math
from typing import Tuple

import torch


class PriorGenerationError(Exception):
    pass


class DeepTruncNormLogScaledSampler:
    """Samples noise from a truncated normal distribution upon calling"""

    def __init__(self, mean_min, mean_max, is_int, min_val):
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.is_int = is_int
        self.min_val = min_val

    def __call__(self, shape: torch.Size | None = None):
        log_min = math.log(self.mean_min)
        log_max = math.log(self.mean_max)
        size = shape or (1,)
        sigma = torch.exp((log_max - log_min) * torch.rand(size) + log_min)
        mu = torch.exp((log_max - log_min) * torch.rand(size) + log_min)

        loi = torch.distributions.Normal(mu, sigma)
        round = (lambda x: torch.round(x).int()) if self.is_int else (lambda x: x)
        samples = round(loi.icdf((torch.rand(size) - 1) * (1 - loi.cdf(torch.zeros(size))) + 1) + self.min_val)
        samples = torch.where(samples > self.min_val, samples, self.min_val * torch.ones_like(samples))
        samples = samples if shape is not None else samples.item()
        return samples


class UniformSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor:
        if shape is None:
            ret = torch.rand((1,)).item()
        else:
            ret = torch.rand(shape)
        return ret * (self.high - self.low) + self.low


### Categorical ###
def num2cat(numerical_tensor: torch.Tensor, max_categories: int, ordered_p: float) -> torch.Tensor:
    """
    Convert continuous numeric data into categorical data with optional randomization.

    This function transforms a continuous numeric tensor into a categorical tensor
    by creating class boundaries. It can also randomize the ordering of categories
    based on the provided probability parameter.

    Args:
        numerical_tensor: Input tensor with continuous numeric values, shape: (seq_len,)
        max_categories: Maximum number of categories to create
        ordered_p: Probability to keep categories ordered (vs randomized) Higher values mean categories are more likely
                  to remain ordered

    Returns:
        Categorical tensor as float values
    """

    if numerical_tensor.ndim != 1:
        raise PriorGenerationError("Input tensor must be 1-dimensional")

    categorical = torch.empty_like(numerical_tensor)
    num_classes = torch.randint(2, max(3, max_categories + 1), (1,), device=numerical_tensor.device).item()

    if torch.rand(1) > 0.5:
        numerical_tensor = -numerical_tensor

    sorted_num, categorical = torch.unique(numerical_tensor, return_inverse=True)
    if len(sorted_num) >= num_classes + 1:
        # Randomly select boundary values
        boundary_indices = torch.randperm(len(sorted_num) - 2, device=numerical_tensor.device)[: num_classes - 1]
        cls_bdry: torch.Tensor = sorted_num[1:-1][boundary_indices]

        # Convert to categorical by counting how many boundaries each value exceeds
        reshaped_num = numerical_tensor.reshape((-1, 1))  # shape: (seq_len, 1)
        reshaped_bdry = cls_bdry.reshape(1, -1)  # shape: (1, num_classes - 1)
        categorical = (reshaped_num > reshaped_bdry).sum(dim=1)  # shape: (seq_len,)

    if torch.rand(1) > ordered_p:
        classes = torch.arange(0, num_classes, device=categorical.device)
        random_classes = torch.randperm(num_classes, device=categorical.device).type(categorical.type())
        categorical = ((categorical.unsqueeze(-1) == classes) * random_classes).sum(-1)

    return categorical.float()


###### Controlling overlap and heterogeneity


def degree_heterogeneity(
    y0: torch.Tensor,
    y1: torch.Tensor,
    deg_hetero: float,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    For deg_hetero = 1, y1 and y0 remain unchanged. For deg_hetero = 0, y1 - y0 is the same for all individuals.
    For 0 < deg_hetero < 1, y1 and y0 are changed such that the CATE is a linear interpolation between the
    original CATE and the constant ATE. The code is from
    https://github.com/bradyneal/realcause/blob/master/models/base.py
    """
    if deg_hetero < 0 or deg_hetero > 1:
        raise PriorGenerationError(f"deg_hetero not in [0, 1], got {deg_hetero}")
    y1_mean = y1.mean()
    y0_mean = y0.mean()
    ate = y1_mean - y0_mean

    alpha = torch.rand(y1.shape, device=y1.device)

    y1_scale = alpha + (1 - alpha) * deg_hetero
    y1_shift = (1 - deg_hetero) * (y0 + ate) * (1 - alpha)

    y0_scale = (1 - alpha) + alpha * deg_hetero
    y0_shift = (1 - deg_hetero) * (y1 - ate) * alpha

    return (y0_scale, y0_shift), (y1_scale, y1_shift)


def scale_causal_effect(
    y0: torch.Tensor,
    y1: torch.Tensor,
    causal_effect_scale: float,
) -> torch.Tensor:
    """
    Scales the causal effect E[y1 - y0] by the given `causal_effect_scale`. The code is from
    https://github.com/bradyneal/realcause/blob/master/models/base.py
    """
    if causal_effect_scale <= 0:
        raise PriorGenerationError(f"causal_effect_scale must be positive, got {causal_effect_scale}")
    ate = (y1 - y0).mean()
    return causal_effect_scale / ate
