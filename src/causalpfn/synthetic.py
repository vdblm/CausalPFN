"""Lightweight synthetic data-generating processes shared by training and benchmarks."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

SamplerType = Callable[[tuple], np.ndarray]


class LaplaceSampler:
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        return np.random.laplace(self.loc, self.scale, shape)


class UniformSampler:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        values = np.random.rand() if shape is None else np.random.rand(*shape)
        return values * (self.high - self.low) + self.low


class GaussianSampler:
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        return np.random.normal(self.loc, self.scale, shape)


class UniformIntegerSampler:
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | int:
        return np.random.randint(self.low, self.high + 1, shape)


class GeneralizedLinearDataset(ABC):
    """Synthetic DGP with linear feature functions and optional nonlinearities."""

    def __init__(
        self,
        n_samples: int = 2048,
        x_dim_dist: Callable[[], int] = UniformIntegerSampler(5, 10),
        noise_samplers: list[SamplerType] | SamplerType | None = None,
        weight_sampler: list[SamplerType] | SamplerType = UniformSampler(-5.0, 5.0),
        covariate_sampler: list[SamplerType] | SamplerType = UniformSampler(-2.0, 2.0),
        standardize_treatment: bool = True,
        standardize_outcome: bool = True,
        standardize_covariates: bool = False,
    ) -> None:
        self.x_dim_dist = x_dim_dist
        if noise_samplers is None:
            noise_samplers = [
                GaussianSampler(0.0, 1.0),
                UniformSampler(-1.0, 1.0),
                LaplaceSampler(0, 1.0),
            ]
        self.noise_samplers = noise_samplers if isinstance(noise_samplers, list) else [noise_samplers]
        self.weight_sampler = weight_sampler if isinstance(weight_sampler, list) else [weight_sampler]
        self.covariate_sampler = covariate_sampler if isinstance(covariate_sampler, list) else [covariate_sampler]
        self.n_samples = n_samples
        self.standardize_treatment = standardize_treatment
        self.standardize_outcome = standardize_outcome
        self.standardize_covariates = standardize_covariates

    def sample_exogenous_noise(self, shape) -> np.ndarray:
        sampler = self.noise_samplers[np.random.randint(len(self.noise_samplers))]
        return sampler(shape)

    def sample_weights(self, shape) -> np.ndarray:
        sampler = self.weight_sampler[np.random.randint(len(self.weight_sampler))]
        return sampler(shape)

    def sample_covariates(self, shape) -> np.ndarray:
        sampler = self.covariate_sampler[np.random.randint(len(self.covariate_sampler))]
        return sampler(shape)

    @abstractmethod
    def covariates2features(self, covariates):
        raise NotImplementedError

    @abstractmethod
    def post_nonlinear(self, random_variable):
        raise NotImplementedError

    def get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)

        n_dims = self.x_dim_dist()
        covariates = self.sample_covariates((self.n_samples, n_dims))
        covariate_features = self.covariates2features(covariates)
        features_dims = covariate_features.shape[1]

        w_T = self.sample_weights((features_dims,))
        treatment_pre_logits = np.einsum("np,p->n", covariate_features, w_T)
        treatment_logits = self.post_nonlinear(treatment_pre_logits) + self.sample_exogenous_noise((self.n_samples,))
        if self.standardize_treatment:
            treatment_logits = (treatment_logits - treatment_logits.mean()) / (treatment_logits.std() + 1e-20)
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))
        treatments = np.random.binomial(1, treatment_probs, size=self.n_samples)

        w_Y0 = self.sample_weights((features_dims,))
        w_Y1 = self.sample_weights((features_dims,))
        E_y0 = self.post_nonlinear(np.einsum("np,p->n", covariate_features, w_Y0))
        E_y1 = self.post_nonlinear(np.einsum("np,p->n", covariate_features, w_Y1))
        y0 = E_y0 + self.sample_exogenous_noise((self.n_samples,))
        y1 = E_y1 + self.sample_exogenous_noise((self.n_samples,))

        outcomes = np.where(treatments == 1, y1, y0)
        if self.standardize_outcome:
            outcomes_mean, outcomes_std = outcomes.mean(), outcomes.std() + 1e-20
        else:
            outcomes_mean, outcomes_std = 0, 1
        outcomes = (outcomes - outcomes_mean) / outcomes_std
        y0, y1 = (y0 - outcomes_mean) / outcomes_std, (y1 - outcomes_mean) / outcomes_std
        E_y0, E_y1 = (E_y0 - outcomes_mean) / outcomes_std, (E_y1 - outcomes_mean) / outcomes_std

        if self.standardize_covariates:
            covariates = (covariates - covariates.mean(axis=0)) / (covariates.std(axis=0) + 1e-20)

        return covariates, treatments, treatment_probs, y0, y1, E_y0, E_y1, outcomes


__all__ = [
    "GaussianSampler",
    "GeneralizedLinearDataset",
    "LaplaceSampler",
    "SamplerType",
    "UniformIntegerSampler",
    "UniformSampler",
]
