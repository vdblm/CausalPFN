import numpy as np
import torch

from benchmarks.polynomial import PolynomialDataset
from causalpfn.training import priors
from causalpfn.training.priors import BackdoorDGPMetaDataset, table_generators
from causalpfn.training.priors.table_generators import SyntheticTableGenerator, TableGenerator


class GaussianTableGenerator(TableGenerator):
    def __init__(self):
        super().__init__(name="gaussian", device="cpu")

    def _sample_table(self, n_samples: int, n_columns: int) -> torch.Tensor:
        return torch.randn(n_samples, n_columns)


class RandomizedTreatmentGenerator(TableGenerator):
    def __init__(self):
        super().__init__(name="randomized-treatment", device="cpu")

    def _sample_conditional_table(self, input: torch.Tensor, n_columns: int) -> torch.Tensor:
        return torch.zeros(input.shape[0], n_columns)


def test_synthetic_backdoor_prior_produces_padded_causal_batch():
    torch.manual_seed(7)
    prior = BackdoorDGPMetaDataset(
        X_y0_E_y0_y1_E_y1_generator=GaussianTableGenerator(),
        T_given_X_generator=RandomizedTreatmentGenerator(),
        max_n_covariates=4,
        layer_norm_covariates_prob=0.5,
        n_samples=128,
        overlap_dist=lambda: 1.0,
        treatment_standardize_p=0.5,
        degree_heterogeneity_dist=lambda: 1.0,
        device="cpu",
        ate_scale_dist=None,
        name="test-prior",
        post_padding_n_cols=8,
    )

    sample = next(iter(prior))

    assert sample["X"].shape == (128, 8)
    assert set(sample) == {"X", "t", "y", "y0", "y1", "E_y0", "E_y1", "propensities"}
    assert all(torch.isfinite(value).all() for value in sample.values())
    assert set(sample["t"].unique().tolist()) == {0.0, 1.0}


def test_removed_prior_types_are_not_exported():
    assert set(priors.__all__) == {
        "BackdoorDGPMetaDataset",
        "DeepTruncNormLogScaledSampler",
        "MetaDataset",
        "PriorGenerationError",
        "UniformSampler",
    }
    assert set(table_generators.__all__) == {"SyntheticTableGenerator", "TableGenerator"}


def test_synthetic_treatment_generator_selects_linear_with_ten_percent_probability(monkeypatch):
    generator = SyntheticTableGenerator(
        device="cpu",
        n_layer_dist=lambda: 1,
        n_hidden_dist=lambda: 2,
        dense_prob_dist=lambda: 1.0,
        init_std_dist=lambda: 1.0,
        noise_std_dist=lambda: 0.0,
        n_block_max_dist=None,
        categorical_columns_prob=0.0,
        categorical_columns_ordered_prob=0.0,
        independent_prob=0.0,
        linear_probability=0.1,
    )
    draws = iter([torch.tensor([0.5]), torch.tensor([0.05]), torch.tensor([0.5]), torch.tensor([0.5])])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws))
    monkeypatch.setattr(generator, "_sample_linear_conditional_table", lambda input, n_columns: input.new_ones(3, 1))
    monkeypatch.setattr(
        generator,
        "_sample_nonlinear_conditional_table",
        lambda input, n_columns: input.new_zeros(3, 1),
    )

    inputs = torch.randn(3, 2)
    assert torch.equal(generator.sample_conditional_table(inputs, 1), torch.ones(3, 1))
    assert torch.equal(generator.sample_conditional_table(inputs, 1), torch.zeros(3, 1))


def test_generalized_linear_dataset_accepts_zero_seed():
    dataset = PolynomialDataset(n_tables=1, test_ratio=0.2, n_samples=32)
    first = dataset.get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(seed=0)
    second = dataset.get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(seed=0)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_polynomial_defaults_and_covariate_standardization():
    dataset = PolynomialDataset(n_tables=25, n_samples=64)
    covariates = dataset.get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(seed=3)[0]

    assert len(dataset) == 25
    assert dataset.test_ratio == 0.2
    assert np.allclose(covariates.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(covariates.std(axis=0), 1.0, atol=1e-7)
