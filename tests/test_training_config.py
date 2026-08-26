from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

EXPERIMENTS = [
    "simple_configuration",
    "synthetic_backdoor",
    "ablation_polynomial_backdoor",
    "ablation_sinusoidal_backdoor",
]
PINNED_REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_retained_experiments_compose_with_warm_start_and_generate_a_prior(experiment):
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="train",
            overrides=[f"+experiment={experiment}", "train_meta_dataset.n_samples=32"],
        )

    assert config.model.path is None
    assert config.model.revision == PINNED_REVISION
    assert config.model.obj._target_.endswith("load_pretrained_in_context_model")

    prior = instantiate(config.train_meta_dataset)
    sample = next(iter(prior))
    assert sample["X"].shape == (32, 99)
    assert set(sample) == {"X", "t", "y", "y0", "y1", "E_y0", "E_y1", "propensities"}
    assert all(torch.isfinite(value).all() for value in sample.values())


def test_random_initialization_remains_an_explicit_override():
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="train",
            overrides=["+experiment=synthetic_backdoor", "model=tabdpt_long_context"],
        )

    assert config.model.obj._target_.endswith("InContextModel")
