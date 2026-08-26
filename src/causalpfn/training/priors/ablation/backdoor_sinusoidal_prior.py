from typing import List

import numpy as np
import torch

from causalpfn.synthetic import GeneralizedLinearDataset, SamplerType
from causalpfn.training.priors.meta_dataset import MetaDataset


class BackdoorSinusoidalMetaDataset(GeneralizedLinearDataset, MetaDataset):

    def __init__(
        self,
        frequency_sampler: List[SamplerType] | SamplerType,  # This is a measure of model complexity
        post_padding_n_cols: int,
        *args,
        name: str = "BackdoorSinusoidalMetaDataset",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        GeneralizedLinearDataset.__init__(self, *args, **kwargs)
        MetaDataset.__init__(self, name=name, post_padding_n_cols=post_padding_n_cols)

        # for the frequency
        if isinstance(frequency_sampler, list):
            self.frequency_sampler = frequency_sampler
        else:
            self.frequency_sampler = [frequency_sampler]

        self.device = device

    def sample_frequency(self) -> int:
        chosen_frequency_sampler = self.frequency_sampler[np.random.randint(len(self.frequency_sampler))]
        return chosen_frequency_sampler((1,))[0]

    def covariates2features(self, covariates):
        return covariates

    def post_nonlinear(self, random_variable):
        frequency = self.sample_frequency()
        return random_variable + np.sin(frequency * random_variable)

    # override the sample method of MetaDataset
    def get_sample(self) -> dict:

        covariates, treatments, propensities, y0, y1, E_y0, E_y1, outcomes = (
            self.get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes()
        )

        numpy_dict = dict(
            X=covariates,
            t=treatments,
            y=outcomes,
            y0=y0,
            y1=y1,
            E_y0=E_y0,
            E_y1=E_y1,
            propensities=propensities,
        )

        return {k: torch.tensor(v, device=self.device).float() for k, v in numpy_dict.items()}
