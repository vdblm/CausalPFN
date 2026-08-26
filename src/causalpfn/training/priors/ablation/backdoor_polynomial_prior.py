from typing import List

import numpy as np
import torch

from causalpfn.synthetic import GeneralizedLinearDataset, SamplerType, UniformIntegerSampler
from causalpfn.training.priors.meta_dataset import MetaDataset


class BackdoorPolynomialMetaDataset(GeneralizedLinearDataset, MetaDataset):

    def __init__(
        self,
        post_padding_n_cols: int,
        *args,
        degree_sampler: List[SamplerType] | SamplerType = UniformIntegerSampler(2, 4),
        name: str = "BackdoorPolynomialMetaDataset",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        GeneralizedLinearDataset.__init__(self, *args, **kwargs)
        MetaDataset.__init__(self, name=name, post_padding_n_cols=post_padding_n_cols)

        # for the degree
        if isinstance(degree_sampler, list):
            self.degree_sampler = degree_sampler
        else:
            self.degree_sampler = [degree_sampler]

        self.device = device

    def sample_degree(self) -> int:
        chosen_degree_sampler = self.degree_sampler[np.random.randint(len(self.degree_sampler))]
        return chosen_degree_sampler((1,))[0]

    def covariates2features(self, covariates):
        degree = self.sample_degree()
        features = np.concatenate([covariates**i for i in range(1, degree + 1)], axis=1)
        return features

    def post_nonlinear(self, random_variable):
        return random_variable

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
