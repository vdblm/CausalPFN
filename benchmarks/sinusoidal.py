# A synthetic sinusoidal dataset for evaluating CATE estimation methods.

from typing import List, Tuple

import numpy as np

from .base import ATE_Dataset, CATE_Dataset, EvalDatasetCatalog, GeneralizedLinearDataset, SamplerType


class SinusoidalDataset(GeneralizedLinearDataset, EvalDatasetCatalog):

    def __init__(
        self,
        n_tables: int,
        test_ratio: float,
        frequency_sampler: List[SamplerType] | SamplerType,
        *args,
        seed: int = 42,
        **kwargs,
    ) -> None:
        GeneralizedLinearDataset.__init__(self, *args, **kwargs)
        EvalDatasetCatalog.__init__(self, n_tables=n_tables, name="Sinusoidal")

        # for the frequency
        if isinstance(frequency_sampler, list):
            self.frequency_sampler = frequency_sampler
        else:
            self.frequency_sampler = [frequency_sampler]

        self.n_tables = n_tables

        self.test_ratio = test_ratio

        self.seeds = [seed + i for i in range(self.n_tables)]

    def sample_frequency(self) -> int:
        chosen_frequency_sampler = self.frequency_sampler[np.random.randint(len(self.frequency_sampler))]
        return chosen_frequency_sampler((1,))[0]

    def covariates2features(self, covariates):
        return covariates

    def post_nonlinear(self, random_variable):
        frequency = self.sample_frequency()
        scale = np.std(random_variable)
        return random_variable + np.sin(frequency * random_variable) * scale

    def __getitem__(self, index) -> Tuple[CATE_Dataset, ATE_Dataset]:
        if index >= self.n_tables:
            raise IndexError("Index out of range for the dataset catalog")

        covariates, treatments, _, _, _, E_y0, E_y1, outcomes = self.get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(
            self.seeds[index]
        )

        cate = E_y1 - E_y0

        indices = np.random.permutation(covariates.shape[0])
        split_idx = int(len(indices) * (1 - self.test_ratio))
        X_train, t_train, y_train = (
            covariates[indices[:split_idx]],
            treatments[indices[:split_idx]],
            outcomes[indices[:split_idx]],
        )
        X_test, cate_test = covariates[indices[split_idx:]], cate[indices[split_idx:]]

        cate_dataset = CATE_Dataset(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_test=X_test,
            true_cate=cate_test,
        )

        ate_dataset = ATE_Dataset(
            X=covariates,
            t=treatments,
            y=outcomes,
            true_ate=float((E_y1 - E_y0).mean()),
        )

        return cate_dataset, ate_dataset
