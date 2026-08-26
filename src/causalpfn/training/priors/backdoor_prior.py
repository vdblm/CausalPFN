from typing import Callable

import torch

from .meta_dataset import MetaDataset
from .table_generators import TableGenerator
from .utils import PriorGenerationError, degree_heterogeneity, scale_causal_effect


def T_logits_to_propensities_and_T(treatment_logits: torch.Tensor, standardize_T: bool, overlap: float):
    if standardize_T:
        treatment_logits = (treatment_logits - treatment_logits.mean(dim=0)) / (treatment_logits.std(dim=0) + 1e-20)
    p = torch.sigmoid(treatment_logits)
    propensities = overlap * p + (p >= 0.5) * (1 - overlap)
    treatments = (propensities > torch.rand_like(propensities)).float()
    return propensities, treatments


class BackdoorDGPMetaDataset(MetaDataset):
    """
    A data generating process for the backdoor setting. Here, we implement a method to return observational samples
    X, T, Y, along with the potential outcomes Y0, Y1, and the expected potential outcomes E[Y0 | X], E[Y1 | X] as
    the CEPO loss requires this to train the model.

    To do so, we first sample a table from a joint distribution, select some of the columns as covariates at random,
    and we also select 4 columns to construct our potential outcomes. Among these four columns, two of them act
    as E[Y0 | X] and E[Y1 | X], while the other two act as the noise terms eta_0 and eta_1 that model the
    aleatoric uncertainty in P(Y1 | X) and P(Y0 | X). We add the noise terms to the expected potential outcomes to
    get the final potential outcomes.

    Moreover, we sample the treatment assignment P(T | X) from a separate table generator to ensures the ignorability
    assumption (backdoor criteria) holds. This function takes in a set of covariates and returns the treatment logits,
    which are then passed through a sigmoid function to get the propensities. Here, we also control the overlap
    post-hoc, similar to RealCause.

    Note that we may make the treatment assignment randomized like an RCT with a specific distribution independent of
    the covariates, or a treatment assignment based on a simple or complicated function of the covariates.

    Args:
        X_y0_E_y0_y1_E_y1_generator:
            A table generator that samples from the covariates and potential outcome distribution.

        T_given_X_generator: A table generator that samples logits for the treatment model P(T | X).

        max_n_covariates: The maximum number of covariates to sample.

        layer_norm_covariates_prob: The probability for whether to apply layer normalization to covariates.

        n_samples: The number of samples to generate.

        overlap_dist:
            A distribution of the degree of overlap (between 0 and 1). If 1., leave treatment untouched;
            if 0., push p(T = 1 | x) to 0. for all covariates x, where p(T = 1 | x) < 0.5 and
            and push p(T = 1 | x) to 1. for all covariates x, where p(T = 1 | x) >= 0.5.
            if 0 < overlap < 1, do a linear interpolation of the above

        treatment_standardize_p:
             A callable that returns whether to standardize the treatment logits before passing it into sigmoid.

        degree_heterogeneity_dist:
            A distribution for the degree of heterogeneity (between 0 and 1). When deg_hetero=1., y1 and y0 remain
            unchanged. When deg_hetero=0, y1 - y0 is the same for all individuals.

        device: The device to use for sampling (e.g., "cpu" or "cuda").

        ate_scale_dist: A distribution for the scale of the causal effect (the value of ATE, optional).
    """

    def __init__(
        self,
        X_y0_E_y0_y1_E_y1_generator: TableGenerator,
        T_given_X_generator: TableGenerator,
        max_n_covariates: int,
        layer_norm_covariates_prob: float,
        n_samples: int,
        overlap_dist: Callable[[], float],
        treatment_standardize_p: float,
        degree_heterogeneity_dist: Callable[[], float],
        device: str,
        ate_scale_dist: Callable[[], float] | None = None,
        *args,
        **kwargs,
    ):
        self.X_y0_E_y0_y1_E_y1_generator = X_y0_E_y0_y1_E_y1_generator
        self.T_given_X_generator = T_given_X_generator

        self.max_n_covariates = max_n_covariates
        self.layer_norm_covariates_prob = layer_norm_covariates_prob
        self.n_samples = n_samples
        self.overlap_dist = overlap_dist
        self.treatment_standardize_p = treatment_standardize_p
        self.degree_heterogeneity_dist = degree_heterogeneity_dist
        self.device = device

        self.ate_scale_dist = ate_scale_dist

        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_sample(self) -> dict:

        n_covariates = torch.randint(1, self.max_n_covariates + 1, (1,), device=self.device).item()
        n_columns = n_covariates + 4  # 4 columns for potential outcomes

        # sample the potential outcomes and covariates
        X_y0_E_y0_y1_E_y1 = self.X_y0_E_y0_y1_E_y1_generator.sample_table(
            n_samples=self.n_samples,
            n_columns=n_columns,
        )

        # permute the columns to avoid any ordering effect
        X_y0_E_y0_y1_E_y1 = X_y0_E_y0_y1_E_y1[:, torch.randperm(n_columns, device=self.device)]

        covariates = X_y0_E_y0_y1_E_y1[:, :n_covariates]
        potential_outcomes = X_y0_E_y0_y1_E_y1[:, n_covariates:]

        # covariates normalize
        covariates = (covariates - covariates.mean(dim=0)) / (covariates.std(dim=0) + 1e-20)
        if torch.rand(1) < self.layer_norm_covariates_prob:
            covariates = torch.nn.functional.layer_norm(covariates, covariates.shape[1:])

        # expected potential outcomes and noises
        E_y0, eta_0 = potential_outcomes[:, :1], potential_outcomes[:, 2:3]
        E_y1, eta_1 = potential_outcomes[:, 1:2], potential_outcomes[:, 3:4]

        # standardize the noise
        eta_0, eta_1 = (
            (eta_0 - eta_0.mean(dim=0)) / (eta_0.std(dim=0) + 1e-20),
            (eta_1 - eta_1.mean(dim=0)) / (eta_1.std(dim=0) + 1e-20),
        )

        # scale noises
        eta_0 *= torch.std(E_y0) * torch.rand(1, device=self.device)
        eta_1 *= torch.std(E_y1) * torch.rand(1, device=self.device)

        # center the noises
        eta_0 = eta_0 * torch.randn_like(eta_0, device=self.device)
        eta_1 = eta_1 * torch.randn_like(eta_1, device=self.device)

        y0, y1 = E_y0 + eta_0, E_y1 + eta_1

        # degree of heterogeneity
        (y0_scale, y0_shift), (y1_scale, y1_shift) = degree_heterogeneity(
            E_y0,
            E_y1,
            self.degree_heterogeneity_dist(),
        )

        # apply the deg_hetero
        y0 = y0 * y0_scale + y0_shift
        y1 = y1 * y1_scale + y1_shift

        E_y0 = E_y0 * y0_scale + y0_shift
        E_y1 = E_y1 * y1_scale + y1_shift

        # size of causal effect
        if self.ate_scale_dist is not None:
            scale = scale_causal_effect(E_y0, E_y1, self.ate_scale_dist())
            y0, y1 = y0 * scale, y1 * scale
            E_y0, E_y1 = E_y0 * scale, E_y1 * scale

        # sample a separate treatment model to ensure ignorability
        treatment_logits = self.T_given_X_generator.sample_conditional_table(input=covariates, n_columns=1)

        overlap = self.overlap_dist()

        if overlap < 0 or overlap > 1:
            raise PriorGenerationError("Overlap must be between 0 and 1")

        treatment_standardize = torch.rand(1).item() < self.treatment_standardize_p
        propensities, treatments = T_logits_to_propensities_and_T(
            treatment_logits=treatment_logits,
            standardize_T=treatment_standardize,
            overlap=overlap,
        )
        outcomes = treatments * y1 + (1 - treatments) * y0

        return dict(
            X=covariates,
            t=treatments.squeeze(),
            y=outcomes.squeeze(),
            y0=y0.squeeze(),
            y1=y1.squeeze(),
            E_y0=E_y0.squeeze(),
            E_y1=E_y1.squeeze(),
            propensities=propensities.squeeze(),
        )
