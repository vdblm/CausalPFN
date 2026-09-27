from __future__ import annotations

import numpy as np
import pytest
import torch

from causalpfn import CATEEstimator


@pytest.mark.slow
def test_estimate_ate_ci_matches_real_point_estimate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    T = np.concatenate([np.zeros(150), np.ones(30)]).astype(np.float32)
    Y = (0.5 * X[:, 0] + 0.8 * T + rng.normal(size=len(X))).astype(np.float32)
    X_query = X[T == 1][:8]

    estimator = CATEEstimator(device="cpu", verbose=False)
    estimator.fit(X, T, Y)
    expected_ate = estimator.estimate_ate(X_query)

    torch.manual_seed(0)
    result = estimator.estimate_ate_CI(X_query, alpha=0.05, n_samples=32)

    assert result["ate"] == expected_ate
    assert result["lower_bound"].shape == (1,)
    assert result["upper_bound"].shape == (1,)
    assert np.isfinite(result["lower_bound"]).all()
    assert np.isfinite(result["upper_bound"]).all()
    assert result["lower_bound"][0] <= result["upper_bound"][0]
