from __future__ import annotations

import numpy as np
import pytest

import causalpfn.causal_estimator as causal_estimator_module
from causalpfn import CATEEstimator


def _brute_force_nearest(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(reference))
    indices = np.arange(len(reference))
    return np.stack([np.lexsort((indices, np.abs(reference - value)))[:k] for value in query])


@pytest.mark.slow
def test_cate_matches_brute_force_neighbour_search(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2_200, 5)).astype(np.float32)
    treatment_effect = (np.sin(X[:, 0]) + 0.5 * X[:, 1]).astype(np.float32)
    T = rng.binomial(1, 0.5, size=len(X)).astype(np.float32)
    baseline = (X[:, 0] - X[:, 1] + rng.normal(0, 0.1, size=len(X))).astype(np.float32)
    Y = baseline + treatment_effect * T

    estimator = CATEEstimator(device="cpu", verbose=False)
    estimator.fit(X, T, Y)
    X_query = X[:16]
    actual = np.asarray(estimator.estimate_cate(X_query))

    monkeypatch.setattr(causal_estimator_module, "nearest_indices_1d", _brute_force_nearest)
    expected = np.asarray(estimator.estimate_cate(X_query))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
