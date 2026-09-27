from __future__ import annotations

import numpy as np

from causalpfn import CATEEstimator


def test_estimate_ate_ci_uses_the_deterministic_point_estimate(monkeypatch):
    estimator = CATEEstimator(device="cpu", verbose=False)
    X = np.zeros((3, 2), dtype=np.float32)
    expected_lower = np.array([-0.5])
    expected_upper = np.array([1.5])

    monkeypatch.setattr(
        estimator,
        "_estimate_ate_cate_CI",
        lambda X, alpha, n_samples: {
            "ate_lower_bound": expected_lower,
            "ate_upper_bound": expected_upper,
        },
    )
    monkeypatch.setattr(estimator, "estimate_ate", lambda X: 0.75)

    result = estimator.estimate_ate_CI(X, alpha=0.05, n_samples=64)

    assert result["ate"] == 0.75
    np.testing.assert_array_equal(result["lower_bound"], expected_lower)
    np.testing.assert_array_equal(result["upper_bound"], expected_upper)
