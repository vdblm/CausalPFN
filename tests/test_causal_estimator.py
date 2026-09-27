import numpy as np
import pytest

from causalpfn import CATEEstimator


@pytest.mark.parametrize("n_queries", [1, 4])
def test_estimate_cate_ci_returns_one_bound_per_query(monkeypatch, n_queries):
    estimator = CATEEstimator.__new__(CATEEstimator)
    lower_bound = np.arange(n_queries, dtype=np.float32)[None, :]
    upper_bound = lower_bound + 1

    monkeypatch.setattr(
        estimator,
        "_estimate_ate_cate_CI",
        lambda X, alpha, n_samples: {
            "cate_lower_bound": lower_bound,
            "cate_upper_bound": upper_bound,
        },
    )

    result = estimator.estimate_cate_CI(np.zeros((n_queries, 2), dtype=np.float32))

    assert result["lower_bound"].shape == (n_queries,)
    assert result["upper_bound"].shape == (n_queries,)
    np.testing.assert_array_equal(result["lower_bound"], lower_bound[0])
    np.testing.assert_array_equal(result["upper_bound"], upper_bound[0])
