import numpy as np
import pytest

from causalpfn import CATEEstimator


class StubCATEEstimator(CATEEstimator):
    def __init__(self):
        pass

    def _estimate_ate_cate_CI(self, X, alpha=0.05, n_samples=10_000):
        lower_bound = np.arange(X.shape[0], dtype=np.float32)[None, :]
        return {
            "cate_lower_bound": lower_bound,
            "cate_upper_bound": lower_bound + 1,
        }


@pytest.mark.parametrize("n_queries", [1, 4])
def test_estimate_cate_ci_returns_one_bound_per_query(n_queries):
    estimator = StubCATEEstimator()
    expected_lower_bound = np.arange(n_queries, dtype=np.float32)
    expected_upper_bound = expected_lower_bound + 1

    result = estimator.estimate_cate_CI(np.zeros((n_queries, 2), dtype=np.float32))

    assert result["lower_bound"].shape == (n_queries,)
    assert result["upper_bound"].shape == (n_queries,)
    np.testing.assert_array_equal(result["lower_bound"], expected_lower_bound)
    np.testing.assert_array_equal(result["upper_bound"], expected_upper_bound)
