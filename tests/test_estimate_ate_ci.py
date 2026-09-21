"""``CATEEstimator.estimate_ate_CI`` must return an ``"ate"`` point estimate,
not raise ``KeyError``.

Before this fix, ``_estimate_ate_cate_CI`` computed ``ate_samples`` but never
put a point estimate in its returned dict, so ``estimate_ate_CI``'s
``output["ate"]`` lookup raised ``KeyError`` on every single call.

The downstream `cfms` repo carried a regression test asserting this bug was
still present (`test_upstream_estimate_ate_ci_is_still_broken`, so that if it
ever started passing, the workaround calling the internal helper directly
could be replaced with the public method again). That test is inverted here
into a positive correctness test, since the fix now lives in this package.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalpfn import CATEEstimator


def _data(true_effect, n_c=900, n_t=150, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal(size=(n_c, 5)), rng.normal(size=(n_t, 5)) + 0.3]
    ).astype(np.float32)
    T = np.concatenate([np.zeros(n_c), np.ones(n_t)]).astype(np.float32)
    Y = (X[:, 0] * 0.5 + true_effect * T + rng.normal(scale=1.0, size=n_c + n_t)).astype(
        np.float32
    )
    return X, T, Y


def test_estimate_ate_ci_no_longer_raises_keyerror():
    """If this starts failing, the fix in _estimate_ate_cate_CI regressed."""
    X, T, Y = _data(0.0, n_c=300, n_t=60)
    est = CATEEstimator(device="cpu", verbose=False, num_neighbours=60)
    est.fit(X, T, Y)
    ci = est.estimate_ate_CI(X[T == 1], alpha=0.05, n_samples=64)
    assert "ate" in ci
    assert np.isfinite(np.asarray(ci["ate"])).all()


@pytest.mark.slow
def test_interval_brackets_the_point_estimate():
    X, T, Y = _data(0.8)
    est = CATEEstimator(device="cpu", verbose=False)
    est.fit(X, T, Y)
    ci = est.estimate_ate_CI(X[T == 1], alpha=0.05, n_samples=512)
    point = float(np.asarray(ci["ate"]))
    lo = float(np.asarray(ci["lower_bound"]).reshape(-1)[0])
    hi = float(np.asarray(ci["upper_bound"]).reshape(-1)[0])
    assert lo < point < hi
    assert np.isfinite([point, lo, hi]).all()


@pytest.mark.slow
def test_a_true_null_produces_an_interval_containing_zero():
    """The point of intervals here is to bound a null, so a genuine null has
    to come back as an interval straddling zero rather than a bare 'no effect'."""
    X, T, Y = _data(0.0)
    est = CATEEstimator(device="cpu", verbose=False)
    est.fit(X, T, Y)
    ci = est.estimate_ate_CI(X[T == 1], alpha=0.05, n_samples=512)
    lo = float(np.asarray(ci["lower_bound"]).reshape(-1)[0])
    hi = float(np.asarray(ci["upper_bound"]).reshape(-1)[0])
    assert lo <= 0.0 <= hi
