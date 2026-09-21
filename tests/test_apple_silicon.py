"""CausalPFN must actually run on Apple Silicon, not just avoid raising.

Before this fix, ``import causalpfn`` (or the first large tensor op inside
``fit``/``predict``) segfaulted the interpreter on macOS/arm64: ``faiss`` and
``torch`` each bundle their own OpenMP runtime, and having both loaded in one
process corrupts memory. ``_flat_l2.IndexFlatL2`` removes faiss entirely, so
these tests assert the platform genuinely works end-to-end, not merely that
some import succeeds.

Adapted from the downstream `cfms` repo's test suite for its (now
unnecessary) workaround (layer6ai-labs/cfms#2, by Max De Marzi), rewritten
against ``causalpfn.CATEEstimator``/``ATEEstimator`` directly since the fix
now lives in this package instead of a downstream shim.

Marked `slow` -- the first run downloads the pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalpfn import ATEEstimator, CATEEstimator


def _known_effect_dataset(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    ps = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    T = (rng.uniform(size=n) < ps).astype(np.float32)
    tau = 2.0 + 0.8 * X[:, 2]
    Y = (X[:, 0] - 0.5 * X[:, 1] + tau * T + rng.normal(scale=0.4, size=n)).astype(np.float32)
    return X, T, Y, tau


@pytest.mark.slow
def test_recovers_a_known_confounded_effect():
    X, T, Y, tau = _known_effect_dataset()

    cate_est = CATEEstimator(device="cpu", verbose=False)
    cate_est.fit(X, T, Y)
    tau_hat = np.asarray(cate_est.estimate_cate(X)).reshape(-1)

    ate_est = ATEEstimator(device="cpu", verbose=False)
    ate_est.fit(X, T, Y)
    ate_hat = float(np.asarray(ate_est.estimate_ate()).reshape(-1)[0])

    assert tau_hat.shape == (len(X),)
    assert np.isfinite(tau_hat).all()
    # Treatment is confounded through X0/X1, so this fails if the adjustment
    # is not happening -- a naive difference in means is materially biased here.
    assert ate_hat == pytest.approx(tau.mean(), rel=0.10)
    # Heterogeneity is driven by X2; the ranking must survive, not just the mean.
    assert np.corrcoef(tau_hat, tau)[0, 1] > 0.9


@pytest.mark.slow
def test_estimates_are_deterministic_across_calls():
    """Removing faiss must not introduce run-to-run drift: two fits on
    identical inputs have to agree exactly, or nothing built on this is
    reproducible."""
    X, T, Y, _ = _known_effect_dataset(n=600, seed=5)

    a = CATEEstimator(device="cpu", verbose=False)
    a.fit(X, T, Y)
    first = np.asarray(a.estimate_cate(X))

    b = CATEEstimator(device="cpu", verbose=False)
    b.fit(X, T, Y)
    second = np.asarray(b.estimate_cate(X))

    np.testing.assert_array_equal(first, second)


@pytest.mark.slow
def test_fit_and_predict_do_not_segfault_on_apple_silicon():
    """The regression this whole fix is about: fit() succeeding is not
    enough on its own, since the old "reorder the imports" attempt made
    fit() succeed while estimate_cate() still crashed. Exercise both."""
    X, T, Y, _ = _known_effect_dataset(n=300, seed=11)
    est = CATEEstimator(device="cpu", verbose=False)
    est.fit(X, T, Y)
    tau_hat = np.asarray(est.estimate_cate(X))
    assert np.isfinite(tau_hat).all()
