from __future__ import annotations

import numpy as np
import pytest

from causalpfn import CATEEstimator


# Produced by CausalPFN 0.1.4 with faiss.IndexFlatL2 on the fixed data below.
LEGACY_FAISS_CATE = np.array(
    [
        -0.17869198,
        -0.91237175,
        1.129458,
        -0.5724335,
        -0.5114931,
        -0.07886714,
        0.7077167,
        0.8638369,
        0.9209615,
        0.64013636,
        0.60120267,
        -0.71298707,
        -1.1388453,
        0.55269885,
        -1.4786408,
        0.43212175,
    ],
    dtype=np.float32,
)


@pytest.mark.slow
def test_cate_matches_legacy_faiss_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2_200, 5)).astype(np.float32)
    treatment_effect = (np.sin(X[:, 0]) + 0.5 * X[:, 1]).astype(np.float32)
    T = rng.binomial(1, 0.5, size=len(X)).astype(np.float32)
    baseline = (X[:, 0] - X[:, 1] + rng.normal(0, 0.1, size=len(X))).astype(np.float32)
    Y = baseline + treatment_effect * T

    estimator = CATEEstimator(device="cpu", verbose=False)
    estimator.fit(X, T, Y)
    actual = np.asarray(estimator.estimate_cate(X[: len(LEGACY_FAISS_CATE)]))

    np.testing.assert_allclose(actual, LEGACY_FAISS_CATE, rtol=1e-5, atol=1e-5)
