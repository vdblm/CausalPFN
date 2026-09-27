from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import pytest

from causalpfn import CATEEstimator


APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def test_import_does_not_load_faiss():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import causalpfn; assert 'faiss' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.skipif(not APPLE_SILICON, reason="Apple Silicon regression test")
def test_fit_and_predict_do_not_segfault():
    """Exercise both locations where the duplicate OpenMP runtime crashed."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    T = rng.binomial(1, 0.5, size=300).astype(np.float32)
    Y = (X[:, 0] - X[:, 1] + T + rng.normal(scale=0.1, size=300)).astype(np.float32)

    estimator = CATEEstimator(device="cpu", verbose=False)
    estimator.fit(X, T, Y)
    estimate = np.asarray(estimator.estimate_cate(X))

    assert estimate.shape == (len(X),)
    assert np.isfinite(estimate).all()
