"""
Standalone test of the fixed causalpfn branch (rahmani-hossein/CausalPFN@fix/apple-silicon-openmp-and-ate-ci),
deliberately WITHOUT causal_bench's faiss_shim/macos_compat workaround, to prove the two bugs
are fixed at the source:
  1. Apple Silicon OpenMP segfault (faiss/torch import order)
  2. estimate_ate_CI KeyError('ate')
"""
import faulthandler
faulthandler.enable()

import numpy as np
from causalpfn import CATEEstimator


def log(msg):
    print(msg, flush=True)


rng = np.random.RandomState(42)
n = 500
X = rng.randn(n, 5).astype(np.float32)
T = rng.binomial(1, 0.5, n).astype(np.float32)
tau_true = np.sin(X[:, 0]) + 0.5 * X[:, 1]
Y = X[:, 0] + T * tau_true + rng.randn(n).astype(np.float32) * 0.1

log("Fitting CATEEstimator (this is where the old code segfaulted on Apple Silicon)...")
est = CATEEstimator(device="cpu", verbose=False)
est.fit(X, T, Y)
log("Fit succeeded, no segfault.")

log("Calling estimate_cate (this is what actually invokes faiss.IndexFlatL2)...")
tau_hat = est.estimate_cate(X)
pehe = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
log(f"estimate_cate OK. PEHE={pehe:.4f}")

log("Calling estimate_ate_CI (this raised KeyError('ate') on the old code)...")
ci = est.estimate_ate_CI(X, alpha=0.05)
log(f"estimate_ate_CI succeeded: { {k: np.asarray(v).ravel()[:1] for k, v in ci.items()} }")

assert "ate" in ci, "Missing 'ate' key -- bug not fixed"
log("\nALL CHECKS PASSED: no segfault, and estimate_ate_CI returned a valid 'ate' key.")
