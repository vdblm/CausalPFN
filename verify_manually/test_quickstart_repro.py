"""
Reproduce notebooks/Foundation_models_quickstart.ipynb's exact scenario (SEED=42)
against the corrected causalpfn branch (fix/apple-silicon-drop-faiss), with NO shim
of any kind -- this branch doesn't import faiss at all.

Reference (verified Colab GPU run, from the notebook's own markdown cell):
  True ATE (known only in this simulation): 1.967
  ATE_hat=1.911  True_ATE=1.967  PEHE=0.237
"""
import faulthandler
faulthandler.enable()

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)
n = 1500

recency, monetary, age = rng.normal(0, 1, (3, n)).astype(np.float32)
X = np.column_stack([recency, monetary, age])

propensity = 1 / (1 + np.exp(-(0.8 * monetary - 0.6 * recency)))
T = rng.binomial(1, propensity).astype(np.float32)

true_cate = (2.0 + 1.5 * recency - 0.75 * age).astype(np.float32)
Y0 = (5.0 + 2.0 * monetary - 0.5 * age + rng.normal(0, 1, n)).astype(np.float32)
Y = np.where(T == 1, Y0 + true_cate, Y0).astype(np.float32)

X_ctx, X_qry, T_ctx, T_qry, Y_ctx, Y_qry, _, cate_qry = train_test_split(
    X, T, Y, true_cate, test_size=0.3, random_state=SEED
)
true_ate = float(true_cate.mean())
print(f"True ATE (known only in this simulation): {true_ate:.3f}", flush=True)

import sys  # noqa: E402
assert "faiss" not in sys.modules, "faiss should never be imported by the fixed branch"
try:
    import faiss  # noqa: E402
    raise AssertionError("faiss is installed in this env but should not be -- test isolation broken")
except ModuleNotFoundError:
    print("Confirmed: faiss is not installed in this environment at all.", flush=True)

from causalpfn import CATEEstimator, ATEEstimator  # noqa: E402

device = "cpu"

cate_estimator = CATEEstimator(device=device, verbose=False)
cate_estimator.fit(X_ctx, T_ctx, Y_ctx)
cate_hat = np.asarray(cate_estimator.estimate_cate(X_qry)).reshape(-1)
print("estimate_cate succeeded, no segfault.", flush=True)

ate_estimator = ATEEstimator(device=device, verbose=False)
ate_estimator.fit(X_ctx, T_ctx, Y_ctx)
ate_hat = float(np.asarray(ate_estimator.estimate_ate()).reshape(-1)[0])

pehe = float(np.sqrt(np.mean((cate_hat - cate_qry) ** 2)))

print(f"ATE_hat={ate_hat:.3f}  True_ATE={true_ate:.3f}  PEHE={pehe:.3f}", flush=True)
print("Reference (Colab GPU run): ATE_hat=1.911  True_ATE=1.967  PEHE=0.237", flush=True)

cate_ci_estimator = CATEEstimator(device=device, verbose=False)
cate_ci_estimator.fit(X_ctx, T_ctx, Y_ctx)
ci = cate_ci_estimator.estimate_ate_CI(X_qry, alpha=0.05)
assert "ate" in ci
print(f"estimate_ate_CI succeeded: ate={float(np.asarray(ci['ate'])):.3f} "
      f"[{float(np.asarray(ci['lower_bound']).ravel()[0]):.3f}, "
      f"{float(np.asarray(ci['upper_bound']).ravel()[0]):.3f}]", flush=True)

print("\nALL CHECKS PASSED.", flush=True)
