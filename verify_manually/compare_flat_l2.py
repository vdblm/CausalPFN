"""Compare the new causalpfn._flat_l2.IndexFlatL2 against real faiss.IndexFlatL2, torch-free."""
import importlib.util
import sys

import faiss
import numpy as np

spec = importlib.util.spec_from_file_location(
    "flat_l2_new", "/Users/hosseinrahmani/Documents/CausalPFN/src/causalpfn/_flat_l2.py"
)
flat_l2_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat_l2_new)

rng = np.random.RandomState(0)


def compare(nb, nq, k, label):
    xb = rng.randn(nb, 1).astype(np.float32)
    xq = rng.randn(nq, 1).astype(np.float32)

    ref = faiss.IndexFlatL2(1)
    ref.add(xb)
    ref_d, ref_i = ref.search(xq, k)

    new = flat_l2_new.IndexFlatL2(1)
    new.add(xb)
    new_d, new_i = new.search(xq, k)

    idx_match = np.array_equal(ref_i, new_i)
    dist_close = np.allclose(ref_d, new_d, rtol=0, atol=1e-4)
    dist_exact = np.array_equal(ref_d, new_d)
    print(f"[{label}] nb={nb} nq={nq} k={k}: idx_match={idx_match} dist_exact={dist_exact} dist_close={dist_close}")
    if not idx_match:
        print("  ref_i[:3]=", ref_i[:3])
        print("  new_i[:3]=", new_i[:3])
    if not dist_close:
        print("  MAX ABS DIFF:", np.max(np.abs(ref_d.astype(np.float64) - new_d.astype(np.float64))))
    return idx_match and dist_close


results = []
results.append(compare(nb=50, nq=20, k=5, label="normal"))
results.append(compare(nb=5, nq=10, k=5, label="k==ntotal"))
results.append(compare(nb=3, nq=10, k=10, label="k>ntotal (sentinel padding)"))
results.append(compare(nb=200, nq=100, k=15, label="larger"))
results.append(compare(nb=1, nq=5, k=3, label="single-point arm"))

print()
if all(results):
    print("ALL COMPARISONS PASSED: new IndexFlatL2 matches real faiss.IndexFlatL2")
    sys.exit(0)
else:
    print("MISMATCH DETECTED")
    sys.exit(1)
