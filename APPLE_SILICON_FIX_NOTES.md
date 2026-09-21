# Apple Silicon segfault + `estimate_ate_CI` KeyError — working notes

**Status: not yet opened as a PR. Do not push/open a PR from this branch until told to.**
This file is a working note for local review, not intended to be part of the
upstream PR diff — delete it before opening the PR.

## The two bugs

1. **Segfault on Apple Silicon.** `causal_estimator.py` imports both `faiss`
   and `torch`. Each bundles its own private OpenMP runtime
   (`libomp.dylib`). Loading both into one process is unsafe on Apple
   Silicon: whichever loads first claims the runtime, and the other's
   parallel ops corrupt memory. The corruption doesn't crash immediately —
   it surfaces later, wherever that runtime's code next executes.
2. **`estimate_ate_CI` raises `KeyError: 'ate'` on every call.**
   `_estimate_ate_cate_CI` computes `ate_samples` but never puts a point
   estimate in its returned dict; `estimate_ate_CI` reads `output["ate"]`,
   which never existed.

Both were originally root-caused by external contributor **Max De Marzi**,
in [layer6ai-labs/cfms#2](https://github.com/layer6ai-labs/cfms/pull/2) — a
downstream repo that carries a workaround for both (a NumPy stand-in for
FAISS, `causal_bench/faiss_shim.py`, and a routed-around CI helper call).

## A wrong fix was tried first — worth remembering why

The first attempt at an upstream fix was "just import `torch` before
`faiss`" (alphabetical order was the accidental cause of `faiss` claiming
the OpenMP runtime first). This was **not actually verified before being
acted on** — it was proposed, then a PR was opened claiming it worked,
without running the code. It turned out to be wrong:

> Reordering only changes *which* library's OpenMP runtime wins — both
> still get loaded into the process either way. It doesn't remove the
> collision, it just relocates it. Verified directly: with only the import
> reorder applied, `CATEEstimator.fit()` succeeds (no crash during weight
> loading), but the very next call, `estimate_cate()`, segfaults instead —
> at the `faiss.IndexFlatL2` call site inside `_predict_cepo`.

That PR was closed and the incorrect claim to Max was retracted. Lesson:
**don't claim a fix works without running it** — this note exists partly
so that doesn't happen again on this branch.

## The actual fix (this branch)

Remove `faiss` from `causalpfn` entirely, rather than trying to make two
OpenMP runtimes coexist:

- `_predict_cepo` is the *only* call site, and only for `IndexFlatL2(1)` —
  an exact, brute-force, one-dimensional L2 index over weak-learner effect
  estimates.
- `src/causalpfn/_flat_l2.py` reimplements just that in NumPy (~90 lines),
  API-compatible with the subset of `faiss.IndexFlatL2` actually used
  (`add`, `search`, sentinel padding when `k > ntotal`).
- `faiss-cpu` is dropped from `pyproject.toml` (core + dev deps) — it's no
  longer imported anywhere in the package.
- `estimate_ate_CI`'s dict now includes `"ate": ate_samples.mean()`.

This removes the double-OpenMP-runtime problem on **every** platform, not
just Apple Silicon, with no accuracy cost (see verification below) and no
`OMP_NUM_THREADS=1` performance penalty (~1.5x slower k-NN, which is why
the downstream repo didn't just set that instead).

## Verification already done (see `verify_manually/` in this repo)

1. **`verify_manually/test_fix.py`** — fits, calls `estimate_cate()`
   (the exact call that segfaulted under the wrong fix), calls
   `estimate_ate_CI()`. Run with `-X faulthandler` so a real crash shows
   the exact line. Result: exit 0, no crash, on real Apple Silicon
   hardware, with `faiss` confirmed absent from the environment entirely.
2. **`verify_manually/compare_flat_l2.py`** — cross-checks the new
   `_flat_l2.IndexFlatL2` against real `faiss.IndexFlatL2` (in a
   torch-free venv — the only way real faiss and this test can coexist
   without the segfault) across 5 scenarios: normal, `k == ntotal`,
   `k > ntotal` (sentinel padding), a larger case, and a single-point
   treatment arm. Returned neighbor indices matched exactly in every case
   (indices are all `_predict_cepo` uses; distances are discarded but were
   confirmed numerically close too).
3. **`verify_manually/test_quickstart_repro.py`** — reproduces
   `notebooks/Foundation_models_quickstart.ipynb`'s exact `SEED=42`
   scenario from the downstream `cfms` repo, using the real pretrained
   model (no shim of any kind — this branch doesn't import `faiss` at
   all). Result matched the notebook's documented Colab GPU reference
   exactly: `ATE_hat=1.911, True_ATE=1.967, PEHE=0.237`.

See each script's own docstring/comments for exact setup commands
(isolated `uv venv`s, install-from-local-branch, etc.).

### Real test suite added (this repo had none)

`tests/` now has 13 pytest tests, ported from Max De Marzi's 16-test suite
against the downstream `cfms` repo's workaround (`layer6ai-labs/cfms#2`),
adapted to test this package directly:
- `test_flat_l2.py` — unit tests + real-faiss cross-check (torch-free
  subprocess), 7 tests
- `test_apple_silicon.py` — end-to-end `CATEEstimator`/`ATEEstimator` on
  Apple Silicon, 3 tests (`slow`, downloads weights)
- `test_estimate_ate_ci.py` — the downstream repo's "still broken"
  regression test inverted into a positive correctness test, 3 tests
  (1 fast, 2 `slow`)

All 13 pass on real Apple Silicon: `pytest tests/` from the repo root
(needs `pip install -e . pytest`, plus `faiss-cpu` if you want the
cross-check to actually run rather than skip).

## Before opening the PR

- [ ] Re-run all three verification scripts fresh (don't trust a stale
      run — re-verify against the current state of this branch)
- [ ] Delete this file — it's a working note, not upstream-PR content
- [ ] Confirm with the person running this whether to proceed
- [ ] PR should credit Max De Marzi's original root-cause work
      ([layer6ai-labs/cfms#2](https://github.com/layer6ai-labs/cfms/pull/2))
