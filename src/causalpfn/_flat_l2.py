"""Exact brute-force 1-D L2 nearest-neighbour search, without FAISS.

Why this exists
----------------
``faiss-cpu`` and ``torch`` each ship their own private OpenMP runtime
(``libomp.dylib``). On Apple Silicon, having both loaded in the same process
is unsafe: whichever loads first claims the runtime, and the other's
parallel loops end up corrupting memory. The corruption doesn't crash
immediately -- it surfaces later, at the first large tensor op that uses the
loser's copy, which can land anywhere (model weight loading, or a later
FAISS call) depending on which library's parallel region runs first.
Reordering the ``faiss``/``torch`` imports only changes *where* the crash
happens, not *whether* it happens, because both OpenMP runtimes still get
loaded into the process either way.

``CausalEstimator._predict_cepo`` is the only place this package touches
FAISS, and only for ``IndexFlatL2(1)`` -- an exact, brute-force,
one-dimensional L2 index over weak-learner effect estimates. That's small
enough that a plain NumPy implementation removes the second OpenMP runtime
entirely (so torch keeps every thread) with no accuracy or meaningful
performance cost, and works identically on every platform, not just Apple
Silicon.

Semantics reproduced from ``faiss.IndexFlatL2``
------------------------------------------------
* ``search`` returns ``(distances, indices)`` with **squared** L2 distances.
* Results are ordered by increasing distance.
* When ``k > ntotal``, missing slots are padded with index ``-1`` and
  distance ``FLT_MAX`` (not ``inf`` -- FAISS uses the float32 max as its
  sentinel); this package does request more neighbours than a treatment arm
  holds when that arm is small, so the sentinel path is live, not
  theoretical.
"""

from __future__ import annotations

import numpy as np

__all__ = ["IndexFlatL2"]

_FLT_MAX = float(np.finfo(np.float32).max)


class IndexFlatL2:
    """Exact brute-force L2 index. API-compatible subset of faiss.IndexFlatL2."""

    def __init__(self, d: int):
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        self.d = int(d)
        self.is_trained = True
        self._xb = np.empty((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    def add(self, x) -> None:
        xb = np.ascontiguousarray(x, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.d:
            raise ValueError(f"expected [n, {self.d}] array, got shape {xb.shape}")
        self._xb = xb.copy() if self.ntotal == 0 else np.vstack([self._xb, xb])

    def search(self, x, k: int):
        xq = np.ascontiguousarray(x, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.d:
            raise ValueError(f"expected [n, {self.d}] array, got shape {xq.shape}")
        k = int(k)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        nq, nb = xq.shape[0], self.ntotal
        kk = min(k, nb)

        dist = np.full((nq, k), _FLT_MAX, dtype=np.float32)
        idx = np.full((nq, k), -1, dtype=np.int64)
        if nq == 0 or kk == 0:
            return dist, idx

        # ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2, in float64 so the expansion
        # does not lose the small differences this is being asked to rank.
        xq64, xb64 = xq.astype(np.float64), self._xb.astype(np.float64)
        d2 = (
            (xq64 * xq64).sum(1)[:, None]
            - 2.0 * (xq64 @ xb64.T)
            + (xb64 * xb64).sum(1)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)  # clamp expansion round-off below zero

        if kk < nb:
            part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            part_d = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(part_d, axis=1, kind="stable")
            top = np.take_along_axis(part, order, axis=1)
        else:
            top = np.argsort(d2, axis=1, kind="stable")[:, :kk]

        idx[:, :kk] = top
        dist[:, :kk] = np.take_along_axis(d2, top, axis=1).astype(np.float32)
        return dist, idx
