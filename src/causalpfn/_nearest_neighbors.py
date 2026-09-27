"""Small, dependency-free nearest-neighbour helpers used by CausalPFN."""

from __future__ import annotations

import numpy as np


def nearest_indices_1d(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` nearest reference values per query.

    In one dimension, the nearest values form a contiguous window after the
    reference values are sorted. Finding that window avoids materializing the
    quadratic query-by-reference distance matrix. Exact ties are resolved by
    the original reference index, making results deterministic.

    ``k`` is capped at the number of reference values, so no sentinel indices
    are returned.

    Args:
        reference: One-dimensional array with shape ``(n_reference,)``.
        query: One-dimensional array with shape ``(n_query,)``.
        k: Requested number of neighbours per query.

    Returns:
        Integer indices into ``reference`` with shape
        ``(n_query, min(k, n_reference))``. Neighbour order within a row is not
        part of the API; callers consume each row as a set.
    """
    reference = np.asarray(reference)
    query = np.asarray(query)
    if reference.ndim != 1 or query.ndim != 1:
        raise ValueError("reference and query must be one-dimensional")
    if reference.size == 0:
        raise ValueError("reference must contain at least one value")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("reference and query must contain only finite values")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    k = min(int(k), reference.size)
    if query.size == 0:
        return np.empty((0, k), dtype=np.intp)
    if k == reference.size:
        return np.broadcast_to(np.arange(reference.size), (query.size, reference.size)).copy()

    # Calculate window boundaries in float64 so close float32 effects remain
    # distinguishable. Stable sorting preserves input order within value ties.
    order = np.argsort(reference, kind="stable")
    values = reference[order].astype(np.float64, copy=False)
    query_values = query.astype(np.float64, copy=False)
    boundaries = values[:-k] + (values[k:] - values[:-k]) / 2
    starts = np.searchsorted(boundaries, query_values, side="left")
    neighbours = order[starts[:, None] + np.arange(k)]

    # A contiguous window is unique unless an excluded endpoint is tied with
    # its farthest included endpoint. Resolve only those uncommon rows with a
    # linear selection, using the lower original index as the tie-breaker.
    threshold = np.maximum(
        np.abs(values[starts] - query_values),
        np.abs(values[starts + k - 1] - query_values),
    )
    left_distance = np.full(query.size, np.inf)
    right_distance = np.full(query.size, np.inf)
    has_left = starts > 0
    has_right = starts + k < reference.size
    left_distance[has_left] = np.abs(values[starts[has_left] - 1] - query_values[has_left])
    right_distance[has_right] = np.abs(values[starts[has_right] + k] - query_values[has_right])
    ambiguous = np.minimum(left_distance, right_distance) <= threshold

    original_indices = np.arange(reference.size)
    reference_values = reference.astype(np.float64, copy=False)
    for query_value in np.unique(query_values[ambiguous]):
        distances = np.abs(reference_values - query_value)
        candidate = np.argpartition(distances, k - 1)[:k]
        radius = distances[candidate].max()
        closer = original_indices[distances < radius]
        tied = original_indices[distances == radius]
        selected = np.concatenate([closer, tied[: k - closer.size]])
        neighbours[query_values == query_value] = selected

    return neighbours
