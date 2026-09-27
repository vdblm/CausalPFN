from __future__ import annotations

import numpy as np
import pytest

from causalpfn._nearest_neighbors import nearest_indices_1d


def _brute_force(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Small reference implementation with the documented tie-break."""
    k = min(k, len(reference))
    indices = np.arange(len(reference))
    return np.stack(
        [np.lexsort((indices, np.abs(reference - value)))[:k] for value in query]
    )


def test_matches_brute_force_with_duplicates_and_ties():
    rng = np.random.default_rng(7)
    reference = np.round(rng.normal(size=300), decimals=1)
    query = np.round(rng.normal(size=100), decimals=1)

    actual = nearest_indices_1d(reference, query, k=64)
    expected = _brute_force(reference, query, k=64)

    for actual_row, expected_row in zip(actual, expected):
        np.testing.assert_array_equal(np.sort(actual_row), np.sort(expected_row))


def test_ties_prefer_lower_input_indices():
    reference = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    query = np.array([-1.0, 1.0, 3.0])

    actual = nearest_indices_1d(reference, query, k=2)

    np.testing.assert_array_equal(actual, [[0, 1], [0, 1], [3, 4]])


def test_k_is_capped_without_sentinel_indices():
    actual = nearest_indices_1d(
        np.array([0.0, 1.0]), np.array([-1.0, 0.5, 2.0]), k=10
    )

    assert actual.shape == (3, 2)
    assert (actual >= 0).all()
    np.testing.assert_array_equal(np.sort(actual, axis=1), [[0, 1], [0, 1], [0, 1]])


def test_large_search_does_not_materialize_pairwise_distances():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=50_000)
    query = rng.normal(size=10_000)

    actual = nearest_indices_1d(reference, query, k=32)

    assert actual.shape == (10_000, 32)
    assert actual.min() >= 0
    assert actual.max() < len(reference)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        nearest_indices_1d(np.array([0.0]), np.array([0.0]), k=k)


def test_rejects_empty_or_non_finite_reference():
    with pytest.raises(ValueError, match="at least one"):
        nearest_indices_1d(np.array([]), np.array([0.0]), k=1)
    with pytest.raises(ValueError, match="finite"):
        nearest_indices_1d(np.array([np.nan]), np.array([0.0]), k=1)
