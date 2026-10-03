"""Tests for conformity scores.

The convention under test: **larger score = less conforming**, and the interval
is built by inverting ``score(y) <= q``. A sign error here inverts every interval
in the package while leaving all the code syntactically valid, so the direction
is asserted explicitly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import MethodError
from methods.scores import (
    absolute_score,
    aps_score,
    laplace_score,
    normalized_score,
    signed_score,
    solve_interval_from_score,
)


def test_absolute_score_is_symmetric_and_non_negative():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.5, 2.0])
    s = absolute_score(y, p)
    assert np.all(s >= 0)
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    # Symmetry: swapping prediction and target must not change the score.
    np.testing.assert_allclose(absolute_score(p, y), s)


def test_score_direction_larger_means_worse():
    """A worse prediction must score higher, or the quantile inverts."""
    good = absolute_score(np.array([1.0]), np.array([1.0]))
    bad = absolute_score(np.array([1.0]), np.array([5.0]))
    assert bad[0] > good[0]


def test_signed_score_matches_absolute_on_upper_side():
    y = np.array([1.0, 3.0])
    p = np.array([2.0, 1.0])
    np.testing.assert_allclose(signed_score(y, p), np.array([1.0, 2.0]))


def test_normalized_score_divides_by_scale():
    y = np.array([1.0, 1.0])
    p = np.array([2.0, 2.0])
    s = normalized_score(y, p, np.array([1.0, 2.0]))
    np.testing.assert_allclose(s, [1.0, 0.5])


def test_normalized_score_handles_ties_at_zero_scale():
    """A zero scale must not produce inf; the clip floor keeps it finite."""
    s = normalized_score(np.array([1.0]), np.array([2.0]), 0.0)
    assert np.all(np.isfinite(s))
    assert s[0] > 0


def test_normalized_score_broadcasts_scalar_scale():
    y = np.array([1.0, 2.0, 3.0])
    p = np.zeros(3)
    s = normalized_score(y, p, 2.0)
    np.testing.assert_allclose(s, y / 2.0)


def test_laplace_score_range_and_ordering():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    labels = np.array([0, 1, 0])
    s = laplace_score(proba, labels)
    assert np.all((s >= 0) & (s <= 1))
    # A confident correct prediction scores lower than an unsure one.
    assert s[0] < s[2]


def test_aps_score_last_class_is_exactly_one():
    """The APS convention: including the *final* class in the prefix gives 1.0.

    "Last" means lowest-ranked under the descending sort, not "label index 2".
    The true label must actually be the smallest probability in the row.
    """
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])[:, ::-1].copy()
    proba = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    last = np.array([2, 2])  # probability 0.1 in both rows -> ranked last
    s = aps_score(proba, last)
    np.testing.assert_allclose(s, np.ones(2), atol=0.0)


def test_aps_score_is_cumulative_mass():
    proba = np.array([[0.5, 0.3, 0.2]])
    s = aps_score(proba, np.array([1]))
    # true label is 2nd in descending order -> cumulative mass 0.5 + 0.3
    np.testing.assert_allclose(s, [0.8])


def test_aps_score_ties_deterministic_without_rng():
    proba = np.array([[0.5, 0.5]])
    a = aps_score(proba, np.array([0]))
    b = aps_score(proba, np.array([0]))
    assert a[0] == b[0]


def test_scores_reject_misaligned_shapes():
    with pytest.raises(MethodError):
        absolute_score(np.zeros(3), np.zeros(4))
    with pytest.raises(MethodError):
        absolute_score(np.array([]), np.array([]))
    with pytest.raises(MethodError):
        laplace_score(np.zeros((3, 2)), np.zeros(4))


def test_solve_interval_absolute_inverts_correctly():
    y_pred = np.array([10.0, 20.0])
    lo, hi = solve_interval_from_score("absolute", y_pred, 2.0)
    np.testing.assert_allclose(lo, [8.0, 18.0])
    np.testing.assert_allclose(hi, [12.0, 22.0])


def test_solve_interval_normalized_scales_bounds():
    y_pred = np.array([10.0])
    lo, hi = solve_interval_from_score("normalized", y_pred, 2.0, scale=np.array([3.0]))
    np.testing.assert_allclose(lo, [4.0])
    np.testing.assert_allclose(hi, [16.0])


def test_solve_interval_rejects_unknown_and_missing_scale():
    with pytest.raises(MethodError):
        solve_interval_from_score("bogus", np.array([1.0]), 1.0)
    with pytest.raises(MethodError):
        solve_interval_from_score("normalized", np.array([1.0]), 1.0)
