"""Tests for the conformal quantile: the finite-sample correction.

These are the tests that matter most in the package. Every validity claim in
Confforge bottoms out in :func:`methods.quantiles.conformal_quantile`, and the
three ways to get it subtly wrong -- dropping the ``n+1``, interpolating, and
wrapping at ``k == 0`` -- all produce code that *runs* and returns a plausible
number.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from core.errors import DataError, MethodError
from methods.quantiles import (
    conformal_index,
    conformal_quantile,
    conformal_quantile_batch,
    is_infinite_quantile,
)


def test_index_uses_n_plus_one_correction():
    """k must be ceil((n+1)(1-alpha)), never ceil(n(1-alpha))."""
    n, alpha = 100, 0.1
    assert conformal_index(n, alpha) == math.ceil((n + 1) * (1 - alpha))
    # The naive formula gives 90; the correct one gives 91.
    assert conformal_index(n, alpha) != math.ceil(n * (1 - alpha))


def test_index_is_one_based():
    """Index is 1-based so it can address scores[k-1] directly."""
    assert conformal_index(10, 0.1) >= 1


def test_index_clips_at_one_never_wraps_to_zero():
    """k == 0 would wrap to scores[-1] (the maximum) and silently over-cover."""
    # (n+1)(1-alpha) < 1 requires alpha > n/(n+1).
    k = conformal_index(10, 0.999)
    assert k == 1, "must clip to 1 rather than reaching 0"
    scores = np.arange(1.0, 11.0)
    q = conformal_quantile(scores, 0.999)
    assert q == 1.0, "must return the smallest score, not the largest"


def test_index_never_exceeds_n_when_quantile_exists():
    """k may exceed n for a thin calibration set -- that is the +inf case.

    Asserting ``k <= n`` unconditionally would be wrong: ``n=1, alpha=0.1``
    gives ``k = ceil(2*0.9) = 2 > 1``, and that is precisely the situation
    :func:`conformal_quantile` reports as an infinite quantile rather than
    silently clipping the confidence level.
    """
    for n in (5, 19, 100, 1000):
        for alpha in (0.01, 0.1, 0.5, 0.9):
            k = conformal_index(n, alpha)
            assert k >= 1
            if k <= n:
                assert conformal_quantile(np.arange(float(n)), alpha) == float(k - 1)


def test_index_rejects_non_positive_n():
    with pytest.raises(DataError):
        conformal_index(0, 0.1)
    with pytest.raises(DataError):
        conformal_index(-5, 0.1)


def test_index_rejects_invalid_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(MethodError):
            conformal_index(10, bad)


def test_quantile_returns_observed_value_no_interpolation():
    """The conformal quantile is an order statistic, never a blend of two."""
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    q = conformal_quantile(scores, 0.2)
    assert q in set(scores.tolist()), "must be an actually observed score"


def test_quantile_is_permutation_invariant():
    """Reordering the calibration set must not change the quantile's value."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    reference = conformal_quantile(scores, 0.1)
    for _ in range(5):
        shuffled = rng.permutation(scores)
        assert conformal_quantile(shuffled, 0.1) == reference


def test_quantile_is_monotone_in_alpha():
    """Larger alpha (less confidence) must never demand a *wider* interval.

    The quantile is therefore non-increasing in alpha.
    """
    rng = np.random.default_rng(1)
    scores = np.abs(rng.normal(size=300))
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    quantiles = [conformal_quantile(scores, a) for a in alphas]
    assert all(b <= a for a, b in pairwise(quantiles)), (
        f"quantiles must be non-increasing in alpha, got {quantiles}"
    )


def test_quantile_rejects_empty_and_non_finite():
    with pytest.raises(DataError):
        conformal_quantile(np.array([]), 0.1)
    with pytest.raises(DataError):
        conformal_quantile(np.array([1.0, np.nan, 3.0]), 0.1)
    with pytest.raises(DataError):
        conformal_quantile(np.array([1.0, np.inf]), 0.1)


def test_thin_calibration_infinite_or_raise():
    """k > n has no order statistic: raise, or return +inf with permission."""
    scores = np.arange(1.0, 4.0)  # n = 3
    with pytest.raises(MethodError):
        conformal_quantile(scores, 0.1)  # k = ceil(4*0.9) = 4 > 3
    q = conformal_quantile(scores, 0.1, allow_infinite=True)
    assert is_infinite_quantile(q)


def test_batch_matches_scalar():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=150)
    alphas = [0.05, 0.1, 0.2, 0.3]
    batch = conformal_quantile_batch(scores, alphas)
    for a, got in zip(alphas, batch, strict=True):
        assert got == conformal_quantile(scores, a)


def test_batch_preserves_input_order():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=100)
    alphas = [0.3, 0.1, 0.2]
    batch = conformal_quantile_batch(scores, alphas)
    for a, got in zip(alphas, batch, strict=True):
        assert got == conformal_quantile(scores, a)


def test_monte_carlo_coverage_guarantee():
    """Empirical coverage >= 1 - alpha - 0.02 for a large calibration set.

    This is the guarantee itself, tested end to end: draw exchangeable
    calibration and test scores, calibrate, and check the induced interval
    covers at least the nominal rate.
    """
    alpha = 0.1
    rng = np.random.default_rng(42)
    n_calib = 2000
    trials = 400
    hits = 0
    total = 0
    for _ in range(trials):
        # The conformity score for split conformal IS the absolute residual, so
        # the calibration scores must be absolute too. Feeding signed normals
        # while testing |test| <= q compares a one-sided quantile against a
        # two-sided event and under-covers at ~0.78 -- a good illustration of how
        # a score/interval mismatch produces a plausible but wrong number.
        calib = np.abs(rng.normal(size=n_calib))
        test = abs(rng.normal())
        q = conformal_quantile(calib, alpha)
        hits += int(test <= q)
        total += 1
    coverage = hits / total
    assert coverage >= 1 - alpha - 0.02, f"coverage {coverage} below {1 - alpha - 0.02}"


def test_interval_nesting():
    """A larger alpha must contain the interval of a smaller one."""
    rng = np.random.default_rng(4)
    scores = np.abs(rng.normal(size=500))
    order = [0.01, 0.05, 0.1, 0.2, 0.4]
    qs = [conformal_quantile(scores, a) for a in order]
    assert all(b <= a for a, b in pairwise(qs)), (
        f"wider-confidence intervals must nest inside narrower ones, got {qs}"
    )


def test_is_infinite_quantile_discriminates():
    assert is_infinite_quantile(float("inf"))
    assert not is_infinite_quantile(1.0)
    assert not is_infinite_quantile(float("nan"))


def test_conformal_quantile_deterministic():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=250)
    values = {conformal_quantile(scores, 0.1) for _ in range(20)}
    assert len(values) == 1
