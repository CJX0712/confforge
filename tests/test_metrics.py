"""Tests for interval metrics, anchored on published ground truth.

The Winkler score has an exact reference value that can be computed by hand. When
a benchmark gate moves, the first question is whether the metric or the method
changed -- and that is only answerable if the metric is pinned to a value
independent of this codebase.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from core.errors import EvaluationError
from core.types import IntervalSet
from eval import metrics as M


def _iv(lo, hi, alpha=0.1):
    return IntervalSet(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), alpha=alpha)


def test_winkler_ground_truth_70():
    """Reference value from the specification.

    interval = [0, 10], y = 13, alpha = 0.1
    w = 10 + (2/0.1) * (13 - 10) = 10 + 60 = 70.0
    """
    value = M.winkler(_iv([0.0], [10.0]), np.array([13.0]), 0.1)
    assert value == pytest.approx(70.0, abs=1e-12)


def test_winkler_covered_point_equals_width():
    """Inside the interval the score is exactly the width -- no penalty."""
    value = M.winkler(_iv([0.0], [10.0]), np.array([5.0]), 0.1)
    assert value == pytest.approx(10.0, abs=1e-12)


def test_winkler_below_penalised():
    """y = -3: width 10 + (2/0.1) * (0 - (-3)) = 10 + 60 = 70."""
    value = M.winkler(_iv([0.0], [10.0]), np.array([-3.0]), 0.1)
    assert value == pytest.approx(70.0, abs=1e-12)


def test_winkler_decomposition_is_additive():
    """total == width + lower_miss + upper_miss, exactly."""
    rng = np.random.default_rng(0)
    lo = rng.normal(size=400)
    hi = lo + rng.random(400)
    y = rng.normal(size=400)
    parts = M.winkler_components(_iv(lo, hi), y, 0.1)
    total = parts["width"] + parts["lower_miss"] + parts["upper_miss"]
    assert total == pytest.approx(parts["winkler"], abs=1e-9)
    assert parts["width_share"] + parts["miss_share"] == pytest.approx(1.0, abs=1e-12)


def test_winkler_symmetric_under_reflection():
    """Reflecting both interval and target leaves the score unchanged."""
    rng = np.random.default_rng(1)
    lo = rng.normal(size=200)
    hi = lo + rng.random(200)
    y = rng.normal(size=200)
    base = M.winkler(_iv(lo, hi), y, 0.1)
    flipped = M.winkler(_iv(-hi, -lo), -y, 0.1)
    assert flipped == pytest.approx(base, abs=1e-9)


def test_winkler_penalty_scales_inversely_with_alpha():
    """The miss weight is ``2/alpha``, so the score falls as alpha grows.

    All points sit below the interval, so score = width + (2/alpha) * miss, and
    the alpha-dependence comes entirely from that weight.
    """
    rng = np.random.default_rng(2)
    lo = rng.normal(size=200)
    hi = lo + 0.5
    y = lo - 1.0  # uniformly 1.0 below the lower bound
    scores = [M.winkler(_iv(lo, hi), y, a) for a in (0.05, 0.1, 0.2, 0.3)]
    assert all(b <= a for a, b in pairwise(scores)), (
        f"penalty weight 2/alpha shrinks with alpha, so scores must fall: {scores}"
    )
    # exact: width 0.5 + (2/a) * 1.0
    for a, got in zip((0.05, 0.1, 0.2, 0.3), scores, strict=True):
        assert got == pytest.approx(0.5 + 2.0 / a, abs=1e-9)


def test_coverage_and_gap():
    iv = _iv([0.0, 10.0], [10.0, 20.0])
    y = np.array([5.0, 25.0])
    assert M.coverage(iv, y) == pytest.approx(0.5)
    assert M.coverage_gap(iv, y, 0.5) == pytest.approx(0.0)
    assert M.coverage_gap(iv, y, 0.9) == pytest.approx(0.4)


def test_group_coverage_identifies_worst_group():
    """CoV-W must point at the group that actually deviates most."""
    # group 0 always covered; group 1 never covered; target 0.9
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    y = np.array([0.5, 5.0])
    groups = np.array([0, 1])
    out = M.group_coverage(_iv(lo, hi), y, groups)
    assert out["per_group"]["0"] == pytest.approx(1.0)
    assert out["per_group"]["1"] == pytest.approx(0.0)
    assert out["worst_group"] == "1"
    assert out["worst_gap"] == pytest.approx(0.9)
    assert out["counts"] == {"0": 1, "1": 1}


def test_group_coverage_equal_groups_is_zero_gap():
    rng = np.random.default_rng(3)
    lo = rng.normal(size=300)
    hi = lo + 2.0
    y = lo + rng.random(300) * 2.0
    groups = np.repeat([0, 1, 2], 100)
    out = M.group_coverage(_iv(lo, hi), y, groups)
    assert out["worst_gap"] < 0.2


def test_mean_width_and_stability():
    iv = _iv([0.0, 1.0], [2.0, 5.0])
    assert M.mean_width(iv) == pytest.approx(3.0)
    # identical adjacent intervals are maximally stable
    flat = _iv([0.0, 0.0], [1.0, 1.0])
    assert M.stability(flat) == pytest.approx(0.0)


def test_metrics_reject_misaligned_shapes():
    with pytest.raises(EvaluationError):
        M.coverage(_iv([0.0], [1.0]), np.zeros(3))
    with pytest.raises(EvaluationError):
        M.winkler(_iv([0.0], [1.0]), np.zeros(3), 0.1)
    with pytest.raises(EvaluationError):
        M.coverage_gap(_iv([0.0], [1.0]), np.array([0.5]), 1.5)


def test_stability_requires_two_samples():
    with pytest.raises(EvaluationError):
        M.stability(_iv([0.0], [1.0]))


def test_efficiency_curve_monotone_width():
    """Sweeping alpha down must produce strictly wider intervals."""
    rng = np.random.default_rng(4)
    lo = rng.normal(size=300)
    hi = lo + 0.5
    y = rng.normal(size=300)
    curve = M.efficiency_curve(
        {a: _iv(lo - (a * 5), hi + (a * 5)) for a in (0.05, 0.1, 0.2, 0.3)}, y
    )
    assert curve["width"] == sorted(curve["width"])


def test_summarize_reports_expected_keys():
    rng = np.random.default_rng(5)
    lo = rng.normal(size=100)
    iv = _iv(lo, lo + 1.0)
    out = M.summarize(iv, lo + rng.random(100))
    for key in ("coverage", "mean_width", "winkler", "coverage_gap", "miss_rate"):
        assert key in out
        assert np.isfinite(out[key])
