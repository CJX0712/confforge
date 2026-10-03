"""Tests for the optional MAPIE backend.

Two things are being defended here.

**Graceful degradation.** MAPIE is an optional dependency. When it is missing the
package must still work, and the benchmark must record a *skip with a reason*
rather than crashing or silently dropping the cells.

**The 1.5.0 return contract.** MAPIE <= 1.0 returned ``(y_min, y_max)``; 1.5.0
returns ``(y_pred, intervals)`` with ``intervals`` of shape ``(n, 2, n_alpha)``.
Writing the old unpacking against the new version raises nothing and returns
wrong numbers everywhere, so the extraction is tested against a synthetic tensor
of the documented shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import BackendUnavailableError, MethodError
from core.seed import set_all
from data.synthetic import make
from methods import backends_mapie as B
from methods.methods import build_method

MAPIE_ONLY = pytest.mark.skipif(not B.MAPIE_AVAILABLE, reason="mapie not installed")


def test_backend_status_reports_availability():
    status = B.backend_status()
    assert isinstance(status["available"], bool)
    if not status["available"]:
        assert status["reason"], "an unavailable backend must state why"


def test_missing_backend_raises_backend_error_not_import_error():
    """The failure mode must be a typed, catchable error."""
    if B.MAPIE_AVAILABLE:
        pytest.skip("mapie is installed")
    with pytest.raises(BackendUnavailableError):
        B.build_mapie_method("mapie_split")


def test_build_mapie_method_rejects_unknown_name():
    if not B.MAPIE_AVAILABLE:
        pytest.skip("mapie not installed")
    with pytest.raises(MethodError):
        B.build_mapie_method("not_a_mapie_method")


@MAPIE_ONLY
def test_new_api_extracts_bounds_from_axis_0():
    """lo = intervals[:, 0, k], hi = intervals[:, 1, k] under MAPIE >= 1.5.0.

    The legacy misbinding -- treating the whole ``(n, 2, k)`` tensor as one
    bound -- is the exact bug this test exists to prevent.
    """
    n, k = 7, 3
    lo = np.arange(float(n)).reshape(n, 1, 1) * np.ones((1, 1, k))
    hi = lo + 10.0
    tensor = np.concatenate([lo, hi], axis=1)  # (n, 2, k)
    for level in range(k):
        got_lo, got_hi = B._extract_bounds(tensor, level)
        np.testing.assert_allclose(got_lo, lo[:, 0, level])
        np.testing.assert_allclose(got_hi, hi[:, 0, level])
        assert got_lo.shape == (n,)
        assert np.all(got_hi > got_lo)


@MAPIE_ONLY
def test_extract_bounds_rejects_ambiguous_rank():
    """A tensor that is neither (n,2) nor (n,2,k) must raise, not guess."""
    with pytest.raises(MethodError):
        B._extract_bounds(np.zeros((4, 3, 2)))
    with pytest.raises(MethodError):
        B._extract_bounds(np.zeros((4, 2, 2, 2)))


@MAPIE_ONLY
def test_extract_bounds_rejects_out_of_range_level():
    with pytest.raises(MethodError):
        B._extract_bounds(np.zeros((4, 2, 1)), level_index=5)


@MAPIE_ONLY
def test_extract_bounds_accepts_legacy_two_column_layout():
    legacy = np.stack([np.zeros(4), np.ones(4)], axis=1)  # (n, 2)
    lo, hi = B._extract_bounds(legacy)
    np.testing.assert_allclose(lo, np.zeros(4))
    np.testing.assert_allclose(hi, np.ones(4))


@MAPIE_ONLY
def test_api_selftest_passes_with_mapie_installed():
    result = B.api_selftest()
    assert result.get("ok") is True, result.get("reason")
    shape = result.get("intervals_shape")
    assert shape is not None and len(shape) == 3 and shape[1] == 2


@MAPIE_ONLY
def test_mapie_split_reproduces_our_split_absolute():
    """Cross-validation: the Tier-0 backend must agree with our own baseline.

    Exact agreement is the point. If these two ever diverge, one of them has
    misread the theory, and that is precisely the signal the backend exists to
    provide.
    """
    set_all(11)
    bundle = make("homoscedastic", seed=11, n_train=400, n_calib=300, n_val=200, n_test=600)
    split = bundle.split

    ours = build_method("split_absolute", seed=11)
    ours.fit(split.x_train, split.y_train, alpha=0.1)
    ours.fit_calibrate(split.x_calib, split.y_calib)
    our_iv = ours.predict_interval(split.x_test, 0.1)

    theirs = build_method("mapie_split", seed=11)
    theirs.fit(split.x_train, split.y_train, alpha=0.1)
    theirs.fit_calibrate(split.x_calib, split.y_calib)
    their_iv = theirs.predict_interval(split.x_test, 0.1)

    assert our_iv.coverage(split.y_test) == pytest.approx(their_iv.coverage(split.y_test), abs=1e-9)
    assert our_iv.mean_width == pytest.approx(their_iv.mean_width, abs=1e-6)


@MAPIE_ONLY
def test_mapie_cqr_runs_and_returns_ordered_bounds():
    set_all(11)
    bundle = make("heteroscedastic", seed=11, n_train=400, n_calib=300, n_val=200, n_test=400)
    split = bundle.split
    method = build_method("mapie_cqr", seed=11)
    method.fit(split.x_train, split.y_train, alpha=0.1)
    method.fit_calibrate(split.x_calib, split.y_calib, alphas=(0.1,))
    interval = method.predict_interval(split.x_test, 0.1)
    assert np.all(interval.upper >= interval.lower - 1e-12)
    assert 0.0 <= interval.coverage(split.y_test) <= 1.0


@MAPIE_ONLY
def test_mapie_cqr_base_family_is_supported():
    """MAPIE 1.5.0 rejects non-quantile base models; assert ours is accepted."""
    assert "GradientBoostingRegressor" in B.MapieCQR.SUPPORTED_BASES
    method = B.MapieCQR(seed=1)
    est = method._base_estimator()
    assert type(est).__name__ in B.MapieCQR.SUPPORTED_BASES


@MAPIE_ONLY
def test_mapie_methods_require_calibration():
    set_all(11)
    bundle = make("homoscedastic", seed=11, n_train=200, n_calib=150, n_val=100, n_test=200)
    method = build_method("mapie_split", seed=11)
    method.fit(bundle.split.x_train, bundle.split.y_train, alpha=0.1)
    with pytest.raises(BackendUnavailableError):
        method.predict_interval(bundle.split.x_test, 0.1)
