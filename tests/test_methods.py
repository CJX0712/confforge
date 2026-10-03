"""Tests for the eight conformal methods.

Two things are checked for every method: the shared contract (fit ->
fit_calibrate -> predict_interval, with errors raised in the right order) and
empirical coverage close to nominal on small synthetic data.

The coverage tolerance here is deliberately loose (+/- 0.05) because these are
*small* problems: with a few hundred calibration points the conformal quantile
itself carries visible sampling noise, and a tight bound would make the suite
fail intermittently for reasons that have nothing to do with the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import DataError, MethodError, NotCalibratedError
from core.seed import set_all
from core.types import IntervalSet
from data.synthetic import default_groups, group_slices, make
from methods.methods import METHOD_ORDER, build_method

ALPHA = 0.1
NUMPY_METHODS = [
    "split_absolute",
    "split_normalized",
    "cqr",
    "mondrian",
    "vennfuse",
    "aci",
]
GROUPED = {"mondrian", "vennfuse"}


def _bundle(name="heteroscedastic", seed=11):
    return make(name, seed=seed, n_train=400, n_calib=300, n_val=200, n_test=800)


def _fit_predict(method_name, bundle, seed=11, alpha=ALPHA):
    split = bundle.split
    grouped = method_name in GROUPED
    g_calib = (
        (
            group_slices(bundle)["calib"]
            if bundle.name == "grouped_hetero"
            else default_groups(bundle, split.x_train)["calib"]
        )
        if grouped
        else None
    )
    g_test = (
        (
            group_slices(bundle)["test"]
            if bundle.name == "grouped_hetero"
            else default_groups(bundle, split.x_train)["test"]
        )
        if grouped
        else None
    )

    method = build_method(method_name, seed=seed)
    method.fit(split.x_train, split.y_train, alpha=alpha)
    if grouped:
        method.fit_calibrate(split.x_calib, split.y_calib, groups=g_calib)
        return method, method.predict_interval(split.x_test, alpha, groups=g_test)
    method.fit_calibrate(split.x_calib, split.y_calib)
    return method, method.predict_interval(split.x_test, alpha)


@pytest.mark.parametrize("name", NUMPY_METHODS)
def test_method_emits_valid_ordered_interval(name):
    set_all(11)
    bundle = _bundle()
    _, interval = _fit_predict(name, bundle)
    assert isinstance(interval, IntervalSet)
    assert len(interval) == len(bundle.split.y_test)
    assert np.all(interval.upper >= interval.lower - 1e-12)
    assert np.all(np.isfinite(interval.width))


@pytest.mark.parametrize("name", NUMPY_METHODS)
def test_method_hits_nominal_coverage(name):
    set_all(11)
    bundle = _bundle()
    _, interval = _fit_predict(name, bundle)
    cov = interval.coverage(bundle.split.y_test)
    assert abs(cov - (1 - ALPHA)) <= 0.05, f"{name} coverage {cov:.4f} off nominal"


@pytest.mark.parametrize("name", NUMPY_METHODS)
def test_method_requires_calibration_before_predicting(name):
    set_all(11)
    bundle = _bundle()
    method = build_method(name, seed=11)
    method.fit(bundle.split.x_train, bundle.split.y_train, alpha=ALPHA)
    with pytest.raises(NotCalibratedError):
        method.predict_interval(bundle.split.x_test, ALPHA)


@pytest.mark.parametrize("name", NUMPY_METHODS)
def test_method_requires_fit_before_calibrating(name):
    set_all(11)
    bundle = _bundle()
    method = build_method(name, seed=11)
    with pytest.raises(NotCalibratedError):
        method.fit_calibrate(bundle.split.x_calib, bundle.split.y_calib)


def test_unknown_method_rejected():
    with pytest.raises(MethodError):
        build_method("not_a_method")


def test_every_declared_method_can_be_built():
    for name in METHOD_ORDER:
        method = build_method(name, seed=1)
        assert getattr(method, "name", name) == name


def test_calibration_rejects_misaligned_groups():
    set_all(11)
    bundle = _bundle("grouped_hetero")
    method = build_method("mondrian", seed=11)
    method.fit(bundle.split.x_train, bundle.split.y_train, alpha=ALPHA)
    with pytest.raises(DataError):
        method.fit_calibrate(
            bundle.split.x_calib, bundle.split.y_calib, groups=np.zeros(3, dtype=int)
        )


def test_cqr_needs_a_smaller_correction_than_split():
    """The size-invariant reason CQR is narrower.

    ``q`` is *not* always negative -- at small training sizes the quantile anchor
    under-covers and conformal has to widen it (measured: q = +0.62 on
    heteroscedastic with n_train=400, turning negative only around n_train=1500).
    What holds at every size is the comparison that matters: the CQR anchor
    requires a far smaller additive correction than the split anchor, because it
    has already localised most of the spread conditionally.
    """
    set_all(11)
    bundle = _bundle("heteroscedastic")
    split = bundle.split
    cqr, _ = _fit_predict("cqr", bundle)
    q_cqr = float(cqr._calibrated["quantiles"][ALPHA])

    base = build_method("split_absolute", seed=11)
    base.fit(split.x_train, split.y_train, alpha=ALPHA)
    base.fit_calibrate(split.x_calib, split.y_calib)
    q_split = float(base._calibrated["quantiles"][ALPHA])

    assert q_cqr < q_split, f"CQR q={q_cqr:.4f} should be well below split q={q_split:.4f}"


def test_cqr_shrinks_relative_to_its_own_base_window():
    """CQR's total width must beat the raw window plus the split correction."""
    set_all(11)
    bundle = _bundle("heteroscedastic")
    split = bundle.split
    cqr, interval = _fit_predict("cqr", bundle)
    lo, hi = cqr.bundle.predict_quantiles(split.x_test)
    raw_width = float(np.mean(hi - lo))
    base = build_method("split_absolute", seed=11)
    base.fit(split.x_train, split.y_train, alpha=ALPHA)
    base.fit_calibrate(split.x_calib, split.y_calib)
    naive = raw_width + 2.0 * float(base._calibrated["quantiles"][ALPHA])
    assert interval.mean_width < naive, (
        f"CQR width {interval.mean_width:.4f} should beat naive {naive:.4f}"
    )


def test_cqr_interval_never_inverts_under_strong_negative_q():
    """The feasibility clamp must hold even for an extreme negative quantile."""
    from methods.estimators import ModelBundle

    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([0.2, 1.2, 2.2])
    clamped = ModelBundle.clamp_correction(lo, hi, -5.0)
    lower, upper = lo - clamped, hi + clamped
    assert np.all(upper >= lower - 1e-12)
    # clamping to the boundary yields a point interval, which is the honest
    # consequence of an over-aggressive correction
    np.testing.assert_allclose(upper - lower, np.zeros(3), atol=1e-12)


def test_clamp_correction_accepts_per_sample_array():
    from methods.estimators import ModelBundle

    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 2.0])
    q = np.array([-0.5, -5.0])
    clamped = ModelBundle.clamp_correction(lo, hi, q)
    assert clamped[0] == pytest.approx(-0.5)
    assert clamped[1] == pytest.approx(-1.0)


def test_vennfuse_switches_are_independent():
    """Each switch must actually change the fitted state, not just a flag."""
    set_all(11)
    bundle = _bundle("grouped_hetero")
    split = bundle.split
    g_calib = group_slices(bundle)["calib"]
    g_test = group_slices(bundle)["test"]
    widths = {}
    for label, kwargs in (
        ("full", {}),
        ("no_cqr", {"use_cqr": False}),
        ("no_mondrian", {"use_mondrian": False}),
    ):
        m = build_method("vennfuse", seed=11, **kwargs)
        m.fit(split.x_train, split.y_train, alpha=ALPHA)
        m.fit_calibrate(split.x_calib, split.y_calib, groups=g_calib)
        iv = m.predict_interval(split.x_test, ALPHA, groups=g_test)
        widths[label] = iv.mean_width
    assert widths["full"] > 0
    assert widths["no_cqr"] != pytest.approx(widths["full"], abs=1e-9)
    assert widths["no_mondrian"] != pytest.approx(widths["full"], abs=1e-9)


def test_aci_alpha_update_direction():
    """Under-coverage must *decrease* alpha (widen); over-coverage must increase it.

    A sign error here is invisible in a single step and catastrophic over a
    stream, so the direction is asserted directly.
    """
    from methods.methods import aci_alpha_update

    # heavily under-covered (err=0.9) with a 0.1 target -> alpha must fall
    assert aci_alpha_update(0.1, 0.9, gamma=0.1, target=0.1) < 0.1
    # heavily over-covered (err=0.0) -> alpha must rise
    assert aci_alpha_update(0.1, 0.0, gamma=0.1, target=0.1) > 0.1
    # perfectly on target -> unchanged
    assert aci_alpha_update(0.1, 0.1, gamma=0.1, target=0.1) == pytest.approx(0.1)


def test_aci_alpha_update_respects_clip():
    from methods.methods import aci_alpha_update

    assert aci_alpha_update(0.49, 0.0, gamma=10.0, target=0.5, lo=0.01, hi=0.5) <= 0.5
    assert aci_alpha_update(0.02, 1.0, gamma=10.0, target=0.0, lo=0.01, hi=0.5) >= 0.01


def test_aci_alpha_update_validates_error_rate():
    from methods.methods import aci_alpha_update

    with pytest.raises(MethodError):
        aci_alpha_update(0.1, 1.5, gamma=0.01)
    with pytest.raises(MethodError):
        aci_alpha_update(0.1, 0.1, gamma=-1.0)


def test_aci_online_update_changes_alpha():
    set_all(11)
    bundle = _bundle("homoscedastic")
    method, interval = _fit_predict("aci", bundle)
    before = method.alpha
    # feed points known to be far outside the interval -> alpha must fall
    y_far = interval.upper + 100.0
    after = method.update_alpha(y_far, interval)
    assert after < before
    y_inside = 0.5 * (interval.lower + interval.upper)
    assert method.update_alpha(y_inside, interval) > after


def test_methods_are_deterministic():
    set_all(11)
    bundle = _bundle()
    for name in NUMPY_METHODS:
        _, first = _fit_predict(name, bundle)
        set_all(11)
        _, second = _fit_predict(name, bundle)
        np.testing.assert_allclose(first.lower, second.lower, rtol=0, atol=0)
        np.testing.assert_allclose(first.upper, second.upper, rtol=0, atol=0)


def test_wider_alpha_gives_wider_intervals():
    set_all(11)
    bundle = _bundle()
    method = build_method("split_absolute", seed=11)
    split = bundle.split
    method.fit(split.x_train, split.y_train, alpha=ALPHA)
    method.fit_calibrate(split.x_calib, split.y_calib)
    tight = method.predict_interval(split.x_test[:200], 0.05)  # 95% interval
    loose = method.predict_interval(split.x_test[:200], 0.30)  # 70% interval
    assert tight.mean_width > loose.mean_width, (
        "a smaller alpha demands higher confidence and must be wider"
    )


def test_switches_are_instance_scoped_not_class_globals():
    """Building an ablated variant must not leak into the next construction.

    A class-attribute switch would make ``VennFuse.use_cqr`` a process-wide
    global, so every later ``build_method("vennfuse")`` would silently inherit
    the ablation.
    """
    ablated = build_method("vennfuse", seed=1, use_cqr=False)
    assert ablated.use_cqr is False
    fresh = build_method("vennfuse", seed=1)
    assert fresh.use_cqr is True
    assert type(ablated).use_cqr is True, "class default must remain untouched"


def test_unknown_switch_is_rejected():
    with pytest.raises(MethodError):
        build_method("vennfuse", seed=1, not_a_switch=True)
