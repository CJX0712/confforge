"""Tests for the synthetic data-generating processes.

The properties under test are the ones a benchmark silently depends on:
reproducibility, strict split disjointness, correct group labels, and -- the one
that actually bit during development -- that ``covariate_shift`` really shifts.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import DataError
from core.seed import set_all
from data.synthetic import (
    DGP_NAMES,
    GROUPED_DGP,
    default_groups,
    group_slices,
    make,
    make_stream,
)

SMALL = {"n_train": 200, "n_calib": 120, "n_val": 100, "n_test": 300}


def test_every_dgp_generates_and_is_disjoint():
    for name in DGP_NAMES:
        bundle = make(
            name,
            seed=11,
            n_calib=25 if name == "small_calib" else 120,
            **{"n_train": 200, "n_val": 100, "n_test": 300},
        )
        bundle.split.assert_disjoint()
        sizes = bundle.split.sizes()
        assert sizes["train"] == 200 and sizes["test"] == 300
        assert bundle.split.x_test.shape[1] == 4
        assert np.all(np.isfinite(bundle.split.y_test))


def test_same_seed_is_bit_identical():
    """Reproducibility is the contract; PYTHONHASHSEED must not affect it."""
    for name in DGP_NAMES:
        a = make(name, seed=23, **SMALL)
        b = make(name, seed=23, **SMALL)
        np.testing.assert_array_equal(a.split.x_train, b.split.x_train)
        np.testing.assert_array_equal(a.split.y_calib, b.split.y_calib)
        np.testing.assert_array_equal(a.split.x_test, b.split.x_test)


def test_different_seeds_differ():
    a = make("homoscedastic", seed=11, **SMALL)
    b = make("homoscedastic", seed=37, **SMALL)
    assert not np.array_equal(a.split.x_test, b.split.x_test)


def test_generation_is_order_independent():
    """Regenerating a dataset after an unrelated one must be unaffected.

    This is what ``core.seed.spawn`` buys over a shared stateful generator: with
    a cached stream, inserting an unrelated draw would silently shift the data.
    """
    first = make("homoscedastic", seed=11, **SMALL)
    make("nonlinear", seed=99, n_train=50, n_calib=50, n_val=50, n_test=50)
    again = make("homoscedastic", seed=11, **SMALL)
    np.testing.assert_array_equal(first.split.x_test, again.split.x_test)


def test_covariate_shift_actually_shifts_the_test_split():
    """Regression test: the shift must apply to TEST only.

    The generator originally applied the translation to every segment, which
    relocated the whole dataset and produced no shift at all -- the DGP was a
    byte-identical duplicate of ``homoscedastic`` and nothing failed.

    ``assert_disjoint()`` does **not** catch this: translating all four segments
    leaves the rows just as disjoint as before, they are merely all moved. The
    property has to be asserted on the mean of the shifted feature, which is what
    the two tests below do -- the coarse one here, the knob-exact one next.
    """
    bundle = make("covariate_shift", seed=11, n_train=800, n_calib=400, n_val=300, n_test=800)
    split = bundle.split
    assert abs(split.x_test[:, 0].mean() - split.x_train[:, 0].mean()) > 2.0, (
        "test covariates should be translated well beyond the training range"
    )
    plain = make("homoscedastic", seed=11, n_train=800, n_calib=400, n_val=300, n_test=800)
    np.testing.assert_allclose(
        split.x_train[:, 0].mean(), plain.split.x_train[:, 0].mean(), atol=0.1
    )
    assert not np.array_equal(split.x_test, plain.split.x_test)


def test_covariate_shift_knob_moves_only_the_test_mean():
    """The ``shift`` knob must move the test mean by exactly that amount.

    This is the precise form of the guard above, and it is the test that would
    fail if ``_TEST_ONLY`` were dropped from ``data/synthetic.py``. Measuring the
    *difference* against the same-seed homoscedastic baseline cancels the sampling
    noise, so the assertion is about the mechanism rather than about luck:

    * train/calib/val means must match the baseline to within sampling error
      (i.e. they did **not** move at all);
    * the test mean must sit ``shift`` above the baseline's test mean.
    """
    for shift in (2.0, 3.5):
        bundle = make(
            "covariate_shift",
            seed=23,
            n_train=2000,
            n_calib=1000,
            n_val=1000,
            n_test=2000,
            shift=shift,
        )
        base = make(
            "homoscedastic",
            seed=23,
            n_train=2000,
            n_calib=1000,
            n_val=1000,
            n_test=2000,
        )
        for segment in ("train", "calib", "val"):
            got = getattr(bundle.split, f"x_{segment}")[:, 0].mean()
            want = getattr(base.split, f"x_{segment}")[:, 0].mean()
            assert abs(got - want) < 0.15, (
                f"{segment} moved by {got - want:+.3f} under shift={shift}; "
                "the translation must not touch non-test segments"
            )
        got_test = bundle.split.x_test[:, 0].mean()
        want_test = base.split.x_test[:, 0].mean()
        assert got_test - want_test == pytest.approx(shift, abs=0.15), (
            f"test mean moved by {got_test - want_test:+.3f}, expected {shift}"
        )


def test_grouped_hetero_has_real_groups():
    bundle = make(GROUPED_DGP, seed=11, **SMALL)
    slices = group_slices(bundle)
    for segment, arr in slices.items():
        assert arr is not None
        assert len(arr) == bundle.split.sizes()[segment]
    test_groups = slices["test"]
    assert test_groups.min() == 0
    assert test_groups.max() >= 2, "expected at least 3 latent groups"
    counts = np.bincount(test_groups)
    assert len(counts) == 3
    assert counts.min() > 10, "every group should be populated"


def test_small_calib_uses_thin_calibration():
    bundle = make("small_calib", seed=11, n_train=300, n_calib=25, n_val=100, n_test=300)
    assert bundle.split.sizes()["calib"] == 25


def test_default_groups_are_derived_from_train_only():
    """Group edges must not depend on calibration or test covariates."""
    bundle = make("heteroscedastic", seed=11, **SMALL)
    groups = default_groups(bundle, bundle.split.x_train)
    for segment in ("train", "calib", "val", "test"):
        assert len(groups[segment]) == bundle.split.sizes()[segment]
        assert set(np.unique(groups[segment])).issubset({0, 1, 2})


def test_unknown_dgp_and_knob_rejected():
    with pytest.raises(DataError):
        make("does_not_exist", seed=1, **SMALL)
    with pytest.raises(DataError):
        make("homoscedastic", seed=1, not_a_knob=3, **SMALL)
    with pytest.raises(DataError):
        make("homoscedastic", seed=1, n_train=0, n_calib=10, n_val=10, n_test=10)


def test_nonlinear_frequency_knob_changes_the_signal():
    """A higher frequency must make the mean function genuinely harder."""
    low = make("nonlinear", seed=11, freq=3.0, **SMALL)
    high = make("nonlinear", seed=11, freq=16.0, **SMALL)
    assert not np.array_equal(low.split.y_test, high.split.y_test)
    # the oscillation count differs, so the spread of x1-driven structure differs
    assert high.split.y_test.std() != low.split.y_test.std()


def test_stream_has_drift_and_is_marked():
    bundle = make_stream(seed=11, n_steps=300, n_train=400, n_calib=300)
    assert bundle.is_stream
    assert bundle.stream_x.shape[0] == 300
    assert 0 < bundle.meta["drift_index"] < 300
    assert bundle.meta["drift_factor"] == 3.0


def test_stream_noise_actually_increases_after_the_jump():
    """Verify the drift on *residuals*; raw y std is dominated by the mean fn."""
    from data.synthetic import _mean_homoscedastic

    bundle = make_stream(seed=11, n_steps=400, n_train=800, n_calib=400)
    resid = bundle.stream_y - _mean_homoscedastic(bundle.stream_x)
    idx = bundle.meta["drift_index"]
    before = resid[idx - 100 : idx].std()
    after = resid[idx : idx + 100].std()
    assert after > 2.0 * before, f"residual std {before:.3f} -> {after:.3f}"


def test_stream_rejects_bad_arguments():
    with pytest.raises(DataError):
        make_stream(name="nope", n_steps=100, n_train=100, n_calib=100)
    with pytest.raises(DataError):
        make_stream(n_steps=100, n_train=100, n_calib=100, drift_at=1.5)
    with pytest.raises(DataError):
        make_stream(n_steps=10, n_train=100, n_calib=100)


def test_set_all_resets_streams_deterministically():
    set_all(123)
    a = make("homoscedastic", seed=11, **SMALL)
    set_all(123)
    b = make("homoscedastic", seed=11, **SMALL)
    np.testing.assert_array_equal(a.split.x_train, b.split.x_train)
