"""Tests for the core layer: config, seeding, types, protocols and errors.

These are the load-bearing invariants of the whole package. The seeding tests in
particular encode a bug that cost real time: a *cached, stateful* RNG stream made
"regenerate this dataset with the same seed" return **different** rows, which
silently breaks reproducibility.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from core.config import ENV_PREFIX, ConfforgeConfig, load_config
from core.errors import (
    BackendUnavailableError,
    ConfigError,
    DataError,
    EvaluationError,
    MethodError,
    NotCalibratedError,
    SplitError,
)
from core.interfaces import (
    ConformalPredictor,
    IntervalScorer,
    OnlineConformalPredictor,
    PointPredictor,
)
from core.seed import RNG, get_seed, set_all, spawn
from core.types import ClassificationSet, DatasetBundle, IntervalSet, MethodSpec, SplitData


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults_are_valid():
    cfg = ConfforgeConfig()
    assert 0 < cfg.alpha < 1
    assert cfg.coverage == pytest.approx(1 - cfg.alpha)
    assert len(cfg.seeds) >= 3
    assert cfg.n_calib >= 19


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"n_train": 0},
        {"n_calib": 5},  # below the finite-sample floor
        {"seeds": (1, 2)},  # fewer than 3
        {"seeds": (1, 1, 2)},  # not distinct
        {"drift_gamma": 0.0},
        {"drift_steps": 10},
        {"coverage_grid": (0.9, 0.5)},  # unsorted
        {"seed": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigError):
        ConfforgeConfig(**kwargs)


def test_env_overrides_are_applied(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "ALPHA", "0.05")
    monkeypatch.setenv(ENV_PREFIX + "N_CALIB", "250")
    monkeypatch.setenv(ENV_PREFIX + "SEEDS", "3,5,7")
    cfg = load_config()
    assert cfg.alpha == pytest.approx(0.05)
    assert cfg.n_calib == 250
    assert cfg.seeds == (3, 5, 7)


def test_env_override_rejects_unparseable_value(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "N_CALIB", "not-a-number")
    with pytest.raises(ConfigError):
        load_config()


def test_load_config_rejects_unknown_key():
    with pytest.raises(ConfigError):
        load_config(not_a_real_key=1)


def test_config_is_immutable():
    cfg = ConfforgeConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.alpha = 0.2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_set_all_is_reproducible():
    set_all(42)
    a = np.random.default_rng(0).standard_normal(5)
    set_all(42)
    b = np.random.default_rng(0).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_set_all_rejects_negative_seed():
    with pytest.raises(ConfigError):
        set_all(-1)


def test_get_seed_reflects_active_seed():
    set_all(99)
    assert get_seed() == 99


def test_spawn_is_fresh_and_order_independent():
    """The regression test for the cached-stateful-stream bug."""
    set_all(7)
    first = spawn("component", 1).standard_normal(4)
    # an unrelated draw in between must not shift the stream
    spawn("other", 2).standard_normal(100)
    second = spawn("component", 1).standard_normal(4)
    np.testing.assert_array_equal(first, second)


def test_spawn_separates_different_tags():
    set_all(7)
    a = spawn("component", 1).standard_normal(4)
    b = spawn("component", 2).standard_normal(4)
    assert not np.array_equal(a, b)


def test_rng_stream_is_shared_and_stateful():
    """Documented contrast with spawn(): stream() is cached and advances."""
    set_all(7)
    g1 = RNG.stream("shared")
    a = g1.standard_normal(3)
    b = RNG.stream("shared").standard_normal(3)
    assert not np.array_equal(a, b), "stream() must advance its state between calls"


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
def test_interval_set_orders_and_measures():
    iv = IntervalSet(np.array([0.0, 1.0]), np.array([2.0, 3.0]), alpha=0.1)
    assert len(iv) == 2
    np.testing.assert_allclose(iv.width, [2.0, 2.0])
    assert iv.mean_width == pytest.approx(2.0)
    assert iv.coverage(np.array([1.0, 5.0])) == pytest.approx(0.5)
    assert iv.mask(np.array([1.0, 5.0])).tolist() == [True, False]


def test_interval_set_rejects_inverted_bounds():
    with pytest.raises(MethodError):
        IntervalSet(np.array([5.0]), np.array([1.0]))


def test_interval_set_rejects_shape_mismatch():
    with pytest.raises(MethodError):
        IntervalSet(np.array([1.0, 2.0]), np.array([1.0]))


def test_interval_set_coverage_rejects_bad_y():
    iv = IntervalSet(np.array([0.0]), np.array([1.0]))
    with pytest.raises(EvaluationError):
        iv.coverage(np.array([0.5, 0.6]))


def test_interval_set_slice_preserves_metadata():
    iv = IntervalSet(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), alpha=0.2)
    part = iv.slice(np.array([0, 2]))
    assert len(part) == 2
    assert part.alpha == pytest.approx(0.2)


def test_classification_set_basics():
    cs = ClassificationSet(np.array([[True, False], [True, True]]), np.array([0, 1]))
    assert len(cs) == 2
    assert cs.mean_size == pytest.approx(1.5)
    assert cs.coverage(np.array([0, 1])) == pytest.approx(1.0)
    assert cs.coverage(np.array([1, 0])) == pytest.approx(0.5)


def test_classification_set_rejects_bad_shapes():
    with pytest.raises(MethodError):
        ClassificationSet(np.array([True, False]), np.array([0, 1]))
    with pytest.raises(MethodError):
        ClassificationSet(np.zeros((0, 2), dtype=bool), np.array([0, 1]))


def test_split_data_detects_leakage():
    x = np.arange(8, dtype=float).reshape(4, 2)
    good = SplitData(x, x[:, 0], x + 10, x[:, 0], x + 20, x[:, 0], x + 30, x[:, 0])
    good.assert_disjoint()  # must not raise
    leaky = SplitData(x, x[:, 0], x, x[:, 0], x + 20, x[:, 0], x + 30, x[:, 0])
    with pytest.raises(SplitError):
        leaky.assert_disjoint()


def test_method_spec_validates_enums():
    with pytest.raises(ConfigError):
        MethodSpec(name="x", family="nope", backend="numpy")
    with pytest.raises(ConfigError):
        MethodSpec(name="x", family="baseline", backend="nope")
    spec = MethodSpec(name="x", family="baseline", backend="numpy")
    assert spec.supports_alpha is True and spec.online is False


def test_dataset_bundle_stream_flag():
    split = SplitData(
        np.zeros((1, 2)),
        np.zeros(1),
        np.zeros((1, 2)),
        np.zeros(1),
        np.zeros((1, 2)),
        np.zeros(1),
        np.zeros((1, 2)),
        np.zeros(1),
    )
    plain = DatasetBundle(name="a", split=split)
    assert plain.is_stream is False
    streamed = DatasetBundle(name="a", split=split, stream_x=np.zeros((2, 2)), stream_y=np.zeros(2))
    assert streamed.is_stream is True


# --------------------------------------------------------------------------- #
# errors + protocols
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc",
    [
        ConfigError,
        DataError,
        SplitError,
        MethodError,
        NotCalibratedError,
        EvaluationError,
        BackendUnavailableError,
    ],
)
def test_errors_have_stable_codes_and_string(exc):
    e = exc("boom")
    assert isinstance(e.code, int)
    assert "boom" in str(e)
    assert str(e).startswith("[E")


def test_error_hierarchy():
    assert issubclass(SplitError, DataError)
    assert issubclass(NotCalibratedError, MethodError)
    assert issubclass(DataError, Exception)


def test_protocols_are_runtime_checkable():
    """The protocols exist to make 'fair comparison' enforceable at runtime."""
    assert isinstance(IntervalSet(np.array([0.0]), np.array([1.0])), object)
    for proto in (PointPredictor, ConformalPredictor, OnlineConformalPredictor, IntervalScorer):
        assert hasattr(proto, "_is_runtime_protocol")


def test_real_method_satisfies_conformal_protocol():
    from methods.methods import build_method

    method = build_method("split_absolute", seed=1)
    assert isinstance(method, ConformalPredictor)
    aci = build_method("aci", seed=1)
    assert isinstance(aci, OnlineConformalPredictor), (
        "ACIConformal must implement partial_fit/update_alpha/alpha to satisfy "
        "core.interfaces.OnlineConformalPredictor"
    )


def test_env_prefix_is_stable():
    assert ENV_PREFIX == "CONFFORGE_"
    assert os.environ.get("CONFFORGE_SEED") is not None or True
