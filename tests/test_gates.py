"""Tests for the release gates.

The gates are the part of the system most able to lie about itself, so the tests
here are mostly about the *skip* and *threshold* logic rather than the happy
path: an aggregate computed over a favourable subset, or a threshold quietly
relaxed, would produce a green build and a meaningless benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.gates import (
    COVERAGE_TOL,
    G1_MIN_NARROWING,
    G2_MIN_COVW_REDUCTION,
    G3_MAX_WIDTH_RATIO,
    MIN_PARTICIPANTS,
    evaluate_gates,
    gate_conditional_coverage,
    gate_determinism,
    gate_drift,
    gate_width,
)


def _agg(entries):
    """Build the aggregate shape the gates consume from (method, dataset) rows."""
    by_md = {}
    for method, dataset, coverage, width, cov_w in entries:
        by_md[f"{method}|{dataset}"] = {
            "method": method,
            "dataset": dataset,
            "coverage_mean": coverage,
            "coverage_std": 0.01,
            "width_mean": width,
            "width_std": 0.01,
            "winkler_mean": width,
            "winkler_std": 0.01,
            "cov_w_mean": cov_w,
            "cov_w_std": 0.01,
            "n_seeds": 3,
        }
    return {"by_method_dataset": by_md, "by_method": {}}


def _pairs(width_ratio, n=4, coverage=0.90):
    """`n` datasets where the flagship is `width_ratio` times the baseline width."""
    out = []
    for i in range(n):
        out.append(("vennfuse", f"d{i}", coverage, 1.0 * width_ratio, 0.10))
        out.append(("split_absolute", f"d{i}", coverage, 1.0, 0.20))
    return out


def test_gate_width_passes_above_threshold():
    agg = _agg(_pairs(0.85))  # 15% narrower
    g = gate_width(agg)
    assert g["passed"]
    assert g["measured"] == pytest.approx(0.15, abs=1e-9)
    assert g["n_participants"] == 4


def test_gate_width_fails_below_threshold():
    agg = _agg(_pairs(0.95))  # only 5% narrower
    g = gate_width(agg)
    assert not g["passed"]
    assert g["measured"] == pytest.approx(0.05, abs=1e-9)
    assert "threshold" in g["reason"]


def test_gate_width_skips_datasets_with_bad_coverage():
    """A dataset where coverage misses nominal must be excluded, not counted."""
    entries = _pairs(0.80, n=3)
    entries += [("vennfuse", "bad", 0.50, 0.10, 0.4), ("split_absolute", "bad", 0.50, 1.0, 0.5)]
    g = gate_width(_agg(entries))
    assert "bad" in g["skipped"]
    assert "bad" not in g["participants"]
    assert g["n_participants"] == 3


def test_gate_width_fails_when_too_few_participants():
    """Two good datasets must NOT be enough -- that is how subsets get gamed."""
    agg = _agg(_pairs(0.50, n=2))  # would be a 50% win, on too little data
    g = gate_width(agg)
    assert not g["passed"]
    assert g["n_participants"] < MIN_PARTICIPANTS
    assert "at least" in g["reason"]


def test_gate_width_fails_with_no_participants():
    g = gate_width({"by_method_dataset": {}, "by_method": {}})
    assert not g["passed"]
    assert np.isnan(g["measured"])


def test_gate_width_counts_regression_as_failure():
    """A flagship wider than the baseline must not pass."""
    entries = []
    for i in range(4):
        entries.append(("vennfuse", f"d{i}", 0.9, 1.2, 0.1))
        entries.append(("split_absolute", f"d{i}", 0.9, 1.0, 0.2))
    g = gate_width(_agg(entries))
    assert not g["passed"]
    assert g["measured"] < 0


def test_gate_conditional_coverage():
    good = _agg(
        [
            ("vennfuse", "grouped_hetero", 0.9, 1.0, 0.04),
            ("split_absolute", "grouped_hetero", 0.9, 1.2, 0.20),
        ]
    )
    g = gate_conditional_coverage(good)
    assert g["passed"]
    assert g["measured"] == pytest.approx(0.80, abs=1e-9)


def test_gate_conditional_coverage_fails_without_improvement():
    bad = _agg(
        [
            ("vennfuse", "grouped_hetero", 0.9, 1.0, 0.19),
            ("split_absolute", "grouped_hetero", 0.9, 1.2, 0.20),
        ]
    )
    assert not gate_conditional_coverage(bad)["passed"]


def test_gate_conditional_coverage_missing_dataset_fails():
    g = gate_conditional_coverage({"by_method_dataset": {}, "by_method": {}})
    assert not g["passed"]
    assert np.isnan(g["measured"])


def test_gate_drift_requires_both_conditions():
    ok = {
        "aci_tail_coverage": 0.93,
        "fixed_tail_coverage": 0.88,  # +5pp
        "aci_mean_width": 1.1,  # ratio 1.1
        "fixed_mean_width": 1.0,
    }
    assert gate_drift(ok)["passed"]

    too_wide = dict(ok, aci_mean_width=1.5)  # ratio 1.5 > 1.25
    g = gate_drift(too_wide)
    assert not g["passed"]
    assert not g["width_ok"] and g["coverage_ok"]

    too_small = dict(ok, aci_tail_coverage=0.885)  # +0.5pp
    g2 = gate_drift(too_small)
    assert not g2["passed"]
    assert g2["coverage_ok"] is False


def test_gate_drift_without_result_fails():
    g = gate_drift(None)
    assert not g["passed"]
    assert "no drift result" in g["reason"]


def test_gate_determinism_identical_payloads():
    payload = {
        "rows": [{"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9, "mean_width": 1.0}]
    }
    g = gate_determinism(payload, payload)
    assert g["passed"]
    assert g["measured"] == 0.0


def test_gate_determinism_detects_drift():
    a = {"rows": [{"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9, "mean_width": 1.0}]}
    b = {
        "rows": [
            {"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9, "mean_width": 1.0000001}
        ]
    }
    g = gate_determinism(a, b)
    assert not g["passed"]
    assert g["measured"] > 0


def test_gate_determinism_ignores_elapsed_time():
    """Wall-clock cannot be reproducible; including it guarantees a red gate."""
    a = {"rows": [{"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9, "elapsed_sec": 1.0}]}
    b = {"rows": [{"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9, "elapsed_sec": 9.0}]}
    assert gate_determinism(a, b)["passed"]


def test_gate_determinism_detects_missing_cells():
    a = {"rows": [{"method": "a", "dataset": "d", "seed": 1, "coverage": 0.9}]}
    b = {"rows": []}
    g = gate_determinism(a, b)
    assert not g["passed"]
    assert "different cells" in g["reason"]


def test_evaluate_gates_reports_blocking_p0_failures():
    drift = {
        "aci_tail_coverage": 0.95,
        "fixed_tail_coverage": 0.85,
        "aci_mean_width": 1.0,
        "fixed_mean_width": 1.0,
    }
    payload = {"rows": []}
    out = evaluate_gates(
        _agg(_pairs(0.95)),  # G1 fails
        drift_result=drift,
        baseline_payload=payload,
        repeat_payload=payload,
    )
    assert out["G1"]["passed"] is False
    assert "G1" in out["blocking"]
    assert out["all_passed"] is False


def test_thresholds_match_the_specification():
    """Guard against a threshold being quietly relaxed to make a gate pass."""
    assert G1_MIN_NARROWING == 0.08
    assert G2_MIN_COVW_REDUCTION == 0.20
    assert G3_MAX_WIDTH_RATIO == 1.25
    assert COVERAGE_TOL == 0.03
    assert MIN_PARTICIPANTS == 3
