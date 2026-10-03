"""Tests for the correctness invariants.

The invariant checker is the package's early-warning system for the failures that
averages hide: a NaN on one seed, an inverted interval on one outlier, a group
index out of range. These tests verify the checker actually *catches* those --
a checker that never fires is indistinguishable from no checker at all.
"""

from __future__ import annotations

import numpy as np

from core.types import IntervalSet
from eval import invariants as I


def _iv(lo, hi, alpha=0.1):
    return IntervalSet(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), alpha=alpha)


def test_clean_interval_produces_no_violations():
    lo = np.zeros(50)
    iv = _iv(lo, lo + 1.0)
    y = lo + 0.5
    assert I.check_cell(iv, y, alpha=0.1, n_calib=500) == []


def test_one_sided_infinite_bounds_are_flagged():
    """A one-sided infinity is always a bug; two-sided is a legal degradation.

    A thin calibration set legitimately yields the whole line, so both bounds go
    infinite together. Only an interval that is infinite on *one* side indicates
    a broken construction.
    """
    lo = np.array([-np.inf, -np.inf, 0.0])
    hi = np.array([np.inf, 1.0, 2.0])  # row 1: lower infinite, upper finite -> a bug
    iv = IntervalSet(lo, hi, alpha=0.1)
    violations = I.check_cell(iv, np.array([0.0, 0.5, 1.0]), alpha=0.1)
    assert any("one-sided" in v["invariant"] for v in violations)


def test_two_sided_infinite_bounds_are_allowed():
    iv = IntervalSet(np.array([-np.inf]), np.array([np.inf]), alpha=0.1)
    assert I.check_cell(iv, np.array([0.0]), alpha=0.1) == []


def test_shape_mismatch_is_flagged():
    iv = _iv([0.0, 1.0], [1.0, 2.0])
    violations = I.check_cell(iv, np.array([0.5]), alpha=0.1)
    assert any(v["invariant"] == "shape agreement" for v in violations)


def test_conformal_index_violation_is_flagged():
    """k outside [1, n_calib] means the quantile does not exist."""
    lo = np.zeros(10)
    iv = _iv(lo, lo + 1.0)
    y = lo + 0.5
    # n_calib=1 with alpha=0.1 gives k=2 > 1
    violations = I.check_cell(iv, y, alpha=0.1, n_calib=1)
    assert any("conformal index" in v["invariant"] for v in violations)
    # a healthy calibration size produces nothing
    assert I.check_cell(iv, y, alpha=0.1, n_calib=500) == []


def test_result_row_nan_is_flagged():
    from core.types import MetricRow

    row = MetricRow(
        method="m",
        dataset="d",
        seed=1,
        alpha=0.1,
        coverage=float("nan"),
        mean_width=1.0,
        winkler=1.0,
    )
    violations = I.check_result_row(row)
    assert any("finite" in v["invariant"] for v in violations)


def test_result_row_negative_width_is_flagged():
    from core.types import MetricRow

    row = MetricRow(
        method="m",
        dataset="d",
        seed=1,
        alpha=0.1,
        coverage=0.9,
        mean_width=-1.0,
        winkler=1.0,
    )
    assert any("mean_width" in v["invariant"] for v in I.check_result_row(row))


def test_result_row_negative_winkler_is_flagged():
    from core.types import MetricRow

    row = MetricRow(
        method="m",
        dataset="d",
        seed=1,
        alpha=0.1,
        coverage=0.9,
        mean_width=1.0,
        winkler=-1.0,
    )
    assert any("winkler" in v["invariant"] for v in I.check_result_row(row))


def test_check_all_over_rows_is_clean():
    """A clean run must say so *and* prove it looked.

    The old version of this test was ``assert I.check_all(rows) == []``, which
    was vacuous: ``check_all`` used to return a bare list and could not report
    whether it had examined anything, so a checker that skipped every cell
    returned exactly the same ``[]``. It now returns a dict carrying
    ``cells_checked``, and this test asserts the positive count as well -- so a
    regression to "checks silently skipped" fails here.
    """
    from core.types import MetricRow

    rows = [
        MetricRow("m", "d", 1, 0.1, 0.9, 1.0, 1.2),
        MetricRow("m", "d", 2, 0.1, 0.91, 1.1, 1.25),
    ]
    lo = np.zeros(20)
    out = I.check_all(
        rows,
        intervals={("m", "d", 1): _iv(lo, lo + 1.0), ("m", "d", 2): _iv(lo, lo + 1.0)},
        targets={("m", "d", 1): lo + 0.5, ("m", "d", 2): lo + 0.5},
        n_calibs={("m", "d", 1): 500, ("m", "d", 2): 500},
    )
    assert out["violations"] == []
    assert out["cells_checked"] == 2, "clean is only meaningful if cells were examined"
    assert out["cells_missing"] == 0


def test_check_all_reports_zero_checks_when_not_wired():
    """Regression guard for the D0 defect.

    Called with no interval dictionaries, ``check_all`` must report
    ``cells_checked == 0`` rather than a reassuring empty violation list. This
    is the exact failure the release report hid: the benchmark printed
    ``invariants: []`` while checking nothing.
    """
    from core.types import MetricRow

    rows = [MetricRow("m", "d", 1, 0.1, 0.9, 1.0, 1.2)]
    out = I.check_all(rows)
    assert out["violations"] == []
    assert out["cells_checked"] == 0
    assert out["cells_missing"] == 1


# --------------------------------------------------------------------------- #
# NEGATIVE tests: inject a defect, prove the checker notices.
#
# These are the tests that would have caught D0. They are written against
# `check_all` (not `check_cell`) on purpose, because D0 lived in the wiring
# between the two -- a perfect `check_cell` that nothing ever calls with data.
# --------------------------------------------------------------------------- #
def _one_cell_row():
    from core.types import MetricRow

    return MetricRow("m", "d", 1, 0.1, 0.9, 1.0, 1.2)


def _names(violations):
    return {v["invariant"] for v in violations}


def test_negative_check_all_catches_one_sided_infinity():
    """Lower = -inf with a finite upper bound cannot be a legal degradation."""
    row = _one_cell_row()
    bad = _iv([-np.inf, 0.0, 1.0], [1.0, 1.0, 1.5])
    out = I.check_all([row], intervals={("m", "d", 1): bad}, targets={("m", "d", 1): np.zeros(3)})
    assert out["cells_checked"] == 1
    assert "no one-sided infinite bounds" in _names(out["violations"]), (
        "injected one-sided infinity was NOT detected"
    )


def test_negative_check_all_catches_shape_mismatch():
    row = _one_cell_row()
    good = _iv([0.0, 1.0], [1.0, 2.0])
    out = I.check_all([row], intervals={("m", "d", 1): good}, targets={("m", "d", 1): np.zeros(7)})
    assert "shape agreement" in _names(out["violations"])


def test_negative_check_all_catches_nan_width():
    row = _one_cell_row()
    lo = np.zeros(4)
    hi = lo + 1.0
    hi[2] = np.nan
    bad = IntervalSet(lo, hi, alpha=0.1)
    out = I.check_all([row], intervals={("m", "d", 1): bad}, targets={("m", "d", 1): lo})
    assert _names(out["violations"]), "NaN width must be reported"


def test_negative_check_all_catches_out_of_range_conformal_index():
    """D2: ``n_calib`` must be forwarded, or the k-identity never runs.

    n_calib=1 with alpha=0.1 gives k = ceil(2*0.9) = 2 > 1, which is a genuine
    violation. Passing ``intervals`` without ``n_calibs`` used to skip it
    silently, so this test asserts both the wired and the unwired behaviour.
    """
    row = _one_cell_row()
    lo = np.zeros(10)
    good = _iv(lo, lo + 1.0)
    ivs = {("m", "d", 1): good}
    tgs = {("m", "d", 1): lo + 0.5}

    wired = I.check_all([row], intervals=ivs, targets=tgs, n_calibs={("m", "d", 1): 1})
    assert any("conformal index" in n for n in _names(wired["violations"])), (
        "k-identity must fire when n_calib makes k exceed n"
    )

    unwired = I.check_all([row], intervals=ivs, targets=tgs)
    assert not any("conformal index" in n for n in _names(unwired["violations"])), (
        "without n_calibs the k-identity is intentionally skipped"
    )


def test_negative_check_all_catches_nan_in_a_row():
    """A NaN metric on a row must surface even without interval data."""
    from core.types import MetricRow

    row = MetricRow("m", "d", 1, 0.1, float("nan"), 1.0, 1.2)
    out = I.check_all([row])
    assert any("finite" in n for n in _names(out["violations"]))


def test_negative_check_all_catches_missing_dictionary_entry():
    """A cell absent from the dictionaries is counted as missing, not clean."""
    row = _one_cell_row()
    out = I.check_all([row], intervals={}, targets={})
    assert out["cells_checked"] == 0
    assert out["cells_missing"] == 1


def test_check_all_with_intervals_and_targets():
    from core.types import MetricRow

    lo = np.zeros(20)
    iv = _iv(lo, lo + 1.0)
    rows = [MetricRow("m", "d", 1, 0.1, 0.9, 1.0, 1.2)]
    out = I.check_all(
        rows,
        intervals={("m", "d", 1): iv},
        targets={("m", "d", 1): lo + 0.5},
        n_calibs={("m", "d", 1): 500},
    )
    assert out["violations"] == []
    assert out["cells_checked"] == 1


def test_invariant_catalogue_is_documented():
    assert len(I.INVARIANTS) >= 6
    joined = " ".join(I.INVARIANTS)
    assert "ordered" in joined
    assert "conformal index" in joined


def test_report_artefact_records_that_checks_were_performed(tmp_path):
    """The count must reach the JSON, not just live on the report object.

    ``BenchmarkReport.to_dict()`` emits a fixed key set, so assigning
    ``report.invariant_checks_performed`` is not sufficient -- without the
    explicit copy in ``save_report`` the field silently never reaches disk and
    the artefact again shows only an empty ``invariants`` list.
    """
    import json

    from core.types import BenchmarkReport, MetricRow
    from eval.benchmark import save_report

    report = BenchmarkReport(rows=[MetricRow("m", "d", 1, 0.1, 0.9, 1.0, 1.2)])
    report.invariant_checks_performed = 1
    report.invariant_rows_checked = 1
    path = save_report(report, tmp_path / "b.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["invariant_checks_performed"] == 1
    assert payload["invariant_rows_checked"] == 1
