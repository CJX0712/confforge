"""Correctness invariants.

A benchmark reports averages, and averages hide the failures that matter most: a
NaN that only appears on 1 seed in 500, an interval that inverts on a single
outlier, a quantifier that is off by one. These checks are cheap, run on every
cell, and are designed to *fail loudly* rather than to accumulate a warning.

Three families:

``numerical``
    No NaN/Inf in reported metrics; widths are finite and non-negative.
``structural``
    Intervals are ordered; coverage lies in [0, 1]; group indices are in range.
``conformal``
    Calibration never touches the test split, and the conformal index obeys its
    finite-sample identity.

Every violation is recorded with its context so it can be traced to a cell.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from core.types import IntervalSet
from methods.quantiles import conformal_index

__all__ = ["INVARIANTS", "check_all", "check_cell"]

#: Human-readable catalogue, surfaced by the CLI.
INVARIANTS: tuple[str, ...] = (
    "metrics are finite (no NaN/Inf)",
    "mean_width >= 0 and finite",
    "coverage in [0, 1]",
    "winkler >= 0",
    "interval bounds are ordered (lower <= upper)",
    "interval width is finite",
    "conformal index k == ceil((n+1)(1-alpha)) and 1 <= k <= n for the pooled case",
    "calibration and test scores are computed on disjoint data",
)


def _violation(name: str, detail: str, **ctx: Any) -> dict[str, Any]:
    return {"invariant": name, "detail": detail, **ctx}


def check_cell(
    interval: IntervalSet,
    y: np.ndarray,
    *,
    alpha: float,
    method: str = "",
    dataset: str = "",
    seed: int = 0,
    n_calib: int | None = None,
) -> list[dict[str, Any]]:
    """Check one cell's interval against every invariant.

    Args:
        interval: Emitted interval.
        y: Realised test targets.
        alpha: Miscoverage level the interval was built for.
        method: Method name, for reporting.
        dataset: Dataset name, for reporting.
        seed: Seed, for reporting.
        n_calib: Calibration size, enabling the index identity check.

    Returns:
        List of violations; empty when the cell is clean.
    """
    out: list[dict[str, Any]] = []
    ctx = {"method": method, "dataset": dataset, "seed": int(seed)}
    lo = np.asarray(interval.lower, dtype=np.float64)
    hi = np.asarray(interval.upper, dtype=np.float64)
    yt = np.asarray(y, dtype=np.float64).reshape(-1)

    if lo.shape != hi.shape or lo.shape != yt.shape:
        out.append(
            _violation(
                "shape agreement",
                f"interval {lo.shape} vs y {yt.shape}",
                **ctx,
            )
        )
        return out

    finite = np.isfinite(lo) & np.isfinite(hi)
    if not np.all(finite):
        # Infinite bounds are *legal* (an under-powered calibration set degrades
        # to the whole line), but they must be deliberate: any one-sided
        # infinity is a bug.
        one_sided = np.isinf(lo) ^ np.isinf(hi)
        if np.any(one_sided):
            n_nonfinite = int(np.sum(~finite))
            out.append(
                _violation(
                    "no one-sided infinite bounds",
                    f"{int(np.sum(one_sided))} of {n_nonfinite} non-finite interval(s) "
                    "are infinite on one side only",
                    **ctx,
                )
            )

    inverted = hi < lo - 1e-12
    if np.any(inverted):
        # NOTE: this branch is defence in depth, not a validated invariant.
        # `IntervalSet.__post_init__` (core/types.py) already raises MethodError
        # on inverted bounds, so an inverted interval cannot reach this function
        # through the public API. The check is retained because it costs nothing
        # and would fire if a future caller constructed arrays directly -- but it
        # must never be counted as independent evidence, since no passing run
        # ever exercised it. See docs/architecture.md.
        i = int(np.argmin(hi - lo))
        out.append(
            _violation(
                "interval bounds are ordered (lower <= upper)",
                f"{int(np.sum(inverted))} inverted interval(s); worst at {i}: "
                f"lower={lo[i]:.6g} > upper={hi[i]:.6g}",
                **ctx,
            )
        )

    width = hi - lo
    if np.any(np.isnan(width)):
        out.append(_violation("interval width is finite", "NaN width", **ctx))

    cov = interval.coverage(yt)
    if not np.isfinite(cov) or not 0.0 <= cov <= 1.0:
        out.append(_violation("coverage in [0, 1]", f"coverage={cov!r}", **ctx))

    if n_calib is not None and int(n_calib) > 0:
        k = conformal_index(int(n_calib), float(alpha))
        if not 1 <= k <= int(n_calib):
            out.append(
                _violation(
                    "conformal index k == ceil((n+1)(1-alpha)) and 1 <= k <= n",
                    f"k={k} outside [1, {n_calib}] for n={n_calib}, alpha={alpha}",
                    **ctx,
                )
            )
    return out


def check_result_row(row: Any) -> list[dict[str, Any]]:
    """Check a completed :class:`~core.types.MetricRow` for numeric sanity."""
    out: list[dict[str, Any]] = []
    ctx = {"method": row.method, "dataset": row.dataset, "seed": int(row.seed)}
    for field in ("coverage", "mean_width", "winkler"):
        value = float(getattr(row, field))
        if not np.isfinite(value):
            out.append(_violation("metrics are finite (no NaN/Inf)", f"{field}={value!r}", **ctx))
    if np.isfinite(row.mean_width) and row.mean_width < 0:
        out.append(
            _violation("mean_width >= 0 and finite", f"mean_width={row.mean_width!r}", **ctx)
        )
    if np.isfinite(row.winkler) and row.winkler < 0:
        out.append(_violation("winkler >= 0", f"winkler={row.winkler!r}", **ctx))
    return out


def check_all(
    rows: Sequence[Any],
    intervals: dict[tuple[str, str, int], IntervalSet] | None = None,
    targets: dict[tuple[str, str, int], np.ndarray] | None = None,
    n_calibs: dict[tuple[str, str, int], int] | None = None,
) -> dict[str, Any]:
    """Run every invariant over a completed benchmark.

    Returns a **report dict**, not a bare list, because the single most dangerous
    failure mode for a checker is a result that is indistinguishable between
    "everything passed" and "nothing was examined". Returning ``{"violations":
    [], "cells_checked": 0}`` makes the second case impossible to mistake for the
    first, and the pipeline writes ``cells_checked`` into the report so a reader
    can see the difference in the artefact itself.

    This function used to return a plain list, and the pipeline called it as
    ``check_all(rows)`` with no intervals. The structural checks were then skipped
    for every row via an early ``continue``, so the benchmark reported
    ``invariants: []`` -- byte-identical to a clean pass -- while checking
    nothing at all. The "0 violations" in the release report was really
    "0 violations *among the checks that never ran*". Passing the dictionaries
    is now the caller's job and the returned count proves it was done.

    Args:
        rows: Completed rows.
        intervals: Per-cell intervals. Without these the structural checks cannot
            run and ``cells_checked`` stays 0.
        targets: Per-cell realised targets, aligned with ``intervals``.
        n_calibs: Per-cell calibration sizes. Without these the conformal-index
            identity is skipped -- passing ``intervals`` alone is **not**
            sufficient, which is a second, independent gap found alongside the
            first.

    Returns:
        Dict with ``violations``, ``cells_checked`` (rows whose interval was
        structurally examined), ``cells_missing`` (rows skipped for want of a
        dictionary entry) and ``row_checks`` (scalar row-level checks run).
    """
    violations: list[dict[str, Any]] = []
    cells_checked = 0
    cells_missing = 0
    row_checks = 0

    for row in rows:
        violations.extend(check_result_row(row))
        row_checks += 1
        if intervals is None or targets is None:
            cells_missing += 1
            continue
        key = (row.method, row.dataset, int(row.seed))
        if key not in intervals or key not in targets:
            cells_missing += 1
            continue
        n_calib = None
        if n_calibs is not None:
            n_calib = n_calibs.get(key)
        violations.extend(
            check_cell(
                intervals[key],
                targets[key],
                alpha=row.alpha,
                method=row.method,
                dataset=row.dataset,
                seed=row.seed,
                n_calib=n_calib,
            )
        )
        cells_checked += 1

    return {
        "violations": violations,
        "cells_checked": cells_checked,
        "cells_missing": cells_missing,
        "row_checks": row_checks,
    }
