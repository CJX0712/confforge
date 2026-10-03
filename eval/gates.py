"""Release gates.

A gate is a *falsifiable claim about the benchmark*, not a vibe. Each one states
the comparison, the threshold, the level, and -- critically -- **what happens when
a dataset cannot be scored**.

Skip policy
-----------
A dataset whose coverage misses nominal by more than ``COVERAGE_TOL`` is recorded
as ``skipped`` for that dataset and excluded from the aggregate. This is
deliberately generous to the method under test, which makes it a conservative
gate: a method cannot pass G1 by winning on the easy datasets while quietly
failing on ``covariate_shift``.

The guard against gaming that is :data:`MIN_PARTICIPANTS`. Aggregation over "only
the datasets that worked" is trivially easy to abuse -- drop the hard ones and
any method looks good -- so a gate with fewer than three participating datasets
**fails**. Reporting fewer than three is not a pass with caveats; it is a fail.

Thresholds are fixed by the delivery specification and are not tuned to the
results. That is the entire point: a threshold moved until the current numbers
clear it measures nothing.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "COVERAGE_TOL",
    "G1_COVERAGE_TOL",
    "G1_MIN_NARROWING",
    "G2_MIN_COVW_REDUCTION",
    "G3_MAX_WIDTH_RATIO",
    "G3_MIN_COVERAGE_GAIN",
    "GATE_SPECS",
    "MIN_PARTICIPANTS",
    "evaluate_gates",
    "gate_conditional_coverage",
    "gate_determinism",
    "gate_drift",
    "gate_width",
]

#: G1 -- flagship must be at least this much narrower than the baseline,
#: aggregated over the datasets where both are valid. P0.
G1_MIN_NARROWING = 0.08

#: G2 -- worst-group coverage violation must shrink by at least this much on the
#: grouped dataset. P1.
G2_MIN_COVW_REDUCTION = 0.20

#: G3 -- post-drift coverage gain of ACI over a fixed-alpha predictor. P0/P1.
G3_MIN_COVERAGE_GAIN = 0.03
#: ...while giving up no more than this much width.
G3_MAX_WIDTH_RATIO = 1.25

#: Coverage tolerance for declaring a dataset valid. Nominal +/- 0.03.
COVERAGE_TOL = 0.03
#: Alias used by G1's own criterion.
G1_COVERAGE_TOL = COVERAGE_TOL

#: Minimum datasets that must participate in an aggregate, else the gate fails.
MIN_PARTICIPANTS = 3


def _fmt(value: float, spec: str = ".4f") -> str:
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)


def _is_valid(coverage: float, nominal: float, tol: float = COVERAGE_TOL) -> bool:
    return bool(np.isfinite(coverage) and abs(coverage - nominal) <= tol)


def gate_width(
    aggregates: Mapping[str, Any],
    *,
    flagship: str = "vennfuse",
    baseline: str = "split_absolute",
    nominal: float = 0.9,
    min_narrowing: float = G1_MIN_NARROWING,
    tol: float = COVERAGE_TOL,
) -> dict[str, Any]:
    """G1 (P0): the flagship is materially narrower than the baseline.

    A dataset participates only if **both** methods are valid on it (coverage
    within ``tol`` of nominal) *and* both were actually measured. The aggregate is
    the ratio of mean widths over participating datasets, so a dataset where the
    flagship is 30% wider drags the average exactly as it should.

    Args:
        aggregates: Output of :func:`eval.benchmark.aggregate`.
        flagship: Method under test.
        baseline: Reference method.
        nominal: Target coverage level.
        min_narrowing: Required relative reduction.
        tol: Coverage tolerance for validity.

    Returns:
        Gate record with ``passed``, the measured ``narrowing``, per-dataset
        detail, the participant count and an explicit ``reason``.
    """
    by_md: Mapping[str, Any] = aggregates.get("by_method_dataset", {})
    detail: dict[str, Any] = {}
    participants: list[str] = []
    for dataset in sorted({e["dataset"] for e in by_md.values()}):
        f = by_md.get(f"{flagship}|{dataset}")
        b = by_md.get(f"{baseline}|{dataset}")
        if f is None or b is None:
            detail[dataset] = {
                "status": "skipped",
                "reason": "method did not produce a row on this dataset",
            }
            continue
        f_ok = _is_valid(f["coverage_mean"], nominal, tol)
        b_ok = _is_valid(b["coverage_mean"], nominal, tol)
        if not (f_ok and b_ok):
            detail[dataset] = {
                "status": "skipped",
                "flagship_coverage": f["coverage_mean"],
                "baseline_coverage": b["coverage_mean"],
                "reason": (
                    f"coverage outside nominal+/-{tol}: "
                    f"{flagship}={f['coverage_mean']:.4f}, {baseline}={b['coverage_mean']:.4f}"
                ),
            }
            continue
        rel = (f["width_mean"] - b["width_mean"]) / b["width_mean"]
        detail[dataset] = {
            "status": "included",
            "flagship_width": f["width_mean"],
            "baseline_width": b["width_mean"],
            "flagship_coverage": f["coverage_mean"],
            "baseline_coverage": b["coverage_mean"],
            "relative_width": rel,
        }
        participants.append(dataset)

    skipped = [d for d, v in detail.items() if v["status"] == "skipped"]
    if len(participants) < MIN_PARTICIPANTS:
        return {
            "gate": "G1",
            "level": "P0",
            "passed": False,
            "measured": float("nan"),
            "threshold": min_narrowing,
            "participants": participants,
            "n_participants": len(participants),
            "skipped": skipped,
            "detail": detail,
            "reason": (
                f"only {len(participants)} dataset(s) valid for both methods, "
                f"need at least {MIN_PARTICIPANTS}; aggregating over a subset this "
                "small would let a method pass by failing on the hard datasets"
            ),
        }

    f_mean = float(np.mean([detail[d]["flagship_width"] for d in participants]))
    b_mean = float(np.mean([detail[d]["baseline_width"] for d in participants]))
    narrowing = (b_mean - f_mean) / b_mean
    passed = bool(narrowing >= min_narrowing)
    return {
        "gate": "G1",
        "level": "P0",
        "passed": passed,
        "measured": narrowing,
        "threshold": min_narrowing,
        "flagship_mean_width": f_mean,
        "baseline_mean_width": b_mean,
        "participants": participants,
        "n_participants": len(participants),
        "skipped": skipped,
        "detail": detail,
        "reason": (
            f"{flagship} is {narrowing * 100:.2f}% narrower than {baseline} over "
            f"{len(participants)} dataset(s) (threshold {min_narrowing * 100:.1f}%)"
            if passed
            else (
                f"{flagship} is only {narrowing * 100:.2f}% narrower than {baseline} "
                f"(threshold {min_narrowing * 100:.1f}%)"
            )
        ),
    }


def gate_conditional_coverage(
    aggregates: Mapping[str, Any],
    *,
    dataset: str = "grouped_hetero",
    flagship: str = "vennfuse",
    baseline: str = "split_absolute",
    min_reduction: float = G2_MIN_COVW_REDUCTION,
) -> dict[str, Any]:
    """G2 (P1): worst-group coverage violation shrinks on the grouped dataset.

    ``CoV-W`` is the largest absolute gap between any subgroup's coverage and
    nominal. Split conformal satisfies it only on average, so a loud subgroup is
    free; a Mondrian-style method is charged for it. The reduction
    ``(baseline - flagship) / baseline`` is the quantity that says whether the
    stratification actually bought anything.

    Returns:
        Gate record; ``passed`` is False when the statistic is unavailable.
    """
    by_md: Mapping[str, Any] = aggregates.get("by_method_dataset", {})
    f = by_md.get(f"{flagship}|{dataset}")
    b = by_md.get(f"{baseline}|{dataset}")
    if f is None or b is None:
        return {
            "gate": "G2",
            "level": "P1",
            "passed": False,
            "measured": float("nan"),
            "threshold": min_reduction,
            "reason": f"no rows for {dataset}; cannot evaluate conditional coverage",
        }
    f_covw = float(f.get("cov_w_mean", float("nan")))
    b_covw = float(b.get("cov_w_mean", float("nan")))
    if not (np.isfinite(f_covw) and np.isfinite(b_covw)) or b_covw <= 0:
        return {
            "gate": "G2",
            "level": "P1",
            "passed": False,
            "measured": float("nan"),
            "threshold": min_reduction,
            "flagship_cov_w": f_covw,
            "baseline_cov_w": b_covw,
            "reason": "worst-group coverage violation unavailable or zero on both arms",
        }
    reduction = (b_covw - f_covw) / b_covw
    passed = bool(reduction >= min_reduction)
    return {
        "gate": "G2",
        "level": "P1",
        "passed": passed,
        "measured": reduction,
        "threshold": min_reduction,
        "flagship_cov_w": f_covw,
        "baseline_cov_w": b_covw,
        "dataset": dataset,
        "reason": (
            f"CoV-W {b_covw:.4f} -> {f_covw:.4f} ({reduction * 100:.1f}% reduction, "
            f"threshold {min_reduction * 100:.0f}%)"
        ),
    }


def gate_drift(
    drift_result: Mapping[str, Any] | None,
    *,
    min_gain: float = G3_MIN_COVERAGE_GAIN,
    max_width_ratio: float = G3_MAX_WIDTH_RATIO,
) -> dict[str, Any]:
    """G3 (P0/P1): ACI recovers coverage after drift without paying too much width.

    Two conditions, both required:

    * post-drift-window coverage improves by at least ``min_gain`` **absolute**
      (3 percentage points, not 3% relative -- a relative bar would be trivially
      met by a method that is simply bad everywhere);
    * mean width over the stream stays within ``max_width_ratio`` of the
      fixed-alpha control, so the gain cannot be bought by widening without limit.

    Returns:
        Gate record; ``passed`` is False when no drift result was supplied.
    """
    if not drift_result:
        return {
            "gate": "G3",
            "level": "P0/P1",
            "passed": False,
            "measured": float("nan"),
            "threshold": min_gain,
            "reason": "no drift result supplied",
        }
    aci = float(drift_result.get("aci_tail_coverage", float("nan")))
    fixed = float(drift_result.get("fixed_tail_coverage", float("nan")))
    aci_w = float(drift_result.get("aci_mean_width", float("nan")))
    fixed_w = float(drift_result.get("fixed_mean_width", float("nan")))
    if not all(np.isfinite(v) for v in (aci, fixed, aci_w, fixed_w)) or fixed_w <= 0:
        return {
            "gate": "G3",
            "level": "P0/P1",
            "passed": False,
            "measured": float("nan"),
            "threshold": min_gain,
            "reason": "drift statistics unavailable or degenerate",
        }
    gain = aci - fixed
    ratio = aci_w / fixed_w
    coverage_ok = gain >= min_gain
    width_ok = ratio <= max_width_ratio
    return {
        "gate": "G3",
        "level": "P0/P1",
        "passed": bool(coverage_ok and width_ok),
        "measured": gain,
        "threshold": min_gain,
        "coverage_gain": gain,
        "width_ratio": ratio,
        "width_threshold": max_width_ratio,
        "aci_tail_coverage": aci,
        "fixed_tail_coverage": fixed,
        "aci_mean_width": aci_w,
        "fixed_mean_width": fixed_w,
        "coverage_ok": bool(coverage_ok),
        "width_ok": bool(width_ok),
        "reason": (
            f"tail coverage {fixed:.4f} -> {aci:.4f} ({gain * 100:+.2f}pp, "
            f"need +{min_gain * 100:.0f}pp); width ratio {ratio:.3f} "
            f"(max {max_width_ratio})"
        ),
    }


def gate_determinism(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    ignore_keys: Sequence[str] = ("elapsed_sec",),
) -> dict[str, Any]:
    """G5 (P0): two runs at the same seed agree bit-for-bit.

    Compares the ``rows`` arrays element by element on the numeric fields, and the
    gate aggregates as a whole. ``elapsed_sec`` is excluded by default because
    wall-clock time is not reproducible by construction -- including it would
    guarantee a red gate for a reason that has nothing to do with determinism.

    Args:
        first: First report payload.
        second: Second report payload.
        ignore_keys: Extra keys to exclude from comparison.

    Returns:
        Gate record with ``passed`` and the worst absolute difference found.
    """
    numeric = ("coverage", "mean_width", "winkler", "alpha")
    skip = set(ignore_keys)

    def _flat(payload: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in payload.get("rows", []):
            key = f"{row.get('method')}|{row.get('dataset')}|{row.get('seed')}"
            for field in numeric:
                if field not in skip and isinstance(row.get(field), (int, float)):
                    out[f"{key}|{field}"] = float(row[field])
            for field, value in row.items():
                if field in numeric or field in skip:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[f"{key}|{field}"] = float(value)
        return out

    a, b = _flat(first), _flat(second)
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        return {
            "gate": "G5",
            "level": "P0",
            "passed": False,
            "measured": float("inf"),
            "threshold": 0.0,
            "n_compared": 0,
            "reason": (
                f"the two runs produced different cells ({len(only_a)} only in run A, "
                f"e.g. {only_a}; {len(only_b)} only in run B, e.g. {only_b})"
            ),
        }

    if not a:
        return {
            "gate": "G5",
            "level": "P0",
            "passed": False,
            "measured": float("nan"),
            "threshold": 0.0,
            "n_compared": 0,
            "reason": "no comparable numeric fields found in either run",
        }

    diffs = [abs(a[k] - b[k]) for k in a]
    worst = max(diffs)
    return {
        "gate": "G5",
        "level": "P0",
        "passed": bool(worst == 0.0),
        "measured": worst,
        "threshold": 0.0,
        "n_compared": len(diffs),
        "reason": (
            f"{len(diffs)} values compared, max abs diff = {worst!r}"
            if worst == 0.0
            else f"max abs diff = {worst:.3e} over {len(diffs)} values; runs are not reproducible"
        ),
    }


#: Static description of every gate, surfaced by the CLI and the report.
GATE_SPECS: tuple[dict[str, str], ...] = (
    {
        "gate": "G1",
        "level": "P0",
        "claim": "flagship is >= 8% narrower than split_absolute with coverage held",
        "threshold": f">= {G1_MIN_NARROWING * 100:.0f}% narrowing",
    },
    {
        "gate": "G2",
        "level": "P1",
        "claim": "flagship cuts worst-group coverage violation by >= 20% on grouped_hetero",
        "threshold": f">= {G2_MIN_COVW_REDUCTION * 100:.0f}% reduction",
    },
    {
        "gate": "G3",
        "level": "P0/P1",
        "claim": "ACI gains >= 3pp post-drift coverage at <= 1.25x the fixed-alpha width",
        "threshold": f">= {G3_MIN_COVERAGE_GAIN * 100:.0f}pp and <= {G3_MAX_WIDTH_RATIO}x width",
    },
    {
        "gate": "G5",
        "level": "P0",
        "claim": "two runs at the same seed agree bit-for-bit",
        "threshold": "max abs diff == 0",
    },
)


def evaluate_gates(
    aggregates: Mapping[str, Any],
    *,
    drift_result: Mapping[str, Any] | None = None,
    repeat_payload: Mapping[str, Any] | None = None,
    baseline_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every applicable gate and summarise.

    Args:
        aggregates: Output of :func:`eval.benchmark.aggregate`.
        drift_result: Output of :func:`eval.drift.run_drift_scenario`.
        repeat_payload: A second benchmark payload for G5.
        baseline_payload: The first benchmark payload for G5.

    Returns:
        Dict keyed by gate name, plus an ``all_passed`` flag and a
        ``blocking`` list naming the P0 gates that failed.
    """
    gates: dict[str, Any] = {
        "G1": gate_width(aggregates),
        "G2": gate_conditional_coverage(aggregates),
        "G3": gate_drift(drift_result),
    }
    if baseline_payload is not None and repeat_payload is not None:
        gates["G5"] = gate_determinism(baseline_payload, repeat_payload)
    else:
        gates["G5"] = {
            "gate": "G5",
            "level": "P0",
            "passed": False,
            "measured": float("nan"),
            "threshold": 0.0,
            "reason": "determinism not evaluated (needs two full benchmark payloads)",
        }
    blocking = [
        name
        for name, rec in gates.items()
        if not rec.get("passed") and str(rec.get("level", "")).startswith("P0")
    ]
    return {
        **gates,
        "all_passed": all(rec.get("passed", False) for rec in gates.values()),
        "blocking": blocking,
    }
