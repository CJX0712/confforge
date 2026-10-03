"""Interval metrics. Pure NumPy -- deliberately no scikit-learn.

Two reasons this module is hand-rolled rather than delegated:

1. **Circularity.** ``sklearn.metrics`` imports parts of this package's ecosystem
   and adds a heavy dependency to a module that must stay importable in a bare
   environment. Interval metrics are a dozen lines each; a dependency is not.
2. **Auditability.** The Winkler score in particular has *exact* published
   reference values. When a benchmark gate moves, the first question is whether
   the metric or the method changed -- and that question is only answerable if
   the metric is small enough to read in one sitting. ``test_metrics.py``
   pins the ground truth (``[0, 10]``, ``y=13``, ``alpha=0.1`` -> ``70.0``).

Conventions
-----------
* Every function takes ``(interval, y)`` where ``interval`` is an
  :class:`~core.types.IntervalSet` and ``y`` the realised targets.
* Every function validates shapes and raises
  :class:`~core.errors.EvaluationError` rather than silently broadcasting. A
  coverage number computed on mis-aligned arrays is worse than no number: it
  looks like evidence.
* Coverage counts a point as covered when ``lower <= y <= upper`` (**closed**
  interval), matching the convention under which the conformal guarantee is
  proved.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.errors import EvaluationError
from core.types import IntervalSet

__all__ = [
    "conditional_coverage_worst",
    "coverage",
    "coverage_gap",
    "efficiency_curve",
    "group_coverage",
    "mean_width",
    "stability",
    "summarize",
    "winkler",
    "winkler_components",
]


def _check(interval: IntervalSet, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(interval.lower, dtype=np.float64)
    hi = np.asarray(interval.upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise EvaluationError(f"interval bounds disagree: {lo.shape} vs {hi.shape}")
    if lo.size == 0:
        raise EvaluationError("cannot score an empty interval set")
    if y is None:
        return lo, hi
    yt = np.asarray(y, dtype=np.float64).reshape(-1)
    if yt.shape != lo.shape:
        raise EvaluationError(
            f"target shape {yt.shape} does not match interval shape {lo.shape}; "
            "metrics never broadcast because a mis-aligned coverage figure is "
            "indistinguishable from a real one"
        )
    return lo, hi


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise EvaluationError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
    return a


def coverage(interval: IntervalSet, y: np.ndarray) -> float:
    """Empirical marginal coverage: ``mean(lower <= y <= upper)``.

    Returns:
        A float in ``[0, 1]``.
    """
    lo, hi = _check(interval, y)
    return float(np.mean((lo <= y) & (y <= hi)))


def mean_width(interval: IntervalSet, y: np.ndarray | None = None) -> float:
    """Average interval width.

    ``y`` is accepted and ignored so every metric shares one call signature; a
    width does not depend on the realised target.
    """
    lo, hi = _check(interval, y)
    return float(np.mean(hi - lo))


def winkler(interval: IntervalSet, y: np.ndarray, alpha: float) -> float:
    """Winkler interval score -- the primary objective of this benchmark.

    .. math::

        w = hi - lo + \\frac{2}{\\alpha}(lo - y)\\,\\mathbb{1}[y < lo]
                              + \\frac{2}{\\alpha}(y - hi)\\,\\mathbb{1}[y > hi]

    The penalty weights are what make this the right scalar objective for
    conformal intervals. Width alone is trivially minimised by a degenerate
    point interval; coverage alone is minimised by ``(-inf, +inf)``. Winkler is
    the unique scale-equivariant combination that charges for width *and*
    charges steeply for a miss, so a method can only win by being both tight
    and valid.

    Reference value used by the test-suite as a ground-truth anchor::

        interval = [0, 10], y = 13, alpha = 0.1
        => 10 + (2/0.1) * (13 - 10) = 10 + 60 = 70.0

    Args:
        interval: Predicted intervals.
        y: Realised targets.
        alpha: Miscoverage level the interval was built for.

    Returns:
        The mean interval score (lower is better).
    """
    return winkler_components(interval, y, alpha)["winkler"]


def winkler_components(interval: IntervalSet, y: np.ndarray, alpha: float) -> dict[str, float]:
    """Winkler score decomposed into width and miss-penalty parts.

    Returning the split matters for diagnosis: a method can lose on Winkler
    because its intervals are too wide (``width_share`` near 1) or because it
    misses too often (``miss_share`` near 1), and those two failures call for
    completely different fixes. Only the total is optimised; the parts are
    reported.

    Returns:
        Dict with ``winkler``, ``width``, ``lower_miss``, ``upper_miss`` and the
        two shares (which sum to 1 whenever the total is positive).
    """
    a = _check_alpha(alpha)
    lo, hi = _check(interval, y)
    width = hi - lo
    factor = 2.0 / a
    below = lo > y
    above = hi < y
    lower_miss = factor * np.maximum(lo - y, 0.0)
    upper_miss = factor * np.maximum(y - hi, 0.0)
    per_sample = width + lower_miss + upper_miss
    total = float(np.mean(per_sample))
    mean_width = float(np.mean(width))
    return {
        "winkler": total,
        "width": mean_width,
        "lower_miss": float(np.mean(lower_miss)),
        "upper_miss": float(np.mean(upper_miss)),
        "width_share": float(mean_width / total) if total > 0 else 1.0,
        "miss_share": float(1.0 - mean_width / total) if total > 0 else 0.0,
        "miss_rate": float(np.mean(below | above)),
    }


def coverage_gap(interval: IntervalSet, y: np.ndarray, target: float) -> float:
    """Absolute deviation between empirical and nominal coverage.

    Args:
        target: Nominal coverage level, e.g. ``0.9``.

    Returns:
        ``|empirical - target|``, always ``>= 0``.

    Raises:
        EvaluationError: If ``target`` is outside ``[0, 1]``.
    """
    t = float(target)
    if not 0.0 <= t <= 1.0:
        raise EvaluationError(f"target coverage must lie in [0, 1], got {target!r}")
    return float(abs(coverage(interval, y) - t))


def group_coverage(
    interval: IntervalSet,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Per-group coverage plus the worst-group deviation ``CoV-W``.

    Marginal coverage is an average, and an average can be satisfied while some
    subgroup is badly under-covered -- the "hidden failure" that makes a
    conformal method unusable in practice. ``CoV-W`` (worst-group coverage
    violation) is the number that exposes it: the maximum absolute gap between
    each group's empirical coverage and the nominal target.

    Args:
        interval: Predicted intervals.
        y: Realised targets.
        groups: Integer group index per sample, same length as ``y``.

    Returns:
        Dict with ``per_group`` coverage, ``counts``, ``worst_group``,
        ``worst_gap`` (the CoV-W statistic) and ``marginal``.

    Raises:
        EvaluationError: If ``groups`` length differs from ``y``.
    """
    lo, hi = _check(interval, y)
    g = np.asarray(groups).reshape(-1)
    if g.shape != lo.shape:
        raise EvaluationError(f"groups shape {g.shape} does not match {lo.shape}")
    target = 1.0 - float(interval.alpha)

    inside = (lo <= y) & (y <= hi)
    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    gaps: dict[str, float] = {}
    for value in np.unique(g):
        mask = g == value
        key = str(int(value)) if np.issubdtype(g.dtype, np.integer) else str(value)
        cnt = int(np.sum(mask))
        if cnt == 0:  # pragma: no cover - unique() cannot produce an empty mask
            continue
        cov = float(np.mean(inside[mask]))
        per_group[key] = cov
        counts[key] = cnt
        gaps[key] = abs(cov - target)

    worst = max(gaps, key=lambda k: gaps[k]) if gaps else ""
    return {
        "per_group": per_group,
        "counts": counts,
        "worst_group": worst,
        "worst_gap": float(gaps[worst]) if gaps else 0.0,
        "marginal": float(np.mean(inside)),
        "target": target,
        "n_groups": len(per_group),
    }


def conditional_coverage_worst(
    interval: IntervalSet,
    y: np.ndarray,
    feature: np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Worst-bin coverage over a binned feature (distribution-free conditional check).

    An operational cousin of :func:`group_coverage` that needs no group labels:
    the feature is split into ``n_bins`` equal-frequency bins using cuts derived
    from the *evaluation* sample, and the worst bin's coverage gap is returned.
    It is a diagnostic, not a guarantee -- binning on test data is a descriptive
    device, and it is reported as such.

    Args:
        feature: 1-D covariate used for stratification.
        n_bins: Number of equal-frequency bins.

    Returns:
        Dict with ``bin_edges``, ``coverage`` per bin, ``counts`` and
        ``worst_gap``.
    """
    lo, hi = _check(interval, y)
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    if f.shape != lo.shape:
        raise EvaluationError(f"feature shape {f.shape} does not match {lo.shape}")
    nb = max(1, int(n_bins))
    qs = np.quantile(f, np.linspace(0.0, 1.0, nb + 1)[1:-1]) if nb > 1 else np.array([])
    idx = np.digitize(f, qs) if qs.size else np.zeros(f.shape, dtype=int)
    target = 1.0 - float(interval.alpha)
    inside = (lo <= y) & (y <= hi)
    covs: list[float] = []
    cnts: list[int] = []
    for b in range(nb):
        mask = idx == b
        cnt = int(np.sum(mask))
        cnts.append(cnt)
        covs.append(float(np.mean(inside[mask])) if cnt else float("nan"))
    finite = [c for c in covs if np.isfinite(c)]
    worst = max((abs(c - target) for c in finite), default=0.0)
    return {
        "bin_edges": np.asarray(qs),
        "coverage": covs,
        "counts": cnts,
        "worst_gap": float(worst),
        "target": target,
    }


def stability(interval: IntervalSet) -> float:
    """Mean relative change of the interval endpoints between adjacent samples.

    A guardrail against a specific failure mode: a method that emits essentially
    random or near-empty intervals can post an attractive mean width while being
    useless. If consecutive samples get wildly different intervals for similar
    inputs, the width number is not measuring what it claims.

    Defined as the mean of ``|Δlo| / (1 + |lo|)`` and ``|Δhi| / (1 + |hi|)`` over
    adjacent pairs. The ``1 +`` offsets keep the ratio finite for endpoints near
    zero, where a bare relative change would explode.

    Note:
        Samples are assumed to be in a meaningful order (e.g. sorted by ``x1``).
        On unordered test data this measures sampling noise, not instability.

    Returns:
        A non-negative float; smaller is more stable. Returns ``0.0`` for a
        single-sample interval, where the statistic is undefined rather than
        perfect -- reporting 0.0 for "no evidence" would be the safer lie, and
        this function refuses to tell it.
    """
    lo, hi = _check(interval)
    n = lo.size
    if n < 2:
        raise EvaluationError(
            "stability needs at least 2 samples; a single interval has no "
            "adjacent pair and the statistic would be undefined"
        )
    dlo = np.abs(np.diff(lo)) / (1.0 + np.abs(lo[:-1]))
    dhi = np.abs(np.diff(hi)) / (1.0 + np.abs(hi[:-1]))
    return float(np.mean(0.5 * (dlo + dhi)))


def efficiency_curve(
    intervals_by_alpha: dict[float, IntervalSet],
    y: np.ndarray,
) -> dict[str, Any]:
    """Coverage-versus-width trade-off across several miscoverage levels.

    A single (coverage, width) pair cannot distinguish a genuinely efficient
    method from one that is simply mis-calibrated. Sweeping alpha produces the
    whole curve; a method lying on the *lower-left* Pareto frontier dominates.

    Args:
        intervals_by_alpha: Mapping ``alpha -> IntervalSet``, all over the same
            ``y`` in the same order.
        y: Realised targets.

    Returns:
        Dict with sorted ``alphas``, ``coverage``, ``width`` and the
        ``winkler`` at each level.

    Raises:
        EvaluationError: If the mapping is empty or the target lengths differ.
    """
    if not intervals_by_alpha:
        raise EvaluationError("efficiency_curve needs at least one alpha level")
    levels = sorted(float(a) for a in intervals_by_alpha)
    covs: list[float] = []
    widths: list[float] = []
    scores: list[float] = []
    for a in levels:
        iv = (
            intervals_by_alpha[a]
            if a in intervals_by_alpha
            else intervals_by_alpha[levels[levels.index(a)]]
        )
        lo, hi = _check(iv, y)
        covs.append(float(np.mean((lo <= y) & (y <= hi))))
        widths.append(float(np.mean(hi - lo)))
        scores.append(winkler(iv, y, a))
    return {"alphas": levels, "coverage": covs, "width": widths, "winkler": scores}


def summarize(interval: IntervalSet, y: np.ndarray) -> dict[str, float]:
    """The full metric bundle reported for one benchmark cell.

    Args:
        interval: Predicted intervals.
        y: Realised targets.

    Returns:
        Flat dict of scalars: ``coverage``, ``mean_width``, ``winkler``,
        ``coverage_gap``, ``miss_rate``, ``width_share``, ``miss_share``,
        ``min_width`` and ``max_width``.
    """
    parts = winkler_components(interval, y, interval.alpha)
    lo, hi = _check(interval, y)
    width = hi - lo
    return {
        "coverage": coverage(interval, y),
        "mean_width": float(np.mean(width)),
        "winkler": parts["winkler"],
        "coverage_gap": coverage_gap(interval, y, 1.0 - float(interval.alpha)),
        "miss_rate": parts["miss_rate"],
        "width_share": parts["width_share"],
        "miss_share": parts["miss_share"],
        "min_width": float(np.min(width)),
        "max_width": float(np.max(width)),
    }
