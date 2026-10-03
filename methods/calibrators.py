"""Calibration strategies: pooled, stratified (Mondrian), and shrunk.

All of these return a *scalar or per-group quantile* computed by
:func:`methods.quantiles.conformal_quantile`. Nothing here inverts a score or
builds an interval; that is :mod:`methods.methods`' job. Keeping the split
sharp is what makes the validity argument auditable: the only place a
finite-sample correction is applied is the single function the quantile module
owns.

The interesting one is :func:`stratified_quantile`, which carries a deliberate
validity caveat documented at its definition.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError, MethodError
from methods.estimators import SHRINKAGE_LAMBDA
from methods.quantiles import conformal_index, conformal_quantile

__all__ = [
    "pooled_quantile",
    "quantile_table",
    "resolve_group_quantile",
    "shrink",
    "stratified_quantile",
]


def pooled_quantile(scores: np.ndarray, alpha: float) -> float:
    """Single global conformal quantile over all calibration scores.

    Args:
        scores: Conformity scores from the calibration split.
        alpha: Miscoverage level.

    Returns:
        The conformal quantile, or ``+inf`` when the calibration set is too thin
        to support ``alpha``.
    """
    return conformal_quantile(scores, alpha, allow_infinite=True)


def shrink(
    group_value: float,
    n_group: int,
    global_value: float,
    lam: float = SHRINKAGE_LAMBDA,
) -> float:
    """James-Stein style shrinkage of a group statistic toward the pooled one.

    .. math:: \\hat{q}_g = \\frac{n_g \\hat{q}_g + \\lambda \\hat{q}_{\\text{global}}}{n_g + \\lambda}

    With ``n_g -> inf`` this returns the group value; with ``n_g = 0`` it returns
    the pooled value. The weight ``n_g / (n_g + lambda)`` is the bias-variance
    knob: it says "trust the group once it has about ``lambda`` observations".

    Args:
        group_value: Statistic computed within the group.
        n_group: Number of calibration points behind ``group_value``.
        global_value: Pooled statistic used as the shrinkage target.
        lam: Shrinkage strength; ``0`` disables shrinkage entirely.

    Returns:
        The shrunk statistic.

    Raises:
        MethodError: If ``n_group`` is negative or ``lam`` is negative.

    Note:
        An infinite ``group_value`` -- which is what a cell too thin to support
        ``alpha`` produces -- collapses to the pooled value rather than staying
        infinite. The naive weighted average would return
        ``(n * inf + lam * g) / (n + lam) = inf``, so one under-powered cell
        poisons the whole table and emits unbounded intervals. Falling back to the
        pooled quantile is both finite and *conservative*, which is the correct
        behaviour: a cell we cannot calibrate is a cell we must cover globally.
    """
    n = int(n_group)
    if n < 0:
        raise MethodError(f"group size must be non-negative, got {n_group}")
    strength = float(lam)
    if strength < 0:
        raise MethodError(f"shrinkage strength must be non-negative, got {lam}")
    gv = float(group_value)
    if not np.isfinite(gv):
        return float(global_value)
    if strength == 0.0:
        return gv
    return float((n * gv + strength * float(global_value)) / (n + strength))


def stratified_quantile(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    *,
    lam: float = SHRINKAGE_LAMBDA,
    min_cell: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Mondrian (group-conditional) conformal quantiles with thin-cell shrinkage.

    Each group is calibrated separately, so a method can afford a *narrow* interval
    in the quiet group while still covering the noisy one. The cost is variance:
    per-group quantiles are estimated from fewer points, and a group with three
    calibration samples has a "conformal quantile" equal to its maximum, which is
    a valid but useless interval.

    Two guards address that:

    ``lam``
        Shrink each group quantile toward the pooled quantile (see :func:`shrink`).
    ``min_cell``
        Groups with fewer than ``min_cell`` calibration points are *not*
        calibrated at all; they inherit the pooled quantile.

    .. warning:: Validity caveat -- read before quoting a guarantee
       Per-group conformal prediction is valid **only if** the group assignment
       is a function of information available independently of calibration and
       test outcomes, *and* each group is calibrated on enough points. With
       shrinkage toward the pooled quantile, exact per-group validity is traded
       for variance reduction: a shrunk quantile is no longer an order statistic
       of that group's scores, so the finite-sample proof does not transfer
       verbatim. What survives is a *practical* improvement in conditional
       coverage, which is exactly what gate G2 measures. Reported group coverage
       is therefore an empirical measurement, never a claimed guarantee.

    Args:
        scores: Conformity scores of shape ``(n_calib,)``.
        groups: Integer group index per calibration point, shape ``(n_calib,)``.
        alpha: Miscoverage level.
        lam: Shrinkage strength toward the pooled quantile.
        min_cell: Minimum calibration count for a group to be calibrated on its own.

    Returns:
        ``(group_ids, group_quantiles)`` -- two parallel arrays covering only the
        groups that had enough data to be calibrated individually.

    Raises:
        DataError: If scores and groups disagree in length.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if s.shape != g.shape:
        raise DataError(f"scores {s.shape} and groups {g.shape} must be the same length")
    if s.size == 0:
        raise DataError("cannot stratify an empty calibration set")

    gq = pooled_quantile(s, alpha)
    ids: list[int] = []
    values: list[float] = []
    for value in np.unique(g):
        mask = g == value
        n_g = int(np.sum(mask))
        if n_g < max(1, int(min_cell)):
            continue
        if n_g < 2:
            # A single point has no order statistic to speak of; the shrunk value
            # would be dominated by the prior anyway.
            continue
        q_g = conformal_quantile(s[mask], alpha, allow_infinite=True)
        ids.append(int(value))
        values.append(shrink(q_g, n_g, gq, lam=lam))
    if not ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.asarray(ids, dtype=np.int64), np.asarray(values, dtype=np.float64)


def quantile_table(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    *,
    lam: float = SHRINKAGE_LAMBDA,
) -> dict[str, float]:
    """Full group -> quantile lookup including the pooled fallback.

    The returned mapping always contains the sentinel key ``"__global__"``. A
    group absent from the mapping is not an error: :func:`resolve_group_quantile`
    falls back to the pooled value, which is precisely the behaviour wanted for a
    test point whose group never appeared in calibration (a new subgroup, or the
    covariate-shift case where test points land outside the training support).
    """
    ids, values = stratified_quantile(scores, groups, alpha, lam=lam)
    table: dict[str, float] = {"__global__": pooled_quantile(scores, alpha)}
    for gid, q in zip(ids, values, strict=True):
        table[str(int(gid))] = float(q)
    return table


def resolve_group_quantile(table: dict[str, float], groups: np.ndarray) -> np.ndarray:
    """Broadcast a quantile table over the groups of a *new* batch.

    Args:
        table: Output of :func:`quantile_table`.
        groups: Group indices of the samples to predict for.

    Returns:
        Quantile of shape ``(n,)``; samples whose group is absent from ``table``
        receive the pooled ``"__global__"`` value.
    """
    g = np.asarray(groups).reshape(-1)
    default = float(table.get("__global__", np.inf))
    out = np.full(g.shape, default, dtype=np.float64)
    for value in np.unique(g):
        key = str(int(value))
        if key in table:
            out[g == value] = float(table[key])
    return out


def calibration_report(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    """Per-cell counts and achieved index, for the benchmark report.

    Worth recording because "group 2 used the pooled fallback" is exactly the kind
    of fact that explains a surprising coverage number later, and it is invisible
    in the width column.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    cells: dict[str, dict[str, float]] = {}
    for value in np.unique(g):
        mask = g == value
        n_g = int(np.sum(mask))
        key = str(int(value))
        cells[key] = {
            "n": float(n_g),
            # k can exceed n_g for a thin cell, which is exactly the case where
            # the conformal quantile is undefined and the pooled fallback applies.
            "k": float(conformal_index(n_g, alpha)) if n_g > 0 else 0.0,
            "sufficient": float(n_g >= 2),
        }
    return {"n_calib": int(s.size), "alpha": float(alpha), "cells": cells}
