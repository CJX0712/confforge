"""The single entry point for every conformal quantile in Confforge.

Why this module exists
----------------------
Conformal validity rests entirely on using the *order statistic* with the
finite-sample corrected index

.. math:: k = \\lceil (n+1)(1-\\alpha) \\rceil

Every other way of taking a quantile is subtly wrong: ``np.quantile(s, 1-alpha)``
omits the ``n+1`` correction and under-covers at small ``n``; the default
``method="linear"`` interpolates between observations, which the conformal
argument does not license; and an unguarded ``scores[k-1]`` silently wraps to
``scores[-1]`` (the *maximum*) when ``k == 0``.

By routing all quantile computation through :func:`conformal_quantile` we make
those three failure modes structurally impossible rather than merely tested for.

References
----------
Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018), *Distribution-Free Predictive
Inference for Regression*, JASA 113(523):1094-1111, Theorem 1.
Vovk, Gammerman, Shafer (2005), Theorem 1.
"""

from __future__ import annotations

import math

import numpy as np

from core.errors import DataError, MethodError

__all__ = [
    "conformal_index",
    "conformal_quantile",
    "conformal_quantile_batch",
    "is_infinite_quantile",
]


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise DataError("conformity scores must be non-empty")
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise DataError(
            f"conformity scores contain {bad} non-finite value(s); drop or repair them "
            "before calibration, otherwise the effective sample size used by the index "
            "formula no longer matches the data"
        )
    return arr


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise MethodError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
    return a


def conformal_index(n: int, alpha: float) -> int:
    """Return the 1-based rank ``k = max(1, ceil((n+1)(1-alpha)))``.

    The ``n+1`` (not ``n``) is not a correction but a consequence of the proof:
    the test point participates in the ranking, so the exact coverage is
    ``ceil((n+1)(1-alpha)) / (n+1)``.

    Args:
        n: Number of calibration scores actually used for the quantile.
        alpha: Miscoverage level (the *total* one, not a per-tail value).

    Returns:
        The 1-based order-statistic rank, clipped below at 1 so that a request
        for near-certain coverage returns the *smallest* score rather than
        wrapping around to the largest.

    Raises:
        DataError: If ``n`` is not positive.
        MethodError: If ``alpha`` is outside ``(0, 1)``.
    """
    if int(n) <= 0:
        raise DataError(f"calibration size must be positive, got {n}")
    a = _validate_alpha(alpha)
    return max(1, math.ceil((int(n) + 1) * (1.0 - a)))


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    *,
    allow_infinite: bool = False,
) -> float:
    """Return the conformal quantile of ``scores`` at miscoverage ``alpha``.

    Args:
        scores: Conformity scores of shape ``(n,)``. Must be finite.
        alpha: Miscoverage level, e.g. ``0.1`` for a 90% interval.
        allow_infinite: When ``True``, return ``+inf`` instead of raising if the
            calibration set is too small for the requested ``alpha`` (i.e.
            ``k > n``). Returning ``+inf`` degrades the interval to the whole
            line, which *preserves* validity; silently clipping the quantile
            level to 1.0 would not.

    Returns:
        The ``k``-th smallest score, exactly an observed value (no interpolation).

    Raises:
        DataError: If scores are empty or non-finite, or ``n <= 0``.
        MethodError: If ``alpha`` is invalid, or the quantile does not exist and
            ``allow_infinite`` is False.
    """
    arr = _validate_scores(scores)
    a = _validate_alpha(alpha)
    n = arr.size
    k = conformal_index(n, a)
    if k > n:
        # k > n  <=>  the calibration set cannot support this confidence level.
        if allow_infinite:
            return float("inf")
        needed = math.ceil(1.0 / (1.0 - a)) - 1
        raise MethodError(
            f"conformal quantile does not exist: n_calib={n} is too small for "
            f"alpha={a} (need at least {needed}). Increase the calibration set, "
            "relax alpha, or pass allow_infinite=True to accept an infinite bound."
        )
    # Partial selection: O(n) instead of O(n log n) for the full sort.
    return float(np.partition(arr, k - 1)[k - 1])


def conformal_quantile_batch(
    scores: np.ndarray,
    alphas: np.ndarray | list[float],
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    """Vectorised :func:`conformal_quantile` over several miscoverage levels.

    Args:
        scores: Conformity scores of shape ``(n,)``.
        alphas: Miscoverage levels, any order.
        allow_infinite: Forwarded to :func:`conformal_quantile`.

    Returns:
        Array of quantiles with the same shape as ``alphas``.
    """
    arr = _validate_scores(scores)
    levels = np.asarray(alphas, dtype=np.float64).reshape(-1)
    out = np.empty(levels.shape, dtype=np.float64)
    for i, a in enumerate(levels):
        out[i] = conformal_quantile(arr, float(a), allow_infinite=allow_infinite)
    return out


def is_infinite_quantile(value: float, tol: float = 1e-12) -> bool:
    """Whether a quantile is the ``+inf`` sentinel produced by a thin calibration set."""
    return not np.isfinite(value) and float(value) > tol
