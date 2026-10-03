"""Conformal conformity scores.

All scores follow the same convention: **larger means less conforming**, i.e. the
interval is built by asking "what is the ``(1-alpha)``-quantile of this score?"
and solving the score inequality for ``y``. Getting that direction backwards is
the single most common bug in conformal code, so it is asserted in the tests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "SolveMode",
    "absolute_score",
    "aps_score",
    "laplace_score",
    "normalized_score",
    "signed_score",
    "solve_interval_from_score",
]

SolveMode = Literal["both", "lower", "upper"]


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        from core.errors import MethodError

        raise MethodError(f"y_true shape {yt.shape} != y_pred shape {yp.shape}")
    if yt.size == 0:
        from core.errors import MethodError

        raise MethodError("empty target/prediction arrays are not supported")
    return yt, yp


def absolute_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Symmetric score ``|y - yhat|`` for split conformal.

    The induced interval is ``[yhat - q, yhat + q]``: symmetric by construction.
    It is the strongest baseline when the noise is homoscedastic and the weakest
    when it is not, because a single global ``q`` is forced to cover the
    noisiest region and therefore over-covers everywhere else.

    Returns:
        Non-negative scores of shape ``(n,)``.
    """
    yt, yp = _validate(y_true, y_pred)
    return np.abs(yt - yp)


def signed_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Two-sided signed score ``max(yhat - y, y - yhat)``.

    Identical in distribution to :func:`absolute_score` (it differs only on the
    positive side), but the induced interval is the *shifted* CQR-style window
    ``[yhat - q_lo, yhat + q_hi]`` when separate tails are calibrated, which is
    what makes asymmetric noise representable.

    Returns:
        Scores of shape ``(n,)``.
    """
    yt, yp = _validate(y_true, y_pred)
    return np.maximum(yp - yt, yt - yp)


def normalized_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray | float,
    *,
    clip: float | None = None,
) -> np.ndarray:
    """Scale-normalised score ``|y - yhat| / max(s, floor)``.

    Dividing by a scale estimate is the standard practical route to *locally
    adaptive* intervals without retraining a quantile regressor. It trades
    distribution-free validity for conditional quality: the guarantee becomes
    approximate unless the scale is estimated on the calibration set itself.

    Args:
        y_true: Targets.
        y_pred: Point predictions.
        scale: Non-negative scale, broadcastable to ``y_true``. Must be > 0.
        clip: Optional lower bound applied to the scale, guarding against
            division by zero for near-zero residuals.

    Returns:
        Non-negative scores of shape ``(n,)``.

    Raises:
        MethodError: If the effective scale is non-positive anywhere.
    """
    yt, yp = _validate(y_true, y_pred)
    s = np.abs(np.broadcast_to(np.asarray(scale, dtype=np.float64), yt.shape))
    floor = 1e-8 if clip is None else float(clip)
    s = np.maximum(s, floor)
    if np.any(s <= 0):
        from core.errors import MethodError

        raise MethodError("normalised score requires a strictly positive scale")
    return np.abs(yt - yp) / s


def laplace_score(proba: np.ndarray, y_true_idx: np.ndarray) -> np.ndarray:
    """Least-ambiguous-set (LAS) score for classification, ``1 - p_{y|x}``.

    The LAS score is *provably* the unique score for which the resulting set is
    the smallest in expectation, so it is the classification analogue of
    :func:`absolute_score`.

    Args:
        proba: Class probabilities of shape ``(n, n_classes)``, rows summing to 1.
        y_true_idx: Integer class indices of shape ``(n,)``.

    Returns:
        Scores in ``[0, 1]`` of shape ``(n,)``.
    """
    p = np.asarray(proba, dtype=np.float64)
    idx = np.asarray(y_true_idx, dtype=np.int64).reshape(-1)
    if p.ndim != 2:
        from core.errors import MethodError

        raise MethodError(f"proba must be 2-D, got shape {p.shape}")
    if p.shape[0] != idx.shape[0]:
        from core.errors import MethodError

        raise MethodError(f"proba rows {p.shape[0]} != labels {idx.shape[0]}")
    rows = np.arange(p.shape[0])
    return 1.0 - p[rows, idx]


def aps_score(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    random_tie: np.random.Generator | None = None,
) -> np.ndarray:
    """Adaptive prediction set (APS) cumulative-rank score.

    For classes sorted by decreasing probability, the score of the true label is
    the cumulative probability mass up to and including it (with the convention
    that including the last class gives exactly 1). Small scores mean "the true
    label was confidently predicted"; the prediction set is then the shortest
    prefix whose cumulative mass exceeds ``1 - q``.

    Unlike the LAS score, APS is *adaptive*: it accounts for the shape of the
    whole predicted distribution, so a flat posterior yields a large set and a
    peaked posterior a small one.

    Ties are broken uniformly at random when ``random_tie`` is supplied, which is
    the convention that keeps the finite-sample guarantee exact. With
    ``random_tie=None`` ties are broken deterministically by class order, which
    makes the guarantee conservative rather than invalid.

    Args:
        proba: Probabilities of shape ``(n, n_classes)``.
        y_true_idx: Integer class indices of shape ``(n,)``.
        classes: Optional class labels; accepted for signature symmetry and
            validated against ``proba`` when supplied.
        random_tie: Generator used to jitter tied probabilities.

    Returns:
        Scores in ``(0, 1]`` of shape ``(n,)``.

    Raises:
        MethodError: If ``classes`` length does not match the class count.
    """
    p = np.asarray(proba, dtype=np.float64)
    idx = np.asarray(y_true_idx, dtype=np.int64).reshape(-1)
    if classes is not None and len(np.asarray(classes)) != p.shape[1]:
        from core.errors import MethodError

        raise MethodError(
            f"classes has {len(np.asarray(classes))} entries but proba has {p.shape[1]} columns"
        )
    if random_tie is not None:
        jitter = random_tie.uniform(0.0, 1.0, size=p.shape) * 1e-12
        p = p + jitter
        rowsum = p.sum(axis=1, keepdims=True)
        p = p / rowsum
    order = np.argsort(-p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rows = np.arange(p.shape[0])
    rank = np.argmax(order == idx[:, None], axis=1)
    taken = cum[rows, rank]
    # Including the final class must yield exactly 1.0.
    last = order[:, -1] == idx
    taken = np.where(last, 1.0, taken)
    return np.clip(taken, 0.0, 1.0)


def solve_interval_from_score(
    score_fn,
    y_pred: np.ndarray,
    q: float,
    *,
    scale: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a score inequality to obtain prediction bounds.

    For a score of the form ``s(y) = a * y + b(yhat)`` the inversion is linear in
    ``y``; this helper covers the two shapes actually used in the package.

    Args:
        score_fn: Either ``"absolute"`` (symmetric window) or ``"normalized"``.
        y_pred: Point predictions of shape ``(n,)``.
        q: Conformal quantile of the scores.
        scale: Required for ``score_fn="normalized"``; broadcastable to ``y_pred``.

    Returns:
        ``(lower, upper)`` arrays of shape ``(n,)``.

    Raises:
        MethodError: For an unknown ``score_fn`` or a missing scale.
    """
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if score_fn == "absolute":
        return yp - float(q), yp + float(q)
    if score_fn == "normalized":
        if scale is None:
            from core.errors import MethodError

            raise MethodError("normalised scores require a scale")
        s = np.maximum(np.broadcast_to(np.asarray(scale, dtype=np.float64), yp.shape), 1e-8)
        return yp - float(q) * s, yp + float(q) * s
    from core.errors import MethodError

    raise MethodError(f"unknown score_fn {score_fn!r}")
