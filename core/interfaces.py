"""Protocols that every conformal method and backend adapter must satisfy.

The whole point of this module is *fair comparison*: if a method cannot report a
prediction interval at a requested ``alpha`` on a held-out set, it does not run.
All benchmark comparisons then share one interface and one set of semantics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import IntervalSet

__all__ = [
    "ConformalPredictor",
    "IntervalScorer",
    "OnlineConformalPredictor",
    "PointPredictor",
]


@runtime_checkable
class PointPredictor(Protocol):
    """A plain regression model, deliberately *without* any conformal logic."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> PointPredictor:
        """Fit on the training segment only."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return point predictions of shape ``(n,)``."""
        ...


@runtime_checkable
class ConformalPredictor(Protocol):
    """A two-stage conformal predictor: fit on train, calibrate, then predict.

    The separation is not cosmetic. Conformal validity relies on the calibration
    scores being exchangeable with future scores, which fails as soon as the
    calibration set is reused for model selection. Keeping ``fit_calibrate``
    separate from ``fit`` makes that contract explicit and testable.
    """

    name: str

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConformalPredictor:
        """Fit the underlying point model on the *training* segment."""
        ...

    def fit_calibrate(self, x: np.ndarray, y: np.ndarray) -> ConformalPredictor:
        """Compute conformity scores on the *calibration* segment."""
        ...

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        """Return intervals at ``alpha`` (or at every alpha seen during calibration)."""
        ...


@runtime_checkable
class OnlineConformalPredictor(Protocol):
    """Conformal predictor that also updates its miscoverage level online.

    Used by the adaptive conformal inference (ACI) family, where the target
    coverage must track a drifting data distribution.
    """

    name: str

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> OnlineConformalPredictor:
        """Consume one batch, updating the point model and the running scores."""
        ...

    def update_alpha(self, y_true: np.ndarray, interval: IntervalSet) -> float:
        """Apply the ACI update and return the new miscoverage level."""
        ...

    @property
    def alpha(self) -> float:
        """Current miscoverage level, always inside the configured clip range."""
        ...


@runtime_checkable
class IntervalScorer(Protocol):
    """Metric family operating on intervals and realised targets."""

    @staticmethod
    def coverage(interval: IntervalSet, y: np.ndarray) -> float:
        """Empirical marginal coverage in ``[0, 1]``."""
        ...

    @staticmethod
    def mean_width(interval: IntervalSet, y: np.ndarray | None = None) -> float:
        """Average interval width; smaller is better at equal coverage."""
        ...

    @staticmethod
    def winkler(interval: IntervalSet, y: np.ndarray, alpha: float) -> float:
        """Winkler interval score (lower is better)."""
        ...
