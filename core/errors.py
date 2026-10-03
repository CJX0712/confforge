"""Error taxonomy for Confforge.

Error codes are grouped by layer:

======  ==========================================================
Range   Meaning
======  ==========================================================
E1xx    Configuration / environment
E2xx    Data layer
E3xx    Method layer (conformal predictors)
E4xx    Evaluation layer
E5xx    Backend availability
======  ==========================================================
"""

from __future__ import annotations

__all__ = [
    "BackendUnavailableError",
    "ConfforgeError",
    "ConfigError",
    "DataError",
    "EvaluationError",
    "MethodError",
    "NotCalibratedError",
    "SplitError",
]


class ConfforgeError(Exception):
    """Base class for every error raised by Confforge.

    Attributes:
        code: Stable numeric code used in reports and tests.
    """

    code = 0

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[E{self.code:03d}] {self.message}"


class ConfigError(ConfforgeError):
    """Invalid or missing configuration (E1xx)."""

    code = 100


class DataError(ConfforgeError):
    """Malformed dataset or split (E2xx)."""

    code = 200


class SplitError(DataError):
    """Train / calibration / test segments are inconsistent or leaked (E2xx)."""

    code = 210


class MethodError(ConfforgeError):
    """A conformal method was used incorrectly (E3xx)."""

    code = 300


class NotCalibratedError(MethodError):
    """``predict`` was called before the calibration step (E3xx)."""

    code = 310


class EvaluationError(ConfforgeError):
    """Metric computation received invalid inputs (E4xx)."""

    code = 400


class BackendUnavailableError(ConfforgeError):
    """An optional third-party backend is not installed (E5xx)."""

    code = 500
