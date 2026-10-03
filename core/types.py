"""Core dataclasses shared across data, method and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BenchmarkReport",
    "ClassificationSet",
    "ConformalResult",
    "DatasetBundle",
    "IntervalSet",
    "MethodSpec",
    "MetricRow",
    "SplitData",
]


@dataclass(frozen=True)
class IntervalSet:
    """A batch of prediction intervals plus the point predictions they surround.

    ``lower``/``upper`` are shape ``(n_samples,)``. Widths are always
    non-negative by construction; any method producing ``lower > upper`` has a bug
    and is rejected upstream.
    """

    lower: np.ndarray
    upper: np.ndarray
    point: np.ndarray | None = None
    alpha: float = 0.1

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        hi = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        if lo.shape != hi.shape:
            from core.errors import MethodError

            raise MethodError(f"lower/upper shape mismatch: {lo.shape} vs {hi.shape}")
        if np.any(hi < lo - 1e-12):
            from core.errors import MethodError

            bad = int(np.argmin(hi - lo))
            raise MethodError(
                f"interval {bad} is inverted: lower={lo[bad]:.6g} > upper={hi[bad]:.6g}"
            )
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        if self.point is not None:
            object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64).reshape(-1))

    def __len__(self) -> int:
        return int(self.lower.size)

    @property
    def width(self) -> np.ndarray:
        """Per-sample interval width."""
        return self.upper - self.lower

    @property
    def mean_width(self) -> float:
        return float(np.mean(self.width))

    def mask(self, y: np.ndarray) -> np.ndarray:
        """Boolean coverage indicator for targets ``y``."""
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape != self.lower.shape:
            from core.errors import EvaluationError

            raise EvaluationError(f"y shape {y.shape} != interval shape {self.lower.shape}")
        return (y >= self.lower) & (y <= self.upper)

    def coverage(self, y: np.ndarray) -> float:
        """Empirical marginal coverage."""
        return float(np.mean(self.mask(y)))

    def slice(self, idx: np.ndarray | slice) -> IntervalSet:
        return IntervalSet(
            self.lower[idx],
            self.upper[idx],
            None if self.point is None else self.point[idx],
            self.alpha,
        )


@dataclass(frozen=True)
class ClassificationSet:
    """A batch of conformal prediction *sets* for classification.

    ``matrix`` is a boolean array of shape ``(n_samples, n_classes)``; row ``i``
    is True exactly for the labels admitted into the prediction set.
    """

    matrix: np.ndarray
    classes: np.ndarray
    alpha: float = 0.1

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=bool)
        if mat.ndim != 2:
            from core.errors import MethodError

            raise MethodError(f"set matrix must be 2-D, got shape {mat.shape}")
        if mat.shape[0] == 0:
            from core.errors import MethodError

            raise MethodError("set matrix must contain at least one row")
        object.__setattr__(self, "matrix", mat)
        object.__setattr__(self, "classes", np.asarray(self.classes))

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def sizes(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    @property
    def mean_size(self) -> float:
        return float(np.mean(self.sizes))

    def coverage(self, y: np.ndarray) -> float:
        """Empirical coverage: fraction of labels contained in their set."""
        y = np.asarray(y)
        idx = np.searchsorted(self.classes, y)
        idx = np.clip(idx, 0, len(self.classes) - 1)
        matched = self.classes[idx] == y
        return float(np.mean(self.matrix[np.arange(len(self)), idx] & matched))

    def slice(self, idx: np.ndarray | slice) -> ClassificationSet:
        return ClassificationSet(self.matrix[idx], self.classes, self.alpha)


@dataclass(frozen=True)
class SplitData:
    """A four-way, strictly disjoint split.

    Conformal prediction is only valid when the calibration set is exchangeable
    with the test set and *independent of training*. This dataclass carries the
    index arrays so tests can assert disjointness rather than trusting comments.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_calib: np.ndarray
    y_calib: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    name: str = "unnamed"

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "calib": len(self.y_calib),
            "val": len(self.y_val),
            "test": len(self.y_test),
        }

    def assert_disjoint(self) -> None:
        """Raise if any two segments share a row.

        Uses row hashing, so duplicated rows in the *generator* are caught too.
        """
        from core.errors import SplitError

        seen: dict[bytes, str] = {}
        for tag, x in (
            ("train", self.x_train),
            ("calib", self.x_calib),
            ("val", self.x_val),
            ("test", self.x_test),
        ):
            arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
            for row in arr:
                key = row.tobytes()
                if key in seen:
                    raise SplitError(
                        f"segment {tag!r} shares a row with {seen[key]!r}: "
                        "conformal calibration must be exchangeable with test and "
                        "disjoint from training"
                    )
                seen[key] = tag


@dataclass(frozen=True)
class DatasetBundle:
    """A named synthetic dataset plus the metadata needed to regenerate it."""

    name: str
    split: SplitData
    groups: np.ndarray | None = None
    stream_x: np.ndarray | None = None
    stream_y: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stream(self) -> bool:
        return self.stream_x is not None and self.stream_y is not None


@dataclass(frozen=True)
class MethodSpec:
    """Static description of a conformal method.

    Attributes:
        name: Identifier used in reports and CLI flags.
        family: ``"baseline"``, ``"backend"`` or ``"flagship"``.
        backend: ``"numpy"`` (always available) or ``"sklearn"``/``"mapie"``.
        supports_alpha: Whether the method can emit several alpha at once.
        online: Whether the method consumes a stream (ACI family).
    """

    name: str
    family: str
    backend: str
    supports_alpha: bool = True
    online: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"baseline", "backend", "flagship"}:
            from core.errors import ConfigError

            raise ConfigError(f"unknown method family {self.family!r}")
        if self.backend not in {"numpy", "sklearn", "mapie"}:
            from core.errors import ConfigError

            raise ConfigError(f"unknown backend {self.backend!r}")


@dataclass(frozen=True)
class ConformalResult:
    """Everything a single (method, dataset, seed) run produces."""

    method: str
    dataset: str
    seed: int
    alpha: float
    coverage: float
    mean_width: float
    winkler: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> MetricRow:
        return MetricRow(
            method=self.method,
            dataset=self.dataset,
            seed=self.seed,
            alpha=self.alpha,
            coverage=self.coverage,
            mean_width=self.mean_width,
            winkler=self.winkler,
            extra=dict(self.extra),
        )


@dataclass(frozen=True)
class MetricRow:
    """One benchmark cell: method x dataset x seed."""

    method: str
    dataset: str
    seed: int
    alpha: float
    coverage: float
    mean_width: float
    winkler: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregate view over all rows plus gate bookkeeping."""

    rows: list[MetricRow] = field(default_factory=list)
    gates: dict[str, dict[str, object]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    ablations: list[dict[str, Any]] = field(default_factory=list)
    invariants: list[dict[str, Any]] = field(default_factory=list)

    def add(self, result: ConformalResult) -> None:
        self.rows.append(result.as_row())

    def methods(self) -> list[str]:
        return sorted({r.method for r in self.rows})

    def datasets(self) -> list[str]:
        return sorted({r.dataset for r in self.rows})

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "rows": [
                {
                    "method": r.method,
                    "dataset": r.dataset,
                    "seed": r.seed,
                    "alpha": r.alpha,
                    "coverage": r.coverage,
                    "mean_width": r.mean_width,
                    "winkler": r.winkler,
                    **r.extra,
                }
                for r in self.rows
            ],
            "gates": self.gates,
            "failures": self.failures,
            "ablations": self.ablations,
            "invariants": self.invariants,
        }
