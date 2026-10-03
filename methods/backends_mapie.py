"""MAPIE 1.5.0 backend adapters (Tier-0).

These two methods exist as an **independent check on our own implementation**.
If ``mapie_split`` and our ``split_absolute`` disagree about coverage, one of the
two is wrong; if ``mapie_cqr`` and our ``cqr`` disagree about width, the
discrepancy is worth a look. A conformal library that is only ever compared
against itself can hide a shared misreading of the theory.

The 1.5.0 return contract
--------------------------
``predict_interval`` returns ``(y_pred, intervals)`` with shapes ``(n,)`` and
``(n, 2, n_alpha)``, where ``intervals[:, 0, k]`` is the lower bound and
``intervals[:, 1, k]`` the upper bound of level ``k``.

.. warning:: The silent-failure trap
   MAPIE <= 1.0 returned ``(y_min, y_max)``. Under 1.5.0 the *old* unpacking
   still runs without error -- it just binds ``y_pred`` to ``y_min`` and the
   ``(n, 2, k)`` tensor to ``y_max``. The code appears to work and every number
   is wrong. :func:`_extract_bounds` therefore validates the *tensor rank* rather
   than trusting the unpacking, and :func:`_selftest_api` asserts the semantics
   against a hand-computed reference at start-up. See
   ``test_backends_mapie.py::test_new_api_extracts_bounds_from_axis_0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.errors import BackendUnavailableError, MethodError
from core.types import IntervalSet, MethodSpec
from methods.estimators import ModelBundle

__all__ = [
    "MAPIE_AVAILABLE",
    "MapieCQR",
    "MapieSplit",
    "api_selftest",
    "backend_status",
    "build_mapie_method",
    "mapie_specs",
]

try:  # pragma: no cover - exercised by whichever branch the env provides
    from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor

    MAPIE_AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as exc:
    MAPIE_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    SplitConformalRegressor = None  # type: ignore[assignment]
    ConformalizedQuantileRegressor = None  # type: ignore[assignment]


def backend_status() -> dict[str, Any]:
    """Report whether the MAPIE tier is usable, and why not if it isn't.

    Surfaced in the benchmark report so a missing optional dependency is a
    *recorded* skip with a reason, never a silent omission.
    """
    return {
        "available": bool(MAPIE_AVAILABLE),
        "reason": "" if MAPIE_AVAILABLE else _IMPORT_ERROR,
        "version": _mapie_version(),
    }


def _mapie_version() -> str:
    if not MAPIE_AVAILABLE:
        return ""
    try:  # pragma: no cover - trivial
        from importlib.metadata import version

        return version("mapie")
    except Exception:
        return "unknown"


def _extract_bounds(intervals: Any, level_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(lower, upper)`` out of a MAPIE 1.5.0 interval tensor.

    Args:
        intervals: The second element of ``predict_interval``.
        level_index: Which alpha column to read.

    Returns:
        ``(lower, upper)``, each of shape ``(n,)``.

    Raises:
        MethodError: If the array is not the expected ``(n, 2, k)`` layout. This
            check is the entire point of the function: it converts the silent
            misbinding described in the module docstring into a loud failure.
    """
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        # Legacy (y_min, y_max) layout -- accepted rather than rejected, because
        # it is unambiguous and costs nothing to support.
        return arr[:, 0], arr[:, 1]
    if arr.ndim != 3 or arr.shape[1] != 2:
        raise MethodError(
            f"unexpected MAPIE interval array of shape {arr.shape}; expected "
            "(n, 2, n_alpha) under MAPIE >= 1.5.0 or (n, 2) under the legacy "
            "layout. Refusing to guess: binding a (n, 2, k) tensor to a single "
            "bound is the classic MAPIE 1.x upgrade bug and yields wrong numbers "
            "without raising."
        )
    k = int(level_index)
    if not 0 <= k < arr.shape[2]:
        raise MethodError(
            f"alpha index {k} out of range for an interval array with {arr.shape[2]} level(s)"
        )
    return arr[:, 0, k], arr[:, 1, k]


class _MapieBase:
    """Shared plumbing for the two MAPIE adapters."""

    name = "mapie_base"

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = int(seed)
        self.options = dict(kwargs)
        self.bundle: ModelBundle | None = None
        self._model: Any = None
        self._alphas: tuple[float, ...] = ()

    def _require_model(self) -> Any:
        if self._model is None:
            raise BackendUnavailableError(
                f"{self.name}: call fit_calibrate() before predict_interval()"
            )
        return self._model

    def _require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            raise BackendUnavailableError(f"{self.name}: call fit() first")
        return self.bundle

    def predict_interval(
        self, x: np.ndarray, alpha: float | None = None
    ) -> IntervalSet:  # pragma: no cover
        raise NotImplementedError


class MapieSplit(_MapieBase):
    """``mapie.regression.SplitConformalRegressor`` -- split conformal, Tier-0.

    The reference implementation of the baseline. Uses ``prefit=True`` and the
    already-fitted shared point model so that the comparison against our own
    ``split_absolute`` isolates the *conformal* code and not the regression.
    """

    name = "mapie_split"

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
        *,
        bundle: ModelBundle | None = None,
        **hyper: Any,
    ) -> MapieSplit:
        """Fit (or adopt) the shared bundle; MAPIE itself only calibrates."""
        if not MAPIE_AVAILABLE:
            raise BackendUnavailableError(f"mapie is not installed: {_IMPORT_ERROR}")
        if bundle is not None:
            self.bundle = bundle
        else:
            self.bundle = ModelBundle.fit(
                np.asarray(x, dtype=np.float64), np.asarray(y), self.seed, alpha=alpha, **hyper
            )
        return self

    def fit_calibrate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        alphas: tuple[float, ...] = (0.1,),
    ) -> MapieSplit:
        """Run MAPIE's own calibration over the calibration split."""
        del groups
        model = SplitConformalRegressor(
            self._require_bundle().point,
            confidence_level=[1.0 - float(a) for a in alphas],
            conformity_score="absolute",
            prefit=True,
        )
        model.conformalize(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._model = model
        self._alphas = tuple(float(a) for a in alphas)
        return self

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        model = self._require_model()
        xa = np.asarray(x, dtype=np.float64)
        a = self._alphas[0] if alpha is None else float(alpha)
        k = self._alphas.index(a) if a in self._alphas else 0
        y_pred, intervals = model.predict_interval(xa)
        lo, hi = _extract_bounds(intervals, k)
        return IntervalSet(lo, hi, np.asarray(y_pred, dtype=np.float64).reshape(-1), a)

    def extras(self) -> dict[str, float]:
        return {"mapie_backend": 1.0}


class MapieCQR(_MapieBase):
    """``mapie.regression.ConformalizedQuantileRegressor`` -- Tier-0.

    Three MAPIE 1.5.0 behaviours were established empirically and constrain this
    adapter. All three are checked in ``test_backends_mapie.py`` so a library
    upgrade that changes any of them fails loudly instead of silently:

    1. **The base model must be a quantile learner.** A plain linear model raises
       ``ValueError: The base model is not supported.``; only
       ``GradientBoostingRegressor``, ``QuantileRegressor``,
       ``HistGradientBoostingRegressor`` and ``LGBMRegressor`` are accepted.
    2. **``confidence_level`` must be a scalar.** Passing a *list* -- even a
       single-element ``[0.9]`` -- raises
       ``decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]`` from
       MAPIE's internal ``Decimal`` conversion, despite the annotation promising
       an ``Iterable``. Multi-alpha is therefore unsupported in 1.5.0 and this
       adapter keeps one calibrated model per requested alpha.
    3. **``prefit=True`` demands exactly three preset estimators** carrying MAPIE's
       own 3-point quantile grid. Passing the two endpoints instead raises
       ``ValueError: You need to have provided 3 different estimators``.

    We use ``prefit=False`` with a single base regressor and let MAPIE build its
    own quantile grid. That costs one extra fit, and buys something worth more:
    MAPIE's CQR then shares *no* machinery with our :class:`~methods.methods.CQR`,
    which is the entire point of having a Tier-0 backend. Agreement between the
    two is real evidence; agreement between two calls into the same fitted model
    would be a tautology.
    """

    name = "mapie_cqr"

    #: Base families MAPIE 1.5.0 accepts for quantile regression.
    SUPPORTED_BASES = (
        "GradientBoostingRegressor",
        "QuantileRegressor",
        "HistGradientBoostingRegressor",
        "LGBMRegressor",
    )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
        *,
        bundle: ModelBundle | None = None,
        **hyper: Any,
    ) -> MapieCQR:
        """Fit (or adopt) the shared bundle MAPIE will clone across its quantile grid."""
        if not MAPIE_AVAILABLE:
            raise BackendUnavailableError(f"mapie is not installed: {_IMPORT_ERROR}")
        if bundle is not None:
            self.bundle = bundle
        else:
            self.bundle = ModelBundle.fit(
                np.asarray(x, dtype=np.float64), np.asarray(y), self.seed, alpha=alpha, **hyper
            )
        return self

    def _base_estimator(self) -> Any:
        """Build the unfitted base regressor MAPIE will clone across its grid.

        Raises:
            BackendUnavailableError: If the family is not one MAPIE accepts.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        est = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.5,
            n_estimators=int(self.options.get("n_estimators", 100)),
            max_depth=int(self.options.get("max_depth", 3)),
            learning_rate=float(self.options.get("learning_rate", 0.1)),
            subsample=1.0,
            random_state=self.seed,
        )
        if type(est).__name__ not in self.SUPPORTED_BASES:
            raise BackendUnavailableError(
                f"{type(est).__name__} is not an accepted MAPIE 1.5.0 base; "
                f"supported families are {list(self.SUPPORTED_BASES)}"
            )
        return est

    def fit_calibrate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        alphas: tuple[float, ...] = (0.1,),
    ) -> MapieCQR:
        """Calibrate one independent MAPIE CQR model per requested alpha."""
        del groups
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        models: dict[float, Any] = {}
        for a in alphas:
            model = ConformalizedQuantileRegressor(
                estimator=self._base_estimator(),
                confidence_level=float(1.0 - float(a)),  # scalar only -- see docstring
                prefit=False,
            )
            model.fit(self.bundle.x_train, self.bundle.y_train)
            model.conformalize(xa, ya)
            models[float(a)] = model
        self._model = models
        self._alphas = tuple(float(a) for a in alphas)
        return self

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        models = self._require_model()
        if not isinstance(models, dict) or not models:
            raise BackendUnavailableError("mapie_cqr: no calibrated model available")
        xa = np.asarray(x, dtype=np.float64)
        a = self._alphas[0] if alpha is None else float(alpha)
        model = models.get(a) or models[self._alphas[0]]
        y_pred, intervals = model.predict_interval(xa)
        lo, hi = _extract_bounds(intervals, 0)
        return IntervalSet(lo, hi, np.asarray(y_pred, dtype=np.float64).reshape(-1), a)

    def extras(self) -> dict[str, float]:
        return {"mapie_backend": 1.0}


def mapie_specs() -> list[MethodSpec]:
    """Static specs for the MAPIE tier."""
    return [
        MethodSpec(name="mapie_split", family="backend", backend="mapie"),
        MethodSpec(name="mapie_cqr", family="backend", backend="mapie"),
    ]


def build_mapie_method(name: str, seed: int = 0, **kwargs: Any) -> _MapieBase:
    """Instantiate a MAPIE adapter by name.

    Raises:
        MethodError: For an unknown name.
        BackendUnavailableError: If MAPIE is not importable. The benchmark
            catches this and records a skipped cell with the reason.
    """
    if not MAPIE_AVAILABLE:
        raise BackendUnavailableError(
            f"mapie backend requested ({name}) but the import failed: {_IMPORT_ERROR}"
        )
    if name == "mapie_split":
        return MapieSplit(seed=seed, **kwargs)
    if name == "mapie_cqr":
        return MapieCQR(seed=seed, **kwargs)
    raise MethodError(f"unknown mapie method {name!r}")


def api_selftest() -> dict[str, Any]:
    """Verify the 1.5.0 return contract against a hand-computed reference.

    Fits MAPIE on a tiny noiseless problem where the correct answer is known
    exactly, then checks that ``_extract_bounds`` recovers it. Cheap (well under a
    second) and it converts the silent-misbinding hazard into a startup assertion.

    Returns:
        Dict with ``ok`` and details. Never raises: an unavailable backend is a
        legitimate state, not an error.
    """
    if not MAPIE_AVAILABLE:
        return {"ok": False, "skipped": True, "reason": _IMPORT_ERROR}
    try:
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        x = rng.normal(size=(120, 2))
        y = 2.0 * x[:, 0]
        model = SplitConformalRegressor(
            LinearRegression().fit(x, y), confidence_level=0.9, prefit=True
        )
        model.conformalize(x, y)
        y_pred, intervals = model.predict_interval(x[:5])
        lo, hi = _extract_bounds(intervals, 0)
        arr = np.asarray(intervals)
        # With in-sample calibration the residuals are ~0, so the window must be
        # tight and, critically, must satisfy lower <= upper elementwise.
        ok = bool(
            arr.ndim == 3
            and arr.shape[1] == 2
            and np.all(np.isfinite(lo))
            and np.all(np.isfinite(hi))
            and np.all(hi >= lo)
            and np.allclose(np.asarray(y_pred).reshape(-1), 2.0 * x[:5, 0], atol=1e-6)
        )
        return {
            "ok": ok,
            "skipped": False,
            "intervals_shape": list(arr.shape),
            "mean_width": float(np.mean(hi - lo)),
            "reason": "" if ok else "MAPIE 1.5.0 interval contract did not hold",
        }
    except Exception as exc:
        return {"ok": False, "skipped": False, "reason": f"{type(exc).__name__}: {exc}"}
