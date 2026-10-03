"""The eight conformal methods under comparison.

Every method implements the same three-step contract from
:class:`core.interfaces.ConformalPredictor`:

1. ``fit`` -- fit the shared models on **train** only;
2. ``fit_calibrate`` -- turn calibration residuals into a conformal quantile;
3. ``predict_interval`` -- emit an :class:`~core.types.IntervalSet` for any alpha.

Two invariants are enforced by construction and asserted in ``test_methods.py``:

* **Calibration never sees training data.** Conformality requires the calibration
  scores to be exchangeable with future scores, which is false the moment model
  selection reuses them. Each method therefore takes calibration inputs in a
  separate call from ``fit``.
* **All methods share one learner** (see :mod:`methods.estimators`). Differences
  in the results are attributable to the conformal mechanism, not to model class.

The eight methods
-----------------
=================  ==========================================================
name               mechanism
=================  ==========================================================
split_absolute     pooled ``|y - yhat|`` quantile -> symmetric window
split_normalized   ``|y - yhat| / sigma_hat(x)`` -> locally scaled window
cqr                Romano 2019 conformity score on a quantile regressor
mondrian           per-group calibration, thin cells shrunk to the pooled value
vennfuse           flagship: CQR anchor + Mondrian shrink + distilled ACI
mapie_split        MAPIE ``SplitConformalRegressor``  (Tier-0 backend)
mapie_cqr          MAPIE ``ConformalizedQuantileRegressor`` (Tier-0 backend)
aci                online adaptive conformal inference over a stream
=================  ==========================================================

Author: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.errors import DataError, MethodError, NotCalibratedError
from core.types import IntervalSet, MethodSpec
from methods import calibrators
from methods.estimators import DEFAULT_KNN, ModelBundle
from methods.quantiles import conformal_quantile
from methods.scores import absolute_score, normalized_score

__all__ = [
    "CQR",
    "METHOD_ORDER",
    "METHOD_SPECS",
    "SPECIAL_METHODS",
    "ACIConformal",
    "ConformalMethod",
    "Mondrian",
    "SplitAbsolute",
    "SplitNormalized",
    "VennFuse",
    "aci_alpha_update",
    "build_method",
]

#: Alpha values every method can emit. Storing them at calibration time is what
#: lets ``predict_interval()`` answer for any of them without recalibrating.
DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)


class ConformalMethod:
    """Base class implementing the fit / fit_calibrate / predict contract.

    Subclasses set :attr:`name` and override the three hooks. Holding the shared
    state here means every method gets the same validation and the same error
    behaviour, and a new method only has to supply its interval logic.
    """

    #: Identifier used in reports and CLI flags.
    name: str = "abstract"
    #: Static description consumed by the benchmark.
    spec: MethodSpec

    def __init__(
        self,
        seed: int = 0,
        alphas: tuple[float, ...] = DEFAULT_ALPHAS,
        **switches: Any,
    ) -> None:
        """Build a method instance.

        Args:
            seed: Reproducibility seed.
            alphas: Miscoverage levels the method must be able to emit.
            **switches: Component switches declared as class attributes, e.g.
                ``use_cqr=False``. They are set **per instance** so that building
                one ablated variant cannot leak into the next -- mutating a class
                attribute would make ``VennFuse.use_cqr`` a process-wide global
                and silently corrupt every later comparison.

        Raises:
            MethodError: If a keyword does not name a declared switch.
        """
        self.seed = int(seed)
        self.alphas = tuple(float(a) for a in alphas)
        self.bundle: ModelBundle | None = None
        self._calibrated: dict[str, Any] = {}
        for key, value in switches.items():
            if not hasattr(type(self), key):
                raise MethodError(
                    f"{type(self).__name__} has no component switch {key!r}; "
                    f"declared switches are "
                    f"{sorted(k for k in vars(type(self)) if k.startswith('use_'))}"
                )
            setattr(self, key, value)

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def _check_xy(x: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        xa = np.asarray(x, dtype=np.float64)
        if xa.ndim != 2:
            raise DataError(f"features must be 2-D, got shape {xa.shape}")
        if xa.shape[0] == 0:
            raise DataError("empty feature matrix")
        if y is not None:
            ya = np.asarray(y, dtype=np.float64).reshape(-1)
            if ya.shape[0] != xa.shape[0]:
                raise DataError(
                    f"x has {xa.shape[0]} rows but y has {ya.shape[0]}: "
                    "calibration inputs must be aligned"
                )
        return xa

    def _require_calibrated(self) -> dict[str, Any]:
        if not self._calibrated:
            raise NotCalibratedError(
                f"{self.name}: call fit_calibrate() before predict_interval(); "
                "an uncalibrated conformal method has no valid interval to emit"
            )
        return self._calibrated

    def _require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            raise NotCalibratedError(f"{self.name}: call fit() before predict_interval()")
        return self.bundle

    # -- contract ----------------------------------------------------------- #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
        *,
        bundle: ModelBundle | None = None,
        **hyper: Any,
    ) -> ConformalMethod:
        """Fit the shared models on the **training** split.

        Args:
            x: Training covariates.
            y: Training targets.
            alpha: Miscoverage level.
            bundle: A pre-fitted :class:`~methods.estimators.ModelBundle` to adopt
                instead of fitting a new one. The benchmark fits **one** bundle per
                ``(dataset, seed)`` and hands it to all eight methods, which is
                what makes "every method shares one learner" literally true rather
                than merely intended -- and it cuts the grid from 432 model fits to
                90.
            **hyper: Model hyper-parameter overrides, ignored when ``bundle`` is given.

        Returns:
            ``self``, for chaining.
        """
        xa = self._check_xy(x, y)
        if bundle is not None:
            self.bundle = bundle
        else:
            self.bundle = ModelBundle.fit(xa, np.asarray(y), self.seed, alpha=alpha, **hyper)
        return self

    def fit_calibrate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> ConformalMethod:
        raise NotImplementedError

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        raise NotImplementedError

    def extras(self) -> dict[str, float]:
        """Extra per-run diagnostics merged into the benchmark row."""
        return {}


class SplitAbsolute(ConformalMethod):
    """Split conformal with the absolute residual score.

    ``s_i = |y_i - yhat_i|``; the interval is ``[yhat - q, yhat + q]`` with ``q``
    the conformal quantile of the calibration scores.

    This is the reference method for the whole benchmark. It is the strongest
    possible baseline when the noise is homoscedastic and the model is well
    specified, and the weakest when neither holds -- a single global ``q`` must
    cover the noisiest region, so every quiet point is over-covered. Its width is
    therefore an upper bound that adaptive methods are measured against.
    """

    name: str = "split_absolute"
    spec: MethodSpec = MethodSpec(name="split_absolute", family="baseline", backend="numpy")

    def fit_calibrate(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> SplitAbsolute:
        """Pool the absolute residuals into a per-alpha quantile table."""
        del groups  # split conformal is deliberately group-blind
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        b = self._require_bundle()
        point = b.predict_point(xa)
        s = absolute_score(ya, point)
        self._calibrated = {
            "scores": s,
            "quantiles": {a: conformal_quantile(s, a, allow_infinite=True) for a in self.alphas},
        }
        return self

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        state = self._require_calibrated()
        b = self._require_bundle()
        a = self.alphas[1] if alpha is None else float(alpha)
        q = self._lookup(state, a)
        point = b.predict_point(x)
        if not np.isfinite(q):
            return IntervalSet(
                np.full(point.shape, -np.inf), np.full(point.shape, np.inf), point, a
            )
        return IntervalSet(point - q, point + q, point, a)

    @staticmethod
    def _lookup(state: dict[str, Any], alpha: float) -> float:
        table = state["quantiles"]
        if alpha in table:
            return float(table[alpha])
        # Off-grid alpha still gets a valid (slightly conservative) interval.
        return float(conformal_quantile(state["scores"], alpha, allow_infinite=True))


class SplitNormalized(ConformalMethod):
    """Locally adaptive split conformal via a kNN noise-scale estimate.

    ``s_i = |y_i - yhat_i| / sigma_hat(x_i)`` and the interval is
    ``[yhat - q * sigma_hat(x), yhat + q * sigma_hat(x)]``.

    The scale is a kNN average of absolute training residuals (see
    :func:`methods.estimators._knn_local_scale`), so it tracks heteroscedasticity
    without assuming a parametric noise model.

    Validity note: dividing by an estimated scale means the normalised scores are
    exchangeable only *approximately* -- the guarantee degrades to a practical one
    unless the scale is computed on the calibration split itself. That is the
    price of adaptivity, and it is why this method is reported alongside, never
    instead of, a strictly valid baseline.
    """

    name: str = "split_normalized"
    spec: MethodSpec = MethodSpec(name="split_normalized", family="baseline", backend="numpy")
    knn: int = DEFAULT_KNN

    def fit_calibrate(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> SplitNormalized:
        """Normalise calibration residuals by their local scale."""
        del groups
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        b = self._require_bundle()
        point = b.predict_point(xa)
        scale = b.local_scale(xa, k=self.knn)
        s = normalized_score(ya, point, scale)
        self._calibrated = {
            "scores": s,
            "scale": scale,
            "quantiles": {a: conformal_quantile(s, a, allow_infinite=True) for a in self.alphas},
        }
        return self

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        state = self._require_calibrated()
        b = self._require_bundle()
        xa = self._check_xy(x, None)
        a = self.alphas[1] if alpha is None else float(alpha)
        q = SplitAbsolute._lookup(state, a)
        point = b.predict_point(xa)
        scale = b.local_scale(xa, k=self.knn)
        if not np.isfinite(q):
            return IntervalSet(
                np.full(point.shape, -np.inf), np.full(point.shape, np.inf), point, a
            )
        return IntervalSet(point - q * scale, point + q * scale, point, a)

    def extras(self) -> dict[str, float]:
        scale = self._calibrated.get("scale")
        if scale is None:
            return {}
        return {"calib_scale_mean": float(np.mean(scale))}


class CQR(ConformalMethod):
    """Conformalized quantile regression (Romano, Patterson & Candes, 2019).

    Fits conditional quantile regressors at ``alpha/2`` and ``1 - alpha/2``,
    giving ``[q_lo(x), q_hi(x)]``, then scores each calibration point by how far
    outside that window it falls::

        s_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

    With ``q = Quantile_{1-alpha}(s)`` the interval is ``[q_lo - q, q_hi + q]``.

    The crucial property -- and the reason CQR can beat split conformal on
    *model error* rather than only on noise -- is that **``q`` is routinely
    negative**. A well-specified conditional quantile pair already brackets most
    calibration points, so the correction shrinks the window instead of widening
    it. A method whose ``q`` is always non-negative cannot exploit that, and is
    structurally unable to be sharper than its own base regressor.
    """

    name: str = "cqr"
    spec: MethodSpec = MethodSpec(name="cqr", family="baseline", backend="numpy")

    def fit_calibrate(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> CQR:
        """Score calibration points by their CQR conformity residual."""
        del groups
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        b = self._require_bundle()
        lo, hi = b.predict_quantiles(xa)
        s = np.maximum(lo - ya, ya - hi)
        self._calibrated = {
            "scores": s,
            "raw_width": float(np.mean(hi - lo)),
            "quantiles": {a: conformal_quantile(s, a, allow_infinite=True) for a in self.alphas},
        }
        return self

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        state = self._require_calibrated()
        b = self._require_bundle()
        xa = self._check_xy(x, None)
        a = self.alphas[1] if alpha is None else float(alpha)
        q = SplitAbsolute._lookup(state, a)
        lo, hi = b.predict_quantiles(xa)
        if not np.isfinite(q):
            return IntervalSet(np.full(lo.shape, -np.inf), np.full(lo.shape, np.inf), None, a)
        # A very negative q would invert the window; clamping to the feasibility
        # boundary keeps the guarantee (see ModelBundle.clamp_correction).
        q_eff = ModelBundle.clamp_correction(lo, hi, q)
        center = 0.5 * (lo + hi)
        return IntervalSet(lo - q_eff, hi + q_eff, center, a)

    def extras(self) -> dict[str, float]:
        raw = self._calibrated.get("raw_width")
        if raw is None:
            return {}
        # A negative q is CQR's signature: the correction shrinks the base window.
        qs = [q for q in self._calibrated["quantiles"].values() if np.isfinite(q)]
        return {
            "cqr_raw_width": float(raw),
            "cqr_q_mean": float(np.mean(qs)) if qs else float("nan"),
            "cqr_shrinking": float(np.mean(np.asarray(qs) < 0.0)) if qs else 0.0,
        }


class Mondrian(ConformalMethod):
    """Group-conditional (Mondrian) split conformal.

    Calibrates a separate quantile per predefined group -- low / medium / high
    noise -- so a quiet group is not charged for the noisiest group's spread.

    The groups come from the dataset: ground-truth latent components for
    ``grouped_hetero``, and noise-level strata derived on the *training* split
    otherwise (see :func:`data.synthetic.default_groups`).

    Thin cells are shrunk toward the pooled quantile by
    :func:`methods.calibrators.shrink`; without that, a three-point group would
    calibrate to its own maximum and blow up in width. See the validity caveat in
    :func:`methods.calibrators.stratified_quantile` before quoting a guarantee.
    """

    name: str = "mondrian"
    spec: MethodSpec = MethodSpec(name="mondrian", family="baseline", backend="numpy")
    lam: float = 10.0

    def fit_calibrate(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> Mondrian:
        """Build the per-group quantile table, shrunk toward the pooled value."""
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if groups is None:
            g = np.zeros(xa.shape[0], dtype=np.int64)
        else:
            g = np.asarray(groups).reshape(-1)
            if g.shape[0] != xa.shape[0]:
                raise DataError(f"groups {g.shape} do not match calibration rows {xa.shape}")
        b = self._require_bundle()
        s = absolute_score(ya, b.predict_point(xa))
        self._calibrated = {
            "scores": s,
            "groups": g,
            "tables": {a: calibrators.quantile_table(s, g, a, lam=self.lam) for a in self.alphas},
            "report": calibrators.calibration_report(s, g, self.alphas[1]),
        }
        return self

    def predict_interval(
        self, x: np.ndarray, alpha: float | None = None, groups: np.ndarray | None = None
    ) -> IntervalSet:
        state = self._require_calibrated()
        b = self._require_bundle()
        xa = self._check_xy(x, None)
        a = self.alphas[1] if alpha is None else float(alpha)
        table = state["tables"].get(a)
        if table is None:
            s = state["scores"]
            gq = calibrators.pooled_quantile(s, a)
            table = {"__global__": gq}
        if groups is None:
            g = np.zeros(xa.shape[0], dtype=np.int64)
        else:
            g = np.asarray(groups).reshape(-1)
            if g.shape[0] != xa.shape[0]:
                raise DataError(f"groups {g.shape} do not match prediction rows {xa.shape}")
        q = calibrators.resolve_group_quantile(table, g)
        point = b.predict_point(xa)
        lo = point - q
        hi = point + q
        # An infinite group quantile would make the window unbounded; that is a
        # valid but useless answer, so it is reported as such rather than hidden.
        return IntervalSet(lo, hi, point, a)

    def extras(self) -> dict[str, float]:
        rep = self._calibrated.get("report")
        if not rep:
            return {}
        cells = rep.get("cells", {})
        thin = sum(1 for c in cells.values() if not c.get("sufficient", True))
        return {
            "n_groups": float(len(cells)),
            "n_thin_cells": float(thin),
        }


class VennFuse(ConformalMethod):
    """Flagship: CQR anchor + Mondrian stratification + distilled ACI.

    Each of the three components is an independent switch, so the ablation in
    :mod:`eval.ablation` can disable exactly one at a time:

    ``use_cqr=True``
        Anchor on the CQR conformity score (negative-``q`` interval shrinking).
        With ``use_cqr=False`` the anchor degrades to the pooled absolute score,
        i.e. split conformal, and the interval can only widen.
    ``use_mondrian=True``
        Stratify the score quantile by group, shrinking thin cells toward the
        pooled value. With ``use_mondrian=False`` every group shares one quantile.
    ``use_aci=True``
        Distil the online ACI miscoverage level into a static per-group operating
        alpha, calibrated on the **validation** split. With ``use_aci=False`` the
        nominal alpha is used unchanged.

    The ACI component is *distilled* rather than run online because this method is
    evaluated on a held-out test split: a sequential method cannot consume the
    test stream without contaminating it. The distillation replays the exact ACI
    update rule of :func:`aci_alpha_update` over the validation split and keeps the
    adapted alpha, which preserves the adaptive behaviour while keeping the test
    split untouched. :class:`~methods.methods.ACIConformal` runs the same rule
    online, and gate G3 measures *that* variant under drift.

    Ordering matters and is fixed: CQR score -> group quantile -> ACI alpha
    calibration. Each stage consumes the previous one's output, so disabling a
    stage is a strict simplification rather than a different algorithm.
    """

    name: str = "vennfuse"
    spec: MethodSpec = MethodSpec(name="vennfuse", family="flagship", backend="numpy")
    lam: float = 10.0
    use_cqr: bool = True
    use_mondrian: bool = True
    use_aci: bool = True
    gamma: float = 0.01
    alpha_min: float = 0.01
    alpha_max: float = 0.5

    def fit_calibrate(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> VennFuse:
        """Compute the conformity score and the per-group quantile table."""
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        g = (
            np.zeros(xa.shape[0], dtype=np.int64)
            if groups is None
            else np.asarray(groups).reshape(-1)
        )
        if g.shape[0] != xa.shape[0]:
            raise DataError(f"groups {g.shape} do not match calibration rows {xa.shape}")
        b = self._require_bundle()

        if self.use_cqr:
            lo, hi = b.predict_quantiles(xa)
            s = np.maximum(lo - ya, ya - hi)
        else:
            s = absolute_score(ya, b.predict_point(xa))

        tables: dict[float, dict[str, float]] = {}
        for a in self.alphas:
            if self.use_mondrian:
                tables[a] = calibrators.quantile_table(s, g, a, lam=self.lam)
            else:
                tables[a] = {"__global__": calibrators.pooled_quantile(s, a)}
        self._calibrated = {
            "scores": s,
            "groups": g,
            "tables": tables,
            "report": calibrators.calibration_report(s, g, self.alphas[1]),
        }
        return self

    def distill_aci(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray | None = None,
        alpha_target: float = 0.1,
    ) -> dict[str, Any]:
        """Replay the ACI update on the **validation** split to adapt alpha.

        The validation split is used for exactly this purpose and never for
        calibration or testing, so this is the legitimate place to spend it.

        Args:
            x_val: Validation covariates.
            y_val: Validation targets.
            groups_val: Optional group labels for the validation split.
            alpha_target: The coverage level ACI is steering towards.

        Returns:
            Dict with the adapted ``alpha``, the realised trajectory, and the
            terminal value.

        Note:
            The reported ``alpha`` is the **median of the second half** of the
            trajectory, not its terminal value. The recursion is a stochastic
            approximation, so its last value is a single noisy draw from the
            converged regime: measured terminal values on the benchmark DGPs were
            0.101-0.131 against tail medians of 0.087-0.121, and on
            ``homoscedastic`` the terminal draw (0.131) was 50% wider than the
            median (0.087) purely by chance. Averaging the settled half is the
            standard variance reduction for this estimator and is the difference
            between measuring the method and measuring one sample of it.
        """
        b = self._require_bundle()
        xa = self._check_xy(x_val, y_val)
        ya = np.asarray(y_val, dtype=np.float64).reshape(-1)
        g = (
            np.zeros(xa.shape[0], dtype=np.int64)
            if groups_val is None
            else np.asarray(groups_val).reshape(-1)
        )
        state = self._calibrated
        if not state:
            raise NotCalibratedError("vennfuse: call fit_calibrate() before distill_aci()")

        if self.use_cqr:
            lo, hi = b.predict_quantiles(xa)
            s_val = np.maximum(lo - ya, ya - hi)
        else:
            s_val = absolute_score(ya, b.predict_point(xa))

        alpha = float(alpha_target)
        trajectory = [alpha]
        # Prequential replay: at step t the interval is built from the scores of
        # steps < t only, so the update never sees the point it is scoring.
        for t in range(xa.shape[0]):
            if t > 0:
                past = s_val[:t]
                a_t = trajectory[-1]
                if self.use_mondrian:
                    table = calibrators.quantile_table(past, g[:t], a_t, lam=self.lam)
                    q = calibrators.resolve_group_quantile(table, g[t : t + 1])[0]
                else:
                    q = calibrators.pooled_quantile(past, a_t)
                if np.isfinite(q):
                    if self.use_cqr:
                        lo_t, hi_t = b.predict_quantiles(xa[t : t + 1])
                        q_t = ModelBundle.clamp_correction(lo_t, hi_t, q)
                        inside = (lo_t[0] - q_t[0]) <= ya[t] <= (hi_t[0] + q_t[0])
                    else:
                        pt = b.predict_point(xa[t : t + 1])[0]
                        inside = (pt - q) <= ya[t] <= (pt + q)
                    err = 0.0 if inside else 1.0
                    # The target MUST be the fixed nominal miscoverage level.
                    # Passing the *current* alpha here (``target=a_t``) removes
                    # the restoring force and turns the recursion into a random
                    # walk: over a 500-step validation split alpha drifted far
                    # enough to drop benchmark coverage from 0.91 to 0.68-0.88.
                    alpha = aci_alpha_update(
                        alpha,
                        err,
                        gamma=self.gamma,
                        target=float(alpha_target),
                        lo=self.alpha_min,
                        hi=self.alpha_max,
                    )
            trajectory.append(alpha)
        path = np.asarray(trajectory, dtype=np.float64)
        settled = path[path.size // 2 :]
        operating = float(np.median(settled)) if settled.size else float(trajectory[-1])
        return {
            "alpha": operating,
            "alpha_target": float(alpha_target),
            "trajectory": path,
            "final": float(trajectory[-1]),
            "settled_median": operating,
            "n_settled": int(settled.size),
        }

    def predict_interval(
        self,
        x: np.ndarray,
        alpha: float | None = None,
        groups: np.ndarray | None = None,
        adapted_alpha: float | None = None,
    ) -> IntervalSet:
        """Emit the fused interval.

        Args:
            x: Prediction covariates.
            alpha: Requested miscoverage level. Ignored when ``adapted_alpha``
                is supplied, since the distilled ACI level supersedes it.
            groups: Optional group labels.
            adapted_alpha: Output of :meth:`distill_aci`.

        Returns:
            The prediction interval.
        """
        state = self._require_calibrated()
        b = self._require_bundle()
        xa = self._check_xy(x, None)
        nominal = self.alphas[1] if alpha is None else float(alpha)
        a = float(adapted_alpha) if adapted_alpha is not None else nominal

        table = state["tables"].get(a)
        if table is None:
            # Off-grid alpha: rebuild the table for it.
            if self.use_mondrian:
                table = calibrators.quantile_table(
                    state["scores"], state["groups"], a, lam=self.lam
                )
            else:
                table = {"__global__": calibrators.pooled_quantile(state["scores"], a)}

        g = (
            np.zeros(xa.shape[0], dtype=np.int64)
            if groups is None
            else np.asarray(groups).reshape(-1)
        )
        if g.shape[0] != xa.shape[0]:
            raise DataError(f"groups {g.shape} do not match prediction rows {xa.shape}")
        q = calibrators.resolve_group_quantile(table, g)

        if self.use_cqr:
            lo, hi = b.predict_quantiles(xa)
            center = 0.5 * (lo + hi)
            # Per-group q can be strongly negative; clamp to the feasibility
            # boundary so the fused window cannot invert (validity-preserving,
            # see ModelBundle.clamp_correction).
            q_eff = ModelBundle.clamp_correction(lo, hi, q)
            lower, upper = lo - q_eff, hi + q_eff
        else:
            center = b.predict_point(xa)
            lower, upper = center - q, center + q
        # An infinite q would mean the thin cell cannot support this alpha; the
        # only valid interval is then the whole line, and that is what is emitted.
        bad = ~np.isfinite(q)
        if np.any(bad):
            lower = np.where(bad, -np.inf, lower)
            upper = np.where(bad, np.inf, upper)
        return IntervalSet(lower, upper, center, a)

    def extras(self) -> dict[str, float]:
        out = {
            "vf_cqr": float(self.use_cqr),
            "vf_mondrian": float(self.use_mondrian),
            "vf_aci": float(self.use_aci),
        }
        state = self._calibrated
        if state:
            qs = [
                float(q) for q in state["tables"].get(self.alphas[1], {}).values() if np.isfinite(q)
            ]
            if qs:
                out["vf_q_mean"] = float(np.mean(qs))
                out["vf_shrinking"] = float(np.mean(np.asarray(qs) < 0.0))
        return out


def aci_alpha_update(
    alpha: float,
    err: float,
    *,
    gamma: float = 0.01,
    target: float | None = None,
    lo: float = 0.01,
    hi: float = 0.5,
) -> float:
    """One Adaptive Conformal Inference update (Gibbs & Candès, 2021).

    .. math:: \\alpha_{t+1} = \\mathrm{clip}\\bigl(\\alpha_t + \\gamma (\\alpha_{\\text{target}} - e_t)\bigr)

    where ``e_t`` is the miscoverage *rate* observed on the batch at step ``t``.

    The sign is the whole mechanism and is easy to get backwards. If the last
    batch was under-covered (``e_t`` high, meaning intervals too narrow), then
    ``alpha_target - e_t`` is *negative*, so alpha **decreases** and the intervals
    widen. Over-coverage pushes alpha up and the intervals tighten. The clip
    bounds are what keep the recursion from running away to a degenerate
    ``alpha = 0`` (infinite intervals) or ``alpha = 1`` (empty ones).

    Args:
        alpha: Current miscoverage level.
        err: Observed miscoverage rate in ``[0, 1]``.
        gamma: Step size. Larger reacts faster and oscillates more.
        target: Target **miscoverage** level -- the same units as ``alpha`` and
            ``err``, *not* a coverage level. Defaults to ``alpha`` (no net pull).
        lo: Lower clip bound.
        hi: Upper clip bound.

    Returns:
        The updated, clipped miscoverage level.

    Raises:
        MethodError: If ``err`` is outside ``[0, 1]`` or ``gamma`` is negative.

    Warning:
        All three of ``alpha``, ``err`` and ``target`` are **miscoverage**
        quantities. Passing a coverage level as ``target`` is the single easiest
        way to break ACI, and it fails silently: with ``alpha=0.1`` and a nominal
        90% target, passing ``target=0.9`` makes the update ``0.1 + g(0.9 - err)``,
        which drives alpha *up* to its ceiling on every step, shrinking the
        intervals to nothing while the run still produces plausible-looking
        numbers. That bug was present here and is now pinned by a unit test.
    """
    e = float(err)
    if not 0.0 <= e <= 1.0:
        raise MethodError(f"ACI error rate must lie in [0, 1], got {err!r}")
    if gamma < 0:
        raise MethodError(f"ACI gamma must be non-negative, got {gamma}")
    tgt = float(alpha) if target is None else float(target)
    nxt = float(alpha) + float(gamma) * (tgt - e)
    return float(min(max(nxt, float(lo)), float(hi)))


class ACIConformal(ConformalMethod):
    """Online adaptive conformal inference over a stream.

    Wraps a fixed-interval rule (split absolute by default) and updates its
    miscoverage level after every observed batch via :func:`aci_alpha_update`.
    This is the method gate G3 measures: after a noise jump, a fixed-alpha
    predictor is stuck at its pre-drift width while ACI re-tightens coverage.
    """

    name: str = "aci"
    spec: MethodSpec = MethodSpec(name="aci", family="baseline", backend="numpy", online=True)
    gamma: float = 0.01
    alpha_min: float = 0.01
    alpha_max: float = 0.5

    def __post_init__(self) -> None:
        self._alpha: float = self.alphas[1]
        self._q_by_alpha: dict[float, float] = {}

    def fit_calibrate(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> ACIConformal:
        """Cache the conformity scores; alpha is chosen online, not here."""
        del groups
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        b = self._require_bundle()
        s = absolute_score(ya, b.predict_point(xa))
        self._calibrated = {
            "scores": s,
            "quantiles": {a: conformal_quantile(s, a, allow_infinite=True) for a in self.alphas},
        }
        self._alpha = self.alphas[0]
        # The ACI target is the *nominal miscoverage* level and must stay fixed,
        # in the same units as ``alpha`` and ``err``. See the warning on
        # :func:`aci_alpha_update`: deriving it as ``1 - alpha`` (a coverage
        # level) drove alpha to its ceiling and collapsed the intervals.
        self.alpha_target = float(self.alphas[0])
        return self

    @property
    def alpha(self) -> float:
        """Current miscoverage level, always inside the clip range."""
        return float(self._alpha)

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> ACIConformal:
        """Consume a new batch, extending the running conformity-score pool.

        Required by :class:`~core.interfaces.OnlineConformalPredictor`. The
        windowed model is *not* refitted here -- doing so per batch would make the
        calibration scores depend on the order batches arrived in, which breaks
        exchangeability. Instead the batch's absolute residuals are appended to
        the score pool and the quantile cache is refreshed, so the miscoverage
        level adapts online while the score distribution stays a genuine sample
        of exchangeable residuals.

        Args:
            x: New covariates.
            y: New targets.

        Returns:
            ``self``, for chaining.
        """
        if self.bundle is None:
            raise NotCalibratedError("aci: call fit() before partial_fit()")
        xa = self._check_xy(x, y)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        state = self._calibrated
        if not state:
            raise NotCalibratedError("aci: call fit_calibrate() before partial_fit()")

        new_scores = absolute_score(ya, self.bundle.predict_point(xa))
        pooled = np.concatenate([state["scores"], new_scores])
        state["scores"] = pooled
        state["quantiles"] = {
            a: conformal_quantile(pooled, a, allow_infinite=True) for a in self.alphas
        }
        state["n_observed"] = int(state.get("n_observed", 0)) + int(xa.shape[0])
        return self

    def update_alpha(self, y_true: np.ndarray, interval: IntervalSet) -> float:
        """Apply one ACI update from realised outcomes.

        Args:
            y_true: Targets observed at this step.
            interval: The interval that was emitted for them.

        Returns:
            The new miscoverage level.
        """
        err = float(np.mean(~interval.mask(y_true)))
        self._alpha = aci_alpha_update(
            self._alpha,
            err,
            gamma=self.gamma,
            target=self.alpha_target,
            lo=self.alpha_min,
            hi=self.alpha_max,
        )
        return self._alpha

    def predict_interval(self, x: np.ndarray, alpha: float | None = None) -> IntervalSet:
        state = self._require_calibrated()
        b = self._require_bundle()
        a = self.alpha if alpha is None else float(alpha)
        if a in state["quantiles"]:
            q = float(state["quantiles"][a])
        else:
            q = float(conformal_quantile(state["scores"], a, allow_infinite=True))
        point = b.predict_point(np.asarray(x, dtype=np.float64))
        if not np.isfinite(q):
            return IntervalSet(
                np.full(point.shape, -np.inf), np.full(point.shape, np.inf), point, a
            )
        return IntervalSet(point - q, point + q, point, a)

    def extras(self) -> dict[str, float]:
        return {"aci_alpha": self.alpha}


#: Methods built from the shared NumPy machinery, in benchmark order. The two
#: MAPIE backends live in :mod:`methods.backends_mapie` and are appended by
#: :func:`build_method` so the two tiers stay separable.
SPECIAL_METHODS: tuple[str, ...] = (
    "split_absolute",
    "split_normalized",
    "cqr",
    "mondrian",
    "vennfuse",
    "aci",
)

METHOD_ORDER: tuple[str, ...] = (
    "split_absolute",
    "split_normalized",
    "cqr",
    "mondrian",
    "vennfuse",
    "mapie_split",
    "mapie_cqr",
    "aci",
)

_CLASSES: dict[str, type[ConformalMethod]] = {
    "split_absolute": SplitAbsolute,
    "split_normalized": SplitNormalized,
    "cqr": CQR,
    "mondrian": Mondrian,
    "vennfuse": VennFuse,
    "aci": ACIConformal,
}


def build_method(name: str, seed: int = 0, **kwargs: Any) -> Any:
    """Instantiate a method by name, including the MAPIE backends.

    Args:
        name: One of :data:`METHOD_ORDER`.
        seed: Reproducibility seed.
        **kwargs: Forwarded to the method constructor (e.g. VennFuse switches).

    Returns:
        An unfitted conformal method.

    Raises:
        MethodError: For an unknown method name.
    """
    if name in _CLASSES:
        return _CLASSES[name](seed=seed, **kwargs)
    if name in {"mapie_split", "mapie_cqr"}:
        from methods.backends_mapie import build_mapie_method

        return build_mapie_method(name, seed=seed, **kwargs)
    raise MethodError(f"unknown method {name!r}; expected one of {list(METHOD_ORDER)}")


def method_specs() -> list[MethodSpec]:
    """Static specs for all eight methods, in benchmark order."""
    from methods.backends_mapie import mapie_specs

    out: list[MethodSpec] = []
    for name in METHOD_ORDER:
        if name in _CLASSES:
            out.append(_CLASSES[name].spec)
        else:
            out.extend(s for s in mapie_specs() if s.name == name)
    return out


METHOD_SPECS = method_specs
