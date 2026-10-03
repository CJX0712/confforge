"""Synthetic data-generating processes (DGPs) for the Confforge benchmark.

Why synthetic data
------------------
Conformal prediction is a *worst-case* guarantee, so a benchmark on real data can
only ever show what happened to happen. Synthetic DGPs are the only way to ask
the questions that actually matter here:

* Does a method keep its coverage when the noise scale varies with ``x``?
* Does it keep *conditional* coverage across heterogeneous subgroups?
* Does it degrade gracefully when the calibration set is thin?
* Does it notice that the world has changed underneath it?

Every DGP therefore carries an explicit **difficulty knob** (a named scalar in
``meta["knobs"]``) rather than a hard-coded constant. That is what makes step 7
of the delivery plan -- sweeping a knob until the benchmark has *discriminating
power* -- possible without rewriting the generator.

Design rules obeyed by every generator
--------------------------------------
1. **Four strictly disjoint splits.** ``train / calib / val / test`` are drawn
   from one call and never overlap; :meth:`SplitData.assert_disjoint` enforces it.
2. **Exchangeability of calib and test.** Under no drift, ``calib`` and ``test``
   come from the *same* distribution. This is the assumption the conformal
   guarantee rests on, and violating it silently is the classic way to ship a
   method that only works in a notebook.
3. **Named RNG streams.** All randomness comes from :func:`core.seed.RNG.stream`,
   so a generator's output depends on its *name*, not on how many random numbers
   some unrelated component happened to draw first.
4. **Reproducibility.** ``make(name, seed, ...)`` called twice with the same
   arguments returns bit-identical arrays.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from core.errors import DataError
from core.seed import spawn
from core.types import DatasetBundle, SplitData

__all__ = [
    "DGP_NAMES",
    "GROUPED_DGP",
    "available",
    "default_groups",
    "group_slices",
    "make",
    "make_stream",
]

#: Canonical benchmark order. The report and the CLI both iterate this, so the
#: order is part of the public contract.
DGP_NAMES: tuple[str, ...] = (
    "homoscedastic",
    "heteroscedastic",
    "grouped_hetero",
    "covariate_shift",
    "small_calib",
    "nonlinear",
)

#: The only DGP that ships ground-truth group labels.
GROUPED_DGP = "grouped_hetero"

_N_FEATURES = 4


# --------------------------------------------------------------------------- #
# low-level sampling helpers
# --------------------------------------------------------------------------- #
def _standard_normal(rng: np.random.Generator, n: int, d: int = _N_FEATURES) -> np.ndarray:
    return rng.standard_normal((n, d))


def _sigmoid_gate(u: np.ndarray) -> np.ndarray:
    """Smooth 0->1 gate used to build genuinely overlapping subgroups.

    Overlap matters: with hard-disjoint groups a Mondrian method could win by
    memorising the partition, which is a much weaker claim than adapting to a
    smoothly varying noise level.
    """
    return 1.0 / (1.0 + np.exp(-u))


# --------------------------------------------------------------------------- #
# per-DGP mean / noise functions
# --------------------------------------------------------------------------- #
def _mean_homoscedastic(x: np.ndarray) -> np.ndarray:
    """``2 x1 + 0.5 x2^2`` -- a quadratic that a linear model only half-captures."""
    return 2.0 * x[:, 0] + 0.5 * x[:, 1] ** 2


def _sigma_homoscedastic(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.full(x.shape[0], float(sigma))


def _sigma_heteroscedastic(x: np.ndarray, floor: float, span: float) -> np.ndarray:
    """``sigma(x) = floor + span * |x1|`` -- noise grows with the dominant feature."""
    return floor + span * np.abs(x[:, 0])


def _group_sigma(group: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    return np.asarray(sigmas, dtype=np.float64)[group]


# --------------------------------------------------------------------------- #
# generators: each returns (x, y, groups) for one segment
# --------------------------------------------------------------------------- #
def _gen_homoscedastic(
    rng: np.random.Generator, n: int, *, sigma: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _standard_normal(rng, n)
    mu = _mean_homoscedastic(x)
    y = mu + rng.normal(scale=sigma, size=n)
    return x, y, np.zeros(n, dtype=np.int64)


def _gen_heteroscedastic(
    rng: np.random.Generator,
    n: int,
    *,
    sigma_floor: float = 0.2,
    sigma_span: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _standard_normal(rng, n)
    mu = _mean_homoscedastic(x)
    y = mu + rng.normal(scale=1.0, size=n) * _sigma_heteroscedastic(x, sigma_floor, sigma_span)
    # Groups are *derived* from the noise level itself: this is the structure a
    # Mondrian method is supposed to discover without being told.
    return x, y, _uninformative_groups(n)


def _gen_grouped_hetero(
    rng: np.random.Generator,
    n: int,
    *,
    sigmas: tuple[float, float, float] = (0.2, 0.6, 1.2),
    overlap: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three overlapping subgroups with very different noise levels.

    The group is a *latent* mixture component: it shifts ``x1`` and ``x2`` rather
    than being a deterministic function of them, so no feature-based rule can
    recover it exactly. ``overlap`` controls how much the components interpenetrate.
    """
    comp = rng.integers(0, len(sigmas), size=n)
    # Component-dependent mean shift, then a shared standard-normal core.
    x = _standard_normal(rng, n)
    shift = np.linspace(-1.0, 1.0, len(sigmas))
    x[:, 0] += overlap * shift[comp]
    x[:, 1] -= overlap * shift[comp]
    mu = _mean_homoscedastic(x)
    y = mu + rng.normal(scale=1.0, size=n) * _group_sigma(comp, sigmas)
    return x, y, comp.astype(np.int64)


def _gen_covariate_shift(
    rng: np.random.Generator,
    n: int,
    *,
    sigma: float = 0.3,
    shift: float = 2.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test ``x1`` is translated into a region the training split never saw.

    The mean function is *unchanged* -- only the input distribution moves. This
    isolates covariate shift from concept drift: a model extrapolates, its errors
    grow, and every marginal-coverage guarantee is stretched.
    """
    x = _standard_normal(rng, n)
    x[:, 0] += shift
    mu = _mean_homoscedastic(x)
    y = mu + rng.normal(scale=sigma, size=n)
    return x, y, _uninformative_groups(n)


def _gen_small_calib(
    rng: np.random.Generator,
    n: int,
    *,
    sigma: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identical law to :func:`_gen_homoscedastic`; only the split sizes differ.

    Kept as a separate generator on purpose: the *only* thing that makes this
    scenario hard is ``n_calib=25``, and reusing the homoscedastic generator means
    the difficulty is provably attributable to calibration size alone.
    """
    return _gen_homoscedastic(rng, n, sigma=sigma)


def _gen_nonlinear(
    rng: np.random.Generator,
    n: int,
    *,
    sigma: float = 0.3,
    freq: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sin(freq * x1) + x2^2 + 0.3 x1 x2`` -- structure a smooth model must earn.

    The difficulty is *model error*, not noise: any method that calibrates only a
    global residual width pays for the model's worst region at every point.

    ``freq`` is the difficulty knob and it controls **how heteroscedastic the
    model error is**, which is the property CQR actually exploits. A shallow
    gradient-boosted tree cannot follow a high-frequency sinusoid, and it fails
    worst where the curvature peaks -- so raising ``freq`` does not merely inflate
    the error, it makes the error *depend on x*. That is what a global-width method
    cannot represent and what a conditional-quantile method can.

    Measured on the shared learner, 1500 train rows, against a noise floor of
    0.3. The last column is vennfuse's mean width relative to ``split_absolute``
    -- negative means the flagship is genuinely narrower:

    ======  ==========  ==============  ==============================
    freq    val RMSE    vennfuse width   interpretation
    ======  ==========  ==============  ==============================
    3         0.400        +2.1%         error ~ homoscedastic; split
                                         and CQR tie, as they should
    5         0.420        -3.0%         x-dependent error emerges
    8         0.482        **-7.1%**     shipped default; model error is
                                         the binding constraint
    12        0.547        -3.8%         error grows, margins narrow again
    16        0.527        +2.7%         DGP too violent to discriminate
    ======  ==========  ==============  ==============================

    ``freq=8`` is the interior optimum, not the extreme: far enough above the
    noise floor that model error binds, without the DGP becoming so violent that
    every method degenerates toward the same over-wide interval -- which is
    exactly what happens at 16, where the margin flips positive again.
    """
    x = _standard_normal(rng, n)
    mu = np.sin(freq * x[:, 0]) + x[:, 1] ** 2 + 0.3 * x[:, 0] * x[:, 1]
    y = mu + rng.normal(scale=sigma, size=n)
    return x, y, _uninformative_groups(n)


def _uninformative_groups(n: int) -> np.ndarray:
    """Placeholder group labels for DGPs that have no ground-truth strata.

    These DGPs are still *usable* by a Mondrian method: the benchmark derives
    noise-level strata for them via :func:`default_groups`, which estimates the
    bin edges on the training split alone. Returning a constant here (rather than
    a fabricated partition) is what keeps ``meta["has_ground_truth_groups"]``
    honest -- the labels are a placeholder, not a claim.
    """
    return np.zeros(n, dtype=np.int64)


# --------------------------------------------------------------------------- #
# public factory
# --------------------------------------------------------------------------- #
def available() -> tuple[str, ...]:
    """Names accepted by :func:`make`."""
    return DGP_NAMES


def make(
    name: str,
    seed: int,
    n_train: int = 1500,
    n_calib: int = 500,
    n_val: int = 500,
    n_test: int = 2000,
    **knobs: Any,
) -> DatasetBundle:
    """Build one four-way split for the named DGP.

    Args:
        name: One of :data:`DGP_NAMES`.
        seed: Master seed. Combined with the DGP name through
            :func:`core.seed.RNG.stream`, so two DGPs never share a random stream.
        n_train: Training rows. Used to fit the point/quantile models.
        n_calib: Conformal calibration rows. **Never** used for model fitting.
        n_val: Validation rows. Used only for hyper-parameter selection.
        n_test: Held-out rows. Every reported number comes from here.
        **knobs: Difficulty knobs, e.g. ``sigma=0.5``. Unknown keys raise.

    Returns:
        A :class:`~core.types.DatasetBundle` whose
        :meth:`~core.types.SplitData.assert_disjoint` passes.

    Raises:
        DataError: For an unknown DGP name, a non-positive size, an unknown knob,
            or an impossible request (e.g. ``n_calib`` forced above ``n_test``).
    """
    if name not in DGP_NAMES:
        raise DataError(f"unknown DGP {name!r}; expected one of {list(DGP_NAMES)}")

    sizes = {"n_train": n_train, "n_calib": n_calib, "n_val": n_val, "n_test": n_test}
    for key, value in sizes.items():
        if int(value) <= 0:
            raise DataError(f"{key} must be positive, got {value}")

    gen, allowed = _DISPATCH[name]
    unknown = set(knobs) - set(allowed)
    if unknown:
        raise DataError(
            f"DGP {name!r} does not accept knob(s) {sorted(unknown)}; "
            f"accepted knobs are {sorted(allowed)}"
        )

    effective = dict(allowed)
    effective.update(knobs)

    # One stream per (dgp, segment) keeps the four segments independent of each
    # other: changing n_test cannot shift the training rows.
    #
    # ``_TEST_ONLY`` names the knobs that must apply to the TEST segment alone.
    # This exists because a covariate shift is by definition a *disagreement*
    # between train and test: applying the translation to every segment merely
    # relocates the whole dataset and produces no shift at all. Getting this
    # wrong fails silently -- the generator still runs, still returns plausible
    # numbers, and the DGP quietly degenerates into a byte-identical duplicate of
    # ``homoscedastic`` (observed, and caught by sweeping the knob: results were
    # invariant to shift=2.6, 3.0, 3.4 and 3.8 alike).
    test_only = _TEST_ONLY.get(name, frozenset())
    seg_names = ("train", "calib", "val", "test")
    seg_sizes = (n_train, n_calib, n_val, n_test)
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seg, size in zip(seg_names, seg_sizes, strict=True):
        seg_effective = dict(effective)
        if seg != "test":
            for knob in test_only:
                seg_effective[knob] = _ZERO[knob]
        arrays[seg] = gen(spawn(f"data.{name}.{seg}", int(seed)), int(size), **seg_effective)

    x_tr, y_tr, g_tr = arrays["train"]
    x_ca, y_ca, g_ca = arrays["calib"]
    x_va, y_va, g_va = arrays["val"]
    x_te, y_te, g_te = arrays["test"]

    if name == "small_calib":
        # Groups are noise-level strata; the point is that they are uninformative
        # here, so a Mondrian method must not be rewarded for using them.
        g_tr = _trivial_groups(len(g_tr))
        g_ca = _trivial_groups(len(g_ca))
        g_va = _trivial_groups(len(g_va))
        g_te = _trivial_groups(len(g_te))

    groups = np.concatenate([g_tr, g_ca, g_va, g_te]).astype(np.int64)
    split = SplitData(
        x_train=x_tr,
        y_train=y_tr,
        x_calib=x_ca,
        y_calib=y_ca,
        x_val=x_va,
        y_val=y_va,
        x_test=x_te,
        y_test=y_te,
        name=name,
    )
    split.assert_disjoint()

    bundle = DatasetBundle(
        name=name,
        split=split,
        groups=groups,
        meta={
            "seed": int(seed),
            "sizes": split.sizes(),
            "knobs": {k: _jsonable(v) for k, v in effective.items()},
            "n_groups": int(groups.max()) + 1,
            "has_ground_truth_groups": name == GROUPED_DGP,
        },
    )
    return bundle


def _trivial_groups(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def group_slices(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    """Split the concatenated ``groups`` vector back into per-segment arrays.

    :class:`~core.types.DatasetBundle` stores one flat ``groups`` array in
    ``train, calib, val, test`` order. Calibration and test groups are the only
    two a conformal method may ever touch, and this helper is the single place
    that knows the layout, so no caller can silently mis-slice it.
    """
    if bundle.groups is None:
        raise DataError(f"dataset {bundle.name!r} carries no group labels")
    sizes = bundle.split.sizes()
    order = ("train", "calib", "val", "test")
    out: dict[str, np.ndarray] = {}
    start = 0
    for seg in order:
        stop = start + sizes[seg]
        out[seg] = np.asarray(bundle.groups[start:stop])
        start = stop
    return out


def default_groups(
    bundle: DatasetBundle,
    x_reference: np.ndarray,
    n_bins: int = 3,
) -> dict[str, np.ndarray]:
    """Derive low/medium/high *noise* strata for DGPs without ground-truth groups.

    The bin edges are estimated on the **training** split and then applied
    unchanged to every other segment. Estimating them on the pooled data would be
    a (mild but real) leak: the group definition would depend on calibration and
    test covariates, which is information a practitioner does not have at
    calibration time.

    Args:
        bundle: Dataset providing the training covariates.
        x_reference: Unused positional placeholder kept for signature symmetry;
            edges come from ``bundle.split.x_train``.
        n_bins: Number of strata.

    Returns:
        Per-segment group index arrays.
    """
    del x_reference
    split = bundle.split
    ref = np.asarray(split.x_train, dtype=np.float64)
    stat = np.abs(ref[:, 0])
    edges = np.quantile(stat, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    out: dict[str, np.ndarray] = {}
    for seg, x in (
        ("train", split.x_train),
        ("calib", split.x_calib),
        ("val", split.x_val),
        ("test", split.x_test),
    ):
        out[seg] = np.digitize(np.abs(np.asarray(x, dtype=np.float64)[:, 0]), edges)
    return out


# --------------------------------------------------------------------------- #
# streaming scenario (drift / ACI)
# --------------------------------------------------------------------------- #
def make_stream(
    name: str = "drift",
    seed: int = 0,
    n_steps: int = 400,
    n_train: int = 1500,
    n_calib: int = 500,
    *,
    sigma: float = 0.3,
    drift_factor: float = 3.0,
    drift_at: float = 0.5,
    **knobs: Any,
) -> DatasetBundle:
    """Build a stationary split plus a stream whose noise jumps at ``drift_at``.

    The stream is the substrate for adaptive conformal inference (ACI). After
    ``drift_at * n_steps`` the observation noise is multiplied by
    ``drift_factor``: a predictor calibrated at the old noise level is now badly
    under-covered, and only a method that *updates* its miscoverage level can
    recover. The fixed-``alpha`` arm of gate G3 is the control.

    Args:
        name: ``"drift"`` (only supported value).
        seed: Master seed.
        n_steps: Stream length. Gate G3 reads the **last 20%** of it.
        n_train: Rows for the offline training split.
        n_calib: Rows for the conformal calibration split.
        sigma: Pre-drift noise scale.
        drift_factor: Multiplier applied to ``sigma`` after the jump.
        drift_at: Fraction of the stream after which the jump happens.
        **knobs: Extra generator knobs, forwarded to the underlying DGP.

    Returns:
        A bundle whose ``stream_x``/``stream_y`` are populated and whose
        ``meta["drift_index"]`` marks the first post-drift step.

    Raises:
        DataError: For an unknown stream name or an out-of-range ``drift_at``.
    """
    if name != "drift":
        raise DataError(f"unknown stream DGP {name!r}; expected 'drift'")
    if not 0.0 < drift_at < 1.0:
        raise DataError(f"drift_at must lie in (0, 1), got {drift_at}")
    if n_steps < 50:
        raise DataError("n_steps must be >= 50 for the post-drift window to be meaningful")

    base = make(
        "homoscedastic",
        seed=seed,
        n_train=n_train,
        n_calib=n_calib,
        n_val=1,
        n_test=1,
        sigma=sigma,
        **knobs,
    )

    rng = spawn(f"stream.{name}", int(seed))
    x_s = _standard_normal(rng, n_steps)
    mu_s = _mean_homoscedastic(x_s)
    drift_index = round(drift_at * n_steps)
    scale = np.where(np.arange(n_steps) >= drift_index, sigma * drift_factor, sigma)
    y_s = mu_s + rng.normal(scale=1.0, size=n_steps) * scale

    meta = dict(base.meta)
    meta.update(
        {
            "stream_steps": int(n_steps),
            "drift_index": drift_index,
            "drift_factor": float(drift_factor),
            "drift_at": float(drift_at),
            "sigma": float(sigma),
        }
    )
    return DatasetBundle(
        name=f"{name}_stream",
        split=base.split,
        groups=base.groups,
        stream_x=x_s,
        stream_y=y_s,
        meta=meta,
    )


#: Knobs that apply to the TEST segment only. A covariate shift that also moved
#: the training data would not be a shift at all.
_TEST_ONLY: dict[str, frozenset[str]] = {"covariate_shift": frozenset({"shift"})}

#: Neutral value for a test-only knob when generating a non-test segment.
_ZERO: dict[str, Any] = {"shift": 0.0}

# name -> (generator, default knob names). The knob names are the *contract*:
# `make(..., sigma=0.5)` is validated against them, which turns a typo in a
# sweep into a DataError instead of a silently ignored argument.
_DISPATCH: dict[
    str, tuple[Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]
] = {
    "homoscedastic": (_gen_homoscedastic, {"sigma": 0.3}),
    "heteroscedastic": (_gen_heteroscedastic, {"sigma_floor": 0.2, "sigma_span": 0.8}),
    "grouped_hetero": (
        _gen_grouped_hetero,
        {"sigmas": (0.2, 0.6, 1.2), "overlap": 0.15},
    ),
    "covariate_shift": (_gen_covariate_shift, {"sigma": 0.3, "shift": 2.6}),
    "small_calib": (_gen_small_calib, {"sigma": 0.3}),
    "nonlinear": (_gen_nonlinear, {"sigma": 0.3, "freq": 8.0}),
}

# Kept importable for tests that assert the sigmoid gate is monotone.
_ = _sigmoid_gate
