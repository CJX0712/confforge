"""Component ablations for the VennFuse flagship.

An ablation turns one component off and reports what changed. Its value is
informativeness, not decoration: a component that changes nothing should be
deleted, and a component that makes things *worse* is a finding that belongs in
the report rather than in a footnote.

Three arms, each isolating one switch:

============  ==========================  ====================================
arm           disabled                    expected effect
============  ==========================  ====================================
no_mondrian   group stratification        width rebounds where groups differ
no_aci        online alpha distillation   drift coverage falls back
no_cqr        CQR anchor -> split anchor  width grows under model error
============  ==========================  ====================================

Negative results are first-class output
---------------------------------------
An arm that does *not* behave as predicted is reported with its real numbers and
an interpretation. The three arms below were not all confirmed:

* ``no_aci`` cannot show its intended effect in the offline benchmark, because
  :meth:`~methods.methods.VennFuse.distill_aci` runs on stationary validation
  data where the fixed nominal alpha is already correct. The arm is therefore
  evaluated on the *drift* stream, where the mechanism is actually load-bearing.
* ``no_cqr`` helps on the homoscedastic DGP, where the split anchor is already
  near-optimal. That is not a failure of the ablation; it is the CQR anchor
  costing a little where there is nothing to adapt to.

Both are kept in the output with these explanations attached.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.config import ConfforgeConfig
from core.seed import set_all
from data.synthetic import default_groups, group_slices, make
from eval import metrics as M
from methods.estimators import ModelBundle
from methods.methods import build_method

__all__ = ["ABLATIONS", "ablation_report", "run_ablation", "run_all_ablations"]

#: ``arm name -> (switch kwargs, datasets it is expected to matter on)``.
ABLATIONS: dict[str, dict[str, Any]] = {
    "no_mondrian": {
        "kwargs": {"use_mondrian": False},
        "probe": ("grouped_hetero", "heteroscedastic"),
        "hypothesis": "disabling per-group calibration widens intervals on grouped data",
    },
    "no_aci": {
        "kwargs": {"use_aci": False},
        "probe": ("heteroscedastic", "nonlinear"),
        "hypothesis": (
            "with a stationary validation split the nominal alpha is already "
            "correct, so the arm is expected to be near-neutral here and is "
            "instead judged on the drift stream"
        ),
    },
    "no_cqr": {
        "kwargs": {"use_cqr": False},
        "probe": ("nonlinear", "homoscedastic"),
        "hypothesis": (
            "replacing the CQR anchor with a split anchor loses the ability to "
            "shrink, so width should grow wherever model error varies with x"
        ),
    },
}


def _groups(bundle: Any, segment: str, x_train: np.ndarray) -> np.ndarray:
    if bundle.name == "grouped_hetero" and bundle.groups is not None:
        return group_slices(bundle)[segment]
    return default_groups(bundle, x_train)[segment]


def run_ablation(
    arm: str,
    config: ConfforgeConfig,
    *,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    """Run the flagship with and without one component on a single cell.

    Both variants are fitted from the same seed and the same data, so the
    difference is attributable to the switch and to nothing else.

    Args:
        arm: Key of :data:`ABLATIONS`.
        config: Run configuration.
        dataset: DGP to evaluate on.
        seed: Seed for the shared learner.

    Returns:
        Dict with both arms' metrics and the deltas, or a ``status`` of
        ``"unknown_arm"`` for an unrecognised name.
    """
    if arm not in ABLATIONS:
        return {"arm": arm, "status": "unknown_arm", "known": sorted(ABLATIONS)}
    spec = ABLATIONS[arm]
    set_all(config.seed)

    n_calib = 25 if dataset == "small_calib" else config.n_calib
    bundle = make(
        dataset,
        seed=seed,
        n_train=config.n_train,
        n_calib=n_calib,
        n_val=config.n_val,
        n_test=config.n_test,
    )
    split = bundle.split
    g_calib = _groups(bundle, "calib", split.x_train)
    g_test = _groups(bundle, "test", split.x_train)

    # Both arms see identical data and seed, so they share one learner -- the
    # same rule the benchmark follows, and the reason an ablation isolates its
    # component instead of also measuring a different model fit.
    shared = ModelBundle.fit(split.x_train, split.y_train, seed, alpha=config.alpha)

    results: dict[str, Any] = {}
    # Every arm is "all switches on, except the one under test". Building the
    # ablated variant by starting from the full configuration and flipping a
    # single key guarantees the two arms differ in exactly one component --
    # constructing it from the arm spec alone is how an ablation quietly ends up
    # comparing two differently-configured methods.
    base_switches = {"use_cqr": True, "use_mondrian": True, "use_aci": True}
    variants = {
        "full": dict(base_switches),
        "ablated": {**base_switches, **spec["kwargs"]},
    }
    # One distillation target, computed once with the full method, so the ACI
    # operating point is identical across the two arms. Otherwise the "no_aci"
    # arm would differ in two ways at once and mean nothing.
    distill_ref: float | None = None

    for label, kwargs in variants.items():
        method = build_method("vennfuse", seed=seed, **kwargs)
        method.fit(split.x_train, split.y_train, alpha=config.alpha, bundle=shared)
        method.fit_calibrate(split.x_calib, split.y_calib, groups=g_calib)

        if distill_ref is None:
            distill_ref = float(
                method.distill_aci(
                    split.x_val,
                    split.y_val,
                    _groups(bundle, "val", split.x_train),
                    alpha_target=config.alpha,
                )["alpha"]
            )
        # The ablated arm keeps the same operating point; only the named
        # component is absent.
        adapted = distill_ref if kwargs.get("use_aci", True) else None

        interval = method.predict_interval(
            split.x_test, config.alpha, groups=g_test, adapted_alpha=adapted
        )
        scores = M.summarize(interval, split.y_test)
        gc = M.group_coverage(interval, split.y_test, g_test)
        results[label] = {
            "coverage": scores["coverage"],
            "mean_width": scores["mean_width"],
            "winkler": scores["winkler"],
            "cov_w": gc["worst_gap"],
            "adapted_alpha": adapted,
        }

    full, abl = results["full"], results["ablated"]
    width_delta = (abl["mean_width"] - full["mean_width"]) / full["mean_width"]
    return {
        "arm": arm,
        "status": "ok",
        "dataset": dataset,
        "seed": int(seed),
        "hypothesis": spec["hypothesis"],
        "full": full,
        "ablated": abl,
        "relative_width_change": width_delta,
        "coverage_change": abl["coverage"] - full["coverage"],
        "cov_w_change": abl["cov_w"] - full["cov_w"],
        "confirmed": bool(width_delta > 0.005),
    }


def run_all_ablations(
    config: ConfforgeConfig,
    *,
    datasets: Sequence[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run every arm on its probe datasets and return one record per cell."""
    use_seed = int(seed if seed is not None else config.seeds[0])
    out: list[dict[str, Any]] = []
    for arm, spec in ABLATIONS.items():
        probes = datasets or spec["probe"]
        for dataset in probes:
            try:
                out.append(run_ablation(arm, config, dataset=dataset, seed=use_seed))
            except Exception as exc:
                out.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "seed": use_seed,
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return out


def ablation_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise ablation records into per-arm verdicts.

    An arm is ``confirmed`` when ablating it widens intervals on at least one
    probe dataset, ``refuted`` when it does not, and ``inconclusive`` when the
    arm could not be evaluated at all. Refuted arms are kept in the summary --
    that is the point of recording them.
    """
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        by_arm.setdefault(str(rec.get("arm")), []).append(rec)

    summary: dict[str, Any] = {}
    for arm, recs in by_arm.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        errors = [r for r in recs if r.get("status") == "error"]
        deltas = [float(r["relative_width_change"]) for r in ok]
        summary[arm] = {
            "n_cells": len(recs),
            "n_ok": len(ok),
            "n_error": len(errors),
            "mean_relative_width_change": float(np.mean(deltas)) if deltas else float("nan"),
            "max_relative_width_change": float(max(deltas)) if deltas else float("nan"),
            "verdict": (
                "inconclusive" if not ok else ("confirmed" if max(deltas) > 0.005 else "refuted")
            ),
            "errors": [r.get("reason", "") for r in errors],
        }
    return summary
