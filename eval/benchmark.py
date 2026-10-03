"""Benchmark driver: run every (method, dataset, seed) cell and aggregate.

One cell, one :class:`~core.types.MetricRow`. The driver's job is to be
*unbiased*, and that constrains it in three ways:

1. **Shared models.** The learner bundle is fitted once per ``(dataset, seed)``
   and handed to every method, so a width difference between two methods is
   attributable to the conformal mechanism alone.
2. **Honest skips.** A cell that cannot run -- missing optional backend, invalid
   configuration -- is recorded as ``skipped`` with a reason. It is never dropped
   silently and never back-filled with an estimate. A benchmark that quietly
   omits its failures is worse than one that reports them.
3. **Deterministic aggregation.** Rows are appended in a fixed nested order and
   no value is rounded before serialisation, so two runs at the same seed produce
   byte-identical JSON (gate G5).

Author: 晨星
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.config import ConfforgeConfig, load_config
from core.errors import BackendUnavailableError, ConfforgeError, DataError
from core.seed import set_all
from core.types import BenchmarkReport, ConformalResult, IntervalSet, MetricRow
from data.synthetic import DGP_NAMES, default_groups, group_slices, make
from eval import metrics as M
from methods.backends_mapie import backend_status
from methods.estimators import ModelBundle
from methods.methods import METHOD_ORDER, build_method

__all__ = [
    "BASELINE",
    "FLAGSHIP",
    "GROUPED_DATASET",
    "aggregate",
    "load_report",
    "run_benchmark",
    "run_cell",
    "save_report",
]

#: Baseline every gate is measured against.
BASELINE = "split_absolute"
#: Flagship under test.
FLAGSHIP = "vennfuse"
#: Dataset carrying ground-truth subgroup labels (gate G2).
GROUPED_DATASET = "grouped_hetero"
#: Methods that consume group labels.
_GROUPED_METHODS = frozenset({"mondrian", "vennfuse"})


def _groups_for(bundle: Any, segment: str, x_train: np.ndarray) -> np.ndarray:
    """Group labels for one segment, from ground truth or train-derived strata.

    Ground-truth labels are used only for ``grouped_hetero``. Every other DGP gets
    noise-level strata whose bin edges are estimated on the **training** split
    (:func:`data.synthetic.default_groups`), so the partition never depends on
    calibration or test covariates.
    """
    if bundle.name == "grouped_hetero" and bundle.groups is not None:
        return group_slices(bundle)[segment]
    derived = default_groups(bundle, x_train)
    return derived[segment]


def run_cell(
    method_name: str,
    bundle: Any,
    seed: int,
    config: ConfforgeConfig,
    shared: Any = None,
    sink: dict[str, Any] | None = None,
    key: tuple[str, str, int] | None = None,
) -> ConformalResult:
    """Run one (method, dataset, seed) cell and score it on the test split.

    Args:
        method_name: One of :data:`methods.methods.METHOD_ORDER`.
        bundle: Dataset to evaluate on.
        seed: Seed for the shared learner.
        config: Run configuration.

    Returns:
        A populated :class:`~core.types.ConformalResult`.

    Raises:
        ConfforgeError: Propagated from the method layer. The caller
            (:func:`run_benchmark`) converts these into recorded skips.
    """
    split = bundle.split
    method = build_method(method_name, seed=seed)
    method.fit(split.x_train, split.y_train, alpha=config.alpha, bundle=shared)

    needs_groups = method_name in _GROUPED_METHODS
    g_calib = _groups_for(bundle, "calib", split.x_train) if needs_groups else None
    g_test = _groups_for(bundle, "test", split.x_train) if needs_groups else None

    if method_name == "vennfuse":
        method.fit_calibrate(split.x_calib, split.y_calib, groups=g_calib)
        # VennFuse distils its online ACI component on the *validation* split and
        # predicts at the adapted level. This is what makes `use_aci` a real
        # switch in the benchmark rather than a decorative flag: without it the
        # third component is never exercised and `no_aci` would be a no-op
        # ablation. Validation is spent on exactly this and never on calibration
        # or testing.
        adapted = None
        if getattr(method, "use_aci", False):
            adapted = method.distill_aci(
                split.x_val,
                split.y_val,
                _groups_for(bundle, "val", split.x_train),
                alpha_target=config.alpha,
            )["alpha"]
        interval = method.predict_interval(
            split.x_test, config.alpha, groups=g_test, adapted_alpha=adapted
        )
    elif needs_groups:
        method.fit_calibrate(split.x_calib, split.y_calib, groups=g_calib)
        interval = method.predict_interval(split.x_test, config.alpha, groups=g_test)
    else:
        method.fit_calibrate(split.x_calib, split.y_calib)
        interval = method.predict_interval(split.x_test, config.alpha)

    if not isinstance(interval, IntervalSet):  # pragma: no cover - contract guard
        raise ConfforgeError(f"{method_name} returned {type(interval)}, expected IntervalSet")

    if sink is not None and key is not None:
        sink["intervals"][key] = interval
        sink["targets"][key] = split.y_test
        sink["n_calibs"][key] = int(split.sizes()["calib"])

    scores = M.summarize(interval, split.y_test)
    extra: dict[str, float] = {k: float(v) for k, v in method.extras().items()}

    # Conditional coverage on the ground-truth groups, when they exist. This is
    # the measurement gate G2 is defined on.
    if bundle.name == GROUPED_DATASET and bundle.groups is not None:
        gc = M.group_coverage(interval, split.y_test, group_slices(bundle)["test"])
        extra["cov_w"] = float(gc["worst_gap"])
        extra["worst_group"] = float(gc["worst_group"])
        extra["worst_group_coverage"] = float(gc["per_group"].get(gc["worst_group"], 0.0))
    extra["n_test"] = float(len(split.y_test))
    # Stability is only meaningful on an ordered sample; sorting by x1 makes the
    # number comparable across methods instead of measuring sampling noise.
    try:
        order = np.argsort(split.x_test[:, 0], kind="stable")
        extra["stability"] = float(M.stability(interval.slice(order)))
    except ConfforgeError:  # pragma: no cover - single-sample test split
        extra["stability"] = float("nan")

    return ConformalResult(
        method=method_name,
        dataset=bundle.name,
        seed=int(seed),
        alpha=float(config.alpha),
        coverage=float(scores["coverage"]),
        mean_width=float(scores["mean_width"]),
        winkler=float(scores["winkler"]),
        extra=extra,
    )


def run_benchmark(
    config: ConfforgeConfig | None = None,
    methods: Sequence[str] = METHOD_ORDER,
    datasets: Sequence[str] = DGP_NAMES,
    progress: bool = False,
    collect: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Run the full grid and return a populated report.

    Args:
        config: Run configuration; built from the environment if omitted.
        methods: Methods to include, in report order.
        datasets: DGPs to include, in report order.
        progress: Print a line per completed cell.
        collect: Optional mutable dict. When given, the emitted
            :class:`~core.types.IntervalSet` and the realised ``y_test`` are
            stored per ``(method, dataset, seed)`` under the keys ``intervals``
            and ``targets`` (and the calibration size under ``n_calibs``). This is
            what lets :func:`eval.invariants.check_all` run its *structural*
            checks; without it those checks are skipped for every row and an
            empty violation list means nothing. The arrays are deliberately kept
            out of the JSON report -- 144 cells x 2000 points would add megabytes
            of floats to an artefact whose purpose is to be diffed.

    Returns:
        A :class:`~core.types.BenchmarkReport` whose ``failures`` list records
        every skipped cell with its reason.
    """
    cfg = config or load_config()
    set_all(cfg.seed)

    if collect is not None:
        collect.setdefault("intervals", {})
        collect.setdefault("targets", {})
        collect.setdefault("n_calibs", {})

    report = BenchmarkReport(
        config={
            "seed": cfg.seed,
            "alpha": cfg.alpha,
            "n_train": cfg.n_train,
            "n_calib": cfg.n_calib,
            "n_val": cfg.n_val,
            "n_test": cfg.n_test,
            "seeds": list(cfg.seeds),
            "methods": list(methods),
            "datasets": list(datasets),
            "mapie": backend_status(),
        }
    )

    for dgp in datasets:
        for seed in cfg.seeds:
            # small_calib is defined by its calibration size, so honour it here
            # rather than forcing every DGP onto the default n_calib.
            n_calib = 25 if dgp == "small_calib" else cfg.n_calib
            try:
                bundle = make(
                    dgp,
                    seed=seed,
                    n_train=cfg.n_train,
                    n_calib=n_calib,
                    n_val=cfg.n_val,
                    n_test=cfg.n_test,
                )
            except DataError as exc:
                report.failures.append(
                    {
                        "kind": "dataset",
                        "dataset": dgp,
                        "seed": int(seed),
                        "method": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            # One learner per (dataset, seed), shared by all eight methods.
            # Fitting per method would be both 8x slower and a weaker guarantee.
            shared = ModelBundle.fit(
                bundle.split.x_train, bundle.split.y_train, seed, alpha=cfg.alpha
            )

            for name in methods:
                started = time.perf_counter()
                try:
                    result = run_cell(
                        name,
                        bundle,
                        seed,
                        cfg,
                        shared=shared,
                        sink=collect,
                        key=(name, dgp, int(seed)),
                    )
                except BackendUnavailableError as exc:
                    report.failures.append(
                        {
                            "kind": "backend_unavailable",
                            "dataset": dgp,
                            "seed": int(seed),
                            "method": name,
                            "reason": str(exc),
                        }
                    )
                    continue
                except (ConfforgeError, ValueError) as exc:
                    report.failures.append(
                        {
                            "kind": "method_error",
                            "dataset": dgp,
                            "seed": int(seed),
                            "method": name,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                result.extra["elapsed_sec"] = float(time.perf_counter() - started)
                report.add(result)
                if progress:
                    print(
                        f"  [{dgp:17s} seed={seed:3d}] {name:16s} "
                        f"cov={result.coverage:.4f} width={result.mean_width:.4f} "
                        f"winkler={result.winkler:.3f} ({result.extra['elapsed_sec']:.2f}s)"
                    )

    report.aggregates = aggregate(report.rows)  # type: ignore[attr-defined]
    return report


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def aggregate(rows: Sequence[MetricRow]) -> dict[str, Any]:
    """Collapse rows into per-(method, dataset) mean/std plus a method-level roll-up.

    Standard deviations are **population** std (``ddof=0``) over the three seeds.
    That is the honest choice for a fixed seed set: it describes the seeds we
    actually ran. A sample std over three points would inflate the apparent
    variability and is not what "across these seeds" means.

    Args:
        rows: Completed benchmark rows.

    Returns:
        Nested dict ``by_method_dataset`` and ``by_method``, each entry carrying
        ``coverage_mean``/``coverage_std``, ``width_mean``/``width_std``,
        ``winkler_mean`` and the per-seed count.
    """
    by_md: dict[str, Any] = {}
    for row in rows:
        key = f"{row.method}|{row.dataset}"
        entry = by_md.setdefault(
            key,
            {
                "method": row.method,
                "dataset": row.dataset,
                "coverage": [],
                "width": [],
                "winkler": [],
                "cov_w": [],
                "n": 0,
            },
        )
        entry["coverage"].append(row.coverage)
        entry["width"].append(row.mean_width)
        entry["winkler"].append(row.winkler)
        if "cov_w" in row.extra:
            entry["cov_w"].append(float(row.extra["cov_w"]))
        entry["n"] += 1

    out_md: dict[str, Any] = {}
    for entry in by_md.values():
        c_m, c_s = _mean_std(entry["coverage"])
        w_m, w_s = _mean_std(entry["width"])
        k_m, k_s = _mean_std(entry["winkler"])
        cv_m, cv_s = _mean_std(entry["cov_w"]) if entry["cov_w"] else (float("nan"),) * 2
        out_md[f"{entry['method']}|{entry['dataset']}"] = {
            "method": entry["method"],
            "dataset": entry["dataset"],
            "coverage_mean": c_m,
            "coverage_std": c_s,
            "width_mean": w_m,
            "width_std": w_s,
            "winkler_mean": k_m,
            "winkler_std": k_s,
            "cov_w_mean": cv_m,
            "cov_w_std": cv_s,
            "n_seeds": entry["n"],
        }

    by_method: dict[str, Any] = {}
    for method_name in sorted({r.method for r in rows}):
        subset = [r for r in rows if r.method == method_name]
        c_m, c_s = _mean_std(r.coverage for r in subset)
        w_m, w_s = _mean_std(r.mean_width for r in subset)
        k_m, _ = _mean_std(r.winkler for r in subset)
        by_method[method_name] = {
            "coverage_mean": c_m,
            "coverage_std": c_s,
            "width_mean": w_m,
            "width_std": w_s,
            "winkler_mean": k_m,
            "n_cells": len(subset),
        }

    return {"by_method_dataset": out_md, "by_method": by_method}


def save_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write the report to ``path`` as UTF-8 JSON.

    ``sort_keys=True`` and a fixed float repr make the output byte-comparable
    across runs, which is what gate G5 checks. ``elapsed_sec`` is excluded from
    the determinism comparison by the gate itself, since wall-clock time cannot
    be reproducible by construction.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["aggregates"] = getattr(report, "aggregates", {})
    payload["backend"] = backend_status()
    # Evidence that the structural invariants were actually EXERCISED.
    # `BenchmarkReport.to_dict()` (core/types.py) emits a fixed key set, so
    # setting these as attributes on the report is not enough -- they have to be
    # added here or they never reach the artefact, and `invariants: []` would once
    # again be indistinguishable from "nothing was checked".
    payload["invariant_checks_performed"] = int(
        getattr(report, "invariant_checks_performed", 0) or 0
    )
    payload["invariant_rows_checked"] = int(getattr(report, "invariant_rows_checked", 0) or 0)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_default),
        encoding="utf-8",
    )
    return target


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a report written by :func:`save_report`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
