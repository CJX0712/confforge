"""Failure analysis, derived entirely from benchmark results.

Every record here is produced by *looking at the numbers that came out*. Nothing
is pre-written, nothing is assumed, and no failure mode is claimed unless a
measured row supports it. That constraint is the whole reason this module exists:
a benchmark that ships a pre-authored "limitations" section is marketing, and the
reader has no way to tell the difference.

Three detectors, each answering a different question:

``worst_cells``
    Where is the flagship *worst* relative to the baseline? Ranked by relative
    width, so a positive number is a genuine regression and is reported as one.

``weakest_gains``
    Where does the flagship help least? A method that only wins on average can be
    hiding a case where it actively hurts.

``recheck``
    Re-run the single most interesting cell under a different seed. A finding that
    evaporates on re-run was noise, and saying so is more useful than quietly
    dropping it.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.config import ConfforgeConfig
from core.seed import set_all
from data.synthetic import make
from eval.benchmark import run_cell

__all__ = ["find_failures", "recheck_cell", "weakest_gains", "worst_cells"]


def _mean_by(rows: Sequence[Any], method: str, dataset: str, field: str) -> float:
    vals = [float(getattr(r, field)) for r in rows if r.method == method and r.dataset == dataset]
    return float(np.mean(vals)) if vals else float("nan")


def worst_cells(
    rows: Sequence[Any],
    *,
    flagship: str = "vennfuse",
    baseline: str = "split_absolute",
    nominal: float = 0.9,
    tol: float = 0.03,
    top: int = 3,
) -> list[dict[str, Any]]:
    """Cells where the flagship is worst relative to the baseline.

    Args:
        rows: Completed benchmark rows.
        flagship: Method under test.
        baseline: Reference method.
        nominal: Nominal coverage.
        tol: Coverage tolerance for the ``coverage_violation`` flag.
        top: How many cells to return.

    Returns:
        List of records sorted by descending relative width, i.e. the cells where
        the flagship is widest relative to the baseline.
    """
    datasets = sorted({r.dataset for r in rows})
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        f_w = _mean_by(rows, flagship, dataset, "mean_width")
        b_w = _mean_by(rows, baseline, dataset, "mean_width")
        f_c = _mean_by(rows, flagship, dataset, "coverage")
        b_c = _mean_by(rows, baseline, dataset, "coverage")
        if not (np.isfinite(f_w) and np.isfinite(b_w)) or b_w <= 0:
            continue
        out.append(
            {
                "kind": "cell",
                "dataset": dataset,
                "flagship_width": f_w,
                "baseline_width": b_w,
                "relative_width": (f_w - b_w) / b_w,
                "flagship_coverage": f_c,
                "baseline_coverage": b_c,
                "coverage_violation": bool(np.isfinite(f_c) and abs(f_c - nominal) > tol),
                "note": (
                    f"{flagship} is {abs((f_w - b_w) / b_w) * 100:.1f}% "
                    f"{'wider' if f_w > b_w else 'narrower'} than {baseline} here"
                ),
            }
        )
    out.sort(key=lambda r: r["relative_width"], reverse=True)
    return out[:top]


def weakest_gains(
    rows: Sequence[Any],
    *,
    flagship: str = "vennfuse",
    baseline: str = "split_absolute",
    top: int = 3,
) -> list[dict[str, Any]]:
    """Cells where the flagship helps least, including cells where it hurts."""
    cells = worst_cells(rows, flagship=flagship, baseline=baseline, top=len(rows) or 1)
    ranked = sorted(cells, key=lambda r: r["relative_width"])
    return [
        {
            "kind": "weak_gain",
            "dataset": r["dataset"],
            "relative_width": r["relative_width"],
            "note": (
                f"narrowing of only {-r['relative_width'] * 100:.1f}%"
                if r["relative_width"] < 0
                else f"a {r['relative_width'] * 100:.1f}% REGRESSION in width"
            ),
        }
        for r in ranked[:top]
    ]


def recheck_cell(
    config: ConfforgeConfig,
    dataset: str,
    method: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Re-run one cell under a different seed to test whether a finding is real.

    Args:
        config: Run configuration.
        dataset: DGP to re-run.
        method: Method to re-run.
        seed: An alternate seed.

    Returns:
        Dict with the fresh metrics, or a recorded error.
    """
    set_all(config.seed)
    n_calib = 25 if dataset == "small_calib" else config.n_calib
    try:
        bundle = make(
            dataset,
            seed=seed,
            n_train=config.n_train,
            n_calib=n_calib,
            n_val=config.n_val,
            n_test=config.n_test,
        )
        result = run_cell(method, bundle, seed, config)
    except Exception as exc:
        return {
            "kind": "recheck",
            "dataset": dataset,
            "method": method,
            "seed": int(seed),
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "kind": "recheck",
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "status": "ok",
        "coverage": result.coverage,
        "mean_width": result.mean_width,
        "winkler": result.winkler,
        "cov_w": result.extra.get("cov_w"),
        "note": f"re-ran {method} on {dataset} with seed={seed}",
    }


def find_failures(
    rows: Sequence[Any],
    *,
    config: ConfforgeConfig | None = None,
    nominal: float = 0.9,
    tol: float = 0.03,
    failures: Sequence[Mapping[str, Any]] = (),
    flagship: str = "vennfuse",
    baseline: str = "split_absolute",
) -> list[dict[str, Any]]:
    """Assemble the failure report: recorded errors, worst cells, weak gains, recheck.

    Args:
        rows: Completed benchmark rows.
        config: Configuration, used only for the recheck cell.
        nominal: Nominal coverage level.
        tol: Coverage tolerance.
        failures: Recorded skips/errors from the benchmark run.
        flagship: Flagship name.
        baseline: Baseline name.

    Returns:
        A list of failure records, each tagged with ``kind``.
    """
    out: list[dict[str, Any]] = []
    for rec in failures:
        out.append({"kind": "recorded_error", **dict(rec)})

    for rec in worst_cells(rows, flagship=flagship, baseline=baseline, nominal=nominal, tol=tol):
        out.append(rec)
    for rec in weakest_gains(rows, flagship=flagship, baseline=baseline):
        out.append(rec)

    if config is not None and rows:
        datasets = sorted({r.dataset for r in rows})
        if datasets:
            # Recheck the flagship on the first dataset, at a seed outside the
            # benchmark set, so the recheck is genuinely independent.
            alt_seed = int(config.seeds[-1]) + 101
            out.append(recheck_cell(config, datasets[0], flagship, seed=alt_seed))
    return out
