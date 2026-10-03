"""Command-line entry point.

Subcommands
-----------
``run``       full pipeline, writes ``benchmark.json`` plus an analysis file
``quick``     one dataset, one seed -- a smoke test, not a benchmark
``gates``     re-print the gate thresholds and semantics
``methods``   list the methods and their mechanism
``datasets``  list the DGPs and their difficulty knobs
``selftest``  verify the optional MAPIE backend's API contract

Exit codes
----------
``0`` success. ``1`` a gate failed. ``2`` a usage or configuration error.

A non-zero exit on a failed gate is the point: it makes the gate usable from CI
without anyone having to read the JSON.

Author: 晨星
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.config import load_config
from core.errors import ConfforgeError
from core.seed import set_all
from data.synthetic import DGP_NAMES
from eval.benchmark import GROUPED_DATASET, save_report
from eval.gates import GATE_SPECS
from methods.backends_mapie import api_selftest, backend_status
from methods.methods import METHOD_ORDER
from pipeline.pipeline import format_summary, run_pipeline, save

__all__ = ["build_parser", "main"]

_DEFAULT_OUT = "results/benchmark.json"


def _use_utf8() -> None:
    """Make stdout UTF-8 so box-drawing characters survive a Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # exotic streams
                reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="confforge",
        description="Distribution-free predictive inference: conformal prediction intervals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full benchmark and evaluate every gate")
    run_p.add_argument("--out", default=_DEFAULT_OUT, help=f"output path (default {_DEFAULT_OUT})")
    run_p.add_argument("--alpha", type=float, default=None, help="miscoverage level")
    run_p.add_argument("--seeds", default=None, help="comma-separated seeds (>= 3)")
    run_p.add_argument("--datasets", default=None, help="comma-separated DGP subset")
    run_p.add_argument("--methods", default=None, help="comma-separated method subset")
    run_p.add_argument("--n-train", type=int, default=None)
    run_p.add_argument("--n-calib", type=int, default=None)
    run_p.add_argument("--n-test", type=int, default=None)
    run_p.add_argument(
        "--drift-factor", type=float, default=1.2, help="post-drift noise multiplier (default 1.2)"
    )
    run_p.add_argument("--no-ablations", action="store_true", help="skip the ablation arms")
    run_p.add_argument(
        "--no-determinism",
        action="store_true",
        help="skip the repeat run used by gate G5 (halves the runtime)",
    )
    run_p.add_argument("--quiet", action="store_true", help="suppress per-cell progress")
    run_p.add_argument("--json", action="store_true", help="print the summary as JSON")

    quick_p = sub.add_parser("quick", help="single dataset, single seed smoke test")
    quick_p.add_argument("--dataset", default="heteroscedastic", choices=list(DGP_NAMES))
    quick_p.add_argument("--out", default="results/quick.json")
    quick_p.add_argument("--seed", type=int, default=11)

    sub.add_parser("gates", help="describe the release gates")
    sub.add_parser("methods", help="list the conformal methods")
    sub.add_parser("datasets", help="list the data-generating processes")
    sub.add_parser("selftest", help="check the optional MAPIE backend contract")
    return parser


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_run(args: argparse.Namespace) -> int:
    overrides: dict = {}
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.seeds:
        overrides["seeds"] = tuple(int(s) for s in _split(args.seeds) or ())
    for key, value in (
        ("n_train", args.n_train),
        ("n_calib", args.n_calib),
        ("n_test", args.n_test),
    ):
        if value is not None:
            overrides[key] = value
    try:
        cfg = load_config(**overrides)
    except ConfforgeError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    set_all(cfg.seed)
    result = run_pipeline(
        cfg,
        methods=_split(args.methods) or METHOD_ORDER,
        datasets=_split(args.datasets) or DGP_NAMES,
        verify_determinism=not args.no_determinism,
        run_ablations=not args.no_ablations,
        drift_factor=args.drift_factor,
        progress=not args.quiet,
    )
    out = Path(args.out)
    report_path = save(result, out)
    save_report(result.report, report_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print()
        print(format_summary(result))
        print()
        print(f"report written to {report_path}")
    return 0 if result.all_gates_passed() else 1


def _cmd_quick(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(n_train=400, n_calib=200, n_val=200, n_test=600, seeds=(11, 23, 37))
    except ConfforgeError as exc:  # pragma: no cover - defensive
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    set_all(cfg.seed)
    result = run_pipeline(
        cfg,
        datasets=[args.dataset],
        verify_determinism=False,
        run_ablations=False,
        progress=False,
    )
    path = save(result, args.out)
    print(format_summary(result))
    print(f"\nreport written to {path}")
    return 0 if result.all_gates_passed() else 1


def _cmd_gates(_: argparse.Namespace) -> int:
    print("Release gates")
    print("=" * 72)
    for spec in GATE_SPECS:
        print(f"  {spec['gate']} [{spec['level']}]: {spec['claim']}")
        print(f"      threshold: {spec['threshold']}")
    print()
    print(f"Conditional-coverage dataset: {GROUPED_DATASET}")
    print("Datasets whose coverage misses nominal by more than 0.03 are recorded")
    print("as skipped and excluded from aggregates; an aggregate with fewer than")
    print("3 participating datasets FAILS rather than reporting a partial result.")
    return 0


def _cmd_methods(_: argparse.Namespace) -> int:
    print("Conformal methods")
    print("=" * 72)
    rows = [
        ("split_absolute", "pooled |y-yhat| quantile -> symmetric window", "baseline"),
        ("split_normalized", "|y-yhat| / sigma_hat(x), kNN local scale", "baseline"),
        ("cqr", "Romano 2019 conformity score on a quantile regressor", "baseline"),
        ("mondrian", "per-group calibration, thin cells shrunk to pooled", "baseline"),
        ("vennfuse", "CQR anchor + Mondrian shrink + distilled ACI", "flagship"),
        ("mapie_split", "MAPIE SplitConformalRegressor", "backend"),
        ("mapie_cqr", "MAPIE ConformalizedQuantileRegressor", "backend"),
        ("aci", "online adaptive conformal inference over a stream", "baseline"),
    ]
    width = max(len(r[0]) for r in rows)
    for name, desc, family in rows:
        print(f"  {name:<{width}}  [{family:8s}] {desc}")
    return 0


def _cmd_datasets(_: argparse.Namespace) -> int:
    print("Data-generating processes")
    print("=" * 72)
    for name in DGP_NAMES:
        print(f"  {name}")
    print()
    print("Each accepts difficulty knobs, e.g.:")
    print("  confforge run --datasets homoscedastic")
    print("  data.synthetic.make('nonlinear', seed=11, freq=12.0)")
    return 0


def _cmd_selftest(_: argparse.Namespace) -> int:
    status = backend_status()
    print("MAPIE backend")
    print("=" * 72)
    print(f"  available: {status['available']}")
    print(f"  version:   {status['version'] or '-'}")
    if not status["available"]:
        print(f"  reason:    {status['reason']}")
        print("  Tier-0 backends are optional; the NumPy methods are unaffected.")
        return 0
    result = api_selftest()
    print(f"  api_selftest ok:    {result.get('ok')}")
    print(f"  intervals shape:    {result.get('intervals_shape')}")
    if not result.get("ok"):
        print(f"  reason:             {result.get('reason')}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    _use_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "quick": _cmd_quick,
        "gates": _cmd_gates,
        "methods": _cmd_methods,
        "datasets": _cmd_datasets,
        "selftest": _cmd_selftest,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except ConfforgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
