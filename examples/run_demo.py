"""End-to-end demonstration.

Run with::

    python examples/run_demo.py

Prints the full analysis and writes ``results/benchmark.json``. Exits non-zero if
a P0 gate fails, so it doubles as a CI smoke test.

The demo is the fastest honest answer to "does this actually work?" -- it runs the
real pipeline over the real grid rather than a curated subset, and reports the
real numbers including the ones that are unflattering.

Author: 晨星
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

# Allow `python examples/run_demo.py` from a clean checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from core.seed import set_all
from data.synthetic import DGP_NAMES
from methods.methods import METHOD_ORDER
from pipeline.pipeline import format_summary, run_pipeline, save


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Confforge benchmark end to end.")
    parser.add_argument("--datasets", default=None, help="comma-separated DGP subset")
    parser.add_argument("--methods", default=None, help="comma-separated method subset")
    parser.add_argument("--out", default="results/benchmark.json", help="output report path")
    parser.add_argument(
        "--no-determinism",
        action="store_true",
        help="skip the repeat run behind gate G5 (halves the runtime)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke-test mode: assert the pipeline RAN (artefacts written, rows > 0) "
            "and exit 0 regardless of gate verdicts. Use for a reduced CI grid, "
            "where MIN_PARTICIPANTS makes G1 fail by construction and a gate-based "
            "exit code would be red no matter how correct the code is."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print a summary.

    Returns:
        ``0`` when every gate passes, ``1`` otherwise.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")

    args = _parse(argv)
    config = load_config()
    set_all(config.seed)

    def _split(value: str | None, fallback: tuple[str, ...]) -> list[str]:
        if not value:
            return list(fallback)
        return [v.strip() for v in value.split(",") if v.strip()]

    datasets = _split(args.datasets, DGP_NAMES)
    methods = _split(args.methods, METHOD_ORDER)

    print("Confforge demo")
    print("=" * 72)
    print(
        f"alpha={config.alpha}  n_train={config.n_train}  n_calib={config.n_calib}  "
        f"n_val={config.n_val}  n_test={config.n_test}  seeds={list(config.seeds)}"
    )
    print(f"datasets: {', '.join(datasets)}")
    print()
    started = time.perf_counter()
    result = run_pipeline(
        config,
        methods=methods,
        datasets=datasets,
        verify_determinism=not (args.no_determinism or args.smoke),
        progress=False,
    )
    elapsed = time.perf_counter() - started

    out = save(result, args.out)
    print(format_summary(result))
    print()
    print(f"elapsed: {elapsed:.1f}s")
    print(f"report:  {out}")
    print(f"analysis: {out.with_name(out.stem + '_analysis.json')}")

    gates = result.gate_results
    print()
    print(
        "gate verdicts: "
        + ", ".join(
            f"{name}={'PASS' if rec.get('passed') else 'FAIL'}"
            for name, rec in sorted(gates.items())
            if isinstance(rec, dict) and "passed" in rec
        )
    )

    if args.smoke:
        # Contract: the pipeline ran and produced well-formed artefacts. Gate
        # verdicts are printed for information and deliberately NOT enforced.
        problems: list[str] = []
        if not out.exists():
            problems.append(f"report not written: {out}")
        if not len(result.report.rows):
            problems.append("benchmark produced zero rows")
        if getattr(result.report, "invariant_checks_performed", 0) <= 0:
            problems.append("no structural invariant was actually checked")
        if problems:
            print()
            print("SMOKE FAILED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print()
        print(
            f"SMOKE OK: {len(result.report.rows)} rows, "
            f"{getattr(result.report, 'invariant_checks_performed', 0)} structural "
            "checks performed. Gate verdicts above are informational only."
        )
        return 0

    return 0 if result.all_gates_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
