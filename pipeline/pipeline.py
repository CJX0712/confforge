"""End-to-end orchestration: benchmark -> drift -> ablation -> gates -> report.

This is the only module that knows the *order* of the analysis. Keeping the
sequencing in one place is what makes the run reproducible and reviewable: a
reader can answer "what exactly does a run compute?" by reading a hundred lines
here rather than by tracing control flow through six modules.

Pipeline stages
---------------
1. **Benchmark** -- the full ``method x dataset x seed`` grid.
2. **Invariants** -- correctness checks over every produced row.
3. **Drift** -- the ACI vs fixed-alpha experiment behind G3.
4. **Ablations** -- one arm per VennFuse component.
5. **Gates** -- G1/G2/G3/G5 verdicts.
6. **Failures** -- worst cells and weak gains, derived from the results.
7. **Repeat run** -- the whole benchmark a second time, for G5.

Stage 7 doubles the cost, so it is optional (``verify_determinism``) and enabled
by default in the demo, where proving reproducibility is the point.

Author: 晨星
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.config import ConfforgeConfig, load_config
from core.seed import set_all
from core.types import BenchmarkReport
from data.synthetic import DGP_NAMES
from eval import ablation, drift, failures, gates, invariants
from eval.benchmark import aggregate, run_benchmark, save_report
from methods.methods import METHOD_ORDER

__all__ = ["PipelineResult", "environment_info", "format_summary", "run_pipeline"]


def environment_info() -> dict[str, Any]:
    """Record the environment the run happened in.

    Reproducibility claims are only checkable against an environment, and a
    benchmark that does not record its own Python and NumPy versions cannot be
    re-run meaningfully on a different machine.
    """
    import numpy as np
    import sklearn

    from methods.backends_mapie import backend_status

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "mapie": backend_status(),
    }


class PipelineResult:
    """Everything a pipeline run produced, plus its own timing."""

    def __init__(self, report: BenchmarkReport, elapsed: float) -> None:
        self.report = report
        self.elapsed = float(elapsed)
        self.gate_results: dict[str, Any] = {}
        self.drift: dict[str, Any] | None = None
        self.ablation: list[dict[str, Any]] = []
        self.ablation_summary: dict[str, Any] = {}
        self.invariant_violations: list[dict[str, Any]] = []
        #: Structured invariant outcome: violations, cells_checked, row_checks.
        #: Exists so "clean" and "unchecked" can never be confused.
        self.invariant_checks: dict[str, Any] = {}
        self.repeat_aggregates: dict[str, Any] | None = None
        self.config: ConfforgeConfig | None = None
        self.environment: dict[str, Any] = environment_info()

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_sec": self.elapsed,
            "environment": self.environment,
            "config": self.report.config,
            "gates": self.gate_results,
            "drift": _strip_arrays(self.drift) if self.drift else None,
            "ablations": self.ablation,
            "ablation_summary": self.ablation_summary,
            "invariant_violations": self.invariant_violations,
            "invariant_checks": self.invariant_checks,
            "invariant_checks_performed": getattr(self.report, "invariant_checks_performed", 0),
            "aggregates": getattr(self.report, "aggregates", {}),
        }

    def all_gates_passed(self) -> bool:
        return bool(self.gate_results.get("all_passed", False))


def _strip_arrays(obj: Any) -> Any:
    """Drop the per-seed alpha trajectories so the JSON stays small and diffable."""
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if k != "alpha_trajectory"}
    if isinstance(obj, list):
        return [_strip_arrays(v) for v in obj]
    return obj


def run_pipeline(
    config: ConfforgeConfig | None = None,
    *,
    methods: Sequence[str] = METHOD_ORDER,
    datasets: Sequence[str] = DGP_NAMES,
    verify_determinism: bool = True,
    run_ablations: bool = True,
    drift_factor: float = 1.2,
    progress: bool = False,
) -> PipelineResult:
    """Run every analysis stage and assemble the result.

    Args:
        config: Run configuration; built from the environment if omitted.
        methods: Methods to benchmark.
        datasets: DGPs to benchmark.
        verify_determinism: Re-run the benchmark for G5. Doubles runtime.
        run_ablations: Run the three VennFuse ablations.
        drift_factor: Noise multiplier for the drift scenario. See the note in
            :mod:`eval.drift` on why the shipped gate value is 1.2 rather than
            the 3.0 DGP default.
        progress: Print per-cell progress.

    Returns:
        A :class:`PipelineResult` with the report, gates and every analysis.
    """
    cfg = config or load_config()
    set_all(cfg.seed)
    started = time.perf_counter()

    # Collect the emitted intervals so the structural invariants actually run.
    # Without this the checker is blind and "0 violations" is indistinguishable
    # from "0 checks" -- see eval/invariants.py::check_all.
    collect: dict[str, Any] = {}
    report = run_benchmark(
        cfg, methods=methods, datasets=datasets, progress=progress, collect=collect
    )
    aggregates = aggregate(report.rows)

    result = PipelineResult(report, 0.0)
    result.config = cfg

    invariant_report = invariants.check_all(
        report.rows,
        intervals=collect.get("intervals"),
        targets=collect.get("targets"),
        n_calibs=collect.get("n_calibs"),
    )
    result.invariant_violations = invariant_report["violations"]
    result.invariant_checks = invariant_report
    # Written into the artefact so a reader can tell "checked and clean" from
    # "never checked". Both used to serialise as `invariants: []`.
    report.invariant_checks_performed = int(invariant_report["cells_checked"])
    report.invariant_rows_checked = int(invariant_report["row_checks"])

    if run_ablations:
        result.ablation = ablation.run_all_ablations(cfg)
        result.ablation_summary = ablation.ablation_report(result.ablation)
        report.ablations = list(result.ablation)

    result.drift = drift.run_drift_benchmark(cfg, drift_factor=drift_factor)
    report.invariants = list(result.invariant_violations)

    repeat_payload = None
    baseline_payload = None
    repeat_aggregates = None
    if verify_determinism:
        repeat = run_benchmark(cfg, methods=methods, datasets=datasets, progress=False)
        repeat_aggregates = aggregate(repeat.rows)
        result.repeat_aggregates = repeat_aggregates
        # Compare the two runs through their own serialisable payloads so the
        # gate sees exactly what a reader of the two JSON files would see.
        baseline_payload = _payload(report, aggregates)
        repeat_payload = _payload(repeat, repeat_aggregates)

    result.gate_results = gates.evaluate_gates(
        aggregates,
        drift_result=result.drift,
        baseline_payload=baseline_payload,
        repeat_payload=repeat_payload,
    )
    report.gates = result.gate_results

    report.failures = failures.find_failures(report.rows, config=cfg, failures=report.failures)
    result.elapsed = time.perf_counter() - started
    report.config["elapsed_sec"] = result.elapsed
    report.config["environment"] = result.environment
    return result


def _payload(report: BenchmarkReport, aggregates: dict[str, Any]) -> dict[str, Any]:
    """Serialisable view of a run, matching what :func:`save_report` writes."""
    payload = report.to_dict()
    payload["aggregates"] = aggregates
    return payload


def format_summary(result: PipelineResult) -> str:
    """Render a compact human-readable summary of a pipeline run.

    Uses ASCII status markers rather than emoji so the output survives a Windows
    console whose code page cannot encode them -- a summary that crashes on
    ``cp1252`` is not a summary.
    """
    lines: list[str] = []
    env = result.environment
    lines.append("Confforge pipeline summary")
    lines.append("=" * 72)
    lines.append(
        f"python {env['python']} | numpy {env['numpy']} | "
        f"scikit-learn {env['scikit_learn']} | "
        f"mapie {env['mapie'].get('version') or 'absent'}"
    )
    cfg = result.config
    if cfg is not None:
        lines.append(
            f"config: alpha={cfg.alpha} n_train={cfg.n_train} n_calib={cfg.n_calib} "
            f"n_val={cfg.n_val} n_test={cfg.n_test} seeds={list(cfg.seeds)}"
        )
    lines.append(f"elapsed: {result.elapsed:.1f}s")
    lines.append("")

    lines.append("Gates")
    lines.append("-" * 72)
    for name in ("G1", "G2", "G3", "G5"):
        rec = result.gate_results.get(name)
        if not rec:
            continue
        mark = "PASS" if rec.get("passed") else "FAIL"
        lines.append(f"  [{mark}] {name} ({rec.get('level', '')}): {rec.get('reason', '')}")
    blocking = result.gate_results.get("blocking", [])
    if blocking:
        lines.append(f"  blocking gates: {', '.join(blocking)}")
    lines.append("")

    agg = getattr(result.report, "aggregates", {}).get("by_method_dataset", {})
    if agg:
        lines.append("Mean width / coverage by (method, dataset), 3 seeds")
        lines.append("-" * 72)
        datasets = sorted({e["dataset"] for e in agg.values()})
        methods = sorted({e["method"] for e in agg.values()})
        header = f"{'dataset':18s}" + "".join(f"{m[:14]:>16s}" for m in methods)
        lines.append(header)
        for dataset in datasets:
            row = f"{dataset:18s}"
            for method in methods:
                entry = agg.get(f"{method}|{dataset}")
                if entry is None:
                    row += f"{'-':>16s}"
                else:
                    row += f"{entry['width_mean']:8.3f}/{entry['coverage_mean']:.3f}".rjust(16)
            lines.append(row)
        lines.append("")

    checked = getattr(result.report, "invariant_checks_performed", 0)
    if result.invariant_violations:
        lines.append(
            f"Invariant violations: {len(result.invariant_violations)} "
            f"(structural checks actually run on {checked} cells)"
        )
        for v in result.invariant_violations[:5]:
            lines.append(f"  - {v.get('invariant')}: {v.get('detail')}")
    else:
        lines.append(
            f"Invariant violations: none -- structural checks ACTUALLY RUN on "
            f"{checked} of {len(result.report.rows)} cells"
            + ("" if checked else "  [WARNING: no structural check was performed]")
        )
    return "\n".join(lines)


def save(result: PipelineResult, path: str | Path) -> Path:
    """Persist both the benchmark report and the full pipeline analysis."""
    report_path = save_report(result.report, path)
    pipeline_path = Path(path).with_name(Path(path).stem + "_analysis.json")
    import json

    pipeline_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report_path
