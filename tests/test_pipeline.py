"""End-to-end pipeline and CLI tests.

These run a deliberately *small* configuration. The point is not to re-verify the
benchmark (the other modules do that) but to check that the orchestration wires
the stages together, that the artefacts land on disk, and that the whole thing is
reproducible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.config import load_config
from core.seed import set_all
from eval.benchmark import load_report, run_benchmark, save_report
from pipeline.pipeline import format_summary, run_pipeline

SMALL = {
    "n_train": 300,
    "n_calib": 200,
    "n_val": 150,
    "n_test": 400,
    "seeds": (11, 23, 37),
}


@pytest.fixture(scope="module")
def small_cfg():
    return load_config(**SMALL)


def test_benchmark_produces_one_row_per_cell(small_cfg):
    set_all(small_cfg.seed)
    report = run_benchmark(small_cfg, datasets=["homoscedastic"], progress=False)
    assert len(report.rows) == 8 * 3  # methods x seeds
    assert not report.failures
    for row in report.rows:
        assert 0.0 <= row.coverage <= 1.0
        assert row.mean_width > 0
        assert np.isfinite(row.winkler)


def test_benchmark_is_deterministic(small_cfg):
    set_all(small_cfg.seed)
    a = run_benchmark(small_cfg, datasets=["homoscedastic"], progress=False)
    set_all(small_cfg.seed)
    b = run_benchmark(small_cfg, datasets=["homoscedastic"], progress=False)
    assert len(a.rows) == len(b.rows)
    for ra, rb in zip(a.rows, b.rows, strict=True):
        assert ra.method == rb.method and ra.dataset == rb.dataset and ra.seed == rb.seed
        assert ra.coverage == rb.coverage
        assert ra.mean_width == rb.mean_width
        assert ra.winkler == rb.winkler


def test_report_round_trips_through_json(small_cfg, tmp_path):
    set_all(small_cfg.seed)
    report = run_benchmark(small_cfg, datasets=["homoscedastic"], progress=False)
    path = save_report(report, tmp_path / "benchmark.json")
    assert path.exists()
    payload = load_report(path)
    assert "rows" in payload and "config" in payload
    assert len(payload["rows"]) == len(report.rows)
    # must be valid UTF-8 JSON with no numpy leakage
    json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_runs_and_produces_gates(small_cfg):
    set_all(small_cfg.seed)
    result = run_pipeline(
        small_cfg,
        datasets=["heteroscedastic"],
        methods=["split_absolute", "cqr", "vennfuse"],
        verify_determinism=True,
        run_ablations=True,
        progress=False,
    )
    for gate in ("G1", "G2", "G3", "G5"):
        assert gate in result.gate_results
        assert "passed" in result.gate_results[gate]
    assert result.elapsed > 0
    assert result.drift is not None
    assert isinstance(result.ablation, list)
    assert result.invariant_violations == []
    # D0 regression guard at the pipeline level: an empty violation list only
    # means something if cells were actually examined. Before the fix this was
    # 0 and the report still claimed "0 violations".
    assert result.invariant_checks["cells_checked"] == len(result.report.rows), (
        "structural invariants must be run on every benchmark cell"
    )
    assert getattr(result.report, "invariant_checks_performed", 0) > 0


def test_pipeline_determinism_gate_passes_on_repeat(small_cfg):
    """G5 must actually pass when the pipeline re-runs itself."""
    set_all(small_cfg.seed)
    result = run_pipeline(
        small_cfg,
        datasets=["homoscedastic"],
        methods=["split_absolute", "cqr"],
        verify_determinism=True,
        run_ablations=False,
        progress=False,
    )
    g5 = result.gate_results["G5"]
    assert g5["passed"], g5.get("reason")
    assert g5["measured"] == 0.0
    assert g5["n_compared"] > 0


def test_pipeline_writes_analysis_file(small_cfg, tmp_path):
    from pipeline.pipeline import save

    set_all(small_cfg.seed)
    result = run_pipeline(
        small_cfg,
        datasets=["homoscedastic"],
        methods=["split_absolute"],
        verify_determinism=False,
        run_ablations=False,
        progress=False,
    )
    out = save(result, tmp_path / "benchmark.json")
    assert out.exists()
    analysis = tmp_path / "benchmark_analysis.json"
    assert analysis.exists()
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    assert "gates" in payload and "environment" in payload


def test_format_summary_is_ascii_safe(small_cfg):
    set_all(small_cfg.seed)
    result = run_pipeline(
        small_cfg,
        datasets=["homoscedastic"],
        methods=["split_absolute"],
        verify_determinism=False,
        run_ablations=False,
        progress=False,
    )
    text = format_summary(result)
    text.encode("ascii")  # must not raise: Windows consoles are often cp1252
    assert "Gates" in text
