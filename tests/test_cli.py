"""CLI smoke tests.

The CLI is the package's public surface and its exit code is part of its
contract: ``confforge run`` must exit non-zero when a P0 gate fails, otherwise it
is useless in CI.
"""

from __future__ import annotations

from cli import main


def test_cli_gates_command_runs(capsys):
    assert main(["gates"]) == 0
    out = capsys.readouterr().out
    assert "G1" in out and "G2" in out and "G3" in out and "G5" in out


def test_cli_methods_and_datasets_commands(capsys):
    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    for name in ("split_absolute", "cqr", "mondrian", "vennfuse", "aci"):
        assert name in out
    assert main(["datasets"]) == 0
    assert "homoscedastic" in capsys.readouterr().out


def test_cli_selftest_command_runs(capsys):
    code = main(["selftest"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "MAPIE" in out


def test_cli_rejects_bad_config(capsys):
    """An out-of-range alpha must exit 2 with a message, not a traceback."""
    assert main(["run", "--alpha", "1.5"]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_quick_runs(tmp_path, capsys):
    code = main(["quick", "--dataset", "heteroscedastic", "--out", str(tmp_path / "q.json")])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert (tmp_path / "q.json").exists()
    assert "Gates" in out
