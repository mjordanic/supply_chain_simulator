"""Example setup directories + thin ``main.py`` CLI (graph-engine edition).

Acceptance criteria pinned by these tests:

1. ``main.py run <setup-dir>`` loads a setup directory and writes artifacts.
2. ``setups/three_node_chain/`` and ``setups/two_factories_two_shops/`` run end-to-end.
3. Error paths (missing setup-dir, no subcommand) exit non-zero.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- helpers

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke ``main.py`` as a subprocess; returns the completed process."""
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "main.py"), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


# ----------------------------------------------------------------- main.py run


def test_main_run_three_node_chain(tmp_path):
    """``main.py run <setup-dir>`` runs the example end-to-end and writes artifacts."""
    output = tmp_path / "chain_out"
    proc = _run_cli(
        "run",
        str(_REPO_ROOT / "setups" / "three_node_chain"),
        "--output",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr
    # Core artifacts must exist.
    assert (output / "data" / "nodes.parquet").exists(), "nodes.parquet must be written"
    assert (output / "data" / "run_log.json").exists()
    assert (output / "data" / "products.parquet").exists()
    assert (output / "reports" / "overview.png").exists()
    # Verbatim config snapshot.
    assert (output / "config" / "catalog.csv").exists(), "catalog.csv must be copied to config/"
    assert (output / "config" / "setup.yaml").exists(), "setup.yaml must be copied to config/"


def test_main_run_two_factories_two_shops(tmp_path):
    """``main.py run setups/two_factories_two_shops`` runs end-to-end."""
    output = tmp_path / "2f2s_out"
    proc = _run_cli(
        "run",
        str(_REPO_ROOT / "setups" / "two_factories_two_shops"),
        "--output",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr
    assert (output / "data" / "nodes.parquet").exists()
    assert (output / "data" / "run_log.json").exists()
    assert (output / "config" / "catalog.csv").exists()
    assert (output / "config" / "setup.yaml").exists()


def test_main_run_rejects_missing_setup_dir(tmp_path):
    """Non-existent setup directory errors out with a non-zero exit code."""
    proc = _run_cli("run", str(tmp_path / "no_such_dir"))
    assert proc.returncode != 0
    assert "not found" in (proc.stdout + proc.stderr).lower()


def test_main_run_default_output_folder(tmp_path):
    """Without ``--output``, artifacts land in ``data/<setup-dir-name>`` under cwd."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "main.py"),
            "run",
            str(_REPO_ROOT / "setups" / "three_node_chain"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    expected = tmp_path / "data" / "three_node_chain"
    assert expected.exists()
    assert (expected / "data" / "nodes.parquet").exists()


def test_main_no_subcommand_exits_nonzero():
    """Running main.py with no subcommand prints help and exits non-zero."""
    proc = _run_cli()
    assert proc.returncode != 0
