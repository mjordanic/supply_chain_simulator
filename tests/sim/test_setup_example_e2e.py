"""End-to-end regression/snapshot test for the committed example setup directory.

Acceptance criteria:
- ``uv run python main.py run setups/three_node_chain`` runs and writes outputs
  including nodes.parquet and a verbatim config/ snapshot.
- Running the same setup twice produces bit-identical run logs (determinism).
- nodes.parquet is written and contains the expected node columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sim.runner import Runner
from src.sim.setup_io import load_setup


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXAMPLE_SETUP = _REPO_ROOT / "setups" / "three_node_chain"


# ---------------------------------------------------------------------------
# Example setup exists
# ---------------------------------------------------------------------------


def test_example_setup_dir_exists():
    assert _EXAMPLE_SETUP.is_dir(), f"Example setup dir not found: {_EXAMPLE_SETUP}"
    assert (_EXAMPLE_SETUP / "catalog.csv").is_file()
    assert (_EXAMPLE_SETUP / "setup.yaml").is_file()


# ---------------------------------------------------------------------------
# Load and run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_scenario():
    return load_setup(_EXAMPLE_SETUP)


@pytest.fixture(scope="module")
def example_run_log(example_scenario):
    return Runner(example_scenario).run()


def test_example_loads_valid_scenario(example_scenario):
    assert example_scenario.is_graph
    assert len(example_scenario.catalog) >= 1
    assert len(example_scenario.nodes) == 3
    assert len(example_scenario.edges) == 2


def test_example_run_has_expected_shape(example_run_log, example_scenario):
    assert example_run_log["n_steps"] == example_scenario.n_steps
    assert len(example_run_log["ticks"]) == example_scenario.n_steps


def test_example_run_nodes_all_present(example_run_log):
    last_tick = example_run_log["ticks"][-1]
    node_ids = set(last_tick["node_cash"].keys())
    assert "factory-1" in node_ids
    assert "shop-1" in node_ids
    assert "sink-1" in node_ids


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_example_deterministic(example_scenario):
    """Same setup → bit-identical run logs on two independent runs."""
    log1 = Runner(load_setup(_EXAMPLE_SETUP)).run()
    log2 = Runner(load_setup(_EXAMPLE_SETUP)).run()

    for t in range(log1["n_steps"]):
        t1 = log1["ticks"][t]
        t2 = log2["ticks"][t]
        assert t1["node_cash"] == t2["node_cash"], f"tick {t}: node_cash differs"
        assert t1["node_inventory"] == t2["node_inventory"], (
            f"tick {t}: node_inventory differs"
        )


# ---------------------------------------------------------------------------
# nodes.parquet output
# ---------------------------------------------------------------------------


def test_example_nodes_parquet_written(tmp_path, example_scenario, example_run_log):
    """DataExporter.save_nodes_parquet writes a valid nodes.parquet file."""
    from src.sim.data_exporter import DataExporter

    exporter = DataExporter(example_scenario, example_run_log)
    out_path = exporter.save_nodes_parquet(str(tmp_path))

    assert out_path, "save_nodes_parquet returned empty path"
    assert (tmp_path / "data" / "nodes.parquet").is_file()

    df = pd.read_parquet(tmp_path / "data" / "nodes.parquet")
    assert "node_id" in df.columns
    assert set(df["node_id"]) == {"factory-1", "shop-1", "sink-1"}


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess)
# ---------------------------------------------------------------------------


def test_cli_run_writes_all_artifacts(tmp_path):
    """main.py run <setup-dir> writes nodes.parquet and config/ snapshot."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "main.py"),
            "run",
            str(_EXAMPLE_SETUP),
            "--output",
            str(tmp_path / "out"),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    out = tmp_path / "out"
    assert (out / "data" / "nodes.parquet").is_file(), "nodes.parquet must be written"
    assert (out / "config" / "catalog.csv").is_file(), "catalog.csv must be in config/"
    assert (out / "config" / "setup.yaml").is_file(), "setup.yaml must be in config/"
    assert (out / "data" / "run_log.json").is_file()
    assert (out / "reports" / "overview.png").is_file()
