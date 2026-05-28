"""Issue 14: example scenario files + thin ``main.py`` CLI (graph-engine edition).

Acceptance criteria pinned by these tests:

1. ``scenarios/example_homogeneous.py`` exposes a top-level ``Scenario``
   with graph-mode nodes (``is_graph == True``) that runs end-to-end and
   produces a run-log with the documented shape.
2. ``scenarios/example_paired_comparison.py`` exposes a top-level
   ``Scenario`` whose paired sub-graphs share seeds — the verifiable CRN
   signal that the two policy groups operate on bit-identical world data.
3. ``main.py`` accepts a scenario path argument, dispatches it through
   ``Runner`` + ``DataExporter``, and writes parquet/JSON/PNG artifacts.
   Passing a path that is not a Scenario fails loudly.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.sim.node import IntermediateNode
from src.sim.runner import Runner
from src.sim.scenario import Scenario


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- examples


@pytest.fixture(scope="module")
def homogeneous_module():
    return importlib.import_module("scenarios.example_homogeneous")


@pytest.fixture(scope="module")
def paired_module():
    return importlib.import_module("scenarios.example_paired_comparison")


@pytest.fixture(scope="module")
def offline_module():
    return importlib.import_module("scenarios.example_llm_world_offline")


def test_homogeneous_exposes_scenario_symbol(homogeneous_module):
    assert isinstance(homogeneous_module.scenario, Scenario)


def test_homogeneous_scenario_uses_graph_engine(homogeneous_module):
    """Homogeneous example must be a graph-mode scenario with multiple nodes."""
    scenario = homogeneous_module.scenario
    assert scenario.is_graph, "homogeneous example must be a graph-mode scenario"
    assert len(scenario.nodes) >= 2, "homogeneous example must have at least 2 nodes"
    assert len(scenario.edges) >= 1, "homogeneous example must have at least 1 edge"


def test_homogeneous_has_multiple_intermediate_nodes(homogeneous_module):
    """Multiple shops (IntermediateNodes) share one policy class in homogeneous example."""
    scenario = homogeneous_module.scenario
    shops = [ni for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)]
    assert len(shops) >= 2, "homogeneous example must have at least 2 shop nodes"
    # All shops share the same policy class (homogeneous = same policy config).
    policy_classes = {type(ni.policy).__name__ for ni in shops}
    assert len(policy_classes) == 1, f"all shop nodes should share one policy class; got {policy_classes}"
    # All shop init_seeds must be distinct so per-node init draws differ.
    seeds = [ni.init_seed for ni in shops]
    assert len(set(seeds)) == len(seeds), "shop init_seeds must be distinct"


def test_homogeneous_runs_end_to_end(homogeneous_module):
    """Runner produces a run-log with the expected top-level shape."""
    scenario = homogeneous_module.scenario
    run_log = Runner(scenario).run()
    assert {"global", "ticks", "n_steps"} <= set(run_log.keys())
    assert run_log["n_steps"] == scenario.n_steps
    assert len(run_log["ticks"]) == scenario.n_steps
    assert (
        len(run_log["global"]["time"]["simulation_step"]) == scenario.n_steps + 1
    )


def test_paired_exposes_scenario_symbol(paired_module):
    assert isinstance(paired_module.scenario, Scenario)


def test_paired_scenario_uses_graph_engine(paired_module):
    """Paired example must be a graph-mode scenario."""
    scenario = paired_module.scenario
    assert scenario.is_graph, "paired example must be a graph-mode scenario"
    # Must have an even number of shops for A/B pairing.
    shops = [ni for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)]
    assert len(shops) >= 2
    assert len(shops) % 2 == 0, "paired example must have even number of shops"


def test_paired_runs_end_to_end(paired_module):
    """Runner produces a graph-mode run-log with expected shape."""
    scenario = paired_module.scenario
    run_log = Runner(scenario).run()
    assert {"global", "ticks", "n_steps"} <= set(run_log.keys())
    assert run_log["n_steps"] == scenario.n_steps
    assert (
        len(run_log["global"]["time"]["simulation_step"]) == scenario.n_steps + 1
    )


def test_paired_policy_groups_attach_distinct_policies(paired_module):
    """Both policy groups are present and distinguishable among shop nodes."""
    scenario = paired_module.scenario
    shops = [ni for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)]
    # Collect distinct policy types (should be 2: one per group).
    policy_types = {type(ni.policy).__name__ for ni in shops}
    assert len(policy_types) == 2, (
        f"paired example should have 2 distinct shop policy types; got {policy_types}"
    )


def test_paired_crn_seeds_shared_across_groups(paired_module):
    """CRN: each A/B shop pair must share init_seed (same world, different policy).

    Both policy groups are built from the same seed structure — only the
    policy class differs. This is the CRN signal that makes paired comparison
    valid.
    """
    scenario = paired_module.scenario
    shops = [ni for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)]
    # Group shops by pair index (pair0-a-shop, pair0-b-shop, pair1-a-shop, ...)
    # Seeds of each A/B pair should match.
    from collections import defaultdict
    pair_seeds: dict[int, list[int]] = defaultdict(list)
    for ni in shops:
        # Node IDs follow pattern: pair<N>-<label>-shop
        parts = ni.node.id.split("-")
        if len(parts) >= 3 and parts[0].startswith("pair"):
            pair_idx = int(parts[0][4:])
            pair_seeds[pair_idx].append(ni.init_seed)
    assert len(pair_seeds) >= 1, "expected at least one CRN pair"
    for idx, seeds in pair_seeds.items():
        assert len(set(seeds)) == 1, (
            f"pair {idx}: A and B shops should share init_seed for CRN; got {seeds}"
        )


# ----------------------------------------------------------------- main.py


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke ``main.py`` as a subprocess; returns the completed process."""
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "main.py"), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_main_runs_homogeneous_example(tmp_path):
    """``main.py <path>`` runs the example end-to-end and writes artifacts."""
    output = tmp_path / "homo_out"
    proc = _run_cli(
        "scenarios/example_homogeneous.py",
        "--output",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr
    # Every artifact the exporter promises must exist.
    assert (output / "config" / "scenario.json").exists()
    assert (output / "data" / "run_log.json").exists()
    assert (output / "data" / "products.parquet").exists()
    assert (output / "data" / "stores.parquet").exists()
    assert (output / "data" / "timeseries.parquet").exists()
    assert (output / "reports" / "overview.png").exists()

    # Round-trip the scenario JSON written by the exporter.
    with open(output / "config" / "scenario.json") as f:
        payload = json.load(f)
    assert "world_seed" in payload
    # Graph-mode scenarios carry "nodes" and "edges" instead of "stores".
    assert "nodes" in payload, "graph-mode scenario JSON must include 'nodes'"
    assert "edges" in payload, "graph-mode scenario JSON must include 'edges'"


def test_main_runs_paired_example(tmp_path):
    output = tmp_path / "paired_out"
    proc = _run_cli(
        "scenarios/example_paired_comparison.py",
        "--output",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr
    assert (output / "data" / "run_log.json").exists()


def test_main_rejects_missing_file(tmp_path):
    """Non-existent scenario path errors out with a non-zero exit code."""
    proc = _run_cli(str(tmp_path / "nope.py"))
    assert proc.returncode != 0
    assert "not found" in (proc.stdout + proc.stderr).lower()


def test_main_rejects_module_without_scenario_symbol(tmp_path):
    """A scenario file missing the ``scenario`` symbol fails loudly."""
    bad = tmp_path / "no_scenario.py"
    bad.write_text("# Intentionally empty: no `scenario` symbol.\n")
    proc = _run_cli(str(bad))
    assert proc.returncode != 0
    msg = (proc.stdout + proc.stderr).lower()
    assert "scenario" in msg


def test_main_rejects_wrong_scenario_type(tmp_path):
    """A module exposing ``scenario`` of the wrong type fails loudly."""
    bad = tmp_path / "wrong_type.py"
    bad.write_text("scenario = 42  # not a Scenario\n")
    proc = _run_cli(str(bad))
    assert proc.returncode != 0
    msg = (proc.stdout + proc.stderr).lower()
    assert "scenario" in msg


def test_main_default_output_folder(tmp_path, monkeypatch):
    """Without ``--output``, artifacts land in ``data/<stem>`` under cwd."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "main.py"),
            str(_REPO_ROOT / "scenarios" / "example_homogeneous.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    expected = tmp_path / "data" / "example_homogeneous"
    assert expected.exists()
    assert (expected / "config" / "scenario.json").exists()


# --------------------------------------------------------- offline LLM example


def test_offline_exposes_scenario_symbol(offline_module):
    """example_llm_world_offline exposes a top-level Scenario after from_world refactor."""
    assert isinstance(offline_module.scenario, Scenario)


def test_main_runs_offline_llm_example(tmp_path):
    """main.py runs the offline (no-API-key) LLM scenario and writes artifacts."""
    output = tmp_path / "offline_out"
    proc = _run_cli(
        "scenarios/example_llm_world_offline.py",
        "--output",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr
    assert (output / "config" / "scenario.json").exists()
    assert (output / "data" / "run_log.json").exists()
    assert (output / "data" / "products.parquet").exists()
    assert (output / "data" / "stores.parquet").exists()
