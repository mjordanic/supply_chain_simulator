"""Issue 09: example scenario files + thin ``main.py`` CLI.

Acceptance criteria pinned by these tests:

1. ``scenarios/example_homogeneous.py`` exposes a top-level ``Scenario``
   that runs end-to-end and produces a run-log with the documented shape.
2. ``scenarios/example_paired_comparison.py`` exposes a top-level
   ``Scenario`` whose ``paired(...)`` stores share ``(template, init_seed)``
   step-0 state pairwise — the verifiable signal that the two policy
   groups operate on bit-identical world data.
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

from src.sim.policy import BaselinePolicy
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


def test_homogeneous_scenario_uses_homogeneous_helper(homogeneous_module):
    """All ``StoreInstance``s share one template AND one policy object."""
    stores = homogeneous_module.scenario.stores
    assert len(stores) >= 2, "homogeneous example must have at least 2 stores"
    first = stores[0]
    for s in stores[1:]:
        assert s.template is first.template
        assert s.policy is first.policy
    # All init_seeds must be distinct so per-store init draws differ.
    seeds = [s.init_seed for s in stores]
    assert len(set(seeds)) == len(seeds)


def test_homogeneous_runs_end_to_end(homogeneous_module):
    """Runner produces a run-log with the expected top-level shape."""
    scenario = homogeneous_module.scenario
    run_log = Runner(scenario).run()
    assert {"global", "stores"} <= set(run_log.keys())
    assert len(run_log["stores"]) == len(scenario.stores)
    assert (
        len(run_log["global"]["time"]["simulation_step"]) == scenario.n_steps + 1
    )


def test_paired_exposes_scenario_symbol(paired_module):
    assert isinstance(paired_module.scenario, Scenario)


def test_paired_scenario_uses_paired_helper(paired_module):
    """``paired(...)`` produces interleaved ``[A, B, A, B, ...]`` stores."""
    stores = paired_module.scenario.stores
    assert len(stores) % 2 == 0 and len(stores) >= 2
    # Both policies appear; the two policy slots within a pair differ.
    for i in range(0, len(stores), 2):
        a, b = stores[i], stores[i + 1]
        assert a.policy is not b.policy
        assert a.init_seed == b.init_seed
        assert a.template is b.template


def test_paired_runs_end_to_end(paired_module):
    scenario = paired_module.scenario
    run_log = Runner(scenario).run()
    assert len(run_log["stores"]) == len(scenario.stores)
    assert (
        len(run_log["global"]["time"]["simulation_step"]) == scenario.n_steps + 1
    )


def test_paired_stores_share_step0_state(paired_module):
    """CRN signal: each pair's two stores start step-0 bit-identical.

    ``paired(...)`` shares ``(template, init_seed)`` across each pair, so
    init-rng-driven state (capacity, active SKU set, initial inventory)
    must match within a pair regardless of attached policy.
    """
    scenario = paired_module.scenario
    run_log = Runner(scenario).run()
    stores = run_log["stores"]
    for i in range(0, len(stores), 2):
        a, b = stores[i], stores[i + 1]
        assert a["step0_capacity"] == b["step0_capacity"]
        assert a["step0_inventory"] == b["step0_inventory"]
        assert a["step0_active_items"] == b["step0_active_items"]
        # Step-0 balance is set by the same template draw, so it matches too.
        assert a["balance"][0] == b["balance"][0]


def test_paired_policy_groups_attach_distinct_policies(paired_module):
    """Both ``BaselinePolicy`` groups are present and distinguishable."""
    stores = paired_module.scenario.stores
    policies_a = {id(stores[i].policy) for i in range(0, len(stores), 2)}
    policies_b = {id(stores[i].policy) for i in range(1, len(stores), 2)}
    assert len(policies_a) == 1, "all 'A' slots should share one policy object"
    assert len(policies_b) == 1, "all 'B' slots should share one policy object"
    assert policies_a.isdisjoint(policies_b)
    # Both policies are BaselinePolicy instances by construction in the example.
    a_policy = stores[0].policy
    b_policy = stores[1].policy
    assert isinstance(a_policy, BaselinePolicy)
    assert isinstance(b_policy, BaselinePolicy)


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
    assert "stores" in payload


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
