"""Round-trip test for the "inspect a saved run" workflow (notebook 02a).

A run saved to disk by ``main.py run`` (``DataExporter.export_all`` + a verbatim
``config/`` snapshot) must reload into exactly what a fresh in-memory run produces:

  - ``load_setup(run_dir / "config")`` rebuilds the identical ``Scenario``
    (same ``world_seed`` -> same derived per-node seeds), and
  - ``run_log.json`` round-trips through the pure ``inspect`` helpers to
    byte-identical DataFrames.

Also checks the integrity guarantee notebook 02a §7 relies on: the exported
``timeseries.parquet`` agrees with the run log it was written from.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.sim.data_exporter import DataExporter
from src.sim.inspect import (
    global_timeseries_df,
    node_equity,
    node_timeseries_df,
    per_product_df,
)
from src.sim.runner import Runner
from src.sim.setup_io import load_setup

SETUP_DIR = Path("setups/three_node_chain")


def _save_run_like_cli(scenario, run_log, out: Path) -> None:
    """Reproduce ``main.py run``'s on-disk layout under ``out``."""
    exporter = DataExporter(scenario, run_log)
    exporter.export_all(str(out))
    exporter.save_nodes_parquet(str(out))
    config_dir = out / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SETUP_DIR / "catalog.csv", config_dir / "catalog.csv")
    shutil.copy2(SETUP_DIR / "setup.yaml", config_dir / "setup.yaml")


def test_saved_run_reloads_to_identical_inspect_frames(tmp_path: Path) -> None:
    """Loading a saved run reproduces the fresh run's inspect DataFrames exactly."""
    scenario = load_setup(SETUP_DIR)
    fresh_log = Runner(scenario).run()

    run_dir = tmp_path / "run"
    _save_run_like_cli(scenario, fresh_log, run_dir)

    # The 02a loader: rebuild the scenario from config, read the run log off disk.
    loaded_scenario = load_setup(run_dir / "config")
    loaded_log = json.loads((run_dir / "data" / "run_log.json").read_text())

    assert loaded_log["n_steps"] == fresh_log["n_steps"]
    assert len(loaded_log["ticks"]) == len(fresh_log["ticks"])

    node_ids = [ni.node.id for ni in scenario.nodes]

    pd.testing.assert_frame_equal(
        node_timeseries_df(fresh_log, scenario),
        node_timeseries_df(loaded_log, loaded_scenario),
    )
    pd.testing.assert_frame_equal(
        global_timeseries_df(fresh_log),
        global_timeseries_df(loaded_log),
    )
    for nid in node_ids:
        pd.testing.assert_frame_equal(
            per_product_df(fresh_log, scenario, nid),
            per_product_df(loaded_log, loaded_scenario, nid),
        )
        pd.testing.assert_frame_equal(
            node_equity(fresh_log, scenario, nid),
            node_equity(loaded_log, loaded_scenario, nid),
        )


def test_saved_timeseries_parquet_reconciles_with_run_log(tmp_path: Path) -> None:
    """Exported timeseries.parquet inventory matches the run log it came from."""
    scenario = load_setup(SETUP_DIR)
    run_log = Runner(scenario).run()

    run_dir = tmp_path / "run"
    _save_run_like_cli(scenario, run_log, run_dir)

    ts_pq = pd.read_parquet(run_dir / "data" / "timeseries.parquet")
    nodes_in_pq = sorted(ts_pq["store_id"].unique())
    assert nodes_in_pq, "expected at least one node with per-product inventory rows"

    from_log = pd.concat(
        per_product_df(run_log, scenario, nid).assign(store_id=nid)
        for nid in nodes_in_pq
    )[["store_id", "tick", "pid", "inventory"]]

    merged = ts_pq.merge(
        from_log,
        left_on=["store_id", "simulation_step", "product_id"],
        right_on=["store_id", "tick", "pid"],
        suffixes=("_parquet", "_runlog"),
    )
    # Every parquet inventory row is accounted for, and none disagree.
    assert len(merged) == len(ts_pq)
    assert (merged["inventory_parquet"] == merged["inventory_runlog"]).all()
