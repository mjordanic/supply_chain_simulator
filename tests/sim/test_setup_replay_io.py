"""Tests for setup-dir serialization of replay scenarios (issue 03).

Acceptance criteria:
1. setup.yaml supports a ``sink_replay`` node type referencing a ``series_id``.
2. Demand series persist in a tidy parquet (series_id, tick, qty) inside the setup dir.
3. Roundtrip: write a replay scenario dir → load_setup → Runner.run() → exact replay under flat world.
4. Missing series_id or too-short series fails at load time with a clear error.
5. Existing synthetic setup dirs load unchanged (backward compatible).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.sim.setup_io import load_setup, write_setup


# ---------------------------------------------------------------------------
# Shared YAML / CSV helpers
# ---------------------------------------------------------------------------


_MINIMAL_CATALOG = (
    "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
    "P0001,Widget,widgets,10.0,4.0,peak,,1.0\n"
)

_SERIES = [5, 10, 3, 7, 2]
_N_STEPS = len(_SERIES)


_REPLAY_SETUP_YAML = textwrap.dedent("""\
run:
  n_steps: 5
  start_date: "2024-01-01"
  world_seed: 42

market:
  cycle_len: 365
  cycle_amp: 0.0
  init_demand: 1.0
  init_supply: 1.0
  peak_factor: 1.0
  off_factor: 1.0
  season_months: {}
  regions: [US]
  correlation: 0.0
  trend_update_interval: 100
  min_value: 0.0
  max_value: 10.0
  stage_multipliers: {}
  price_elasticity: 0.0
  promo_multiplier: 1.0
  demand_factor_min: 0.0
  supply_factor_min: 0.0
  cross_inv_lo: 0.0
  cross_inv_hi: 1.0
  cross_factor_range: [1.0, 1.0]
  trend: {kind: constant, value: 1.0}
  demand_shock: {kind: constant, value: 0.0}
  supply_shock: {kind: constant, value: 0.0}
  base_demand: {kind: constant, value: 1.0}

disruption:
  event_prob: 0.0
  types: [natural_disaster]
  regions: [US]
  severity: {kind: constant, value: 0.0}
  duration: {kind: constant, value: 1}

nodes:
  - id: factory-1
    type: factory
    region: US
    produces_product_id: P0001
    unit_cost: 1.0
    capacity_per_tick: 10000
    inventory: 10000
    list_price: 1.0
    cash: 0.0

  - id: shop-1
    type: intermediate
    region: US
    carried_products: [P0001]
    capacity: 100000
    inventory:
      P0001: 10000
    list_prices:
      P0001: 2.0
    min_order_imposed:
      P0001: 0
    cash: 1000000.0

  - id: sink-1
    type: sink_replay
    region: US
    product_id: P0001
    series_id: sink_series
    income_rate: 1000000.0
    cash: 1000000.0

edges:
  - supplier: factory-1
    buyer: shop-1
    lead_time: 0

  - supplier: shop-1
    buyer: sink-1
    lead_time: 0
""")

_SYNTHETIC_SETUP_YAML = textwrap.dedent("""\
run:
  n_steps: 3
  start_date: "2024-01-01"
  world_seed: 7

market:
  cycle_len: 365
  cycle_amp: 0.0
  init_demand: 1.0
  init_supply: 1.0
  peak_factor: 1.0
  off_factor: 1.0
  season_months: {}
  regions: [US]
  correlation: 0.0
  trend_update_interval: 100
  min_value: 0.0
  max_value: 10.0
  stage_multipliers: {}
  price_elasticity: 0.0
  promo_multiplier: 1.0
  demand_factor_min: 0.0
  supply_factor_min: 0.0
  cross_inv_lo: 0.0
  cross_inv_hi: 1.0
  cross_factor_range: [1.0, 1.0]
  trend: {kind: constant, value: 1.0}
  demand_shock: {kind: constant, value: 0.0}
  supply_shock: {kind: constant, value: 0.0}
  base_demand: {kind: constant, value: 1.0}

disruption:
  event_prob: 0.0
  types: [natural_disaster]
  regions: [US]
  severity: {kind: constant, value: 0.0}
  duration: {kind: constant, value: 1}

nodes:
  - id: factory-1
    type: factory
    region: US
    produces_product_id: P0001
    unit_cost: 4.0
    capacity_per_tick: 30
    inventory: 50
    list_price: 4.0
    cash: 0.0

  - id: shop-1
    type: intermediate
    region: US
    carried_products: [P0001]
    capacity: 200
    inventory:
      P0001: 10
    list_prices:
      P0001: 7.0
    min_order_imposed:
      P0001: 0
    cash: 500.0

  - id: sink-1
    type: demand_sink
    region: US
    product_id: P0001
    demand_dist: {kind: constant, value: 5.0}
    income_rate: 100.0
    cash: 1000.0

edges:
  - supplier: factory-1
    buyer: shop-1
    lead_time: 1

  - supplier: shop-1
    buyer: sink-1
    lead_time: 1
""")


def _write_replay_setup(tmp_path: Path) -> Path:
    """Write a minimal replay scenario directory and return its path."""
    setup_dir = tmp_path / "replay_setup"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(_MINIMAL_CATALOG)
    (setup_dir / "setup.yaml").write_text(_REPLAY_SETUP_YAML)

    # Write the demand parquet
    df = pd.DataFrame({
        "series_id": ["sink_series"] * _N_STEPS,
        "tick": list(range(_N_STEPS)),
        "qty": _SERIES,
    })
    df.to_parquet(setup_dir / "demand_series.parquet", index=False)

    return setup_dir


# ---------------------------------------------------------------------------
# AC1: setup.yaml supports sink_replay node type
# ---------------------------------------------------------------------------


class TestSinkReplayNodeType:
    def test_load_setup_accepts_sink_replay_type(self, tmp_path):
        """load_setup accepts a sink_replay node and returns a ReplayDemandSinkNode."""
        from src.sim.replay_demand_sink import ReplayDemandSinkNode

        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        sink_ni = next(ni for ni in scenario.nodes if ni.node.id == "sink-1")
        assert isinstance(sink_ni.node, ReplayDemandSinkNode)

    def test_replay_node_has_correct_series(self, tmp_path):
        """Loaded ReplayDemandSinkNode has the series from the parquet."""
        from src.sim.replay_demand_sink import ReplayDemandSinkNode

        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        sink_ni = next(ni for ni in scenario.nodes if ni.node.id == "sink-1")
        assert isinstance(sink_ni.node, ReplayDemandSinkNode)
        assert list(sink_ni.node.series) == _SERIES

    def test_replay_node_product_id_and_region(self, tmp_path):
        """Loaded replay node has the correct product_id and region."""
        from src.sim.replay_demand_sink import ReplayDemandSinkNode

        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        sink_ni = next(ni for ni in scenario.nodes if ni.node.id == "sink-1")
        assert isinstance(sink_ni.node, ReplayDemandSinkNode)
        assert sink_ni.node.product_id == "P0001"
        assert sink_ni.node.region == "US"


# ---------------------------------------------------------------------------
# AC2: demand series in tidy parquet
# ---------------------------------------------------------------------------


class TestDemandParquet:
    def test_demand_parquet_has_expected_columns(self, tmp_path):
        """demand_series.parquet has series_id, tick, qty columns."""
        setup_dir = _write_replay_setup(tmp_path)
        df = pd.read_parquet(setup_dir / "demand_series.parquet")
        assert set(df.columns) >= {"series_id", "tick", "qty"}

    def test_write_setup_emits_demand_parquet_for_replay_scenario(self, tmp_path):
        """write_setup for a replay scenario emits demand_series.parquet."""
        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        out_dir = tmp_path / "written"
        write_setup(scenario, out_dir)
        assert (out_dir / "demand_series.parquet").is_file()

    def test_written_demand_parquet_has_correct_series(self, tmp_path):
        """write_setup followed by load_setup gives identical series.

        write_setup uses the node id as the series_id in the emitted parquet.
        """
        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        out_dir = tmp_path / "written_series"
        write_setup(scenario, out_dir)
        df = pd.read_parquet(out_dir / "demand_series.parquet")

        # series_id in the written parquet is the node id ("sink-1")
        row = df[df["series_id"] == "sink-1"].sort_values("tick")
        assert list(row["qty"]) == _SERIES


# ---------------------------------------------------------------------------
# AC3: roundtrip write → load_setup → Runner.run() → exact replay
# ---------------------------------------------------------------------------


class TestReplayRoundtrip:
    def test_roundtrip_write_load_run_exact_replay(self, tmp_path):
        """Write replay scenario, load it, run it; verify exact per-tick demand matches series."""
        from src.sim.runner import Runner

        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        # Write then re-load to ensure the full cycle: write_setup → load_setup → run.
        out_dir = tmp_path / "roundtrip"
        write_setup(scenario, out_dir)
        rt_scenario = load_setup(out_dir)

        log = Runner(rt_scenario).run()

        for i, tick_log in enumerate(log["ticks"]):
            sink_flows = [f for f in tick_log["node_flows"] if f["node_id"] == "sink-1"]
            assert len(sink_flows) == 1, f"tick {i}: expected 1 sink flow entry"
            assert sink_flows[0]["demand"] == _SERIES[i], (
                f"tick {i}: expected {_SERIES[i]}, got {sink_flows[0]['demand']}"
            )

    def test_roundtrip_preserves_n_steps_and_world_seed(self, tmp_path):
        """write_setup + load_setup preserves run parameters for replay scenarios."""
        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        out_dir = tmp_path / "run_params"
        write_setup(scenario, out_dir)
        rt = load_setup(out_dir)

        assert rt.n_steps == scenario.n_steps
        assert rt.world_seed == scenario.world_seed
        assert rt.start_date == scenario.start_date

    def test_roundtrip_replay_node_type_preserved(self, tmp_path):
        """After write_setup + load_setup, the sink is still a ReplayDemandSinkNode."""
        from src.sim.replay_demand_sink import ReplayDemandSinkNode

        setup_dir = _write_replay_setup(tmp_path)
        scenario = load_setup(setup_dir)

        out_dir = tmp_path / "type_preserved"
        write_setup(scenario, out_dir)
        rt = load_setup(out_dir)

        sink_ni = next(ni for ni in rt.nodes if ni.node.id == "sink-1")
        assert isinstance(sink_ni.node, ReplayDemandSinkNode)


# ---------------------------------------------------------------------------
# AC4: missing series_id or too-short series fails at load time
# ---------------------------------------------------------------------------


class TestLoadValidation:
    def test_missing_series_id_in_parquet_raises_at_load(self, tmp_path):
        """If series_id from YAML is not in the parquet, load_setup raises ValueError."""
        setup_dir = _write_replay_setup(tmp_path)
        # Overwrite parquet with a different series_id
        df = pd.DataFrame({
            "series_id": ["wrong_id"] * _N_STEPS,
            "tick": list(range(_N_STEPS)),
            "qty": _SERIES,
        })
        df.to_parquet(setup_dir / "demand_series.parquet", index=False)

        with pytest.raises(ValueError, match="sink_series"):
            load_setup(setup_dir)

    def test_series_shorter_than_n_steps_raises_at_load(self, tmp_path):
        """If parquet series has fewer entries than n_steps, load_setup raises ValueError."""
        setup_dir = _write_replay_setup(tmp_path)
        # Write a parquet with only 2 entries but n_steps is 5
        df = pd.DataFrame({
            "series_id": ["sink_series"] * 2,
            "tick": [0, 1],
            "qty": [5, 10],
        })
        df.to_parquet(setup_dir / "demand_series.parquet", index=False)

        with pytest.raises(ValueError, match="series"):
            load_setup(setup_dir)

    def test_missing_demand_parquet_raises_at_load(self, tmp_path):
        """If demand_series.parquet is missing but a sink_replay node exists, raises."""
        setup_dir = _write_replay_setup(tmp_path)
        (setup_dir / "demand_series.parquet").unlink()

        with pytest.raises((ValueError, FileNotFoundError)):
            load_setup(setup_dir)

    def test_sink_replay_missing_series_id_field_raises(self, tmp_path):
        """A sink_replay node without series_id field raises ValueError at load time."""
        setup_dir = _write_replay_setup(tmp_path)

        import yaml
        doc = yaml.safe_load((setup_dir / "setup.yaml").read_text())
        # Remove series_id from the sink_replay node
        for node in doc["nodes"]:
            if node.get("type") == "sink_replay":
                del node["series_id"]
        (setup_dir / "setup.yaml").write_text(yaml.safe_dump(doc))

        with pytest.raises(ValueError, match="series_id"):
            load_setup(setup_dir)


# ---------------------------------------------------------------------------
# AC5: backward compatibility — existing synthetic setup dirs load unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_synthetic_setup_loads_without_demand_parquet(self, tmp_path):
        """A synthetic (non-replay) setup dir loads fine without demand_series.parquet."""
        setup_dir = tmp_path / "synthetic"
        setup_dir.mkdir()
        (setup_dir / "catalog.csv").write_text(_MINIMAL_CATALOG)
        (setup_dir / "setup.yaml").write_text(_SYNTHETIC_SETUP_YAML)
        # No demand_series.parquet — this is a synthetic setup

        scenario = load_setup(setup_dir)
        assert len(scenario.nodes) == 3

    def test_synthetic_setup_write_does_not_emit_demand_parquet(self, tmp_path):
        """write_setup for a synthetic scenario does NOT emit demand_series.parquet."""
        setup_dir = tmp_path / "synthetic"
        setup_dir.mkdir()
        (setup_dir / "catalog.csv").write_text(_MINIMAL_CATALOG)
        (setup_dir / "setup.yaml").write_text(_SYNTHETIC_SETUP_YAML)

        scenario = load_setup(setup_dir)
        out_dir = tmp_path / "synthetic_out"
        write_setup(scenario, out_dir)

        assert not (out_dir / "demand_series.parquet").is_file()
