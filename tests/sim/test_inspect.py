"""Tests for src/sim/inspect.py.

Uses a hand-crafted run-log dict and a tiny 3-node-chain scenario (1 product,
factory → intermediate → sink) to keep tests self-contained and fast.

All assertions are on external behaviour:
  - column sets
  - row cardinality
  - node_type / level / region joins
  - _total-key exclusion in per_product_df
  - equity arithmetic (cash + inventory-at-cost + outstanding-at-cost)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.sim.inspect import (
    global_timeseries_df,
    node_equity,
    node_timeseries_df,
    per_product_df,
)


# ---------------------------------------------------------------------------
# Minimal scenario fixture (hand-built, no engine)
# ---------------------------------------------------------------------------


def _make_scenario():
    """Return a tiny 3-node scenario (factory → intermediate → sink)."""
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import (
        DisruptionParams,
        ItemLifecycleParams,
        MarketParams,
        NodeInstance,
        Scenario,
        load_catalog,
    )

    catalog = load_catalog(
        [
            {
                "name": "Widget",
                "category": "goods",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            }
        ]
    )
    pid = catalog[0].product_id  # "P0000"

    factory = FactoryNode(
        id="fac",
        region="EU",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=4.0,
        capacity_per_tick=50,
        inventory=100,
        list_price=4.0,
        cash=0.0,
    )
    # Manually assign level (normally set by compute_levels in build_world).
    factory.level = 0

    intermediate = IntermediateNode(
        id="wh",
        region="EU",
        init_seed=2,
        carried_products={pid},
        capacity=200,
        tags=["warehouse"],
        inventory={pid: 30},
        pending={},
        list_prices={pid: 7.0},
        min_order_imposed={pid: 0},
        cash=200.0,
    )
    intermediate.level = 1

    sink = DemandSinkNode(
        id="sink",
        region="EU",
        init_seed=3,
        product_id=pid,
        demand_dist=Constant(5),
        income_rate=50.0,
        cash=500.0,
    )
    sink.level = 2

    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    scenario = Scenario(
        catalog=catalog,
        market=MarketParams(
            cycle_len=365,
            cycle_amp=0.0,
            init_demand=1.0,
            init_supply=1.0,
            peak_factor=1.0,
            off_factor=1.0,
            season_months={},
            regions=["EU"],
            correlation=0.0,
            trend_update_interval=100,
            min_value=0.5,
            max_value=2.0,
            stage_multipliers={s: 1.0 for s in stages},
            price_elasticity=0.0,
            promo_multiplier=1.0,
            demand_factor_min=0.1,
            supply_factor_min=0.01,
            cross_inv_lo=0.3,
            cross_inv_hi=0.7,
            cross_factor_range=(0.5, 1.5),
            trend=Constant(1.0),
            demand_shock=Constant(0.0),
            supply_shock=Constant(0.0),
            base_demand=Constant(5),
        ),
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["flood"],
            regions=["EU"],
            severity=Constant(0.1),
            duration=Constant(1),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=stages,
            init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in stages},
        ),
        stores=[],
        n_steps=3,
        start_date=datetime(2024, 1, 1),
        world_seed=1,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=intermediate, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=[
            EdgeSpec(supplier_id="fac", buyer_id="wh", default_lead_time=1),
            EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
        ],
    )
    return scenario, pid


def _make_run_log(pid: str, n_ticks: int = 3) -> dict:
    """Hand-crafted run-log with deterministic values for easy assertions."""
    ticks = []
    for t in range(1, n_ticks + 1):
        ticks.append(
            {
                "tick": t,
                "node_cash": {
                    "fac": float(t * 10),
                    "wh": float(t * 20),
                    "sink": float(t * 30),
                },
                "node_inventory": {
                    # Factory uses _total key only.
                    "fac": {"_total": 100 - t * 5},
                    # Intermediate uses per-pid keys.
                    "wh": {pid: 30 - t * 3},
                    # Sink has empty inventory.
                    "sink": {},
                },
                "node_pending": {
                    "fac": {},
                    "wh": {pid: t * 2},
                    "sink": {},
                },
                "node_orders": {
                    "fac": {},
                    "wh": {pid: t * 1},
                    "sink": {},
                },
            }
        )

    return {
        "n_steps": n_ticks,
        "ticks": ticks,
        "global": {
            "time": {
                "simulation_step": list(range(n_ticks + 1)),
                "simulation_date": [datetime(2024, 1, d + 1) for d in range(n_ticks + 1)],
            },
            "market_supply": {"EU": [1.0] * (n_ticks + 1)},
            "market_demand": {"EU": [1.5] * (n_ticks + 1)},
            "products": {},
            "events": {"occurrences": [None] * n_ticks},
        },
    }


# ---------------------------------------------------------------------------
# node_timeseries_df
# ---------------------------------------------------------------------------


def test_node_timeseries_columns():
    """node_timeseries_df returns exactly the required columns."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_timeseries_df(run_log, scenario)
    expected_cols = {
        "tick", "node_id", "node_type", "region", "level",
        "cash", "inventory_total", "pending_total", "orders_total",
    }
    assert set(df.columns) == expected_cols


def test_node_timeseries_row_count():
    """One row per (tick, node) — 3 ticks × 3 nodes = 9 rows."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_timeseries_df(run_log, scenario)
    assert len(df) == 9


def test_node_timeseries_node_type_join():
    """node_type is populated from scenario metadata for all three node types."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_timeseries_df(run_log, scenario)

    fac_rows = df[df["node_id"] == "fac"]
    wh_rows = df[df["node_id"] == "wh"]
    sink_rows = df[df["node_id"] == "sink"]

    assert (fac_rows["node_type"] == "FactoryNode").all()
    assert (wh_rows["node_type"] == "IntermediateNode").all()
    assert (sink_rows["node_type"] == "DemandSinkNode").all()


def test_node_timeseries_region_join():
    """region is populated from scenario metadata."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_timeseries_df(run_log, scenario)
    assert (df["region"] == "EU").all()


def test_node_timeseries_level_join():
    """level is populated correctly from scenario node metadata."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_timeseries_df(run_log, scenario)

    assert (df[df["node_id"] == "fac"]["level"] == 0).all()
    assert (df[df["node_id"] == "wh"]["level"] == 1).all()
    assert (df[df["node_id"] == "sink"]["level"] == 2).all()


def test_node_timeseries_factory_inventory_total():
    """Factory inventory_total comes from the _total key."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_timeseries_df(run_log, scenario)

    fac_rows = df[df["node_id"] == "fac"].sort_values("tick")
    # tick 1: _total = 95, tick 2: _total = 90, tick 3: _total = 85
    assert list(fac_rows["inventory_total"]) == [95, 90, 85]


def test_node_timeseries_intermediate_inventory_total():
    """IntermediateNode inventory_total is the sum of per-pid values."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_timeseries_df(run_log, scenario)

    wh_rows = df[df["node_id"] == "wh"].sort_values("tick")
    # tick 1: {pid: 27}, tick 2: {pid: 24}, tick 3: {pid: 21}
    assert list(wh_rows["inventory_total"]) == [27, 24, 21]


def test_node_timeseries_sink_inventory_zero():
    """DemandSinkNode has empty inventory dict -> inventory_total == 0."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_timeseries_df(run_log, scenario)
    assert (df[df["node_id"] == "sink"]["inventory_total"] == 0).all()


def test_node_timeseries_pending_and_orders():
    """pending_total and orders_total are summed per tick correctly."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_timeseries_df(run_log, scenario)

    wh_rows = df[df["node_id"] == "wh"].sort_values("tick")
    # pending: tick 1→2, tick 2→4, tick 3→6
    assert list(wh_rows["pending_total"]) == [2, 4, 6]
    # orders: tick 1→1, tick 2→2, tick 3→3
    assert list(wh_rows["orders_total"]) == [1, 2, 3]


def test_node_timeseries_cash():
    """cash column reflects node_cash values from run-log."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_timeseries_df(run_log, scenario)

    fac_rows = df[df["node_id"] == "fac"].sort_values("tick")
    assert list(fac_rows["cash"]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# global_timeseries_df
# ---------------------------------------------------------------------------


def test_global_timeseries_columns():
    """global_timeseries_df returns tick + per-region supply/demand columns."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = global_timeseries_df(run_log)
    expected_cols = {"tick", "market_supply_EU", "market_demand_EU"}
    assert set(df.columns) == expected_cols


def test_global_timeseries_row_count():
    """Includes step 0 pre-run snapshot: n_steps + 1 rows (4 for 3-step run)."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = global_timeseries_df(run_log)
    assert len(df) == 4  # steps 0..3


def test_global_timeseries_values():
    """Values match the hand-crafted series."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=2)
    df = global_timeseries_df(run_log)
    # All supply values are 1.0; all demand values are 1.5.
    assert (df["market_supply_EU"] == 1.0).all()
    assert (df["market_demand_EU"] == 1.5).all()


# ---------------------------------------------------------------------------
# per_product_df
# ---------------------------------------------------------------------------


def test_per_product_columns():
    """per_product_df returns the required columns."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = per_product_df(run_log, scenario, "wh")
    expected_cols = {"tick", "node_id", "pid", "inventory", "pending", "orders"}
    assert set(df.columns) == expected_cols


def test_per_product_row_count_intermediate():
    """Intermediate node: one row per (tick, pid) = 3 ticks × 1 product = 3."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = per_product_df(run_log, scenario, "wh")
    assert len(df) == 3
    assert (df["pid"] == pid).all()


def test_per_product_no_total_key():
    """per_product_df never emits a row with pid == '_total'.

    The factory inventory dict only contains the synthetic '_total' key;
    it should be suppressed, leaving an empty result (no rows, no '_total' pid).
    """
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    # Check factory node which uses _total key — result must be empty.
    df = per_product_df(run_log, scenario, "fac")
    assert len(df) == 0 or "_total" not in df["pid"].values


def test_per_product_factory_empty():
    """Factory has only _total in inventory → per_product_df returns empty DataFrame."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = per_product_df(run_log, scenario, "fac")
    assert len(df) == 0


def test_per_product_sink_empty():
    """Sink has empty inventory → per_product_df returns empty DataFrame."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = per_product_df(run_log, scenario, "sink")
    assert len(df) == 0


def test_per_product_values():
    """Inventory, pending, orders values match the run-log for known ticks."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = per_product_df(run_log, scenario, "wh").sort_values("tick")
    # tick 1: inventory=27, pending=2, orders=1
    row0 = df.iloc[0]
    assert row0["tick"] == 1
    assert row0["inventory"] == 27
    assert row0["pending"] == 2
    assert row0["orders"] == 1


def test_per_product_node_id_column():
    """node_id column is the requested node_id in every row."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = per_product_df(run_log, scenario, "wh")
    assert (df["node_id"] == "wh").all()


# ---------------------------------------------------------------------------
# node_equity
# ---------------------------------------------------------------------------


def test_node_equity_columns():
    """node_equity returns the required columns."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid)
    df = node_equity(run_log, scenario, "wh")
    expected_cols = {
        "tick", "node_id", "cash", "inventory_value",
        "outstanding_value", "equity",
    }
    assert set(df.columns) == expected_cols


def test_node_equity_row_count():
    """One row per tick."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=3)
    df = node_equity(run_log, scenario, "wh")
    assert len(df) == 3


def test_node_equity_arithmetic_intermediate():
    """equity = cash + inventory_value + outstanding_value (unit_cost=4.0)."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=1)
    df = node_equity(run_log, scenario, "wh")
    row = df.iloc[0]
    # tick 1: cash=20, inventory={pid:27}→27*4=108, pending={pid:2}→2*4=8
    assert row["cash"] == pytest.approx(20.0)
    assert row["inventory_value"] == pytest.approx(27 * 4.0)
    assert row["outstanding_value"] == pytest.approx(2 * 4.0)
    assert row["equity"] == pytest.approx(20.0 + 27 * 4.0 + 2 * 4.0)


def test_node_equity_arithmetic_factory():
    """Factory equity uses node.unit_cost for the _total inventory."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=1)
    df = node_equity(run_log, scenario, "fac")
    row = df.iloc[0]
    # tick 1: cash=10, inventory=_total:95→95*4=380, pending={}→0
    assert row["cash"] == pytest.approx(10.0)
    assert row["inventory_value"] == pytest.approx(95 * 4.0)
    assert row["outstanding_value"] == pytest.approx(0.0)
    assert row["equity"] == pytest.approx(10.0 + 95 * 4.0)


def test_node_equity_arithmetic_sink():
    """Sink has no inventory → inventory_value and outstanding_value are 0."""
    scenario, pid = _make_scenario()
    run_log = _make_run_log(pid, n_ticks=1)
    df = node_equity(run_log, scenario, "sink")
    row = df.iloc[0]
    assert row["inventory_value"] == pytest.approx(0.0)
    assert row["outstanding_value"] == pytest.approx(0.0)
    assert row["equity"] == pytest.approx(row["cash"])
