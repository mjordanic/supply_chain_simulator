"""DataExporter graph-mode flow column population (ADR 0019, issue 08).

Verifies that ``timeseries.parquet`` written from a graph-mode run log has
non-null ``sales`` / ``demand`` / ``price`` / ``revenue`` columns for nodes
that appear in the per-tick flow log, and that the exported values reconcile
exactly with the ``flow_frame`` view of the same run log.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.sim.data_exporter import DataExporter
from src.sim.inspect import flow_frame
from src.sim.runner import Runner


def _make_chain_scenario():
    """Factory → Warehouse → Sink, one product, 5 ticks."""
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
    pid = catalog[0].product_id

    stages = ["introduction", "growth", "maturity", "decline", "dead"]

    factory = FactoryNode(
        id="fac", region="EU", init_seed=1,
        produces_product_id=pid, unit_cost=4.0, capacity_per_tick=50,
        inventory=100, list_price=4.0, cash=0.0,
    )
    warehouse = IntermediateNode(
        id="wh", region="EU", init_seed=2,
        carried_products={pid}, capacity=200, tags=["warehouse"],
        inventory={pid: 30}, pending={}, list_prices={pid: 7.0},
        min_order_imposed={pid: 0}, cash=200.0,
    )
    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=3,
        product_id=pid, demand_dist=Constant(5), income_rate=50.0, cash=500.0,
    )

    return Scenario(
        catalog=catalog,
        market=MarketParams(
            cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
            peak_factor=1.0, off_factor=1.0, season_months={}, regions=["EU"],
            correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
            stage_multipliers={s: 1.0 for s in stages}, price_elasticity=0.0,
            promo_multiplier=1.0, demand_factor_min=0.1, supply_factor_min=0.01,
            cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
            trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
            base_demand=Constant(5),
        ),
        disruption=DisruptionParams(
            event_prob=0.0, types=["flood"], regions=["EU"],
            severity=Constant(0.1), duration=Constant(1),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=stages, init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in stages},
        ),
        n_steps=5,
        start_date=datetime(2024, 1, 1),
        world_seed=1,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=warehouse, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=[
            EdgeSpec(supplier_id="fac", buyer_id="wh", default_lead_time=1),
            EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
        ],
    )


def _run_and_export():
    """Run the chain scenario and export to a temp dir. Returns (ts_df, run_log, scenario)."""
    scenario = _make_chain_scenario()
    run_log = Runner(scenario).run()
    tmp = tempfile.mkdtemp()
    DataExporter(scenario, run_log).export_all(tmp)
    ts_df = pd.read_parquet(Path(tmp) / "data" / "timeseries.parquet")
    return ts_df, run_log, scenario


def test_warehouse_sales_demand_price_non_null():
    """Warehouse rows in graph-mode timeseries.parquet have non-null sales/demand/price/revenue."""
    ts_df, _, _ = _run_and_export()
    wh_rows = ts_df[ts_df["store_id"] == "wh"]
    assert len(wh_rows) > 0, "expected warehouse rows"
    assert wh_rows["sales"].notna().all(), "sales should be non-null for warehouse"
    assert wh_rows["demand"].notna().all(), "demand should be non-null for warehouse"
    assert wh_rows["price"].notna().all(), "price should be non-null for warehouse"
    assert wh_rows["revenue"].notna().all(), "revenue should be non-null for warehouse"


def test_revenue_equals_sales_times_price():
    """Exported revenue = sales × price for every non-null row."""
    ts_df, _, _ = _run_and_export()
    # Only rows where all three are populated.
    mask = ts_df["sales"].notna() & ts_df["price"].notna() & ts_df["revenue"].notna()
    rows = ts_df[mask]
    assert len(rows) > 0, "expected at least one non-null revenue row"
    expected = rows["sales"] * rows["price"]
    pd.testing.assert_series_equal(
        rows["revenue"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_flow_columns_reconcile_with_flow_frame():
    """Exported sales/demand/price in timeseries.parquet match flow_frame values."""
    ts_df, run_log, scenario = _run_and_export()
    ff = flow_frame(run_log)

    # Merge parquet and flow_frame on (store_id, simulation_step, product_id).
    # flow_frame uses 'tick' which equals simulation_step for graph-mode runs.
    merged = ts_df.merge(
        ff.rename(columns={"node_id": "store_id", "pid": "product_id", "tick": "simulation_step"}),
        on=["store_id", "simulation_step", "product_id"],
        suffixes=("_pq", "_ff"),
    )
    assert len(merged) > 0, "expected matched rows between parquet and flow_frame"

    # Non-null flow_frame rows must match the parquet.
    non_null_mask = merged["sales_ff"].notna()
    matched = merged[non_null_mask]
    assert len(matched) > 0, "expected at least some matched non-null rows"

    pd.testing.assert_series_equal(
        matched["sales_pq"].reset_index(drop=True).astype(float),
        matched["sales_ff"].reset_index(drop=True).astype(float),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        matched["demand_pq"].reset_index(drop=True).astype(float),
        matched["demand_ff"].reset_index(drop=True).astype(float),
        check_names=False,
    )
    # price in parquet vs price in flow_frame (NaN for sinks)
    price_pq = matched["price_pq"].reset_index(drop=True)
    price_ff = matched["price_ff"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        price_pq.astype(float),
        price_ff.astype(float),
        check_names=False,
    )
