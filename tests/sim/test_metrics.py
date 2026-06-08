"""Tests for src/sim/metrics.py — DataFrame-native operational metrics (ADR 0019).

Covers:
  - business_metrics: per-node and system-wide KPI rows returned
  - service_level: known sales/demand ratio
  - stockout_rate: uses the decision-time stockout boolean
  - inventory_turnover: sum(sales) / max(1, mean(inventory_total))
  - mean_price_pct_of_msrp: price / base_price mean
  - value-preserving: KPI values equal the RunSlice-era values for the same scenario
  - system roll-up: aggregates all selling nodes correctly
  - empty frame: returns empty DataFrame without exception
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from src.sim.metrics import business_metrics


# ---------------------------------------------------------------------------
# Helpers to build minimal flow_frame and node_timeseries_df
# ---------------------------------------------------------------------------


def _make_flow_frame(rows: list[dict]) -> pd.DataFrame:
    cols = ["tick", "node_id", "pid", "sales", "demand", "price", "stockout"]
    return pd.DataFrame(rows, columns=cols)


def _make_ts_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["tick", "node_id", "node_type", "region", "level", "cash",
            "inventory_total", "pending_total", "orders_total"]
    return pd.DataFrame(rows, columns=cols)


class _FakeWare:
    def __init__(self, product_id, base_price):
        self.product_id = product_id
        self.base_price = base_price


class _FakeScenario:
    def __init__(self, catalog):
        self.catalog = catalog


def _simple_scenario():
    return _FakeScenario([_FakeWare("P0001", 20.0), _FakeWare("P0002", 30.0)])


# ---------------------------------------------------------------------------
# service_level
# ---------------------------------------------------------------------------


def test_service_level_known_values():
    """sales=10, demand=15 -> service_level ~= 0.6667 for node A."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": t, "node_id": "A", "pid": "P0001",
         "sales": 10, "demand": 15, "price": 18.0, "stockout": False}
        for t in range(1, 6)
    ])
    ts = _make_ts_df([
        {"tick": t, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 20, "pending_total": 0, "orders_total": 0}
        for t in range(1, 6)
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["service_level"] == pytest.approx(10.0 / 15.0, abs=1e-6)


def test_service_level_perfect_fulfillment():
    """sales == demand -> service_level == 1.0."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 10.0, "stockout": False}
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["service_level"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# stockout_rate -- uses the stockout boolean, not inventory == 0
# ---------------------------------------------------------------------------


def test_stockout_rate_uses_boolean():
    """stockout_rate is the fraction of (node, pid, tick) rows where stockout==True."""
    sc = _simple_scenario()
    rows = [
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 0, "demand": 5, "price": 10.0, "stockout": True},
        {"tick": 2, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 10.0, "stockout": False},
        {"tick": 3, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 10.0, "stockout": False},
        {"tick": 4, "node_id": "A", "pid": "P0001",
         "sales": 0, "demand": 5, "price": 10.0, "stockout": True},
    ]
    ff = _make_flow_frame(rows)
    ts = _make_ts_df([
        {"tick": t, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
        for t in range(1, 5)
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    # 2 out of 4 rows have stockout == True.
    assert row_a["stockout_rate"] == pytest.approx(0.5, abs=1e-6)


def test_stockout_rate_no_stockouts():
    """All stockout==False -> stockout_rate == 0.0."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 10.0, "stockout": False},
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["stockout_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# inventory_turnover
# ---------------------------------------------------------------------------


def test_inventory_turnover_known_value():
    """total_sales=50, mean_inventory=5 -> turnover=10."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": t, "node_id": "A", "pid": "P0001",
         "sales": 10, "demand": 10, "price": 10.0, "stockout": False}
        for t in range(1, 6)
    ])
    ts = _make_ts_df([
        {"tick": t, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 5, "pending_total": 0, "orders_total": 0}
        for t in range(1, 6)
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    # total_sales=50, mean_inv=5 -> 50/5=10.
    assert row_a["inventory_turnover"] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# mean_price_pct_of_msrp
# ---------------------------------------------------------------------------


def test_mean_price_pct_known_value():
    """price=18, msrp=20 (base_price) -> mean_price_pct ~= 0.9."""
    sc = _FakeScenario([_FakeWare("P0001", 20.0)])
    ff = _make_flow_frame([
        {"tick": t, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 18.0, "stockout": False}
        for t in range(1, 4)
    ])
    ts = _make_ts_df([
        {"tick": t, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
        for t in range(1, 4)
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["mean_price_pct_of_msrp"] == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# System-wide roll-up
# ---------------------------------------------------------------------------


def test_system_row_exists():
    """business_metrics always returns a '_system' roll-up row."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 10, "price": 10.0, "stockout": False},
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
    ])
    result = business_metrics(ff, ts, sc)
    assert "_system" in result["node_id"].values


def test_system_service_level_aggregates():
    """'_system' service_level is total_sales / total_demand across all nodes."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 10, "price": 10.0, "stockout": False},
        {"tick": 1, "node_id": "B", "pid": "P0001",
         "sales": 8, "demand": 10, "price": 10.0, "stockout": False},
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": n, "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
        for n in ["A", "B"]
    ])
    result = business_metrics(ff, ts, sc)
    system = result[result["node_id"] == "_system"].iloc[0]
    # total_sales=13, total_demand=20 -> 0.65
    assert system["service_level"] == pytest.approx(13.0 / 20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Demand sinks are excluded (price is None)
# ---------------------------------------------------------------------------


def test_demand_sinks_excluded_from_selling_nodes():
    """Rows with price==None (demand sinks) do not appear in KPI output."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "shop", "pid": "P0001",
         "sales": 5, "demand": 10, "price": 10.0, "stockout": False},
        {"tick": 1, "node_id": "sink", "pid": "P0001",
         "sales": 0, "demand": 10, "price": None, "stockout": False},
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": n, "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
        for n in ["shop", "sink"]
    ])
    result = business_metrics(ff, ts, sc)
    assert "sink" not in result["node_id"].values
    assert "shop" in result["node_id"].values


# ---------------------------------------------------------------------------
# Empty frame
# ---------------------------------------------------------------------------


def test_empty_flow_frame_returns_empty_dataframe():
    """Empty flow_frame -> empty result with correct columns, no exception."""
    sc = _simple_scenario()
    ff = _make_flow_frame([])
    ts = _make_ts_df([])
    result = business_metrics(ff, ts, sc)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "node_id", "service_level", "stockout_rate",
        "inventory_turnover", "mean_price_pct_of_msrp",
    }
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Value-preserving regression: compare against the RunSlice-era values
# ---------------------------------------------------------------------------


def test_value_preserving_vs_run_slice_era():
    """KPI values match the RunSlice-era values for the same hand-checkable scenario.

    Scenario: single selling node, 1 product, 5 ticks.
      sales=10, demand=15, price=18, stockout=False, inventory=20 each tick.
    Expected (same as old RunSlice formulas):
      service_level  = 50/75 ~= 0.6667
      stockout_rate  = 0/5 = 0.0
      inventory_turn = 50/20 = 2.5
      price_pct_msrp = 18/20 = 0.9
    """
    sc = _FakeScenario([_FakeWare("P0001", 20.0)])
    n_ticks = 5
    ff = _make_flow_frame([
        {"tick": t, "node_id": "shop", "pid": "P0001",
         "sales": 10, "demand": 15, "price": 18.0, "stockout": False}
        for t in range(1, n_ticks + 1)
    ])
    ts = _make_ts_df([
        {"tick": t, "node_id": "shop", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 20, "pending_total": 0, "orders_total": 0}
        for t in range(1, n_ticks + 1)
    ])
    result = business_metrics(ff, ts, sc)
    row = result[result["node_id"] == "shop"].iloc[0]

    assert row["service_level"] == pytest.approx(50.0 / 75.0, abs=1e-6)
    assert row["stockout_rate"] == pytest.approx(0.0, abs=1e-6)
    assert row["inventory_turnover"] == pytest.approx(50.0 / 20.0, abs=1e-6)
    assert row["mean_price_pct_of_msrp"] == pytest.approx(18.0 / 20.0, abs=1e-6)


def test_all_values_are_finite():
    """All KPI values in the output are finite (no NaN/inf)."""
    sc = _simple_scenario()
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 0, "demand": 0, "price": 0.0, "stockout": True},
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 0, "pending_total": 0, "orders_total": 0}
    ])
    result = business_metrics(ff, ts, sc)
    for col in ["service_level", "stockout_rate", "inventory_turnover", "mean_price_pct_of_msrp"]:
        for val in result[col]:
            assert math.isfinite(val), f"column {col} has non-finite value {val}"
