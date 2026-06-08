"""Tests for src/sim/metrics.py — DataFrame-native operational metrics (ADR 0019).

Mirrors tests/sim/test_metrics.py; exists to catch any RL-path import regression.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.sim.metrics import business_metrics


def _make_flow_frame(rows):
    cols = ["tick", "node_id", "pid", "sales", "demand", "price", "stockout"]
    return pd.DataFrame(rows, columns=cols)


def _make_ts_df(rows):
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
    return _FakeScenario([_FakeWare("P0001", 20.0)])


def test_service_level_known_values():
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


def test_stockout_rate_uses_boolean():
    sc = _simple_scenario()
    rows = [
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 0, "demand": 5, "price": 10.0, "stockout": True},
        {"tick": 2, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 10.0, "stockout": False},
    ]
    ff = _make_flow_frame(rows)
    ts = _make_ts_df([
        {"tick": t, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
        for t in range(1, 3)
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["stockout_rate"] == pytest.approx(0.5, abs=1e-6)


def test_inventory_turnover_known_value():
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
    assert row_a["inventory_turnover"] == pytest.approx(50.0 / 5.0, abs=1e-6)


def test_mean_price_pct_known_value():
    sc = _FakeScenario([_FakeWare("P0001", 20.0)])
    ff = _make_flow_frame([
        {"tick": 1, "node_id": "A", "pid": "P0001",
         "sales": 5, "demand": 5, "price": 18.0, "stockout": False}
    ])
    ts = _make_ts_df([
        {"tick": 1, "node_id": "A", "node_type": "IntermediateNode",
         "region": "US", "level": 1, "cash": 0.0,
         "inventory_total": 10, "pending_total": 0, "orders_total": 0}
    ])
    result = business_metrics(ff, ts, sc)
    row_a = result[result["node_id"] == "A"].iloc[0]
    assert row_a["mean_price_pct_of_msrp"] == pytest.approx(0.9, abs=1e-6)


def test_empty_flow_frame():
    sc = _simple_scenario()
    ff = _make_flow_frame([])
    ts = _make_ts_df([])
    result = business_metrics(ff, ts, sc)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_all_finite():
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
            assert math.isfinite(val), f"{col} = {val}"
