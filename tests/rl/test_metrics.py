"""Tests for src/sim/metrics.py (formerly src/rl/metrics.py).

Covers:
  - service_level: hand-crafted slice with known sales/demand
  - stockout_rate: hand-crafted inventory with known zero-inventory fraction
  - profit_decomposition: net total adds back to balance delta from same slice
  - empty/zero-demand slices: no divide-by-zero, finite values
  - aggregate_episode: keys match documented schema; values agree with individual calls
  - mean_price_pct_of_msrp and inventory_turnover: formula correctness
"""

from __future__ import annotations

import math

import pytest

from src.sim.metrics import (
    RunSlice,
    aggregate_episode,
    inventory_turnover,
    mean_price_pct_of_msrp,
    profit_decomposition,
    service_level,
    stockout_rate,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal RunSlice objects
# ---------------------------------------------------------------------------


def _simple_slice(
    n_ticks: int = 5,
    n_skus: int = 2,
    sales_val: float = 10.0,
    demand_val: float = 15.0,
    inventory_val: float = 20.0,
    price_val: float = 18.0,
    msrp_val: float = 20.0,
    revenue_val: float | None = None,
    holding_val: float = 1.0,
    order_cost_val: float = 5.0,
    fee_val: float = 2.0,
) -> RunSlice:
    """Build a RunSlice with constant values across all (tick, sku) pairs."""
    if revenue_val is None:
        revenue_val = sales_val * price_val

    return RunSlice(
        sales=[[sales_val] * n_skus for _ in range(n_ticks)],
        demand=[[demand_val] * n_skus for _ in range(n_ticks)],
        inventory=[[inventory_val] * n_skus for _ in range(n_ticks)],
        price=[[price_val] * n_skus for _ in range(n_ticks)],
        msrp=[[msrp_val] * n_skus for _ in range(n_ticks)],
        revenue=[[revenue_val] * n_skus for _ in range(n_ticks)],
        holding_cost=[[holding_val] * n_skus for _ in range(n_ticks)],
        order_cost=[[order_cost_val] * n_skus for _ in range(n_ticks)],
        order_fee=[[fee_val] * n_skus for _ in range(n_ticks)],
    )


# ---------------------------------------------------------------------------
# service_level
# ---------------------------------------------------------------------------


def test_service_level_known_values():
    """sales=10, demand=15 across 5 ticks × 2 SKUs → 100/150 ≈ 0.6667."""
    rs = _simple_slice(n_ticks=5, n_skus=2, sales_val=10.0, demand_val=15.0)
    sl = service_level(rs)
    assert sl == pytest.approx(10.0 / 15.0, abs=1e-6)


def test_service_level_perfect_fulfillment():
    """sales == demand → service_level == 1.0."""
    rs = _simple_slice(sales_val=10.0, demand_val=10.0)
    assert service_level(rs) == pytest.approx(1.0)


def test_service_level_zero_demand():
    """Zero demand → service_level uses max(1, demand) ⇒ no divide-by-zero."""
    rs = _simple_slice(sales_val=0.0, demand_val=0.0)
    sl = service_level(rs)
    assert math.isfinite(sl)
    assert sl == pytest.approx(0.0)


def test_service_level_empty_slice():
    """Empty RunSlice → service_level returns 0.0 (finite)."""
    rs = RunSlice()
    sl = service_level(rs)
    assert math.isfinite(sl)
    assert sl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# stockout_rate
# ---------------------------------------------------------------------------


def test_stockout_rate_known_fraction():
    """K of T ticks have inventory=0; rate should be K/(T*n_skus)."""
    n_ticks = 10
    n_skus = 2
    # 3 ticks have zero inventory for ALL SKUs → 3*2 zeros out of 10*2=20
    inv_rows: list[list[float]] = []
    for t in range(n_ticks):
        if t < 3:
            inv_rows.append([0.0] * n_skus)
        else:
            inv_rows.append([10.0] * n_skus)
    rs = RunSlice(inventory=inv_rows)
    rate = stockout_rate(rs)
    assert rate == pytest.approx(6.0 / 20.0, abs=1e-6)


def test_stockout_rate_partial_skus():
    """Only some SKUs have zero inventory in a tick."""
    # 1 tick, 3 SKUs: [0, 5, 0] → 2 zeros / 3 total
    rs = RunSlice(inventory=[[0.0, 5.0, 0.0]])
    assert stockout_rate(rs) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_stockout_rate_no_stockouts():
    """All positive inventory → stockout_rate == 0.0."""
    rs = _simple_slice(inventory_val=10.0)
    assert stockout_rate(rs) == pytest.approx(0.0)


def test_stockout_rate_empty_slice():
    """Empty RunSlice → stockout_rate returns 0.0 (finite)."""
    rs = RunSlice()
    rate = stockout_rate(rs)
    assert math.isfinite(rate)
    assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# profit_decomposition
# ---------------------------------------------------------------------------


def test_profit_decomposition_net_equals_balance_delta():
    """net_profit == sum(revenue) - sum(holding_cost + order_cost + order_fee).

    We simulate the balance evolution manually and verify the net_profit
    from profit_decomposition equals the total balance change.
    """
    n_ticks = 4
    n_skus = 3
    revenue_per_cell = 30.0
    holding_per_cell = 2.0
    order_cost_per_cell = 8.0
    fee_per_cell = 5.0

    rs = _simple_slice(
        n_ticks=n_ticks,
        n_skus=n_skus,
        revenue_val=revenue_per_cell,
        holding_val=holding_per_cell,
        order_cost_val=order_cost_per_cell,
        fee_val=fee_per_cell,
    )

    decomp = profit_decomposition(rs)
    n_cells = n_ticks * n_skus
    expected_revenue = revenue_per_cell * n_cells
    expected_holding = holding_per_cell * n_cells
    expected_order_cost = order_cost_per_cell * n_cells
    expected_fees = fee_per_cell * n_cells
    expected_net = expected_revenue - (expected_holding + expected_order_cost + expected_fees)

    assert decomp["revenue"] == pytest.approx(expected_revenue, abs=1e-6)
    assert decomp["holding_cost"] == pytest.approx(expected_holding, abs=1e-6)
    assert decomp["order_cost"] == pytest.approx(expected_order_cost, abs=1e-6)
    assert decomp["order_fees"] == pytest.approx(expected_fees, abs=1e-6)
    assert decomp["net_profit"] == pytest.approx(expected_net, abs=1e-6)

    # Balance delta == sum of per-tick balance changes = sum(revenue - total_cost)
    balance_delta = (
        expected_revenue
        - expected_holding
        - expected_order_cost
        - expected_fees
    )
    assert decomp["net_profit"] == pytest.approx(balance_delta, abs=1e-6)


def test_profit_decomposition_keys():
    """profit_decomposition always returns exactly the expected keys."""
    rs = _simple_slice()
    keys = set(profit_decomposition(rs).keys())
    assert keys == {"revenue", "holding_cost", "order_cost", "order_fees", "net_profit"}


def test_profit_decomposition_empty_slice():
    """Empty RunSlice → all values are 0.0 (finite), no exceptions."""
    rs = RunSlice()
    decomp = profit_decomposition(rs)
    for k, v in decomp.items():
        assert math.isfinite(v), f"key={k} value={v} not finite"
    assert decomp["net_profit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_price_pct_of_msrp
# ---------------------------------------------------------------------------


def test_mean_price_pct_of_msrp_known_value():
    """price=18, msrp=20 → 18/20=0.9 for all cells → mean=0.9."""
    rs = _simple_slice(price_val=18.0, msrp_val=20.0)
    assert mean_price_pct_of_msrp(rs) == pytest.approx(0.9, abs=1e-6)


def test_mean_price_pct_varying():
    """Mix of price/msrp ratios: mean computed correctly."""
    # Tick 0: [1.0, 0.5]; Tick 1: [0.5, 1.0]  → mean = (1+0.5+0.5+1)/4 = 0.75
    rs = RunSlice(
        price=[[10.0, 5.0], [5.0, 10.0]],
        msrp=[[10.0, 10.0], [10.0, 10.0]],
    )
    assert mean_price_pct_of_msrp(rs) == pytest.approx(0.75, abs=1e-6)


def test_mean_price_pct_empty_slice():
    """Empty RunSlice → returns 1.0 (graceful fallback, finite)."""
    rs = RunSlice()
    v = mean_price_pct_of_msrp(rs)
    assert math.isfinite(v)
    assert v == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# inventory_turnover
# ---------------------------------------------------------------------------


def test_inventory_turnover_known_value():
    """sales=10 per cell, inventory=5 per cell; 5 ticks × 2 SKUs.
    total_sales = 100; mean_inv = 5.0 → turnover = 100/5 = 20.
    """
    rs = _simple_slice(n_ticks=5, n_skus=2, sales_val=10.0, inventory_val=5.0)
    assert inventory_turnover(rs) == pytest.approx(100.0 / 5.0, abs=1e-6)


def test_inventory_turnover_zero_inventory():
    """Zero mean inventory → uses max(1, mean_inv) = 1 → turnover = total_sales."""
    rs = _simple_slice(n_ticks=2, n_skus=1, sales_val=7.0, inventory_val=0.0)
    assert inventory_turnover(rs) == pytest.approx(14.0 / 1.0, abs=1e-6)


def test_inventory_turnover_empty_slice():
    """Empty RunSlice → returns 0.0 (finite)."""
    rs = RunSlice()
    v = inventory_turnover(rs)
    assert math.isfinite(v)
    assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregate_episode
# ---------------------------------------------------------------------------

_EXPECTED_AGGREGATE_KEYS = {
    "service_level",
    "stockout_rate",
    "mean_price_pct_of_msrp",
    "inventory_turnover",
    "revenue",
    "holding_cost",
    "order_cost",
    "order_fees",
    "net_profit",
}


def test_aggregate_episode_keys():
    """aggregate_episode returns exactly the documented keys."""
    rs = _simple_slice()
    assert set(aggregate_episode(rs).keys()) == _EXPECTED_AGGREGATE_KEYS


def test_aggregate_episode_values_agree_with_individual_calls():
    """aggregate_episode values match individual function calls on same slice."""
    rs = _simple_slice(
        n_ticks=6,
        n_skus=3,
        sales_val=8.0,
        demand_val=12.0,
        inventory_val=15.0,
        price_val=17.0,
        msrp_val=20.0,
        holding_val=1.5,
        order_cost_val=4.0,
        fee_val=3.0,
    )
    agg = aggregate_episode(rs)
    decomp = profit_decomposition(rs)

    assert agg["service_level"] == pytest.approx(service_level(rs))
    assert agg["stockout_rate"] == pytest.approx(stockout_rate(rs))
    assert agg["mean_price_pct_of_msrp"] == pytest.approx(mean_price_pct_of_msrp(rs))
    assert agg["inventory_turnover"] == pytest.approx(inventory_turnover(rs))
    assert agg["revenue"] == pytest.approx(decomp["revenue"])
    assert agg["holding_cost"] == pytest.approx(decomp["holding_cost"])
    assert agg["order_cost"] == pytest.approx(decomp["order_cost"])
    assert agg["order_fees"] == pytest.approx(decomp["order_fees"])
    assert agg["net_profit"] == pytest.approx(decomp["net_profit"])


def test_aggregate_episode_empty_slice_all_finite():
    """Empty RunSlice → all aggregate values are finite (no NaN/inf)."""
    rs = RunSlice()
    for k, v in aggregate_episode(rs).items():
        assert math.isfinite(v), f"aggregate key={k} value={v} not finite"


# ---------------------------------------------------------------------------
# Zero-demand edge cases (no divide-by-zero)
# ---------------------------------------------------------------------------


def test_all_zero_demand_no_exception():
    """Slice where all demand is 0 — must not raise."""
    rs = RunSlice(
        sales=[[0.0, 0.0]] * 10,
        demand=[[0.0, 0.0]] * 10,
        inventory=[[0.0, 0.0]] * 10,
        price=[[20.0, 20.0]] * 10,
        msrp=[[20.0, 20.0]] * 10,
        revenue=[[0.0, 0.0]] * 10,
        holding_cost=[[0.0, 0.0]] * 10,
        order_cost=[[0.0, 0.0]] * 10,
        order_fee=[[0.0, 0.0]] * 10,
    )
    agg = aggregate_episode(rs)
    for k, v in agg.items():
        assert math.isfinite(v), f"key={k} value={v} not finite on zero-demand slice"
    assert agg["service_level"] == pytest.approx(0.0)
    assert agg["stockout_rate"] == pytest.approx(1.0)  # all inventory=0 → all stockouts
