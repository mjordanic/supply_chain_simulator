"""Tests for min-order-aware routing in _default_sink_action.

Issue 04: _default_sink_action must skip a supplier whose min_order exceeds the
quantity the sink can route there, falling through to the next feasible supplier.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.sim.central_table import CentralTable, Offer
from src.sim.distributions import Constant
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.runner import _default_sink_action
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    Ware,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sink(product_id: str = "P0000", cash: float = 1000.0) -> DemandSinkNode:
    return DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=1,
        product_id=product_id,
        demand_dist=Constant(5),
        income_rate=100.0,
        cash=cash,
    )


def _make_table(
    pid: str,
    *,
    warehouse_qty: int = 100,
    warehouse_price: float = 5.0,
    warehouse_min_order: int = 20,
    shop_qty: int = 100,
    shop_price: float = 8.0,
    shop_min_order: int = 1,
) -> CentralTable:
    """Two suppliers: cheap warehouse (high min_order) + pricier shop (low min_order)."""
    table = CentralTable()
    table.publish(
        "warehouse",
        pid,
        Offer(available_qty=warehouse_qty, list_price=warehouse_price, min_order=warehouse_min_order),
    )
    table.publish(
        "shop",
        pid,
        Offer(available_qty=shop_qty, list_price=shop_price, min_order=shop_min_order),
    )
    return table


# ---------------------------------------------------------------------------
# Unit tests for _default_sink_action
# ---------------------------------------------------------------------------

class TestDefaultSinkActionMinOrder:
    """_default_sink_action must skip suppliers whose min_order can't be met."""

    def test_buys_from_cheapest_when_min_order_met(self):
        """When demand_target >= warehouse min_order, buys from the cheap warehouse."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        # demand_target=30 >= warehouse min_order=20 → should buy from warehouse
        table = _make_table(pid, warehouse_min_order=20)
        action = _default_sink_action(
            sink, 30.0, table,
            allowed_supplier_ids={"warehouse", "shop"},
        )
        buys = dict(action["buy"])
        assert "warehouse" in buys, "Expected to buy from cheap warehouse when min_order met"
        assert buys["warehouse"] > 0

    def test_skips_warehouse_when_below_min_order(self):
        """When demand_target < warehouse min_order, should skip to shop."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        # demand_target=5 < warehouse min_order=20 → must skip warehouse, buy from shop
        table = _make_table(pid, warehouse_min_order=20, shop_min_order=1)
        action = _default_sink_action(
            sink, 5.0, table,
            allowed_supplier_ids={"warehouse", "shop"},
        )
        buys = dict(action["buy"])
        assert "warehouse" not in buys, "Should not buy from warehouse when below min_order"
        assert buys.get("shop", 0) > 0, "Should fall through to shop when warehouse min_order is too high"

    def test_no_silent_lost_sale_when_shop_can_fill(self):
        """Demand is met (not a lost sale) when the shop can fill it."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        table = _make_table(pid, warehouse_min_order=20, shop_min_order=1, shop_qty=100)
        action = _default_sink_action(
            sink, 5.0, table,
            allowed_supplier_ids={"warehouse", "shop"},
        )
        total_filled = sum(qty for _, qty in action["buy"])
        assert total_filled == 5, f"Expected 5 units filled, got {total_filled}"

    def test_unmet_demand_when_no_feasible_supplier(self):
        """When all suppliers have min_order > demand_target, demand is genuinely unmet."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        table = CentralTable()
        table.publish(
            "warehouse", pid,
            Offer(available_qty=100, list_price=5.0, min_order=50),
        )
        table.publish(
            "shop", pid,
            Offer(available_qty=100, list_price=8.0, min_order=30),
        )
        # demand_target=5 < both min_orders (50, 30) → no fill
        action = _default_sink_action(
            sink, 5.0, table,
            allowed_supplier_ids={"warehouse", "shop"},
        )
        total_filled = sum(qty for _, qty in action["buy"])
        assert total_filled == 0, "No spurious fill when no feasible supplier exists"

    def test_partial_fill_when_only_shop_can_fill_partial(self):
        """Shop with low stock fills what it can; no extra from warehouse."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        table = CentralTable()
        table.publish(
            "warehouse", pid,
            Offer(available_qty=100, list_price=5.0, min_order=20),
        )
        table.publish(
            "shop", pid,
            Offer(available_qty=3, list_price=8.0, min_order=1),
        )
        action = _default_sink_action(
            sink, 10.0, table,
            allowed_supplier_ids={"warehouse", "shop"},
        )
        buys = dict(action["buy"])
        assert "warehouse" not in buys
        assert buys.get("shop", 0) == 3

    def test_min_order_zero_never_skips(self):
        """A supplier with min_order=0 is never skipped regardless of demand_target."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        table = CentralTable()
        table.publish(
            "factory", pid,
            Offer(available_qty=100, list_price=5.0, min_order=0),
        )
        action = _default_sink_action(
            sink, 1.0, table,
            allowed_supplier_ids={"factory"},
        )
        buys = dict(action["buy"])
        assert buys.get("factory", 0) > 0

    def test_buys_cheapest_first_after_fallthrough(self):
        """After skipping a too-expensive-min-order warehouse, buys cheapest feasible."""
        pid = "P0000"
        sink = _make_sink(product_id=pid)
        table = CentralTable()
        # warehouse (cheapest) has high min_order
        table.publish("warehouse", pid, Offer(available_qty=100, list_price=3.0, min_order=50))
        # shop-a (mid-price) is feasible
        table.publish("shop-a", pid, Offer(available_qty=100, list_price=6.0, min_order=1))
        # shop-b (expensive) is also feasible
        table.publish("shop-b", pid, Offer(available_qty=100, list_price=10.0, min_order=1))
        action = _default_sink_action(
            sink, 5.0, table,
            allowed_supplier_ids={"warehouse", "shop-a", "shop-b"},
        )
        buys = dict(action["buy"])
        # Should buy from shop-a (cheapest feasible), not shop-b
        assert "warehouse" not in buys
        assert buys.get("shop-a", 0) == 5
        assert "shop-b" not in buys


# ---------------------------------------------------------------------------
# Integration test: warehouse→sink / shop→sink lateral scenario
# ---------------------------------------------------------------------------

def _minimal_market_params() -> MarketParams:
    from src.sim.distributions import Constant, Normal
    return MarketParams(
        cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
        peak_factor=1.0, off_factor=1.0, season_months={}, regions=["US"],
        correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
        stage_multipliers={
            "introduction": 1.0, "growth": 1.0, "maturity": 1.0,
            "decline": 1.0, "dead": 0.0,
        },
        price_elasticity=0.0, promo_multiplier=1.0,
        demand_factor_min=0.1, supply_factor_min=0.01,
        cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
        base_demand=Constant(10),
    )


def _minimal_disruption_params() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0, types=["natural_disaster"], regions=["US"],
        severity=Constant(0.1), duration=Constant(1),
    )


def _minimal_lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages, init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _minimal_catalog() -> list[Ware]:
    return load_catalog([{
        "name": "Widget", "category": "test", "related_products": [],
        "base_price": 10.0, "unit_cost": 5.0, "seasonality": "all_season",
    }])


def test_low_demand_sink_routes_to_shop_not_warehouse():
    """Integration: a sink with low demand (below warehouse min_order) fills from the shop.

    Topology: factory → warehouse → sink
                      ↘ shop ↗

    Warehouse has low price + high min_order=20; shop has high price + min_order=1.
    Sink demand=5 (< 20) → must fill from shop, not lose the sale.
    """
    catalog = _minimal_catalog()
    pid = catalog[0].product_id

    factory = FactoryNode(
        id="factory",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=5.0,
        capacity_per_tick=200,
        inventory=500,
        list_price=5.0,
        cash=0.0,
    )
    warehouse = IntermediateNode(
        id="warehouse",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=1000,
        tags=["warehouse"],
        inventory={pid: 500},
        pending={},
        list_prices={pid: 6.0},
        min_order_imposed={pid: 20},  # HIGH min_order — sink can't meet this
        cash=5000.0,
    )
    shop = IntermediateNode(
        id="shop",
        region="US",
        init_seed=3,
        carried_products={pid},
        capacity=1000,
        tags=["shop"],
        inventory={pid: 500},
        pending={},
        list_prices={pid: 9.0},
        min_order_imposed={pid: 1},  # LOW min_order — sink can buy here
        cash=5000.0,
    )
    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=4,
        product_id=pid,
        demand_dist=Constant(5),   # demand=5 < warehouse min_order=20
        income_rate=200.0,
        cash=5000.0,
    )

    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="warehouse", default_lead_time=1),
        EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=1),
        EdgeSpec(supplier_id="warehouse", buyer_id="sink", default_lead_time=1),
        EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=1),
    ]

    from src.sim.runner import Runner, build_world
    scenario = Scenario(
        catalog=catalog,
        market=_minimal_market_params(),
        disruption=_minimal_disruption_params(),
        item_lifecycle=_minimal_lifecycle_params(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=warehouse, init_seed=2),
            NodeInstance(node=shop, init_seed=3),
            NodeInstance(node=sink, init_seed=4),
        ],
        edges=edges,
    )

    # Run the simulation for 10 ticks and confirm sink cash stays non-negative
    # (it can buy), which means it was filling from the shop (not losing sales).
    log = Runner(scenario).run()
    assert len(log["ticks"]) == 10

    # The sink should have a non-negative cash balance (it earned income and spent some)
    for tick_log in log["ticks"]:
        sink_cash = tick_log["node_cash"].get("sink", 0)
        assert sink_cash >= 0, f"Sink cash negative at tick {tick_log['tick']}"

    # Verify no negative inventory anywhere
    for tick_log in log["ticks"]:
        for node_id, inv in tick_log["node_inventory"].items():
            if isinstance(inv, dict):
                for p, qty in inv.items():
                    assert qty >= 0, f"{node_id}[{p}] went negative"
