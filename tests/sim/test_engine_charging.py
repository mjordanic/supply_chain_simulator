"""Engine charging correctness (ADR 0019, issue 03).

Tests assert:
- Holding cost equals closing_on_hand × holding_rate × unit_cost per pid.
- Holding cost is only charged to IntermediateNodes (factories/sinks unaffected).
- A multi-SKU order to one supplier incurs exactly one order_fee.
- Ordering from two suppliers in one tick incurs two order_fees.
- Order fee is only charged to the ordering intermediate.
- Operational KPIs (service/stockout) are not shifted by charging.
"""

from __future__ import annotations

from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Scenario builder helpers
# ---------------------------------------------------------------------------

def _catalog():
    from src.sim.scenario import load_catalog
    return load_catalog(
        [
            {
                "name": "Alpha",
                "category": "goods",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            },
            {
                "name": "Beta",
                "category": "goods",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 6.0,
                "seasonality": "all_season",
            },
        ]
    )


def _flat_market():
    from src.sim.distributions import Constant
    from src.sim.scenario import MarketParams
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return MarketParams(
        cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
        peak_factor=1.0, off_factor=1.0, season_months={}, regions=["EU"],
        correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
        stage_multipliers={s: 1.0 for s in stages}, price_elasticity=0.0,
        promo_multiplier=1.0, demand_factor_min=0.1, supply_factor_min=0.01,
        cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
        base_demand=Constant(0),
    )


def _flat_disruption():
    from src.sim.distributions import Constant
    from src.sim.scenario import DisruptionParams
    return DisruptionParams(
        event_prob=0.0, types=["flood"], regions=["EU"],
        severity=Constant(0.1), duration=Constant(1),
    )


def _flat_lifecycle():
    from src.sim.scenario import ItemLifecycleParams
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages, init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _make_scenario(
    *,
    wh_inventory: dict,
    holding_rate: float,
    order_fee: float,
    wh_cash: float = 10_000.0,
    n_steps: int = 1,
    demand: int = 0,
    n_suppliers: int = 1,
    wh_policy=None,
):
    """Build a minimal factory → warehouse → sink scenario.

    All pids must be real product_id values from _catalog(); the caller
    must build the catalog first to get them.
    """
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario

    catalog = _catalog()
    pids = [w.product_id for w in catalog]
    p0, p1 = pids[0], pids[1]

    # Rebuild market with correct demand constant
    from src.sim.scenario import MarketParams
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    market = MarketParams(
        cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
        peak_factor=1.0, off_factor=1.0, season_months={}, regions=["EU"],
        correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
        stage_multipliers={s: 1.0 for s in stages}, price_elasticity=0.0,
        promo_multiplier=1.0, demand_factor_min=0.1, supply_factor_min=0.01,
        cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
        base_demand=Constant(demand),
    )

    factory1 = FactoryNode(
        id="fac1", region="EU", init_seed=1,
        produces_product_id=p0, unit_cost=4.0, capacity_per_tick=0,
        inventory=100, list_price=4.0, cash=0.0,
    )
    wh = IntermediateNode(
        id="wh", region="EU", init_seed=2,
        carried_products=set(pids),
        capacity=500,
        tags=["warehouse"],
        inventory=dict(wh_inventory),
        pending={},
        list_prices={p0: 10.0, p1: 15.0},
        min_order_imposed={p: 0 for p in pids},
        cash=wh_cash,
        holding_rate=holding_rate,
        order_fee=order_fee,
    )
    if wh_policy is not None:
        wh.policy = wh_policy

    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=3,
        product_id=p0, demand_dist=Constant(demand), income_rate=50.0, cash=1000.0,
    )
    sink.policy = DefaultDemandSinkPolicy(policy_seed=4)

    nodes = [
        NodeInstance(node=factory1, init_seed=1),
        NodeInstance(node=wh, init_seed=2),
        NodeInstance(node=sink, init_seed=3),
    ]
    edges = [
        EdgeSpec(supplier_id="fac1", buyer_id="wh", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
    ]

    if n_suppliers == 2:
        factory2 = FactoryNode(
            id="fac2", region="EU", init_seed=11,
            produces_product_id=p1, unit_cost=6.0, capacity_per_tick=0,
            inventory=100, list_price=6.0, cash=0.0,
        )
        nodes.append(NodeInstance(node=factory2, init_seed=11))
        edges.append(EdgeSpec(supplier_id="fac2", buyer_id="wh", default_lead_time=1))

    return Scenario(
        catalog=catalog,
        market=market,
        disruption=_flat_disruption(),
        item_lifecycle=_flat_lifecycle(),
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
        nodes=nodes,
        edges=edges,
    )


class _OrderingPolicy:
    """Stub IntermediateNode policy that orders from specified suppliers."""

    def __init__(self, supplier_orders: list[tuple[str, str, int]]):
        """supplier_orders: list of (pid, supplier_id, qty)."""
        self._orders = supplier_orders

    def decide(self, obs, table):
        order: dict[str, list] = {}
        for pid, sid, qty in self._orders:
            order.setdefault(pid, []).append((sid, qty))
        return {"order": order, "list_price": {}, "min_order_imposed": {}}


# Get actual pids once for all tests.
_PIDS = [w.product_id for w in _catalog()]
P0, P1 = _PIDS[0], _PIDS[1]


# ---------------------------------------------------------------------------
# Holding cost
# ---------------------------------------------------------------------------


def test_holding_cost_charged_correctly():
    """Holding cost = closing_on_hand × holding_rate × unit_cost per pid.

    Two-product warehouse: p0 has 20 units (unit_cost=4), p1 has 10 units (unit_cost=6).
    No demand, no ordering.  Expected: 20*0.01*4 + 10*0.01*6 = 0.8 + 0.6 = 1.4
    """
    sc = _make_scenario(
        wh_inventory={P0: 20, P1: 10},
        holding_rate=0.01,
        order_fee=0.0,
        wh_cash=500.0,
        demand=0,
        n_steps=1,
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    wh_cash_after = log["ticks"][0]["node_cash"]["wh"]
    expected_holding = 20 * 0.01 * 4.0 + 10 * 0.01 * 6.0
    assert wh_cash_after == pytest.approx(500.0 - expected_holding, abs=1e-6)


def test_holding_cost_only_on_intermediates_not_factories_or_sinks():
    """Factories and sinks are NOT charged holding cost.

    Factory cash decreases only from production (0 units produced here).
    Sink cash grows only by income_rate with no purchases.
    """
    sc = _make_scenario(
        wh_inventory={P0: 10, P1: 10},
        holding_rate=0.05,
        order_fee=0.0,
        wh_cash=1000.0,
        demand=0,
        n_steps=1,
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    tick = log["ticks"][0]

    # Factory: capacity_per_tick=0 → no production cost; starts at cash=0.
    factory_cash = tick["node_cash"]["fac1"]
    assert factory_cash == pytest.approx(0.0, abs=1e-6)

    # Sink: income_rate=50 added, no purchases this tick.
    sink_cash = tick["node_cash"]["sink"]
    assert sink_cash == pytest.approx(1000.0 + 50.0, abs=1e-6)


def test_holding_cost_on_closing_inventory_after_sales():
    """Holding cost is charged on post-sell closing inventory.

    wh starts with 5 of p0 only; sink demands 3; closing on-hand = 2.
    cost = 2 * 0.02 * 4.0 = 0.16.
    """
    sc = _make_scenario(
        wh_inventory={P0: 5},
        holding_rate=0.02,
        order_fee=0.0,
        wh_cash=100.0,
        demand=3,
        n_steps=1,
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    tick = log["ticks"][0]

    closing_inv = tick["node_inventory"]["wh"].get(P0, 0)
    wh_cash = tick["node_cash"]["wh"]

    # wh received cash from sink: 3 * 10.0 = 30.
    # Holding cost on closing: closing_inv * 0.02 * 4.0.
    expected_holding = closing_inv * 0.02 * 4.0
    assert wh_cash == pytest.approx(100.0 + 3 * 10.0 - expected_holding, abs=1e-6)


# ---------------------------------------------------------------------------
# Order fee
# ---------------------------------------------------------------------------


def test_order_fee_single_supplier_multi_sku():
    """A two-SKU order to one supplier incurs exactly one order_fee."""
    sc = _make_scenario(
        wh_inventory={P0: 0, P1: 0},
        holding_rate=0.0,
        order_fee=100.0,
        wh_cash=5000.0,
        demand=0,
        n_steps=1,
        n_suppliers=1,
        wh_policy=_OrderingPolicy([(P0, "fac1", 5), (P1, "fac1", 5)]),
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    tick = log["ticks"][0]
    purchases = tick["purchases"]
    total_paid = sum(p["cash_paid"] for p in purchases if p["buyer_id"] == "wh")
    wh_cash = tick["node_cash"]["wh"]
    # One PO to fac1 → one fee of 100.
    assert wh_cash == pytest.approx(5000.0 - total_paid - 100.0, abs=1e-6)


def test_order_fee_two_suppliers_two_fees():
    """Ordering from two suppliers in one tick incurs two order_fees."""
    sc = _make_scenario(
        wh_inventory={P0: 0, P1: 0},
        holding_rate=0.0,
        order_fee=50.0,
        wh_cash=5000.0,
        demand=0,
        n_steps=1,
        n_suppliers=2,
        wh_policy=_OrderingPolicy([(P0, "fac1", 5), (P1, "fac2", 5)]),
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    tick = log["ticks"][0]
    purchases = tick["purchases"]
    total_paid = sum(p["cash_paid"] for p in purchases if p["buyer_id"] == "wh")
    wh_cash = tick["node_cash"]["wh"]
    # Two distinct suppliers → two fees = 100 total.
    assert wh_cash == pytest.approx(5000.0 - total_paid - 2 * 50.0, abs=1e-6)


def test_order_fee_only_on_ordering_intermediate():
    """Sink buying from wh does NOT trigger an order_fee on the sink.

    The wh has no policy so it never reorders → no order_fee for wh either.
    Sink pays for goods but cash changes only by purchase + income_rate.
    """
    sc = _make_scenario(
        wh_inventory={P0: 20},
        holding_rate=0.0,
        order_fee=200.0,
        wh_cash=5000.0,
        demand=3,
        n_steps=1,
    )
    from src.sim.runner import Runner
    log = Runner(sc).run()
    tick = log["ticks"][0]
    wh_cash = tick["node_cash"]["wh"]
    sink_cash = tick["node_cash"]["sink"]

    # wh: +30 from sink; no holding (rate=0); no fee (no reorder policy).
    assert wh_cash == pytest.approx(5000.0 + 3 * 10.0, abs=1e-6)
    # Sink: −30 paid + 50 income_rate, started at 1000.
    assert sink_cash == pytest.approx(1000.0 - 3 * 10.0 + 50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Operational KPIs unaffected
# ---------------------------------------------------------------------------


def test_operational_flows_unaffected_by_charging():
    """Sales, demand, and stockout are identical with vs without charging.

    Charging changes cash only; it must not affect the allocation math.
    """
    sc_no = _make_scenario(
        wh_inventory={P0: 30},
        holding_rate=0.0,
        order_fee=0.0,
        wh_cash=500.0,
        demand=5,
        n_steps=3,
    )
    sc_ch = _make_scenario(
        wh_inventory={P0: 30},
        holding_rate=0.05,
        order_fee=999.0,
        wh_cash=500.0,
        demand=5,
        n_steps=3,
    )
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame

    ff_no = flow_frame(Runner(sc_no).run())
    ff_ch = flow_frame(Runner(sc_ch).run())

    for col in ["sales", "demand", "stockout"]:
        assert list(ff_no[col]) == list(ff_ch[col]), f"column {col!r} differed after charging"
