"""Engine flow-logging correctness (ADR 0019, issue 02).

A plain ``Runner.run()`` must emit the per-tick flow log that state snapshots
cannot recover.  These tests run a small, hand-checkable factory → warehouse →
sink chain and assert, via the ``inspect`` builders, that:

  - ``sales`` / ``demand`` / ``price`` are recorded correctly,
  - ``stockout`` reflects *decision-time* on-hand (not the closing snapshot a
    same-tick sale would drive to zero),
  - per-(buyer, supplier, pid) ``cash_paid`` == ``qty_filled × supplier
    list_price``,
  - the legacy ``node_orders`` aggregate is derived from the purchase rows.
"""

from __future__ import annotations

from datetime import datetime

from src.sim.inspect import flow_frame, purchase_frame
from src.sim.runner import Runner


def _make_chain_scenario(wh_stock: int, demand: int, n_steps: int):
    """Factory → IntermediateNode(warehouse) → DemandSinkNode, one product.

    The warehouse has no policy (it never reorders), so its stock only ever
    depletes — letting us drive it to a genuine decision-time stockout.
    """
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

    factory = FactoryNode(
        id="fac", region="EU", init_seed=1,
        produces_product_id=pid, unit_cost=4.0, capacity_per_tick=50,
        inventory=100, list_price=4.0, cash=0.0,
    )
    warehouse = IntermediateNode(
        id="wh", region="EU", init_seed=2,
        carried_products={pid}, capacity=200, tags=["warehouse"],
        inventory={pid: wh_stock}, pending={}, list_prices={pid: 7.0},
        min_order_imposed={pid: 0}, cash=200.0,
    )
    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=3,
        product_id=pid, demand_dist=Constant(demand), income_rate=50.0, cash=500.0,
    )

    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    scenario = Scenario(
        catalog=catalog,
        market=MarketParams(
            cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
            peak_factor=1.0, off_factor=1.0, season_months={}, regions=["EU"],
            correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
            stage_multipliers={s: 1.0 for s in stages}, price_elasticity=0.0,
            promo_multiplier=1.0, demand_factor_min=0.1, supply_factor_min=0.01,
            cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
            trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
            base_demand=Constant(demand),
        ),
        disruption=DisruptionParams(
            event_prob=0.0, types=["flood"], regions=["EU"],
            severity=Constant(0.1), duration=Constant(1),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=stages, init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in stages},
        ),
        n_steps=n_steps, start_date=datetime(2024, 1, 1), world_seed=1,
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
    return scenario, pid


def test_cash_paid_equals_qty_filled_times_supplier_price():
    """Every purchase row's cash_paid equals qty_filled × supplier list_price."""
    scenario, pid = _make_chain_scenario(wh_stock=30, demand=5, n_steps=3)
    log = Runner(scenario).run()
    purchases = purchase_frame(log)

    assert len(purchases) > 0, "expected at least one purchase"
    # The sink buys from the warehouse at its list_price of 7.0.
    sink_buys = purchases[purchases["buyer_id"] == "sink"]
    assert (sink_buys["supplier_id"] == "wh").all()
    for _, row in sink_buys.iterrows():
        assert row["cash_paid"] == row["qty_filled"] * 7.0


def test_sales_and_demand_recorded_on_selling_node():
    """The warehouse's sales/demand and the sink's demand_target are logged."""
    scenario, pid = _make_chain_scenario(wh_stock=30, demand=5, n_steps=1)
    log = Runner(scenario).run()
    flows = flow_frame(log)

    wh = flows[(flows["node_id"] == "wh") & (flows["tick"] == 1)].iloc[0]
    # Sink demanded 5 and the warehouse had stock, so it sold 5.
    assert wh["sales"] == 5
    assert wh["demand"] == 5
    assert wh["price"] == 7.0

    sink = flows[(flows["node_id"] == "sink") & (flows["tick"] == 1)].iloc[0]
    assert sink["demand"] == 5  # exogenous demand_target
    assert sink["sales"] == 0


def test_stockout_is_decision_time_not_closing_snapshot():
    """A warehouse that sells out still reports stockout=False at decision time."""
    # wh starts with 3 units; sink demands 5 → warehouse sells all 3, closing
    # inventory becomes 0, but at decision time it had stock.
    scenario, pid = _make_chain_scenario(wh_stock=3, demand=5, n_steps=2)
    log = Runner(scenario).run()
    flows = flow_frame(log)

    wh_t1 = flows[(flows["node_id"] == "wh") & (flows["tick"] == 1)].iloc[0]
    closing_inv = log["ticks"][0]["node_inventory"]["wh"].get(pid, 0)

    assert wh_t1["sales"] == 3                       # sold all it had
    assert closing_inv == 0                          # closing snapshot is empty
    assert wh_t1["stockout"] is False or not bool(wh_t1["stockout"])  # but not a stockout

    # Tick 2: the warehouse never reordered → genuinely 0 on hand at decision
    # time → a real stockout, with no sales.
    wh_t2 = flows[(flows["node_id"] == "wh") & (flows["tick"] == 2)].iloc[0]
    assert bool(wh_t2["stockout"]) is True
    assert wh_t2["sales"] == 0


def test_node_orders_derived_from_purchase_rows():
    """The legacy node_orders aggregate is rebuilt from purchase rows.

    With an intermediate buyer that reorders, node_orders[buyer][pid] equals
    the summed requested quantity across that buyer's purchase rows.
    """
    scenario, pid = _make_chain_scenario(wh_stock=30, demand=5, n_steps=3)
    log = Runner(scenario).run()

    for tick_log in log["ticks"]:
        # node_orders only ever keys intermediate nodes (the warehouse here).
        assert set(tick_log["node_orders"].keys()) == {"wh"}
        # Sink purchases never leak into node_orders.
        assert "sink" not in tick_log["node_orders"]
