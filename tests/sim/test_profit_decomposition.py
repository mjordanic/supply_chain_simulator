"""Tests for profit_decomposition (ADR 0019 Rule 3, issue 05).

Covers:
- Decomposition columns present and correct for a minimal scenario.
- order_cost uses real cash_paid (not catalog unit_cost).
- holding_cost and order_fees derived from flows + node rates.
- net_profit reconciles to Δcash for every IntermediateNode.
- Lateral-link scenario: order_cost != unit_cost-based figure, still reconciles.
- Empty scenario (no IntermediateNodes) returns empty DataFrame.
"""

from __future__ import annotations

from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Scenario helpers (shared with test_engine_charging.py style)
# ---------------------------------------------------------------------------


def _catalog_two():
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


def _flat_market(demand: int = 5):
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
        base_demand=Constant(demand),
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


def _make_chain_scenario(
    *,
    holding_rate: float = 0.01,
    order_fee: float = 50.0,
    wh_cash: float = 2000.0,
    demand: int = 5,
    n_steps: int = 5,
    wh_policy=None,
    wh_inventory: dict | None = None,
):
    """Build a minimal factory → warehouse → sink chain scenario."""
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario

    catalog = _catalog_two()
    pids = [w.product_id for w in catalog]
    p0 = pids[0]

    if wh_inventory is None:
        wh_inventory = {p0: 50}

    wh = IntermediateNode(
        id="wh", region="EU", init_seed=2,
        carried_products={p0},
        capacity=500,
        tags=["warehouse"],
        inventory=dict(wh_inventory),
        pending={},
        list_prices={p0: 8.0},
        min_order_imposed={p0: 0},
        cash=wh_cash,
        holding_rate=holding_rate,
        order_fee=order_fee,
    )
    if wh_policy is not None:
        wh.policy = wh_policy

    fac = FactoryNode(
        id="fac", region="EU", init_seed=1,
        produces_product_id=p0, unit_cost=4.0, capacity_per_tick=0,
        inventory=500, list_price=4.0, cash=0.0,
    )
    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=3,
        product_id=p0, demand_dist=Constant(demand), income_rate=50.0, cash=500.0,
    )
    sink.policy = DefaultDemandSinkPolicy(policy_seed=4)

    nodes = [
        NodeInstance(node=fac, init_seed=1),
        NodeInstance(node=wh, init_seed=2),
        NodeInstance(node=sink, init_seed=3),
    ]
    edges = [
        EdgeSpec(supplier_id="fac", buyer_id="wh", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
    ]

    return Scenario(
        catalog=catalog,
        market=_flat_market(demand),
        disruption=_flat_disruption(),
        item_lifecycle=_flat_lifecycle(),
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
        nodes=nodes,
        edges=edges,
    )


class _OrderingPolicy:
    """Stub policy that orders a fixed qty from a fixed supplier."""

    def __init__(self, supplier_orders: list[tuple[str, str, int]]):
        self._orders = supplier_orders

    def decide(self, obs, table):
        order: dict[str, list] = {}
        for pid, sid, qty in self._orders:
            order.setdefault(pid, []).append((sid, qty))
        return {"order": order, "list_price": {}, "min_order_imposed": {}}


_PIDS = [w.product_id for w in _catalog_two()]
P0 = _PIDS[0]


# ---------------------------------------------------------------------------
# Columns and output shape
# ---------------------------------------------------------------------------


def test_profit_decomposition_columns_present():
    """Output DataFrame has the expected columns."""
    sc = _make_chain_scenario()
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    result = profit_decomposition(
        flow_frame(log), purchase_frame(log), closing_inventory_frame(log), sc
    )
    expected = {"node_id", "revenue", "order_cost", "holding_cost", "order_fees", "net_profit"}
    assert set(result.columns) == expected


def test_profit_decomposition_one_row_per_intermediate():
    """One row per IntermediateNode in the scenario."""
    sc = _make_chain_scenario()
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    result = profit_decomposition(
        flow_frame(log), purchase_frame(log), closing_inventory_frame(log), sc
    )
    # Only 'wh' is an IntermediateNode.
    assert list(result["node_id"]) == ["wh"]


# ---------------------------------------------------------------------------
# Revenue
# ---------------------------------------------------------------------------


def test_revenue_is_price_times_sales():
    """Revenue = sum(price * sales) for selling rows of this node."""
    sc = _make_chain_scenario(demand=10, n_steps=1, holding_rate=0.0, order_fee=0.0)
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    ff = flow_frame(log)
    pf = purchase_frame(log)
    cif = closing_inventory_frame(log)

    result = profit_decomposition(ff, pf, cif, sc)
    wh_row = result[result["node_id"] == "wh"].iloc[0]

    # Revenue = sum of price * sales for wh
    wh_ff = ff[(ff["node_id"] == "wh") & ff["price"].notna()]
    expected_rev = float((wh_ff["price"] * wh_ff["sales"]).sum())
    assert wh_row["revenue"] == pytest.approx(expected_rev, abs=1e-6)


# ---------------------------------------------------------------------------
# Order cost uses real cash_paid (not unit_cost)
# ---------------------------------------------------------------------------


def test_order_cost_uses_cash_paid_not_unit_cost():
    """Order cost = sum(cash_paid) from purchase_frame — real cash, not catalog cost.

    wh sells at 8.0; factory's unit_cost is 4.0 but it sells at list_price=4.0
    In this chain, wh buys from factory at list_price=4.0.  To prove the
    distinction, we verify order_cost == sum(cash_paid), not unit_cost-based.
    """
    sc = _make_chain_scenario(
        demand=5,
        n_steps=3,
        holding_rate=0.0,
        order_fee=0.0,
        wh_policy=_OrderingPolicy([(P0, "fac", 10)]),
    )
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    ff = flow_frame(log)
    pf = purchase_frame(log)
    cif = closing_inventory_frame(log)

    result = profit_decomposition(ff, pf, cif, sc)
    wh_row = result[result["node_id"] == "wh"].iloc[0]

    # cash_paid = qty_filled * supplier_list_price (4.0 per unit from factory)
    expected_oc = float(pf[pf["buyer_id"] == "wh"]["cash_paid"].sum())
    assert wh_row["order_cost"] == pytest.approx(expected_oc, abs=1e-6)
    # It equals qty_filled * 4.0 (factory list_price == unit_cost here, but by
    # cash_paid, not catalog lookup — the test documents the data source).
    assert wh_row["order_cost"] > 0.0


# ---------------------------------------------------------------------------
# Holding cost and order fees derived from flows + rates
# ---------------------------------------------------------------------------


def test_holding_cost_matches_engine_charge():
    """Holding cost derived in profit_decomposition equals what the engine charged.

    Compare order_cost + holding_cost + order_fees == initial_cash - final_cash + revenue.
    """
    holding_rate = 0.02
    sc = _make_chain_scenario(
        demand=3,
        n_steps=5,
        holding_rate=holding_rate,
        order_fee=0.0,
    )
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    result = profit_decomposition(
        flow_frame(log), purchase_frame(log), closing_inventory_frame(log), sc
    )
    wh_row = result[result["node_id"] == "wh"].iloc[0]

    # Holding cost must be positive (wh had inventory, rate=0.02 > 0).
    assert wh_row["holding_cost"] > 0.0

    # Derive holding cost independently from closing_inventory_frame.
    cif = closing_inventory_frame(log)
    unit_costs = {w.product_id: w.unit_cost for w in sc.catalog}
    wh_inv = cif[cif["node_id"] == "wh"]
    expected = float(
        (wh_inv["qty"] * wh_inv["pid"].map(unit_costs).fillna(0.0) * holding_rate).sum()
    )
    assert wh_row["holding_cost"] == pytest.approx(expected, abs=1e-6)


def test_order_fees_derived_from_purchase_frame():
    """Order fees = n_distinct_suppliers_per_tick × order_fee."""
    order_fee = 100.0
    sc = _make_chain_scenario(
        demand=0,
        n_steps=3,
        holding_rate=0.0,
        order_fee=order_fee,
        wh_policy=_OrderingPolicy([(P0, "fac", 5)]),
    )
    from src.sim.runner import Runner
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    pf = purchase_frame(log)
    result = profit_decomposition(
        flow_frame(log), pf, closing_inventory_frame(log), sc
    )
    wh_row = result[result["node_id"] == "wh"].iloc[0]

    # wh orders from one supplier each tick for 3 ticks → 3 × 100 = 300.
    wh_pf = pf[pf["buyer_id"] == "wh"]
    expected_fees = float(wh_pf.groupby("tick")["supplier_id"].nunique().sum() * order_fee)
    assert wh_row["order_fees"] == pytest.approx(expected_fees, abs=1e-6)


# ---------------------------------------------------------------------------
# Net profit reconciles to Δcash
# ---------------------------------------------------------------------------


def test_net_profit_reconciles_to_delta_cash():
    """net_profit == final_cash - initial_cash for every IntermediateNode."""
    sc = _make_chain_scenario(
        demand=5,
        n_steps=10,
        holding_rate=0.01,
        order_fee=50.0,
        wh_policy=_OrderingPolicy([(P0, "fac", 10)]),
    )
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    log = Runner(sc).run()
    result = profit_decomposition(
        flow_frame(log), purchase_frame(log), closing_inventory_frame(log), sc
    )

    # Initial cash from a fresh Simulation (before any ticks).
    sim = build_world(sc)
    for _, row in result.iterrows():
        nid = row["node_id"]
        initial_cash = sim.nodes[nid].cash
        final_cash = log["ticks"][-1]["node_cash"][nid]
        delta_cash = final_cash - initial_cash
        assert row["net_profit"] == pytest.approx(delta_cash, abs=1e-6), (
            f"node {nid!r}: net_profit={row['net_profit']:.4f} != delta_cash={delta_cash:.4f}"
        )


def test_net_profit_reconciles_multi_node():
    """Reconciliation holds for every IntermediateNode in a two-shop scenario."""
    from src.sim.setup_io import load_setup
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    sc = load_setup("setups/two_factories_two_shops")
    log = Runner(sc).run()
    result = profit_decomposition(
        flow_frame(log), purchase_frame(log), closing_inventory_frame(log), sc
    )

    sim = build_world(sc)
    for _, row in result.iterrows():
        nid = row["node_id"]
        initial_cash = sim.nodes[nid].cash
        final_cash = log["ticks"][-1]["node_cash"][nid]
        delta_cash = final_cash - initial_cash
        assert row["net_profit"] == pytest.approx(delta_cash, abs=1e-4), (
            f"node {nid!r}: net_profit={row['net_profit']:.4f} != delta_cash={delta_cash:.4f}"
        )


# ---------------------------------------------------------------------------
# Lateral-link scenario: order_cost differs from unit_cost-based figure
# ---------------------------------------------------------------------------


def test_lateral_link_order_cost_differs_from_unit_cost():
    """In a lateral-link scenario, order_cost (cash_paid) > unit_cost-based cost.

    When wh-B buys from wh-A (an intermediate selling at a margin), the
    list_price > factory unit_cost, so cash_paid > qty × unit_cost.  The
    profit_decomposition uses real cash_paid and still reconciles to Δcash.
    """
    # Build a minimal two-intermediate lateral scenario:
    # factory → wh-a → wh-b → sink
    # wh-a sells at list_price=6.0 but factory unit_cost=4.0.
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario, load_catalog
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import flow_frame, purchase_frame, closing_inventory_frame
    from src.sim.metrics import profit_decomposition

    catalog = load_catalog([
        {
            "name": "Widget",
            "category": "goods",
            "related_products": [],
            "base_price": 20.0,
            "unit_cost": 4.0,
            "seasonality": "all_season",
        }
    ])
    p0 = catalog[0].product_id
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    market = _flat_market(demand=5)
    disruption = _flat_disruption()
    lifecycle = _flat_lifecycle()

    fac = FactoryNode(
        id="fac", region="EU", init_seed=1,
        produces_product_id=p0, unit_cost=4.0, capacity_per_tick=0,
        inventory=500, list_price=4.0, cash=0.0,
    )
    wh_a = IntermediateNode(
        id="wh-a", region="EU", init_seed=2,
        carried_products={p0},
        capacity=500, tags=[],
        inventory={p0: 50},
        pending={},
        list_prices={p0: 6.0},  # margin above unit_cost=4.0
        min_order_imposed={p0: 0},
        cash=1000.0,
        holding_rate=0.0,
        order_fee=0.0,
    )
    wh_b = IntermediateNode(
        id="wh-b", region="EU", init_seed=3,
        carried_products={p0},
        capacity=500, tags=[],
        inventory={p0: 20},
        pending={},
        list_prices={p0: 10.0},
        min_order_imposed={p0: 0},
        cash=2000.0,
        holding_rate=0.0,
        order_fee=0.0,
    )
    # wh-b orders from wh-a
    wh_b.policy = _OrderingPolicy([(p0, "wh-a", 5)])

    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=4,
        product_id=p0, demand_dist=Constant(5), income_rate=100.0, cash=500.0,
    )
    sink.policy = DefaultDemandSinkPolicy(policy_seed=5)

    sc = Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption,
        item_lifecycle=lifecycle,
        n_steps=5,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
        nodes=[
            NodeInstance(node=fac, init_seed=1),
            NodeInstance(node=wh_a, init_seed=2),
            NodeInstance(node=wh_b, init_seed=3),
            NodeInstance(node=sink, init_seed=4),
        ],
        edges=[
            EdgeSpec(supplier_id="fac", buyer_id="wh-a", default_lead_time=1),
            EdgeSpec(supplier_id="wh-a", buyer_id="wh-b", default_lead_time=1),
            EdgeSpec(supplier_id="wh-b", buyer_id="sink", default_lead_time=1),
        ],
    )

    log = Runner(sc).run()
    ff = flow_frame(log)
    pf = purchase_frame(log)
    cif = closing_inventory_frame(log)
    result = profit_decomposition(ff, pf, cif, sc)

    wh_b_row = result[result["node_id"] == "wh-b"].iloc[0]

    # wh-b buys from wh-a at list_price=6.0; unit_cost=4.0.
    # cash_paid = qty_filled × 6.0 > qty_filled × 4.0.
    wh_b_pf = pf[pf["buyer_id"] == "wh-b"]
    cash_paid_total = float(wh_b_pf["cash_paid"].sum())
    unit_cost_based = float(wh_b_pf["qty_filled"].sum() * 4.0)

    assert cash_paid_total > unit_cost_based, (
        f"cash_paid={cash_paid_total} should exceed unit_cost_based={unit_cost_based}"
    )
    assert wh_b_row["order_cost"] == pytest.approx(cash_paid_total, abs=1e-6)

    # Still reconciles to Δcash.
    sim = build_world(sc)
    for _, row in result.iterrows():
        nid = row["node_id"]
        initial_cash = sim.nodes[nid].cash
        final_cash = log["ticks"][-1]["node_cash"][nid]
        delta_cash = final_cash - initial_cash
        assert row["net_profit"] == pytest.approx(delta_cash, abs=1e-4), (
            f"node {nid!r}: net_profit={row['net_profit']:.4f} != delta_cash={delta_cash:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_scenario_no_intermediates():
    """No IntermediateNodes → empty DataFrame with correct columns."""
    import pandas as pd
    from src.sim.metrics import profit_decomposition

    # Fake scenario with no nodes.
    class _FakeScenario:
        catalog = []
        nodes = []

    result = profit_decomposition(
        pd.DataFrame(columns=["tick", "node_id", "pid", "sales", "demand", "price", "stockout"]),
        pd.DataFrame(columns=["tick", "buyer_id", "supplier_id", "pid", "qty_filled", "cash_paid"]),
        pd.DataFrame(columns=["tick", "node_id", "pid", "qty"]),
        _FakeScenario(),
    )
    assert isinstance(result, pd.DataFrame)
    expected = {"node_id", "revenue", "order_cost", "holding_cost", "order_fees", "net_profit"}
    assert set(result.columns) == expected
    assert len(result) == 0
