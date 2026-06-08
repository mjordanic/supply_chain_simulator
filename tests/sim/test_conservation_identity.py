"""Integration test: ADR 0013 Rule 5 cash-conservation identity (issue 06).

Rule 5 (as amended by ADR 0019): within the trade graph, for a run with no
active factory production,

    initial_cash + sink_cash_created
        == node_cash_total + cumulative_void + in_transit_value

where
- initial_cash       = sum of all node cash balances before tick 1.
- sink_cash_created  = n_steps × income_rate per DemandSinkNode.
- node_cash_total    = sum of all node cash balances after the last tick.
- cumulative_void    = total holding cost + total order fees charged to
                       IntermediateNodes over all ticks (derived from the
                       per-tick logged closing inventory + node rates —
                       never logged per-tick directly, per ADR 0019).
- in_transit_value   = sum of pending (paid-for, not-yet-delivered) units
                       × catalog unit_cost at end of run.

Tests:
- conservation_identity_terms returns the four expected keys.
- Void terms are non-zero in the test scenario (confirming issue 03 charges).
- The identity holds within floating-point tolerance (1e-4).
- Derived void matches what the engine actually charged (from cash deltas).
- Tick-by-tick identity holds at every tick (optional stricter check).
"""

from __future__ import annotations

from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Scenario builder — zero-production factory so identity is clean
# ---------------------------------------------------------------------------


def _make_scenario(
    *,
    holding_rate: float = 0.02,
    order_fee: float = 100.0,
    n_steps: int = 10,
    demand: int = 5,
    wh_policy=None,
):
    """Build a factory → warehouse → sink scenario with zero factory production.

    capacity_per_tick=0 means the factory never produces new units during the
    run.  This is required for the ADR 0013 Rule 5 identity to hold exactly:
    with active production the factory injects inventory value outside the
    cash-conservation scope.
    """
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario, load_catalog, MarketParams, DisruptionParams, ItemLifecycleParams

    catalog = load_catalog(
        [
            {
                "name": "Alpha",
                "category": "goods",
                "related_products": [],
                "base_price": 12.0,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            }
        ]
    )
    p0 = catalog[0].product_id
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
    disruption = DisruptionParams(
        event_prob=0.0, types=["flood"], regions=["EU"],
        severity=Constant(0.1), duration=Constant(1),
    )
    lifecycle = ItemLifecycleParams(
        stages=stages, init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )

    # Zero production: capacity_per_tick=0 — factory never creates new units.
    fac = FactoryNode(
        id="fac", region="EU", init_seed=1,
        produces_product_id=p0, unit_cost=4.0, capacity_per_tick=0,
        inventory=500, list_price=4.0, cash=0.0,
    )
    wh = IntermediateNode(
        id="wh", region="EU", init_seed=2,
        carried_products={p0}, capacity=1000, tags=[],
        inventory={p0: 100},
        pending={},
        list_prices={p0: 8.0},
        min_order_imposed={p0: 0},
        cash=500.0,
        holding_rate=holding_rate,
        order_fee=order_fee,
    )
    if wh_policy is not None:
        wh.policy = wh_policy

    sink = DemandSinkNode(
        id="sink", region="EU", init_seed=3,
        product_id=p0, demand_dist=Constant(demand), income_rate=200.0, cash=200.0,
    )
    sink.policy = DefaultDemandSinkPolicy(policy_seed=4)

    return Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption,
        item_lifecycle=lifecycle,
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
        nodes=[
            NodeInstance(node=fac, init_seed=1),
            NodeInstance(node=wh, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=[
            EdgeSpec(supplier_id="fac", buyer_id="wh", default_lead_time=1),
            EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
        ],
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


# ---------------------------------------------------------------------------
# Helper keys
# ---------------------------------------------------------------------------


def test_conservation_identity_terms_returns_expected_keys():
    """conservation_identity_terms returns the four Rule 5 term keys."""
    from src.sim.runner import Runner
    from src.sim.inspect import conservation_identity_terms

    sc = _make_scenario()
    log = Runner(sc).run()
    terms = conservation_identity_terms(log, sc)
    assert set(terms.keys()) == {
        "sink_cash_created",
        "node_cash_total",
        "cumulative_void",
        "in_transit_value",
    }


# ---------------------------------------------------------------------------
# Void terms are non-zero (confirms engine charges from issue 03)
# ---------------------------------------------------------------------------


def test_cumulative_void_is_non_zero():
    """Void terms (holding + fees) are non-zero; confirms engine is charging.

    The test scenario has holding_rate=0.02 and order_fee=100, with initial
    inventory and a policy that orders in some ticks.  If the engine stopped
    charging (regression to pre-issue-03 state), cumulative_void would be 0.
    """
    from src.sim.distributions import Constant
    from src.sim.scenario import load_catalog
    p0 = load_catalog([{"name":"A","category":"g","related_products":[],"base_price":12.,"unit_cost":4.,"seasonality":"all_season"}])[0].product_id

    sc = _make_scenario(
        holding_rate=0.02,
        order_fee=100.0,
        n_steps=10,
        demand=3,
        wh_policy=_OrderingPolicy([(p0, "fac", 10)]),
    )
    from src.sim.runner import Runner
    from src.sim.inspect import conservation_identity_terms

    log = Runner(sc).run()
    terms = conservation_identity_terms(log, sc)

    assert terms["cumulative_void"] > 0.0, (
        f"expected non-zero void but got {terms['cumulative_void']}"
    )


# ---------------------------------------------------------------------------
# Conservation identity holds (main integration test)
# ---------------------------------------------------------------------------


def test_conservation_identity_holds_at_end_of_run():
    """ADR 0013 Rule 5 identity holds at end of run within tolerance.

    With zero factory production:
        initial_cash + sink_cash_created == node_cash_total + cumulative_void

    Payment is at allocation time (ADR 0012), so in-transit goods are already
    paid for and the cash is in the seller's balance.  The identity does NOT
    include an in_transit_value term.
    """
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import conservation_identity_terms

    sc = _make_scenario(holding_rate=0.02, order_fee=100.0, n_steps=10, demand=5)
    sim0 = build_world(sc)
    initial_cash = sum(n.cash for n in sim0.nodes.values())

    log = Runner(sc).run()
    terms = conservation_identity_terms(log, sc)

    lhs = initial_cash + terms["sink_cash_created"]
    rhs = terms["node_cash_total"] + terms["cumulative_void"]
    assert lhs == pytest.approx(rhs, abs=1e-4), (
        f"Conservation identity violated: LHS={lhs:.4f} != RHS={rhs:.4f} "
        f"(diff={abs(lhs - rhs):.4f})\n"
        f"terms: {terms}\ninitial_cash={initial_cash:.4f}"
    )


def test_conservation_identity_holds_with_in_transit():
    """Identity holds even when there are in-transit units at end of run.

    A lead_time=2 on the fac→wh edge means some orders may still be in
    transit when the episode ends (in_transit_value > 0).  The identity
    still holds WITHOUT adding in_transit_value: payment is at allocation
    time (ADR 0012), so the seller's cash already reflects the payment and
    in_transit_value must NOT be added to the RHS.
    """
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario, load_catalog, MarketParams, DisruptionParams, ItemLifecycleParams
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import conservation_identity_terms

    catalog = load_catalog([{"name":"A","category":"g","related_products":[],"base_price":12.,"unit_cost":4.,"seasonality":"all_season"}])
    p0 = catalog[0].product_id
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    market = MarketParams(cycle_len=365,cycle_amp=0.0,init_demand=1.0,init_supply=1.0,peak_factor=1.0,off_factor=1.0,season_months={},regions=["EU"],correlation=0.0,trend_update_interval=100,min_value=0.5,max_value=2.0,stage_multipliers={s:1.0 for s in stages},price_elasticity=0.0,promo_multiplier=1.0,demand_factor_min=0.1,supply_factor_min=0.01,cross_inv_lo=0.3,cross_inv_hi=0.7,cross_factor_range=(0.5,1.5),trend=Constant(1.0),demand_shock=Constant(0.0),supply_shock=Constant(0.0),base_demand=Constant(5))
    disruption = DisruptionParams(event_prob=0.0,types=["flood"],regions=["EU"],severity=Constant(0.1),duration=Constant(1))
    lifecycle = ItemLifecycleParams(stages=stages,init_stage="maturity",default_stage_change_probs={s:0.0 for s in stages})

    fac = FactoryNode(id="fac",region="EU",init_seed=1,produces_product_id=p0,unit_cost=4.0,capacity_per_tick=0,inventory=500,list_price=4.0,cash=0.0)
    wh = IntermediateNode(id="wh",region="EU",init_seed=2,carried_products={p0},capacity=500,tags=[],inventory={p0:100},pending={},list_prices={p0:8.0},min_order_imposed={p0:0},cash=500.0,holding_rate=0.02,order_fee=0.0)
    wh.policy = _OrderingPolicy([(p0, "fac", 20)])
    sink = DemandSinkNode(id="sink",region="EU",init_seed=3,product_id=p0,demand_dist=Constant(5),income_rate=200.0,cash=200.0)
    sink.policy = DefaultDemandSinkPolicy(policy_seed=4)

    sc = Scenario(
        catalog=catalog, market=market, disruption=disruption,
        item_lifecycle=lifecycle, n_steps=5,
        start_date=datetime(2024,1,1), world_seed=42,
        nodes=[NodeInstance(node=fac,init_seed=1),NodeInstance(node=wh,init_seed=2),NodeInstance(node=sink,init_seed=3)],
        # lead_time=2 on fac→wh so wh has in-transit at end of last tick
        edges=[
            EdgeSpec(supplier_id="fac",buyer_id="wh",default_lead_time=2),
            EdgeSpec(supplier_id="wh",buyer_id="sink",default_lead_time=1),
        ],
    )

    sim0 = build_world(sc)
    initial_cash = sum(n.cash for n in sim0.nodes.values())
    log = Runner(sc).run()
    terms = conservation_identity_terms(log, sc)

    # Verify there IS in-transit inventory so the test is meaningful.
    assert terms["in_transit_value"] > 0.0, "Expected in-transit units with lead_time=2"

    # Identity holds WITHOUT in_transit_value (payment is immediate per ADR 0012).
    lhs = initial_cash + terms["sink_cash_created"]
    rhs = terms["node_cash_total"] + terms["cumulative_void"]
    assert lhs == pytest.approx(rhs, abs=1e-4), (
        f"Identity violated with in-transit: LHS={lhs:.4f} != RHS={rhs:.4f}"
    )


# ---------------------------------------------------------------------------
# Derived void matches engine charges
# ---------------------------------------------------------------------------


def test_derived_void_matches_engine_charges():
    """Derived void (from conservation_identity_terms) == engine's cash charge.

    The engine charged void by directly debiting each IntermediateNode's cash.
    We can recover the actual engine charge from the node's cash delta:
        engine_void = (init_cash + revenue - order_cost) - final_cash

    The derived void must match this to confirm the derivation is correct
    (ADR 0019 — void is derived, not logged directly).
    """
    from src.sim.distributions import Constant
    from src.sim.scenario import load_catalog
    p0 = load_catalog([{"name":"A","category":"g","related_products":[],"base_price":12.,"unit_cost":4.,"seasonality":"all_season"}])[0].product_id

    sc = _make_scenario(
        holding_rate=0.02,
        order_fee=100.0,
        n_steps=8,
        demand=3,
        wh_policy=_OrderingPolicy([(p0, "fac", 10)]),
    )
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import conservation_identity_terms, purchase_frame

    sim0 = build_world(sc)
    wh_init_cash = sim0.nodes["wh"].cash
    log = Runner(sc).run()
    wh_final_cash = log["ticks"][-1]["node_cash"]["wh"]

    # Engine-charged void from cash accounting:
    # init_cash + revenue_received - order_cost_paid - engine_void = final_cash
    # → engine_void = init_cash + revenue - order_cost - final_cash
    pf = purchase_frame(log)
    wh_revenue = float(pf[pf["supplier_id"] == "wh"]["cash_paid"].sum())
    wh_order_cost = float(pf[pf["buyer_id"] == "wh"]["cash_paid"].sum())
    engine_void = wh_init_cash + wh_revenue - wh_order_cost - wh_final_cash

    terms = conservation_identity_terms(log, sc)
    derived_void = terms["cumulative_void"]

    assert derived_void == pytest.approx(engine_void, abs=1e-4), (
        f"Derived void {derived_void:.4f} != engine void {engine_void:.4f}"
    )
    # Void must be non-zero (confirms the engine is actually charging).
    assert derived_void > 0.0


# ---------------------------------------------------------------------------
# Tick-by-tick identity (stronger check)
# ---------------------------------------------------------------------------


def test_conservation_identity_holds_tick_by_tick():
    """The identity holds at every tick, not just the final tick.

    At each tick t:
        initial_cash + sink_cash_created_up_to_t
            == node_cash_total_at_t + cumulative_void_up_to_t
    """
    from src.sim.runner import Runner, build_world
    from src.sim.node import DemandSinkNode, IntermediateNode

    sc = _make_scenario(holding_rate=0.01, order_fee=50.0, n_steps=8, demand=5)
    sim0 = build_world(sc)
    initial_cash = sum(n.cash for n in sim0.nodes.values())

    log = Runner(sc).run()

    unit_costs = {w.product_id: w.unit_cost for w in sc.catalog}
    sink_income_per_tick = sum(
        ni.node.income_rate
        for ni in sc.nodes
        if isinstance(ni.node, DemandSinkNode)
    )
    intermediate_rates = {
        ni.node.id: (float(ni.node.holding_rate), float(ni.node.order_fee))
        for ni in sc.nodes
        if isinstance(ni.node, IntermediateNode)
    }

    cum_void = 0.0
    cum_sink = 0.0
    for tick_log in log["ticks"]:
        cum_sink += sink_income_per_tick

        # Accumulate void this tick
        purchases_this_tick = tick_log.get("purchases", [])
        for nid, (hr, of) in intermediate_rates.items():
            inv = tick_log.get("node_inventory", {}).get(nid, {})
            for pid, qty in inv.items():
                if pid != "_total":
                    cum_void += qty * hr * unit_costs.get(pid, 0.0)
            if of > 0.0:
                sups = {p["supplier_id"] for p in purchases_this_tick if p["buyer_id"] == nid}
                cum_void += len(sups) * of

        # Current node_cash_total
        cash_now = sum(tick_log["node_cash"].values())

        lhs = initial_cash + cum_sink
        # Payment is at allocation time (ADR 0012) — no in_transit term needed.
        rhs = cash_now + cum_void
        assert lhs == pytest.approx(rhs, abs=1e-4), (
            f"Identity violated at tick {tick_log['tick']}: "
            f"LHS={lhs:.4f} != RHS={rhs:.4f} (diff={abs(lhs-rhs):.4f})"
        )


# ---------------------------------------------------------------------------
# Multi-node scenario (two intermediates)
# ---------------------------------------------------------------------------


def test_conservation_identity_holds_multi_intermediate():
    """Identity holds with two IntermediateNodes (two sets of void terms).

    Builds a custom two-intermediate scenario (factory→wh-a→wh-b→sink) with
    zero factory production so the identity holds exactly.  Both intermediates
    have non-zero holding_rate and order_fee, exercising multi-node void
    accumulation.
    """
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import DefaultDemandSinkPolicy
    from src.sim.scenario import NodeInstance, Scenario, load_catalog, MarketParams, DisruptionParams, ItemLifecycleParams
    from src.sim.runner import Runner, build_world
    from src.sim.inspect import conservation_identity_terms

    catalog = load_catalog([{"name":"A","category":"g","related_products":[],"base_price":15.,"unit_cost":4.,"seasonality":"all_season"}])
    p0 = catalog[0].product_id
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    market = MarketParams(cycle_len=365,cycle_amp=0.0,init_demand=1.0,init_supply=1.0,peak_factor=1.0,off_factor=1.0,season_months={},regions=["EU"],correlation=0.0,trend_update_interval=100,min_value=0.5,max_value=2.0,stage_multipliers={s:1.0 for s in stages},price_elasticity=0.0,promo_multiplier=1.0,demand_factor_min=0.1,supply_factor_min=0.01,cross_inv_lo=0.3,cross_inv_hi=0.7,cross_factor_range=(0.5,1.5),trend=Constant(1.0),demand_shock=Constant(0.0),supply_shock=Constant(0.0),base_demand=Constant(5))
    disruption = DisruptionParams(event_prob=0.0,types=["flood"],regions=["EU"],severity=Constant(0.1),duration=Constant(1))
    lifecycle = ItemLifecycleParams(stages=stages,init_stage="maturity",default_stage_change_probs={s:0.0 for s in stages})

    fac = FactoryNode(id="fac",region="EU",init_seed=1,produces_product_id=p0,unit_cost=4.0,capacity_per_tick=0,inventory=500,list_price=4.0,cash=0.0)
    wh_a = IntermediateNode(id="wh-a",region="EU",init_seed=2,carried_products={p0},capacity=500,tags=[],inventory={p0:80},pending={},list_prices={p0:6.0},min_order_imposed={p0:0},cash=300.0,holding_rate=0.02,order_fee=50.0)
    wh_b = IntermediateNode(id="wh-b",region="EU",init_seed=3,carried_products={p0},capacity=500,tags=[],inventory={p0:40},pending={},list_prices={p0:9.0},min_order_imposed={p0:0},cash=400.0,holding_rate=0.01,order_fee=30.0)
    wh_b.policy = _OrderingPolicy([(p0, "wh-a", 8)])
    sink = DemandSinkNode(id="sink",region="EU",init_seed=4,product_id=p0,demand_dist=Constant(5),income_rate=200.0,cash=100.0)
    sink.policy = DefaultDemandSinkPolicy(policy_seed=5)

    sc = Scenario(
        catalog=catalog, market=market, disruption=disruption,
        item_lifecycle=lifecycle, n_steps=8,
        start_date=datetime(2024,1,1), world_seed=42,
        nodes=[NodeInstance(node=fac,init_seed=1),NodeInstance(node=wh_a,init_seed=2),NodeInstance(node=wh_b,init_seed=3),NodeInstance(node=sink,init_seed=4)],
        edges=[
            EdgeSpec(supplier_id="fac",buyer_id="wh-a",default_lead_time=1),
            EdgeSpec(supplier_id="wh-a",buyer_id="wh-b",default_lead_time=1),
            EdgeSpec(supplier_id="wh-b",buyer_id="sink",default_lead_time=1),
        ],
    )

    sim0 = build_world(sc)
    initial_cash = sum(n.cash for n in sim0.nodes.values())
    log = Runner(sc).run()
    terms = conservation_identity_terms(log, sc)

    # Both intermediates should contribute to void.
    assert terms["cumulative_void"] > 0.0

    lhs = initial_cash + terms["sink_cash_created"]
    rhs = terms["node_cash_total"] + terms["cumulative_void"]
    assert lhs == pytest.approx(rhs, abs=1e-3)
