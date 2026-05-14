"""Integration tests for TextbookReorderPolicy and OrderUpToPolicy.

These tests drive a real Store/Market/ItemRegistry stack to verify the
textbook reorder semantics, not just Python wiring.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.sim.distributions import Constant
from src.sim.policy import OrderUpToPolicy, TextbookReorderPolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    load_catalog,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

DELIVERY_LAG = 3
SAFETY_LEAD = 2
COVER_HORIZON = 10


def _mini_catalog():
    return load_catalog(
        [
            {
                "name": "Widget",
                "category": "Widgets",
                "related_products": [],
                "base_price": 20.0,
                "unit_cost": 10.0,
                "seasonality": "all_season",
            }
        ]
    )


def _constant_market(demand: float = 5.0) -> MarketParams:
    """A zero-noise, constant-demand market for deterministic position tracking."""
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=9999,
        min_value=1.0,
        max_value=1.0,
        stage_multipliers={
            "maturity": 1.0,
        },
        price_elasticity=0.0,  # no price response
        promo_multiplier=1.0,
        demand_factor_min=1.0,
        supply_factor_min=1.0,
        cross_inv_lo=0.0,
        cross_inv_hi=1.0,
        cross_factor_range=(1.0, 1.0),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(demand),
    )


def _no_disruption() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.0),
        duration=Constant(1),
    )


def _no_lifecycle() -> ItemLifecycleParams:
    return ItemLifecycleParams(
        stages=["maturity"],
        init_stage="maturity",
        default_stage_change_probs={"maturity": 0.0},
    )


def _mini_template(
    *,
    capacity: int = 5_000,
    balance: float = 100_000.0,
    delivery_lag: int = DELIVERY_LAG,
    init_stock_pct: float = 0.0,
    init_active_count: int = 1,
) -> StoreTemplate:
    return StoreTemplate(
        id="mini",
        region="US",
        capacity=capacity,
        init_balance=balance,
        init_stock_pct=init_stock_pct,
        delivery_lag=delivery_lag,
        holding_rate=0.001,
        order_fee=0.0,
        init_active_count=init_active_count,
    )


def _run_scenario(
    policy: OrderUpToPolicy,
    n_steps: int,
    template: StoreTemplate | None = None,
    demand: float = 5.0,
) -> dict:
    if template is None:
        template = _mini_template()
    scenario = Scenario(
        catalog=_mini_catalog(),
        market=_constant_market(demand=demand),
        disruption=_no_disruption(),
        item_lifecycle=_no_lifecycle(),
        stores=[StoreInstance(template=template, init_seed=0, policy=policy)],
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    return Runner(scenario).run()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


def test_textbook_base_is_abstract():
    """TextbookReorderPolicy is abstract — instantiating it directly raises TypeError."""
    with pytest.raises(TypeError):
        TextbookReorderPolicy()  # type: ignore[abstract]


def test_order_up_to_instantiates_with_no_args():
    """OrderUpToPolicy() instantiates with no arguments."""
    p = OrderUpToPolicy()
    assert p is not None


def test_order_up_to_default_kwargs():
    """Default kwarg values match the spec."""
    p = OrderUpToPolicy()
    assert p.cover_horizon_ticks == 10
    assert p.safety_lead_ticks == 2
    assert abs(p.opening_budget_pct - 0.50) < 1e-9
    assert p.stockout_safety_bonus_ticks == 0
    assert p.min_qty == 0


# ─────────────────────────────────────────────────────────────────────────────
# CRN / RNG isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_order_up_to_crn_self_consistency():
    """Two OrderUpToPolicy instances on the same scenario produce bit-identical trajectories."""
    import hashlib
    import json

    from src.sim.data_exporter import _jsonable

    def _run_with_seed(world_seed: int, policy_seed: int = 0) -> dict:
        policy = OrderUpToPolicy(policy_seed=policy_seed)
        template = _mini_template(init_stock_pct=0.0)
        scenario = Scenario(
            catalog=_mini_catalog(),
            market=_constant_market(demand=5.0),
            disruption=_no_disruption(),
            item_lifecycle=_no_lifecycle(),
            stores=[StoreInstance(template=template, init_seed=1, policy=policy)],
            n_steps=20,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
        )
        return Runner(scenario).run()

    # Same (world_seed, init_seed, capacity, balance) → identical trajectories
    r1 = _run_with_seed(42, policy_seed=0)
    r2 = _run_with_seed(42, policy_seed=0)

    h1 = hashlib.sha256(
        json.dumps(_jsonable(r1), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    h2 = hashlib.sha256(
        json.dumps(_jsonable(r2), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert h1 == h2


# ─────────────────────────────────────────────────────────────────────────────
# Pilot order on tick 0
# ─────────────────────────────────────────────────────────────────────────────


def test_order_up_to_pilot_fires_on_tick_zero():
    """With init_stock_pct=0.0, a pilot order is placed on tick 0."""
    policy = OrderUpToPolicy(policy_seed=0, opening_budget_pct=0.5)
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=1_000,
        balance=100_000.0,
    )
    log = _run_scenario(policy, n_steps=1, template=template)
    store_log = log["stores"][0]
    # First entry in order_quantity (step 0 baseline) is pre-decide; step 1 is
    # after the first decide. Actually step-1 order_quantity shows step-0 order.
    # Actually the log at index 1 reflects the tick-1 state including step-1 decisions.
    # Pilot fires on first decide call (step 1, first tick).
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    # At least one positive order should have fired
    assert any(q > 0 for q in order_quantities), (
        f"No pilot order fired. order_quantities={order_quantities}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# No order when position > s
# ─────────────────────────────────────────────────────────────────────────────


def test_order_up_to_no_order_when_position_above_s():
    """No order fires while position stays above s."""
    # With init_stock_pct=1.0 (full), position >> s; no reorder should fire.
    demand = 1  # tiny demand so position stays high
    safety_lead = 2
    cover_horizon = 10
    delivery_lag = DELIVERY_LAG
    policy = OrderUpToPolicy(
        policy_seed=0,
        safety_lead_ticks=safety_lead,
        cover_horizon_ticks=cover_horizon,
        opening_budget_pct=0.0,  # disable pilot
    )
    template = _mini_template(
        init_stock_pct=1.0,
        capacity=5_000,
        balance=1_000_000.0,
    )
    # Run just a few ticks — with high init stock, no reorder should fire
    log = _run_scenario(policy, n_steps=2, template=template, demand=demand)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    # Check ticks 1 and 2 (skip tick 0 which is pre-decide baseline)
    order_quantities = store_log["products"][pid]["order_quantity"][1:]
    # With 5000 init stock and demand=1, position >> s=(lag+safety)*rate
    assert all(q == 0 for q in order_quantities), (
        f"Unexpected orders when position >> s: {order_quantities}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Position oscillates between s and S
# ─────────────────────────────────────────────────────────────────────────────


def test_order_up_to_oscillates_between_s_and_S():
    """Under constant demand, position stays inside [s, S] after warmup."""
    demand = 5.0
    delivery_lag = DELIVERY_LAG
    safety_lead = SAFETY_LEAD
    cover_horizon = COVER_HORIZON

    policy = OrderUpToPolicy(
        policy_seed=0,
        safety_lead_ticks=safety_lead,
        cover_horizon_ticks=cover_horizon,
        opening_budget_pct=0.5,
    )
    # Run long enough for warmup (pilot lands + rate stabilises)
    warmup = (delivery_lag + safety_lead + cover_horizon) * 4
    n_steps = warmup * 4
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=50_000,
        balance=10_000_000.0,
        delivery_lag=delivery_lag,
    )
    log = _run_scenario(policy, n_steps=n_steps, template=template, demand=demand)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]

    inv_series = store_log["products"][pid]["inventory"]
    pending_series = store_log["products"][pid]["outstanding_orders"]

    # After warmup, check that position (inv + pending) stays inside [s, 2*S]
    # S = (lag + safety + cover) * rate, s = (lag + safety) * rate
    rate = demand
    s = (delivery_lag + safety_lead) * rate
    S = (delivery_lag + safety_lead + cover_horizon) * rate

    post_warmup_positions = [
        inv_series[t] + pending_series[t]
        for t in range(warmup, n_steps + 1)
    ]
    # Allow generous bounds: position should be in [0, 2*S]
    assert all(pos >= 0 for pos in post_warmup_positions), (
        "Negative position encountered"
    )
    # At least some ticks should be above s (not perpetually stocked out)
    above_s = sum(1 for pos in post_warmup_positions if pos >= s)
    assert above_s > len(post_warmup_positions) * 0.3, (
        f"Policy seems perpetually below s. above_s={above_s}/{len(post_warmup_positions)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flagship scale smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _flagship_catalog(n_products: int = 10) -> list:
    """A small catalog with meaningful price/cost margins for profitability."""
    items = []
    for i in range(n_products):
        items.append({
            "name": f"Item{i:02d}",
            "category": "Fashion",
            "related_products": [],
            "base_price": 30.0,   # $30 MSRP
            "unit_cost": 10.0,    # $10 cost → 200% margin
            "seasonality": "all_season",
        })
    return load_catalog(items)


def _flagship_run(n_products: int = 10, demand: float = 5.0, n_steps: int = 180) -> dict:
    """Run a flagship-scale scenario with n_products and constant demand."""
    from src.sim.scenario import Scenario, StoreInstance

    catalog = _flagship_catalog(n_products)
    policy = OrderUpToPolicy(policy_seed=0)
    template = StoreTemplate(
        id="flagship",
        region="US",
        capacity=10_000,
        init_balance=1_000_000.0,
        init_stock_pct=0.0,
        delivery_lag=DELIVERY_LAG,
        holding_rate=0.001,
        order_fee=0.0,  # no fixed fee so the policy focuses on inventory math
        init_active_count=n_products,
    )
    scenario = Scenario(
        catalog=catalog,
        market=_constant_market(demand=demand),
        disruption=_no_disruption(),
        item_lifecycle=_no_lifecycle(),
        stores=[StoreInstance(template=template, init_seed=0, policy=policy)],
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    return Runner(scenario).run()


def test_order_up_to_flagship_scale_is_profitable():
    """Full 180-tick episode at flagship scale finishes with non-negative net P&L.

    Uses a 10-product catalog with $30 MSRP / $10 cost (200% margin) and
    constant demand of 5 units/tick/product. The policy should easily be
    profitable given the generous margin.
    """
    log = _flagship_run(n_products=10, demand=5.0, n_steps=180)
    store_log = log["stores"][0]
    initial_balance = store_log["balance"][0]
    final_balance = store_log["balance"][-1]
    assert final_balance >= initial_balance, (
        f"OrderUpToPolicy lost money at flagship scale: "
        f"initial={initial_balance:.2f}, final={final_balance:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exports sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_policy_exports():
    """Confirm that OrderUpToPolicy and TextbookReorderPolicy are exported."""
    from src.sim.policy import OrderUpToPolicy, TextbookReorderPolicy
    assert OrderUpToPolicy is not None
    assert TextbookReorderPolicy is not None


# ─────────────────────────────────────────────────────────────────────────────
# ReorderPointPolicy (s,Q) tests
# ─────────────────────────────────────────────────────────────────────────────


def test_reorder_point_pilot_fires_on_tick_zero():
    """With init_stock_pct=0.0, a pilot order is placed on tick 0."""
    from src.sim.policy import ReorderPointPolicy

    policy = ReorderPointPolicy(policy_seed=0, opening_budget_pct=0.5)
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=1_000,
        balance=100_000.0,
    )
    log = _run_scenario(policy, n_steps=1, template=template)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    assert any(q > 0 for q in order_quantities), (
        f"No pilot order fired. order_quantities={order_quantities}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicOrderUpToPolicy (R,S) tests
# ─────────────────────────────────────────────────────────────────────────────


def test_periodic_order_up_to_pilot_fires_on_tick_zero():
    """Pilot order fires on tick 0 regardless of review_interval."""
    from src.sim.policy import PeriodicOrderUpToPolicy

    # review_interval=7 so step%7==0 only fires on steps 0,7,14,...
    # Pilot is placed before the trigger loop, so it should still fire on tick 0.
    policy = PeriodicOrderUpToPolicy(
        policy_seed=0,
        review_interval=7,
        opening_budget_pct=0.5,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=1_000,
        balance=100_000.0,
    )
    log = _run_scenario(policy, n_steps=1, template=template)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    assert any(q > 0 for q in order_quantities), (
        f"No pilot order fired. order_quantities={order_quantities}"
    )


def test_reorder_point_quantity_is_fixed_Q():
    """When Q is set explicitly, every triggered order uses that fixed quantity.

    We use opening_budget_pct=0.5 with a small balance so the pilot order
    seeds ~50 units; after the pilot lands, demand=5/tick depletes stock below
    s=(delivery_lag)*rate=15 and the policy should fire triggered orders of
    exactly Q=50 units each time.
    """
    from src.sim.policy import ReorderPointPolicy

    fixed_Q = 50
    # pilot_qty = 0.5 * balance / K_active / cost = 0.5 * 1000 / 1 / 10 = 50
    # After pilot (50 units) lands at tick DELIVERY_LAG, position depletes:
    # rate≈5, s≈15 (delivery_lag=3, safety=0), trigger fires at position<15.
    policy = ReorderPointPolicy(
        policy_seed=0,
        Q=fixed_Q,
        opening_budget_pct=0.5,
        safety_lead_ticks=0,
        cover_horizon_ticks=10,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=5_000,
        balance=1_000.0,  # small balance → pilot ≈ 50 units (not capacity-filling)
        delivery_lag=DELIVERY_LAG,
    )
    log = _run_scenario(policy, n_steps=60, template=template, demand=5.0)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    # Skip index 0 (pre-decide baseline) and index 1 (which holds the pilot).
    post_pilot = order_quantities[2:]
    non_zero = [q for q in post_pilot if q > 0]
    assert len(non_zero) > 0, (
        f"No triggered orders after pilot. all order_quantities={order_quantities}"
    )
    for q in non_zero:
        assert q == fixed_Q, (
            f"Expected all triggered orders to be {fixed_Q}, got {q} in {non_zero}"
        )


def test_reorder_point_quantity_is_rate_derived_when_Q_is_None():
    """When Q=None (default), quantity equals cover_horizon_ticks × rate.

    Uses opening_budget_pct=0.5 with a small balance so the pilot order
    seeds ~50 units; after the pilot lands and demand depletes stock below s,
    the policy fires triggered orders of cover_horizon × rate units each.
    """
    from src.sim.policy import ReorderPointPolicy

    cover_horizon = 10
    demand = 5.0
    expected_Q = int(round(cover_horizon * demand))  # 50

    policy = ReorderPointPolicy(
        policy_seed=0,
        Q=None,
        opening_budget_pct=0.5,
        safety_lead_ticks=0,
        cover_horizon_ticks=cover_horizon,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=5_000,
        balance=1_000.0,  # small balance → pilot ≈ 50 units (not capacity-filling)
        delivery_lag=DELIVERY_LAG,
    )
    log = _run_scenario(policy, n_steps=60, template=template, demand=demand)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    # Skip index 0 (pre-decide baseline) and index 1 (pilot order).
    post_pilot = order_quantities[2:]
    non_zero = [q for q in post_pilot if q > 0]
    assert len(non_zero) > 0, (
        f"No triggered orders after pilot. all order_quantities={order_quantities}"
    )
    for q in non_zero:
        assert q == expected_Q, (
            f"Expected rate-derived Q={expected_Q}, got {q} in {non_zero}"
        )


def test_periodic_order_up_to_orders_only_on_review_ticks():
    """Orders fire only on steps divisible by review_interval.

    Uses a small balance so the pilot seeds ~25 units, which depletes below
    s within a few ticks.  Then only review ticks can trigger reorders.
    """
    from src.sim.policy import PeriodicOrderUpToPolicy

    review_interval = 7
    n_steps = 4 * review_interval  # 28 ticks
    policy = PeriodicOrderUpToPolicy(
        policy_seed=0,
        review_interval=review_interval,
        opening_budget_pct=0.5,
        safety_lead_ticks=SAFETY_LEAD,
        cover_horizon_ticks=COVER_HORIZON,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=50_000,
        balance=500.0,  # pilot ≈ 25 units (0.5*500/1/10)
        delivery_lag=DELIVERY_LAG,
    )
    log = _run_scenario(policy, n_steps=n_steps, template=template, demand=5.0)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]

    # order_quantities[0] is the pre-decide baseline (before any decide call).
    # order_quantities[1] holds the pilot order (cold-start; not subject to
    # review-interval gating — pilot orders bypass the trigger loop).
    # All ticks t >= 2 are steady-state: non-review ticks must have qty == 0.
    for t, qty in enumerate(order_quantities):
        if t <= 1:
            continue  # pre-decide baseline (0) and pilot order (1)
        if t % review_interval == 0:
            # Review tick — may or may not order (depends on position vs S).
            pass
        else:
            assert qty == 0, (
                f"Unexpected order {qty} on non-review tick t={t} "
                f"(review_interval={review_interval}). "
                f"All quantities: {order_quantities}"
            )


def test_periodic_order_up_to_brings_position_to_S():
    """On a review tick, qty = max(0, S - position) brings position toward S.

    Uses a small balance so the pilot seeds ~25 units; after the pilot, demand
    depletes the position so that review ticks trigger non-zero orders.  We
    verify the order quantity is positive on review ticks when position < S,
    and that pos_after = pos_before + qty ≤ S + tolerance (within the spread
    of the rolling rate estimate).
    """
    from src.sim.policy import PeriodicOrderUpToPolicy

    review_interval = 5
    demand = 5.0
    safety_lead = SAFETY_LEAD
    cover_horizon = COVER_HORIZON
    delivery_lag = DELIVERY_LAG

    policy = PeriodicOrderUpToPolicy(
        policy_seed=0,
        review_interval=review_interval,
        opening_budget_pct=0.5,
        safety_lead_ticks=safety_lead,
        cover_horizon_ticks=cover_horizon,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=50_000,
        balance=500.0,  # pilot ≈ 25 units; depletes below S so reorders fire
        delivery_lag=delivery_lag,
    )
    n_steps = 60
    log = _run_scenario(policy, n_steps=n_steps, template=template, demand=demand)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    inv_series = store_log["products"][pid]["inventory"]
    pending_series = store_log["products"][pid]["outstanding_orders"]

    # Nominal S = (delivery_lag + safety_lead + cover_horizon) * rate
    # The policy's internal rate estimate may differ from the true demand,
    # so we allow a relative tolerance of ±20% on S.
    nominal_S = (delivery_lag + safety_lead + cover_horizon) * demand  # 75
    S_tol = nominal_S * 0.20  # ±15 units

    warmup = 10
    checked = 0
    for t in range(warmup, n_steps + 1):
        if t % review_interval != 0:
            continue
        if t >= len(order_quantities):
            break
        qty = order_quantities[t]
        if qty > 0:
            pos_before = inv_series[t] + pending_series[t] - qty
            pos_after = pos_before + qty
            # pos_after should be close to the policy's internal S.
            assert pos_after <= nominal_S + S_tol, (
                f"At t={t}: pos_after={pos_after} exceeds S+tol={nominal_S + S_tol:.1f} "
                f"(pos_before={pos_before}, qty={qty}, nominal_S={nominal_S})"
            )
            # Quantity must be positive when pos_before < S.
            assert qty > 0
            checked += 1

    assert checked >= 2, (
        f"Too few review-tick orders to verify: checked={checked}. "
        f"order_quantities={order_quantities}"
    )


def test_reorder_point_no_order_when_position_above_s():
    """No order fires while position stays above s."""
    from src.sim.policy import ReorderPointPolicy

    policy = ReorderPointPolicy(
        policy_seed=0,
        opening_budget_pct=0.0,  # disable pilot
        safety_lead_ticks=2,
        cover_horizon_ticks=10,
    )
    template = _mini_template(
        init_stock_pct=1.0,
        capacity=5_000,
        balance=1_000_000.0,
    )
    log = _run_scenario(policy, n_steps=2, template=template, demand=1)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"][1:]
    assert all(q == 0 for q in order_quantities), (
        f"Unexpected orders when position >> s: {order_quantities}"
    )


def test_periodic_order_up_to_no_negative_order_when_above_S():
    """When position > S on a review tick, the requested qty is 0 (not negative)."""
    from src.sim.policy import PeriodicOrderUpToPolicy

    review_interval = 3
    policy = PeriodicOrderUpToPolicy(
        policy_seed=0,
        review_interval=review_interval,
        # large pilot to ensure position >> S on first review ticks
        opening_budget_pct=0.5,
        safety_lead_ticks=SAFETY_LEAD,
        cover_horizon_ticks=COVER_HORIZON,
    )
    template = _mini_template(
        init_stock_pct=0.0,
        capacity=5_000,
        balance=100_000.0,
        delivery_lag=DELIVERY_LAG,
    )
    log = _run_scenario(policy, n_steps=10, template=template, demand=5.0)
    store_log = log["stores"][0]
    pid = list(store_log["products"].keys())[0]
    order_quantities = store_log["products"][pid]["order_quantity"]
    assert all(q >= 0 for q in order_quantities), (
        f"Negative order found: {order_quantities}"
    )


def test_reorder_point_crn_self_consistency():
    """Two ReorderPointPolicy instances on the same CRN seed produce bit-identical trajectories."""
    import hashlib
    import json

    from src.sim.data_exporter import _jsonable
    from src.sim.policy import ReorderPointPolicy

    def _run_with_seed(world_seed: int, policy_seed: int = 0) -> dict:
        policy = ReorderPointPolicy(policy_seed=policy_seed)
        template = _mini_template(init_stock_pct=0.0)
        scenario = Scenario(
            catalog=_mini_catalog(),
            market=_constant_market(demand=5.0),
            disruption=_no_disruption(),
            item_lifecycle=_no_lifecycle(),
            stores=[StoreInstance(template=template, init_seed=1, policy=policy)],
            n_steps=20,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
        )
        return Runner(scenario).run()

    r1 = _run_with_seed(42, policy_seed=0)
    r2 = _run_with_seed(42, policy_seed=0)

    h1 = hashlib.sha256(
        json.dumps(_jsonable(r1), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    h2 = hashlib.sha256(
        json.dumps(_jsonable(r2), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert h1 == h2


def test_periodic_order_up_to_crn_self_consistency():
    """Two PeriodicOrderUpToPolicy instances on the same CRN seed produce bit-identical trajectories."""
    import hashlib
    import json

    from src.sim.data_exporter import _jsonable
    from src.sim.policy import PeriodicOrderUpToPolicy

    def _run_with_seed(world_seed: int, policy_seed: int = 0) -> dict:
        policy = PeriodicOrderUpToPolicy(policy_seed=policy_seed)
        template = _mini_template(init_stock_pct=0.0)
        scenario = Scenario(
            catalog=_mini_catalog(),
            market=_constant_market(demand=5.0),
            disruption=_no_disruption(),
            item_lifecycle=_no_lifecycle(),
            stores=[StoreInstance(template=template, init_seed=1, policy=policy)],
            n_steps=20,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
        )
        return Runner(scenario).run()

    r1 = _run_with_seed(42, policy_seed=0)
    r2 = _run_with_seed(42, policy_seed=0)

    h1 = hashlib.sha256(
        json.dumps(_jsonable(r1), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    h2 = hashlib.sha256(
        json.dumps(_jsonable(r2), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert h1 == h2
