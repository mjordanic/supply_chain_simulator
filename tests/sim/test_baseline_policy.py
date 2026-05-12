"""T5: BaselinePolicy contract tests (issue 06).

Each test constructs a synthetic observation and verifies one acceptance
criterion from the issue. The full integration with the Runner / Store
observation shape lands in issue 07; these tests target the policy in
isolation against the observation contract documented in
``BaselinePolicy.decide``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.sim.distributions import Uniform
from src.sim.policy import BaselinePolicy


def _baseline_obs(
    *,
    step: int = 0,
    inventory: dict[str, int] | None = None,
    capacity: int = 100,
    pending: dict[str, int] | None = None,
    active: set[str] | None = None,
    prices: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    related: dict[str, list[tuple[str, float]]] | None = None,
    balance: float = 10000.0,
    sales: dict[str, int] | None = None,
    promotions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the full observation shape BaselinePolicy.decide expects."""
    if inventory is None:
        inventory = {"P0000": 10, "P0001": 20, "P0002": 5}
    if pending is None:
        pending = {pid: 0 for pid in inventory}
    if active is None:
        active = set(inventory.keys())
    if prices is None:
        prices = {pid: 20.0 for pid in inventory}
    if costs is None:
        costs = {pid: 10.0 for pid in inventory}
    if related is None:
        related = {pid: [] for pid in inventory}
    if sales is None:
        sales = {pid: 0 for pid in inventory}
    if promotions is None:
        promotions = {}
    return {
        "current_sim_step": step,
        "inventory": inventory,
        "max_capacity": capacity,
        "outstanding_orders": pending,
        "active_products": active,
        "product_prices": prices,
        "related_products": related,
        "balance": balance,
        "unit_costs": costs,
        "sales": sales,
        "promotions": promotions,
    }


def _policy(**overrides: Any) -> BaselinePolicy:
    """BaselinePolicy with sane defaults that pass all four contract tests."""
    defaults = dict(
        policy_seed=42,
        min_qty=5,
        init_qty_factor=0.3,
        promo_len=Uniform(3, 10),
        promo_cd_len=5,
        review_interval=5,
        promo_threshold=0.7,
        target_active_count=3,
        slow_sales_limit=3,
        stock_lo_ratio=0.2,
        stock_hi_ratio=0.6,
        price_up_factor=1.1,
        price_down_factor=0.9,
        history_window=4,
        trend_threshold=0.05,
        cross_price_adj=0.05,
        max_history=50,
        inactive_price_factor=0.5,
        reorder_factor=0.3,
        qty_factor=0.5,
        order_cd_len=10,
        order_cd_jitter=0.3,
        promo_discount=0.7,
    )
    defaults.update(overrides)
    return BaselinePolicy(**defaults)


def test_no_order_during_product_cooldown():
    """An active cooldown on a product blocks any order for it."""
    p = _policy()
    p.order_cd["P0000"] = 50  # cooldown active until step 50
    obs = _baseline_obs(step=10)

    decisions = p.decide(obs)
    assert decisions["order"]["P0000"] == 0


def test_no_order_during_cooldown_after_decision_seeded():
    """When decide seeds a cooldown, the next decide within that window
    returns zero for the same product."""
    p = _policy(min_qty=1)
    # Seed the first-order flag on the policy — observation no longer carries it.
    p.needs_init_order = {"P0000", "P0001"}
    # Make replenishment look attractive: low stock, no pending, plenty of balance.
    obs = _baseline_obs(
        step=0,
        inventory={"P0000": 0, "P0001": 0},
        pending={"P0000": 0, "P0001": 0},
        capacity=100,
        active={"P0000", "P0001"},
        prices={"P0000": 20.0, "P0001": 20.0},
        costs={"P0000": 10.0, "P0001": 10.0},
        related={"P0000": [], "P0001": []},
    )
    decisions = p.decide(obs)
    ordered = {pid for pid, q in decisions["order"].items() if q > 0}
    assert ordered, "test setup should have produced at least one order"

    # Next step is still inside any seeded cooldown. Verify no orders for
    # those products.
    obs_next = _baseline_obs(
        step=1,
        inventory={"P0000": 0, "P0001": 0},
        pending={pid: 0 for pid in ordered},
        capacity=100,
        active={"P0000", "P0001"},
        prices={"P0000": 20.0, "P0001": 20.0},
        costs={"P0000": 10.0, "P0001": 10.0},
        related={"P0000": [], "P0001": []},
    )
    next_decisions = p.decide(obs_next)
    for pid in ordered:
        assert pid in p.order_cd
        if 1 < p.order_cd[pid]:
            assert next_decisions["order"][pid] == 0


def test_orders_never_exceed_free_capacity():
    """Sum of ordered units across all products never exceeds free capacity."""
    p = _policy(min_qty=1)
    inventory = {"P0000": 5, "P0001": 5, "P0002": 5}
    pending = {"P0000": 10, "P0001": 0, "P0002": 0}
    capacity = 100
    p.needs_init_order = set(inventory.keys())  # maximises desired order qty
    obs = _baseline_obs(
        step=0,
        inventory=inventory,
        pending=pending,
        capacity=capacity,
        active=set(inventory.keys()),
        prices={pid: 20.0 for pid in inventory},
        costs={pid: 10.0 for pid in inventory},
        related={pid: [] for pid in inventory},
        balance=1_000_000.0,  # remove budget cap so capacity is the only ceiling
    )

    decisions = p.decide(obs)
    free_capacity = capacity - (sum(inventory.values()) + sum(pending.values()))
    assert sum(decisions["order"].values()) <= free_capacity


def test_orders_respect_capacity_when_inventory_almost_full():
    """When inventory is near capacity the total ordered qty stays at zero."""
    p = _policy(min_qty=5)
    inventory = {"P0000": 95, "P0001": 4, "P0002": 1}
    pending = {pid: 0 for pid in inventory}
    capacity = 100  # free capacity = 0
    p.needs_init_order = set(inventory.keys())
    obs = _baseline_obs(
        step=0,
        inventory=inventory,
        pending=pending,
        capacity=capacity,
        active=set(inventory.keys()),
        prices={pid: 20.0 for pid in inventory},
        costs={pid: 10.0 for pid in inventory},
        related={pid: [] for pid in inventory},
        balance=1_000_000.0,
    )
    decisions = p.decide(obs)
    assert sum(decisions["order"].values()) <= 0


def test_returned_prices_are_at_least_unit_cost():
    """Every returned price is >= the corresponding unit_cost."""
    p = _policy()
    inventory = {"P0000": 10, "P0001": 90, "P0002": 0}
    costs = {"P0000": 10.0, "P0001": 30.0, "P0002": 5.0}
    prices = {"P0000": 20.0, "P0001": 50.0, "P0002": 8.0}
    obs = _baseline_obs(
        step=0,
        inventory=inventory,
        capacity=100,
        active={"P0000", "P0001"},  # P0002 inactive
        prices=prices,
        costs=costs,
        related={
            "P0000": [("P0001", 0.8)],
            "P0001": [("P0000", 0.8)],
            "P0002": [],
        },
        promotions={"P0000": {"discount": 0.5, "duration": 5, "start_step": 0}},
    )
    decisions = p.decide(obs)
    for pid, price in decisions["price"].items():
        assert price >= costs[pid], f"{pid}: price {price} < cost {costs[pid]}"


def test_returned_prices_floor_when_inactive_factor_below_cost():
    """Even when the inactive_price_factor would drive a price below cost,
    the returned price never dips under unit_cost."""
    # inactive_price_factor=0.1; base_price 20; cost 18 -> raw 2.0 -> floored to 18.
    p = _policy(inactive_price_factor=0.1)
    obs = _baseline_obs(
        step=0,
        inventory={"P0000": 10},
        capacity=100,
        active=set(),  # everything inactive
        prices={"P0000": 20.0},
        costs={"P0000": 18.0},
        related={"P0000": []},
    )
    decisions = p.decide(obs)
    assert decisions["price"]["P0000"] >= 18.0


@pytest.mark.parametrize("review_interval", [3, 5, 7])
def test_activate_deactivate_only_on_review_step(review_interval: int):
    """activate / deactivate lists are non-empty only when step % review_interval == 0."""
    p = _policy(review_interval=review_interval, target_active_count=10)
    inventory = {"P0000": 10, "P0001": 5}
    obs_base = dict(
        inventory=inventory,
        capacity=100,
        active={"P0000"},
        prices={pid: 20.0 for pid in inventory},
        costs={pid: 10.0 for pid in inventory},
        related={pid: [] for pid in inventory},
    )

    for step in range(review_interval * 4):
        obs = _baseline_obs(step=step, **obs_base)
        decisions = p.decide(obs)
        if step % review_interval != 0:
            assert decisions["activate"] == [], f"step {step}: activate non-empty"
            assert decisions["deactivate"] == [], f"step {step}: deactivate non-empty"


def test_decide_consumes_only_policy_rng():
    """Two policies seeded identically produce identical action streams.

    This is the "policy_rng only" half of the seeding contract: BaselinePolicy
    must not reach for the global ``random`` module or any other shared RNG.
    """
    inventory = {"P0000": 80, "P0001": 80}  # high stock triggers the promo path
    obs = _baseline_obs(
        step=0,
        inventory=inventory,
        capacity=100,
        active=set(inventory.keys()),
        prices={pid: 20.0 for pid in inventory},
        costs={pid: 10.0 for pid in inventory},
        related={pid: [] for pid in inventory},
    )

    p1 = _policy(promo_threshold=0.5)
    p2 = _policy(promo_threshold=0.5)
    d1 = p1.decide(obs)
    d2 = p2.decide(obs)
    assert d1["promotions"] == d2["promotions"]
    assert d1["order"] == d2["order"]


def test_decide_returns_full_action_dict_shape():
    """The returned action dict carries every key Store.decide expects."""
    p = _policy()
    obs = _baseline_obs()
    decisions = p.decide(obs)

    assert set(decisions.keys()) == {
        "promotions",
        "order",
        "price",
        "activate",
        "deactivate",
    }


def test_kwargs_constructor_no_init_params_dict():
    """BaselinePolicy must accept hyperparameters via plain kwargs.

    The old broadcasting-dict pathway (``init_params=…, live_params=…``)
    is gone; this test guards against its return.
    """
    p = BaselinePolicy(
        policy_seed=0,
        min_qty=10,
        init_qty_factor=0.3,
        promo_len=Uniform(3, 10),
        promo_cd_len=5,
        review_interval=5,
        promo_threshold=0.7,
        target_active_count=3,
        slow_sales_limit=3,
        stock_lo_ratio=0.2,
        stock_hi_ratio=0.6,
        price_up_factor=1.1,
        price_down_factor=0.9,
        history_window=4,
        trend_threshold=0.05,
        cross_price_adj=0.05,
        max_history=50,
        inactive_price_factor=0.5,
        reorder_factor=0.3,
        qty_factor=0.5,
        order_cd_len=10,
        order_cd_jitter=0.3,
        promo_discount=0.7,
    )
    assert p.min_qty == 10
    assert p.review_interval == 5
    assert p.promo_threshold == 0.7
