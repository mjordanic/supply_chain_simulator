"""T1: Determinism / replay tests for the runner skeleton (issue 03).

Three invariants from ADR 0001:

1. Same ``world_seed`` → bit-identical world streams (market_supply,
   market_demand, event occurrences, raw per-step demand draws).
2. Same ``Scenario`` with two different attached ``Policy``s → identical
   world streams. Policy choice cannot leak into the world.
3. Two stores instantiated from the same ``(template, init_seed)`` start at
   bit-identical step 0 (capacity, balance, active SKU set, inventory)
   regardless of which ``Policy`` is attached.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.sim.distributions import Constant, Uniform
from src.sim.policy import NoopPolicy, Policy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    StoreInstance,
    StoreTemplate,
)


class _BurnRngPolicy(Policy):
    """Burns its ``policy_rng`` on every ``decide``.

    If world streams were leaking through the policy RNG, swapping
    ``NoopPolicy`` for this would shift them — so this is the foil for
    test 2.
    """

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        for _ in range(7):
            self.policy_rng.random()
        return {}


def _disruption_with_events() -> DisruptionParams:
    """Disruption params that actually fire events under the test seed."""
    return DisruptionParams(
        event_prob=0.5,
        types=["natural_disaster", "pandemic", "economic_crisis"],
        regions=["US", "EU"],
        severity=Constant(1.0),
        duration=Constant(3),
    )


def _template() -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=Uniform(800, 1200),
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
    )


def test_same_world_seed_bit_identical_streams(make_scenario):
    template = _template()
    s1 = make_scenario(
        stores=[StoreInstance(template=template, init_seed=1, policy=NoopPolicy())],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=42,
    )
    s2 = make_scenario(
        stores=[StoreInstance(template=template, init_seed=1, policy=NoopPolicy())],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=42,
    )

    log1 = Runner(s1).run()
    log2 = Runner(s2).run()

    assert log1["global"]["market_demand"] == log2["global"]["market_demand"]
    assert log1["global"]["market_supply"] == log2["global"]["market_supply"]
    assert (
        log1["global"]["events"]["occurrences"]
        == log2["global"]["events"]["occurrences"]
    )
    # Per-step raw demand traces match too.
    assert log1["stores"][0]["demand_trace"] == log2["stores"][0]["demand_trace"]


def test_different_seeds_diverge(make_scenario):
    """Sanity: different world seeds must NOT produce identical streams."""
    template = _template()
    s1 = make_scenario(
        stores=[StoreInstance(template=template, init_seed=1, policy=NoopPolicy())],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=1,
    )
    s2 = make_scenario(
        stores=[StoreInstance(template=template, init_seed=1, policy=NoopPolicy())],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=2,
    )

    log1 = Runner(s1).run()
    log2 = Runner(s2).run()

    assert log1["global"]["market_demand"] != log2["global"]["market_demand"]


def test_world_streams_invariant_under_policy_swap(make_scenario):
    template = _template()
    s_a = make_scenario(
        stores=[
            StoreInstance(
                template=template, init_seed=1, policy=NoopPolicy(policy_seed=1)
            )
        ],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=42,
    )
    s_b = make_scenario(
        stores=[
            StoreInstance(
                template=template,
                init_seed=1,
                policy=_BurnRngPolicy(policy_seed=999),
            )
        ],
        disruption=_disruption_with_events(),
        n_steps=20,
        world_seed=42,
    )

    log_a = Runner(s_a).run()
    log_b = Runner(s_b).run()

    assert log_a["global"]["market_demand"] == log_b["global"]["market_demand"]
    assert log_a["global"]["market_supply"] == log_b["global"]["market_supply"]
    assert (
        log_a["global"]["events"]["occurrences"]
        == log_b["global"]["events"]["occurrences"]
    )
    # Raw demand draws are world_rng output → also stable across the swap.
    assert log_a["stores"][0]["demand_trace"] == log_b["stores"][0]["demand_trace"]


def test_two_stores_same_init_seed_identical_step0(make_scenario):
    template = _template()
    s = make_scenario(
        stores=[
            StoreInstance(
                template=template, init_seed=7, policy=NoopPolicy(policy_seed=1)
            ),
            StoreInstance(
                template=template,
                init_seed=7,
                policy=_BurnRngPolicy(policy_seed=999),
            ),
        ],
        n_steps=10,
        world_seed=42,
    )
    runner = Runner(s)
    a, b = runner.stores

    assert a.capacity == b.capacity
    assert a.balance == b.balance
    assert a.holding_rate == b.holding_rate
    assert a.order_fee == b.order_fee
    assert a.delivery_lag == b.delivery_lag
    assert a.init_stock_pct == b.init_stock_pct
    # Same init_seed implies same active SKU pick, same order, same allocation.
    assert a.active_items == b.active_items
    assert a.inventory == b.inventory


def test_two_stores_different_init_seed_diverge(make_scenario):
    """Sanity: different init seeds must NOT produce identical step-0 state."""
    template = _template()
    s = make_scenario(
        stores=[
            StoreInstance(template=template, init_seed=1, policy=NoopPolicy()),
            StoreInstance(template=template, init_seed=2, policy=NoopPolicy()),
        ],
        n_steps=10,
        world_seed=42,
    )
    runner = Runner(s)
    a, b = runner.stores
    # At least one of capacity / active SKU set must differ to prove the
    # init_rng is actually keyed on init_seed.
    assert (a.capacity, a.active_items) != (b.capacity, b.active_items)
