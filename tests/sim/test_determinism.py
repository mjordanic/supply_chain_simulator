"""T1: Determinism / replay tests for the runner skeleton (issue 03).

Three invariants from ADR 0001 (legacy Store engine):

1. Same ``world_seed`` → bit-identical world streams (market_supply,
   market_demand, event occurrences, raw per-step demand draws).
2. Same ``Scenario`` with two different attached ``Policy``s → identical
   world streams. Policy choice cannot leak into the world.
3. Two stores instantiated from the same ``(template, init_seed)`` start at
   bit-identical step 0 (capacity, balance, active SKU set, inventory)
   regardless of which ``Policy`` is attached.

Graph-level invariants (issue 09 — re-stated for two-factories-two-shops):

G1. Same ``world_seed`` → identical cash trajectory across all graph nodes.
G2. Same ``world_seed`` + policy swap on one shop → same world_rng and
    allocation_rng sequences (policy swap is orthogonal to world streams).
G3. Two shops instantiated from the same ``init_seed`` start with bit-
    identical step-0 state (inventory, cash, carried_products) regardless
    of attached policy.
G4. Different ``world_seed`` → diverging cash trajectories (sanity check).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from src.sim.distributions import Constant, Normal, Uniform
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    NoopPolicy,
    OrderUpToPolicy,
    Policy,
    StaticFactoryPolicy,
)
from src.sim.runner import GraphRunner, Runner, build_graph_world
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    StoreInstance,
    StoreTemplate,
    load_catalog,
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


# ---------------------------------------------------------------------------
# Graph-level invariants (G1-G4) — two-factories-two-shops scenario
# ---------------------------------------------------------------------------

def _graph_catalog():
    return load_catalog([
        {
            "name": "Widget",
            "category": "general",
            "related_products": [],
            "base_price": 12.0,
            "unit_cost": 5.0,
            "seasonality": "all_season",
        }
    ])


def _graph_market_params() -> MarketParams:
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=100,
        min_value=0.5,
        max_value=2.0,
        stage_multipliers={
            "introduction": 1.0, "growth": 1.0, "maturity": 1.0,
            "decline": 1.0, "dead": 0.0,
        },
        price_elasticity=0.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(15),
    )


def _graph_lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _build_2f2s_scenario(
    world_seed: int = 42,
    n_steps: int = 20,
    shop1_policy=None,
    shop2_policy=None,
    shop1_init_seed: int = 3,
    shop2_init_seed: int = 4,
) -> Scenario:
    """Build a 2-factory / 2-shop / 2-sink graph scenario."""
    catalog = _graph_catalog()
    pid = catalog[0].product_id

    f_lo = FactoryNode(
        id="f-lo", region="US", init_seed=1,
        produces_product_id=pid, unit_cost=5.0, capacity_per_tick=30,
        inventory=100, list_price=5.0, cash=0.0,
    )
    f_lo.policy = StaticFactoryPolicy(capacity_per_tick=30, unit_cost=5.0, policy_seed=10)

    f_hi = FactoryNode(
        id="f-hi", region="US", init_seed=2,
        produces_product_id=pid, unit_cost=8.0, capacity_per_tick=60,
        inventory=100, list_price=8.0, cash=0.0,
    )
    f_hi.policy = StaticFactoryPolicy(capacity_per_tick=60, unit_cost=8.0, policy_seed=11)

    shop1 = IntermediateNode(
        id="shop-1", region="US", init_seed=shop1_init_seed,
        carried_products={pid}, capacity=300, tags=["shop"],
        inventory={pid: 30}, pending={}, list_prices={pid: 12.0},
        min_order_imposed={pid: 0}, cash=1000.0,
    )
    if shop1_policy is None:
        shop1.policy = OrderUpToPolicy(
            cover_horizon_ticks=14, safety_lead_pct_of_lag=1 / 3,
            delivery_lag=3, unit_cost=5.0, list_price_out=12.0, policy_seed=20,
        )
    else:
        shop1.policy = shop1_policy

    shop2 = IntermediateNode(
        id="shop-2", region="US", init_seed=shop2_init_seed,
        carried_products={pid}, capacity=300, tags=["shop"],
        inventory={pid: 30}, pending={}, list_prices={pid: 12.0},
        min_order_imposed={pid: 0}, cash=1000.0,
    )
    if shop2_policy is None:
        shop2.policy = OrderUpToPolicy(
            cover_horizon_ticks=14, safety_lead_pct_of_lag=1 / 3,
            delivery_lag=3, unit_cost=5.0, list_price_out=12.0, policy_seed=21,
        )
    else:
        shop2.policy = shop2_policy

    sink1 = DemandSinkNode(
        id="sink-1", region="US", init_seed=5,
        product_id=pid, demand_dist=Normal(mean=15, std=3),
        income_rate=300.0, cash=1000.0,
    )
    sink1.policy = DefaultDemandSinkPolicy(policy_seed=30)

    sink2 = DemandSinkNode(
        id="sink-2", region="US", init_seed=6,
        product_id=pid, demand_dist=Normal(mean=15, std=3),
        income_rate=300.0, cash=1000.0,
    )
    sink2.policy = DefaultDemandSinkPolicy(policy_seed=31)

    edges = [
        EdgeSpec(supplier_id="f-lo",   buyer_id="shop-1", default_lead_time=4),
        EdgeSpec(supplier_id="f-hi",   buyer_id="shop-1", default_lead_time=2),
        EdgeSpec(supplier_id="f-lo",   buyer_id="shop-2", default_lead_time=4),
        EdgeSpec(supplier_id="f-hi",   buyer_id="shop-2", default_lead_time=2),
        EdgeSpec(supplier_id="shop-1", buyer_id="sink-1", default_lead_time=1),
        EdgeSpec(supplier_id="shop-2", buyer_id="sink-2", default_lead_time=1),
    ]

    return Scenario(
        catalog=catalog,
        market=_graph_market_params(),
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=["US"],
            severity=Constant(0.0),
            duration=Constant(1),
        ),
        item_lifecycle=_graph_lifecycle_params(),
        stores=[],
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=world_seed,
        nodes=[
            NodeInstance(node=f_lo,   init_seed=1),
            NodeInstance(node=f_hi,   init_seed=2),
            NodeInstance(node=shop1,  init_seed=shop1_init_seed),
            NodeInstance(node=shop2,  init_seed=shop2_init_seed),
            NodeInstance(node=sink1,  init_seed=5),
            NodeInstance(node=sink2,  init_seed=6),
        ],
        edges=edges,
    )


def _extract_cash_trajectory(log: dict) -> dict[str, list[float]]:
    """Return {node_id: [cash_t0, cash_t1, ...]} from a GraphRunner log."""
    node_ids: list[str] = []
    if log["ticks"]:
        node_ids = list(log["ticks"][0]["node_cash"].keys())
    return {
        nid: [tick["node_cash"][nid] for tick in log["ticks"]]
        for nid in node_ids
    }


def test_g1_same_world_seed_identical_cash_trajectory():
    """G1: same world_seed → identical cash trajectory across all graph nodes."""
    s1 = _build_2f2s_scenario(world_seed=42, n_steps=20)
    s2 = _build_2f2s_scenario(world_seed=42, n_steps=20)

    log1 = GraphRunner(s1).run()
    log2 = GraphRunner(s2).run()

    traj1 = _extract_cash_trajectory(log1)
    traj2 = _extract_cash_trajectory(log2)

    assert traj1 == traj2, (
        "Cash trajectories diverged despite identical world_seed — "
        "simulation is not deterministic."
    )


def test_g2_policy_swap_orthogonal_to_world_and_allocation_rng():
    """G2: policy swap on one shop leaves world_rng and allocation_rng unchanged.

    Two builds with the same world_seed but a different shop-2 policy must
    produce identical RNG trajectories (world_rng and allocation_rng are
    seeded from world_seed only, not from policy_seed).
    """
    world_seed = 42

    class _BurnShopPolicy(OrderUpToPolicy):
        """Burns extra policy_rng values on every decide call."""

        def decide(self, obs, central_table):
            for _ in range(13):
                self.policy_rng.random()
            return super().decide(obs, central_table)

    s_normal = _build_2f2s_scenario(world_seed=world_seed, n_steps=15)
    s_burn = _build_2f2s_scenario(
        world_seed=world_seed,
        n_steps=15,
        shop2_policy=_BurnShopPolicy(
            cover_horizon_ticks=14, safety_lead_pct_of_lag=1 / 3,
            delivery_lag=3, unit_cost=5.0, list_price_out=12.0, policy_seed=999,
        ),
    )

    g_normal = build_graph_world(s_normal)
    g_burn = build_graph_world(s_burn)

    # world_rng must start identically.
    wv1 = [g_normal.world_rng.random() for _ in range(10)]
    wv2 = [g_burn.world_rng.random() for _ in range(10)]
    assert wv1 == wv2, "world_rng diverged after policy swap at build time"

    # allocation_rng must start identically.
    av1 = [g_normal.allocation_rng.random() for _ in range(10)]
    av2 = [g_burn.allocation_rng.random() for _ in range(10)]
    assert av1 == av2, "allocation_rng diverged after policy swap at build time"


def test_g3_same_init_seed_identical_step0_state():
    """G3: two shops with the same init_seed start with bit-identical step-0 state.

    inventory, cash, and carried_products must be equal regardless of
    which policy is attached.
    """
    s = _build_2f2s_scenario(
        world_seed=42, n_steps=5,
        shop1_init_seed=7, shop2_init_seed=7,   # same init seed
    )
    gsim = build_graph_world(s)
    shop1 = gsim.nodes["shop-1"]
    shop2 = gsim.nodes["shop-2"]

    assert shop1.inventory == shop2.inventory, (
        "Shops with the same init_seed must start with identical inventory"
    )
    assert shop1.cash == shop2.cash, (
        "Shops with the same init_seed must start with identical cash"
    )
    assert shop1.carried_products == shop2.carried_products, (
        "Shops with the same init_seed must start with identical carried_products"
    )


def test_g4_different_world_seed_diverges():
    """G4: different world_seed → diverging cash trajectories (sanity check)."""
    s1 = _build_2f2s_scenario(world_seed=1, n_steps=20)
    s2 = _build_2f2s_scenario(world_seed=9999, n_steps=20)

    log1 = GraphRunner(s1).run()
    log2 = GraphRunner(s2).run()

    traj1 = _extract_cash_trajectory(log1)
    traj2 = _extract_cash_trajectory(log2)

    # At least one node must diverge across the two world seeds.
    assert traj1 != traj2, (
        "Cash trajectories are identical despite different world_seeds — "
        "world_seed is not influencing the simulation."
    )
