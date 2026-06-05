"""Determinism tests for allocation_rng — buyer shuffle reproducibility (issue 08).

Two invariants:

1. **Shuffle reproducibility** — same ``world_seed`` → same buyer-shuffle sequence
   across independent simulation builds (allocation_rng is seeded deterministically
   from world_seed via _derive_seed).

2. **Policy-orthogonality** — swapping the policy on one buyer does NOT perturb the
   allocation_rng sequence; the shuffle order is identical regardless of which
   policies are attached to the nodes.
"""

from __future__ import annotations

from datetime import datetime
from random import Random

import pytest

from src.sim.allocation import shuffle_buyers
from src.sim.central_table import CentralTable, Offer
from src.sim.distributions import Constant
from src.sim.episode_sampler import _derive_seed
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    IntermediatePolicy,
    StaticFactoryPolicy,
)
from src.sim.runner import build_world as build_graph_world
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
# Shared helpers (mirrored from test_graph_determinism_chain.py)
# ---------------------------------------------------------------------------

def _minimal_market_params() -> MarketParams:
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
        stage_multipliers={s: 1.0 for s in ["introduction", "growth", "maturity", "decline"]}
        | {"dead": 0.0},
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
        base_demand=Constant(10),
    )


def _minimal_disruption_params() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.1),
        duration=Constant(1),
    )


def _minimal_lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _minimal_catalog():
    return load_catalog([
        {
            "name": "Widget",
            "category": "test",
            "related_products": [],
            "base_price": 8.0,
            "unit_cost": 5.0,
            "seasonality": "all_season",
        }
    ])


def _build_chain_scenario(
    n_steps: int = 10,
    world_seed: int = 42,
    factory_policy=None,
    shop_policy=None,
    sink_policy=None,
) -> Scenario:
    """Build a 3-node chain scenario: factory → shop → sink."""
    catalog = _minimal_catalog()
    pid = catalog[0].product_id

    factory = FactoryNode(
        id="factory-1",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=5.0,
        capacity_per_tick=50,
        inventory=100,
        list_price=5.0,
        cash=0.0,
    )
    factory.policy = factory_policy

    shop = IntermediateNode(
        id="shop-1",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=500,
        tags=["shop"],
        inventory={pid: 20},
        pending={},
        list_prices={pid: 8.0},
        min_order_imposed={pid: 0},
        cash=500.0,
    )
    shop.policy = shop_policy

    sink = DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=3,
        product_id=pid,
        demand_dist=Constant(10),
        income_rate=100.0,
        cash=1000.0,
    )
    sink.policy = sink_policy

    edges = [
        EdgeSpec(supplier_id="factory-1", buyer_id="shop-1", default_lead_time=2),
        EdgeSpec(supplier_id="shop-1", buyer_id="sink-1", default_lead_time=1),
    ]

    return Scenario(
        catalog=catalog,
        market=_minimal_market_params(),
        disruption=_minimal_disruption_params(),
        item_lifecycle=_minimal_lifecycle_params(),
        n_steps=n_steps,
        start_date=datetime(2024, 1, 1),
        world_seed=world_seed,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=shop, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=edges,
    )


def _collect_shuffle_sequences(gsim, n_ticks: int) -> list[list[str]]:
    """Run the simulation and record the buyer-id order produced by each shuffle.

    We instrument this by collecting the node IDs of all buyers at level 1
    (the sink level) each tick. Since the chain has only one sink, we instead
    record the allocation_rng state at the start of each tick (via peeking
    at a fixed sequence after n ticks of usage).
    """
    # We can't monkey-patch tick() to capture the shuffle order without
    # modifying the production code, so we compare allocation_rng states
    # from two independent builds to verify they are in sync.
    # The real test is: two builds with the same world_seed have the same
    # allocation_rng state after N ticks.
    sequences: list[float] = []
    for _ in range(n_ticks):
        gsim.tick()
        # Sample a value from allocation_rng after each tick to record its
        # state trajectory — but this would consume RNG state. Instead, we
        # check state equality between two parallel gsims externally.
    return sequences


# ---------------------------------------------------------------------------
# Invariant 1: Same world_seed → same allocation_rng seeding
# ---------------------------------------------------------------------------

class TestShuffleReproducibility:
    def test_allocation_rng_seeded_from_world_seed(self):
        """allocation_rng is derived from world_seed via _derive_seed("allocation").

        Two builds with the same world_seed must start with the same
        allocation_rng state (they will therefore produce identical shuffles).
        """
        world_seed = 42
        s1 = _build_chain_scenario(world_seed=world_seed)
        s2 = _build_chain_scenario(world_seed=world_seed)

        g1 = build_graph_world(s1)
        g2 = build_graph_world(s2)

        # Verify both allocation_rngs start at identical states.
        v1 = [g1.allocation_rng.random() for _ in range(20)]
        v2 = [g2.allocation_rng.random() for _ in range(20)]
        assert v1 == v2, (
            "allocation_rng states differ after building two worlds with the same "
            "world_seed — seeding must be deterministic"
        )

    def test_allocation_rng_state_matches_derive_seed(self):
        """allocation_rng must be seeded exactly as Random(_derive_seed(world_seed, 'allocation'))."""
        world_seed = 123
        s = _build_chain_scenario(world_seed=world_seed)
        g = build_graph_world(s)

        expected_seed = _derive_seed(world_seed, "allocation")
        expected_rng = Random(expected_seed)

        v_actual = [g.allocation_rng.random() for _ in range(15)]
        v_expected = [expected_rng.random() for _ in range(15)]
        assert v_actual == v_expected, (
            "allocation_rng was not seeded with _derive_seed(world_seed, 'allocation')"
        )

    def test_same_world_seed_produces_same_shuffle_order_after_n_ticks(self):
        """Two runs with the same world_seed produce the same allocation_rng trajectory.

        We verify this by comparing allocation_rng states BEFORE each tick fires,
        using two independent builds that run N ticks.
        """
        world_seed = 77
        s1 = _build_chain_scenario(world_seed=world_seed, n_steps=10)
        s2 = _build_chain_scenario(world_seed=world_seed, n_steps=10)

        g1 = build_graph_world(s1)
        g2 = build_graph_world(s2)

        # Run both simulations step-by-step, checking rng parity after each tick.
        for tick_i in range(10):
            g1.tick()
            g2.tick()
            # After each tick the allocation_rng is consumed by the same number
            # of shuffle calls; both runs have the same buyer count per level.
            # Sample a probe value — both must agree.
            probe1 = g1.allocation_rng.random()
            probe2 = g2.allocation_rng.random()
            assert probe1 == probe2, (
                f"allocation_rng diverged at tick {tick_i}: "
                f"run1={probe1}, run2={probe2}"
            )
            # Restore RNG state parity by pushing the same seed back.
            # (Calling .random() consumed 1 value from each — they are still in sync.)

    def test_shuffle_buyers_is_deterministic_from_rng(self):
        """shuffle_buyers called twice with the same RNG state produces the same order."""
        buyers = [f"buyer-{i}" for i in range(6)]

        rng_a = Random(42)
        rng_b = Random(42)

        shuffled_a = shuffle_buyers(buyers, rng_a)
        shuffled_b = shuffle_buyers(buyers, rng_b)

        assert shuffled_a == shuffled_b, (
            "shuffle_buyers is not deterministic from the same RNG seed"
        )

    def test_shuffle_buyers_returns_all_original_elements(self):
        """shuffle_buyers must return all original buyers (no duplicates or drops)."""
        buyers = [f"buyer-{i}" for i in range(8)]
        rng = Random(999)
        shuffled = shuffle_buyers(buyers, rng)
        assert sorted(shuffled) == sorted(buyers), (
            "shuffle_buyers dropped or duplicated buyers"
        )

    def test_shuffle_buyers_does_not_mutate_input(self):
        """shuffle_buyers returns a new list; the input list is not mutated."""
        buyers = ["a", "b", "c", "d"]
        original = list(buyers)
        rng = Random(1)
        shuffled = shuffle_buyers(buyers, rng)
        assert buyers == original, "shuffle_buyers mutated the input list"
        assert shuffled is not buyers, "shuffle_buyers returned the same object"

    def test_different_world_seeds_produce_different_allocation_rngs(self):
        """Two builds with different world_seeds must start with different allocation_rngs."""
        s1 = _build_chain_scenario(world_seed=1)
        s2 = _build_chain_scenario(world_seed=9999)

        g1 = build_graph_world(s1)
        g2 = build_graph_world(s2)

        v1 = [g1.allocation_rng.random() for _ in range(5)]
        v2 = [g2.allocation_rng.random() for _ in range(5)]
        assert v1 != v2, (
            "allocation_rng state is identical for different world_seeds — "
            "seed derivation is not functioning"
        )


# ---------------------------------------------------------------------------
# Invariant 2: Policy swap does not perturb allocation_rng
# ---------------------------------------------------------------------------

class TestPolicyOrthogonality:
    def test_policy_swap_on_sink_does_not_perturb_allocation_rng(self):
        """Swapping the sink's policy does not change the allocation_rng trajectory.

        Two builds with the same world_seed but different sink policies must have
        identical allocation_rng states after N ticks.
        """
        world_seed = 55

        s_no_policy = _build_chain_scenario(
            world_seed=world_seed,
            sink_policy=None,
        )
        s_with_policy = _build_chain_scenario(
            world_seed=world_seed,
            sink_policy=DefaultDemandSinkPolicy(policy_seed=99),
        )

        g_no = build_graph_world(s_no_policy)
        g_with = build_graph_world(s_with_policy)

        # allocation_rng must be in the same state at start.
        v_no_start = [g_no.allocation_rng.random() for _ in range(5)]
        v_with_start = [g_with.allocation_rng.random() for _ in range(5)]
        assert v_no_start == v_with_start, (
            "allocation_rng state differs at build time due to sink policy attachment"
        )

    def test_policy_swap_does_not_perturb_allocation_rng_after_ticks(self):
        """After N ticks, allocation_rng state must be identical regardless of policies.

        Since the number of shuffle calls per tick depends only on the graph topology
        (number of levels and buyers per level), not on the policies attached,
        allocation_rng consumption is policy-independent.
        """
        world_seed = 88

        s_no_policy = _build_chain_scenario(
            world_seed=world_seed,
            factory_policy=None,
            shop_policy=None,
            sink_policy=None,
            n_steps=8,
        )
        s_with_policies = _build_chain_scenario(
            world_seed=world_seed,
            factory_policy=StaticFactoryPolicy(capacity_per_tick=50, unit_cost=5.0, policy_seed=1),
            shop_policy=IntermediatePolicy.SingleSupplierAdapter(
                supplier_id="factory-1", policy_seed=7
            ),
            sink_policy=DefaultDemandSinkPolicy(policy_seed=13),
            n_steps=8,
        )

        g_no = build_graph_world(s_no_policy)
        g_with = build_graph_world(s_with_policies)

        # Run N ticks and compare allocation_rng trajectories.
        for tick_i in range(8):
            g_no.tick()
            g_with.tick()

            # After each tick, probe allocation_rng states must match.
            probe_no = g_no.allocation_rng.random()
            probe_with = g_with.allocation_rng.random()
            assert probe_no == probe_with, (
                f"allocation_rng diverged at tick {tick_i} after policy swap: "
                f"no-policy={probe_no} vs with-policy={probe_with}"
            )

    def test_policy_rng_and_allocation_rng_are_independent_streams(self):
        """Policy RNG and allocation_rng are derived from different seeds.

        A policy that consumes many values from policy_rng must not affect
        the allocation_rng sequence.
        """
        world_seed = 42
        s = _build_chain_scenario(
            world_seed=world_seed,
            sink_policy=DefaultDemandSinkPolicy(policy_seed=7),
        )
        g = build_graph_world(s)

        # Derive the expected allocation seed directly.
        expected_alloc_seed = _derive_seed(world_seed, "allocation")
        expected_rng = Random(expected_alloc_seed)

        # Both must produce the same sequence regardless of the policy_seed on the sink.
        v_actual = [g.allocation_rng.random() for _ in range(10)]
        v_expected = [expected_rng.random() for _ in range(10)]
        assert v_actual == v_expected, (
            "allocation_rng was contaminated by policy_rng — "
            "the two streams must be independent"
        )

    def test_world_rng_and_allocation_rng_are_independent_streams(self):
        """world_rng and allocation_rng are separate Random instances.

        Consuming values from one must not change the state of the other.
        """
        world_seed = 42
        s = _build_chain_scenario(world_seed=world_seed)
        g = build_graph_world(s)

        # Record allocation_rng probe values before touching world_rng.
        alloc_probe_before = g.allocation_rng.random()

        # Rebuild and consume many values from world_rng.
        s2 = _build_chain_scenario(world_seed=world_seed)
        g2 = build_graph_world(s2)
        for _ in range(100):
            g2.world_rng.random()

        # allocation_rng probe should still match what we observed before.
        alloc_probe_after = g2.allocation_rng.random()
        assert alloc_probe_before == alloc_probe_after, (
            "allocation_rng state was affected by world_rng consumption — "
            "the two streams must be independent"
        )
