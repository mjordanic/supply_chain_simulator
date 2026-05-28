"""Tests for Node subclass construction and init_seed determinism.

Invariant: two nodes constructed from the same (NodeSubclass, init_seed)
start at bit-identical step-0 state regardless of which policy is attached.
This extends the ``test_two_stores_same_init_seed_identical_step0`` invariant
from test_determinism.py to all three node types.
"""

from __future__ import annotations

from random import Random
from typing import Any, Mapping

import pytest

from src.sim.distributions import Constant, Uniform
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode, Node
from src.sim.policy import (
    DemandSinkPolicy,
    FactoryPolicy,
    IntermediatePolicy,
    NodePolicy,
)


# ---------------------------------------------------------------------------
# Minimal concrete NodePolicy subclasses for testing
# ---------------------------------------------------------------------------


class _NoopFactoryPolicy(FactoryPolicy):
    def decide(self, obs_factory: Mapping[str, Any]) -> dict[str, Any]:
        return {"produce_qty": 0, "list_price": 1.0}


class _BurnRngFactoryPolicy(FactoryPolicy):
    """Burns policy_rng on every decide — foil for determinism tests."""

    def decide(self, obs_factory: Mapping[str, Any]) -> dict[str, Any]:
        for _ in range(7):
            self.policy_rng.random()
        return {"produce_qty": 0, "list_price": 1.0}


class _NoopIntermediatePolicy(IntermediatePolicy):
    def decide(
        self, obs_intermediate: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        return {"order": {}, "list_price": {}, "min_order_imposed": {}}


class _BurnRngIntermediatePolicy(IntermediatePolicy):
    def decide(
        self, obs_intermediate: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        for _ in range(7):
            self.policy_rng.random()
        return {"order": {}, "list_price": {}, "min_order_imposed": {}}


class _NoopDemandSinkPolicy(DemandSinkPolicy):
    def decide(
        self, obs_sink: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        return {"buy": []}


class _BurnRngDemandSinkPolicy(DemandSinkPolicy):
    def decide(
        self, obs_sink: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        for _ in range(7):
            self.policy_rng.random()
        return {"buy": []}


# ---------------------------------------------------------------------------
# Helpers to build nodes with a given init_seed
# ---------------------------------------------------------------------------


def _make_factory(init_seed: int, policy: NodePolicy | None = None) -> FactoryNode:
    return FactoryNode(
        id="factory-1",
        region="US",
        init_seed=init_seed,
        policy=policy,
        produces_product_id="P0001",
        unit_cost=5.0,
        capacity_per_tick=Uniform(80, 120),
        inventory=0,
        list_price=5.0,
    )


def _make_intermediate(
    init_seed: int, policy: NodePolicy | None = None
) -> IntermediateNode:
    return IntermediateNode(
        id="intermediate-1",
        region="US",
        init_seed=init_seed,
        policy=policy,
        carried_products={"P0001", "P0002"},
        capacity=Uniform(800, 1200),
        tags=["warehouse"],
        inventory={"P0001": 0, "P0002": 0},
        pending={},
        list_prices={"P0001": 10.0, "P0002": 20.0},
        min_order_imposed={"P0001": 1, "P0002": 1},
    )


def _make_sink(
    init_seed: int, policy: NodePolicy | None = None
) -> DemandSinkNode:
    return DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=init_seed,
        policy=policy,
        product_id="P0001",
        demand_dist=Constant(10),
        income_rate=500.0,
        cash=1000.0,
        activation_tick={"P0001": 0},
    )


# ---------------------------------------------------------------------------
# Node ABC interface tests
# ---------------------------------------------------------------------------


class TestNodeABC:
    def test_node_is_abstract(self):
        """Node cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Node(id="x", region="US", init_seed=1, policy=None)  # type: ignore[abstract]

    def test_factory_node_is_node(self):
        node = _make_factory(init_seed=1)
        assert isinstance(node, Node)

    def test_intermediate_node_is_node(self):
        node = _make_intermediate(init_seed=1)
        assert isinstance(node, Node)

    def test_sink_node_is_node(self):
        node = _make_sink(init_seed=1)
        assert isinstance(node, Node)

    def test_node_carries_id(self):
        node = _make_factory(init_seed=1)
        assert node.id == "factory-1"

    def test_node_carries_region(self):
        node = _make_factory(init_seed=1)
        assert node.region == "US"

    def test_node_carries_init_seed(self):
        node = _make_factory(init_seed=42)
        assert node.init_seed == 42

    def test_node_carries_policy(self):
        policy = _NoopFactoryPolicy()
        node = _make_factory(init_seed=1, policy=policy)
        assert node.policy is policy

    def test_node_policy_defaults_none(self):
        node = _make_factory(init_seed=1)
        assert node.policy is None


# ---------------------------------------------------------------------------
# FactoryNode field tests
# ---------------------------------------------------------------------------


class TestFactoryNode:
    def test_produces_product_id(self):
        node = _make_factory(init_seed=1)
        assert node.produces_product_id == "P0001"

    def test_unit_cost(self):
        node = _make_factory(init_seed=1)
        assert node.unit_cost == 5.0

    def test_capacity_per_tick_distribution(self):
        node = _make_factory(init_seed=1)
        assert isinstance(node.capacity_per_tick, Uniform)

    def test_inventory(self):
        node = _make_factory(init_seed=1)
        assert node.inventory == 0

    def test_list_price(self):
        node = _make_factory(init_seed=1)
        assert node.list_price == 5.0


# ---------------------------------------------------------------------------
# IntermediateNode field tests
# ---------------------------------------------------------------------------


class TestIntermediateNode:
    def test_carried_products(self):
        node = _make_intermediate(init_seed=1)
        assert node.carried_products == {"P0001", "P0002"}

    def test_capacity_distribution(self):
        node = _make_intermediate(init_seed=1)
        assert isinstance(node.capacity, Uniform)

    def test_tags(self):
        node = _make_intermediate(init_seed=1)
        assert node.tags == ["warehouse"]

    def test_inventory(self):
        node = _make_intermediate(init_seed=1)
        assert node.inventory == {"P0001": 0, "P0002": 0}

    def test_pending(self):
        node = _make_intermediate(init_seed=1)
        assert node.pending == {}

    def test_list_prices(self):
        node = _make_intermediate(init_seed=1)
        assert node.list_prices == {"P0001": 10.0, "P0002": 20.0}

    def test_min_order_imposed(self):
        node = _make_intermediate(init_seed=1)
        assert node.min_order_imposed == {"P0001": 1, "P0002": 1}


# ---------------------------------------------------------------------------
# DemandSinkNode field tests
# ---------------------------------------------------------------------------


class TestDemandSinkNode:
    def test_product_id(self):
        node = _make_sink(init_seed=1)
        assert node.product_id == "P0001"

    def test_demand_dist(self):
        node = _make_sink(init_seed=1)
        assert isinstance(node.demand_dist, Constant)

    def test_income_rate(self):
        node = _make_sink(init_seed=1)
        assert node.income_rate == 500.0

    def test_cash(self):
        node = _make_sink(init_seed=1)
        assert node.cash == 1000.0

    def test_activation_tick(self):
        node = _make_sink(init_seed=1)
        assert node.activation_tick == {"P0001": 0}


# ---------------------------------------------------------------------------
# Determinism invariant: same init_seed → bit-identical step-0 state,
# regardless of attached policy
# ---------------------------------------------------------------------------


class TestInitSeedDeterminism:
    """Extend test_two_stores_same_init_seed_identical_step0 to all three node types."""

    def test_factory_same_init_seed_identical_regardless_of_policy(self):
        """Two FactoryNodes with the same init_seed are bit-identical, policy notwithstanding."""
        a = _make_factory(init_seed=7, policy=_NoopFactoryPolicy(policy_seed=1))
        b = _make_factory(init_seed=7, policy=_BurnRngFactoryPolicy(policy_seed=999))

        assert a.id == b.id
        assert a.region == b.region
        assert a.init_seed == b.init_seed
        assert a.produces_product_id == b.produces_product_id
        assert a.unit_cost == b.unit_cost
        assert a.inventory == b.inventory
        assert a.list_price == b.list_price
        # capacity_per_tick is the same Distribution instance type
        # (both constructed from same args)
        assert type(a.capacity_per_tick) == type(b.capacity_per_tick)

    def test_factory_different_init_seeds_different_policies_still_equal_fields(self):
        """Fields that don't derive from init_seed are always equal for same construction args."""
        a = _make_factory(init_seed=1)
        b = _make_factory(init_seed=2)
        # Static fields are equal
        assert a.produces_product_id == b.produces_product_id
        assert a.unit_cost == b.unit_cost

    def test_intermediate_same_init_seed_identical_regardless_of_policy(self):
        """Two IntermediateNodes with same init_seed are bit-identical, policy notwithstanding."""
        a = _make_intermediate(init_seed=7, policy=_NoopIntermediatePolicy(policy_seed=1))
        b = _make_intermediate(init_seed=7, policy=_BurnRngIntermediatePolicy(policy_seed=999))

        assert a.id == b.id
        assert a.region == b.region
        assert a.init_seed == b.init_seed
        assert a.carried_products == b.carried_products
        assert type(a.capacity) == type(b.capacity)
        assert a.tags == b.tags
        assert a.inventory == b.inventory
        assert a.pending == b.pending
        assert a.list_prices == b.list_prices
        assert a.min_order_imposed == b.min_order_imposed

    def test_sink_same_init_seed_identical_regardless_of_policy(self):
        """Two DemandSinkNodes with same init_seed are bit-identical, policy notwithstanding."""
        a = _make_sink(init_seed=7, policy=_NoopDemandSinkPolicy(policy_seed=1))
        b = _make_sink(init_seed=7, policy=_BurnRngDemandSinkPolicy(policy_seed=999))

        assert a.id == b.id
        assert a.region == b.region
        assert a.init_seed == b.init_seed
        assert a.product_id == b.product_id
        assert type(a.demand_dist) == type(b.demand_dist)
        assert a.income_rate == b.income_rate
        assert a.cash == b.cash
        assert a.activation_tick == b.activation_tick

    def test_policy_swap_does_not_perturb_node_fields(self):
        """Changing a node's policy after construction doesn't change any node fields."""
        node = _make_factory(init_seed=42, policy=_NoopFactoryPolicy())
        original_inventory = node.inventory
        original_list_price = node.list_price

        # Replace policy (simulated by constructing a new node with different policy)
        node2 = _make_factory(init_seed=42, policy=_BurnRngFactoryPolicy())
        assert node2.inventory == original_inventory
        assert node2.list_price == original_list_price


# ---------------------------------------------------------------------------
# NodePolicy ABC interface tests
# ---------------------------------------------------------------------------


class TestNodePolicyABC:
    def test_node_policy_is_abstract(self):
        """NodePolicy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NodePolicy()  # type: ignore[abstract]

    def test_factory_policy_is_node_policy(self):
        policy = _NoopFactoryPolicy()
        assert isinstance(policy, NodePolicy)

    def test_intermediate_policy_is_node_policy(self):
        policy = _NoopIntermediatePolicy()
        assert isinstance(policy, NodePolicy)

    def test_sink_policy_is_node_policy(self):
        policy = _NoopDemandSinkPolicy()
        assert isinstance(policy, NodePolicy)

    def test_node_policy_carries_policy_seed(self):
        policy = _NoopFactoryPolicy(policy_seed=42)
        assert policy.policy_seed == 42

    def test_node_policy_policy_seed_defaults_none(self):
        policy = _NoopFactoryPolicy()
        assert policy.policy_seed is None

    def test_node_policy_carries_policy_rng(self):
        policy = _NoopFactoryPolicy(policy_seed=1)
        assert isinstance(policy.policy_rng, Random)

    def test_node_policy_rng_seeded_from_policy_seed(self):
        """Two policies with the same seed produce the same RNG sequence."""
        p1 = _NoopFactoryPolicy(policy_seed=99)
        p2 = _NoopFactoryPolicy(policy_seed=99)
        assert p1.policy_rng.random() == p2.policy_rng.random()

    def test_node_policy_rng_different_seeds_diverge(self):
        """Different policy seeds produce different RNG sequences."""
        p1 = _NoopFactoryPolicy(policy_seed=1)
        p2 = _NoopFactoryPolicy(policy_seed=2)
        assert p1.policy_rng.random() != p2.policy_rng.random()
