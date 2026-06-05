"""Tests for policy_registry.build_policy and the registry itself.

Acceptance criteria covered:
- Every registered name resolves to its class
- params splatted to the constructor
- Injected node/edge facts (capacity, list_prices, supplier ids, lead times)
  reach the constructed policy and match the node/edge
- Unknown name raises a clear ValueError
"""

from __future__ import annotations

import pytest

from src.sim.policy_registry import build_policy, list_policy_names
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
    StaticFactoryPolicy,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal node + edge fixtures
# ---------------------------------------------------------------------------

def _factory_node(node_id: str = "factory-1", *, capacity: int = 50, unit_cost: float = 5.0):
    from src.sim.node import FactoryNode
    return FactoryNode(
        id=node_id,
        region="US",
        init_seed=1,
        produces_product_id="P0001",
        unit_cost=unit_cost,
        capacity_per_tick=capacity,
        inventory=100,
        list_price=unit_cost,
    )


def _intermediate_node(
    node_id: str = "shop-1",
    *,
    capacity: int = 500,
    carried: set | None = None,
    list_prices: dict | None = None,
):
    from src.sim.node import IntermediateNode
    return IntermediateNode(
        id=node_id,
        region="US",
        init_seed=2,
        carried_products=carried or {"P0001"},
        capacity=capacity,
        tags=["shop"],
        inventory={"P0001": 10},
        list_prices=list_prices or {"P0001": 8.0},
    )


def _sink_node(node_id: str = "sink-1"):
    from src.sim.node import DemandSinkNode
    from src.sim.distributions import Normal
    return DemandSinkNode(
        id=node_id,
        region="US",
        init_seed=3,
        product_id="P0001",
        demand_dist=Normal(mean=10.0, std=2.0),
        income_rate=200.0,
        cash=1000.0,
    )


def _edges(supplier: str = "factory-1", buyer: str = "shop-1", lead_time: int = 2):
    from src.sim.graph import EdgeSpec
    return [EdgeSpec(supplier_id=supplier, buyer_id=buyer, default_lead_time=lead_time)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_policy_names_contains_all_expected():
    names = list_policy_names()
    expected = {
        "order_up_to",
        "reorder_point",
        "periodic_order_up_to",
        "periodic_reorder",
        "single_supplier",
        "static_factory",
        "default_demand_sink",
    }
    assert expected <= set(names), f"Missing names: {expected - set(names)}"


# ---------------------------------------------------------------------------
# build_policy — happy paths
# ---------------------------------------------------------------------------


def test_build_policy_static_factory():
    node = _factory_node(capacity=30, unit_cost=4.0)
    policy = build_policy("static_factory", {}, node=node, edges=[], policy_seed=1)
    assert isinstance(policy, StaticFactoryPolicy)
    # Injected capacity and unit_cost must match the node.
    assert policy.capacity_per_tick == 30
    assert policy.unit_cost == 4.0


def test_build_policy_static_factory_target_inventory_param():
    """target_inventory is a true hyperparameter, not injected."""
    node = _factory_node(capacity=50)
    policy = build_policy(
        "static_factory",
        {"target_inventory": 200},
        node=node,
        edges=[],
        policy_seed=0,
    )
    assert isinstance(policy, StaticFactoryPolicy)
    assert policy.target_inventory == 200


def test_build_policy_order_up_to():
    node = _intermediate_node()
    edges = _edges(lead_time=3)
    policy = build_policy("order_up_to", {}, node=node, edges=edges, policy_seed=7)
    assert isinstance(policy, OrderUpToPolicy)
    # Injected delivery_lag == edge lead_time.
    assert policy.delivery_lag == 3


def test_build_policy_order_up_to_cover_horizon_param():
    """cover_horizon_ticks is a hyperparameter — should be respected."""
    node = _intermediate_node()
    edges = _edges(lead_time=2)
    policy = build_policy(
        "order_up_to",
        {"cover_horizon_ticks": 21},
        node=node,
        edges=edges,
        policy_seed=0,
    )
    assert isinstance(policy, OrderUpToPolicy)
    assert policy.cover_horizon_ticks == 21


def test_build_policy_reorder_point():
    node = _intermediate_node()
    edges = _edges(lead_time=4)
    policy = build_policy("reorder_point", {}, node=node, edges=edges, policy_seed=0)
    assert isinstance(policy, ReorderPointPolicy)
    assert policy.delivery_lag == 4


def test_build_policy_periodic_order_up_to():
    node = _intermediate_node()
    edges = _edges(lead_time=2)
    policy = build_policy(
        "periodic_order_up_to", {}, node=node, edges=edges, policy_seed=0
    )
    assert isinstance(policy, PeriodicOrderUpToPolicy)


def test_build_policy_periodic_reorder():
    node = _intermediate_node()
    edges = _edges(lead_time=2)
    policy = build_policy("periodic_reorder", {}, node=node, edges=edges, policy_seed=0)
    assert isinstance(policy, PeriodicReorderPolicy)


def test_build_policy_single_supplier():
    node = _intermediate_node()
    edges = _edges(supplier="factory-1", buyer="shop-1", lead_time=2)
    policy = build_policy("single_supplier", {}, node=node, edges=edges, policy_seed=0)
    assert isinstance(policy, IntermediatePolicy.SingleSupplierAdapter)  # type: ignore[attr-defined]
    # Injected supplier id must be the one edge supplier.
    assert policy.supplier_id == "factory-1"


def test_build_policy_default_demand_sink():
    node = _sink_node()
    policy = build_policy(
        "default_demand_sink", {}, node=node, edges=[], policy_seed=5
    )
    assert isinstance(policy, DefaultDemandSinkPolicy)
    # policy_seed is stored on the base class.
    assert policy.policy_seed == 5


# ---------------------------------------------------------------------------
# policy_seed injection
# ---------------------------------------------------------------------------


def test_build_policy_policy_seed_is_injected():
    """policy_seed passed to build_policy is wired to the policy instance."""
    node = _factory_node()
    policy = build_policy("static_factory", {}, node=node, edges=[], policy_seed=999)
    assert policy.policy_seed == 999


# ---------------------------------------------------------------------------
# Error: unknown name
# ---------------------------------------------------------------------------


def test_build_policy_unknown_name_raises():
    node = _factory_node()
    with pytest.raises(ValueError, match="no_such_policy"):
        build_policy("no_such_policy", {}, node=node, edges=[], policy_seed=0)


def test_build_policy_unknown_name_lists_registered():
    """Error message from an unknown name mentions the valid registered names."""
    node = _factory_node()
    with pytest.raises(ValueError, match="order_up_to"):
        build_policy("typo_policy", {}, node=node, edges=[], policy_seed=0)
