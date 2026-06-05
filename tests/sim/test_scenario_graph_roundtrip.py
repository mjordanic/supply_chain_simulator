"""Tests for Scenario graph extension: NodeInstance, nodes_df, edges_df, make_nodes.

Key contracts verified:
- NodeInstance.to_dict() and from_dict() preserve fields (policy omitted)
- nodes_df() and edges_df() return DataFrames with correct shape
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import pytest

from src.sim.distributions import Constant, Uniform
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import DemandSinkPolicy, FactoryPolicy, IntermediatePolicy
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    make_nodes,
)


# ---------------------------------------------------------------------------
# Minimal concrete policy for attaching to NodeInstances
# ---------------------------------------------------------------------------


class _NoopFactoryPolicy(FactoryPolicy):
    def decide(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        return {"produce_qty": 0, "list_price": 1.0}


class _NoopSinkPolicy(DemandSinkPolicy):
    def decide(self, obs: Mapping[str, Any], ct: Any) -> dict[str, Any]:
        return {"buy": []}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _minimal_market() -> MarketParams:
    from src.sim.distributions import Normal

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
        trend_update_interval=20,
        min_value=0.1,
        max_value=2.0,
        stage_multipliers={"introduction": 1.0, "growth": 1.2, "maturity": 1.0, "decline": 0.8, "dead": 0.0},
        price_elasticity=-1.0,
        promo_multiplier=1.2,
        demand_factor_min=0.1,
        supply_factor_min=0.1,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.8, 1.2),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(10.0),
    )


def _minimal_disruption() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=[],
        regions=["US"],
        severity=Constant(1.0),
        duration=Constant(1),
    )


def _minimal_lifecycle() -> ItemLifecycleParams:
    return ItemLifecycleParams(
        stages=["introduction", "growth", "maturity", "decline", "dead"],
        init_stage="maturity",
        default_stage_change_probs={"introduction": 0.1, "growth": 0.1, "maturity": 0.1, "decline": 0.1, "dead": 0.0},
    )


def _minimal_catalog():
    from src.sim.scenario import Ware
    return [
        Ware(
            product_id="P0000",
            name="Widget",
            category="widgets",
            related_products=[],
            base_price=10.0,
            unit_cost=5.0,
            seasonality="all_season",
        )
    ]


def _build_graph_scenario(*, with_policy: bool = False) -> Scenario:
    """Build a minimal 3-node graph scenario (factory → intermediate → sink)."""
    factory = FactoryNode(
        id="factory-1",
        region="US",
        init_seed=1,
        produces_product_id="P0000",
        unit_cost=5.0,
        capacity_per_tick=100,
        inventory=0,
        list_price=5.0,
    )
    intermediate = IntermediateNode(
        id="intermediate-1",
        region="US",
        init_seed=2,
        carried_products={"P0000"},
        capacity=500,
        tags=["warehouse"],
        inventory={"P0000": 50},
        pending={},
        list_prices={"P0000": 8.0},
        min_order_imposed={"P0000": 1},
    )
    sink = DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=3,
        product_id="P0000",
        demand_dist=Constant(10),
        income_rate=200.0,
        cash=1000.0,
        activation_tick={"P0000": 0},
    )

    policy_factory = _NoopFactoryPolicy() if with_policy else None
    policy_sink = _NoopSinkPolicy() if with_policy else None

    nodes = [
        NodeInstance(node=factory, init_seed=1, policy=policy_factory),
        NodeInstance(node=intermediate, init_seed=2, policy=None),
        NodeInstance(node=sink, init_seed=3, policy=policy_sink),
    ]
    edges = [
        EdgeSpec(supplier_id="factory-1", buyer_id="intermediate-1", default_lead_time=2),
        EdgeSpec(supplier_id="intermediate-1", buyer_id="sink-1", default_lead_time=1),
    ]

    return Scenario(
        catalog=_minimal_catalog(),
        market=_minimal_market(),
        disruption=_minimal_disruption(),
        item_lifecycle=_minimal_lifecycle(),
        nodes=nodes,
        edges=edges,
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


# ---------------------------------------------------------------------------
# NodeInstance tests
# ---------------------------------------------------------------------------


class TestNodeInstance:
    def test_node_instance_carries_node(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        ni = NodeInstance(node=factory, init_seed=1)
        assert ni.node is factory

    def test_node_instance_carries_init_seed(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        ni = NodeInstance(node=factory, init_seed=99)
        assert ni.init_seed == 99

    def test_node_instance_carries_policy(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        policy = _NoopFactoryPolicy()
        ni = NodeInstance(node=factory, init_seed=1, policy=policy)
        assert ni.policy is policy

    def test_node_instance_policy_defaults_none(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        ni = NodeInstance(node=factory, init_seed=1)
        assert ni.policy is None

    def test_node_instance_to_dict_omits_policy(self):
        """NodeInstance.to_dict() must not include a 'policy' key."""
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        policy = _NoopFactoryPolicy()
        ni = NodeInstance(node=factory, init_seed=1, policy=policy)
        d = ni.to_dict()
        assert "policy" not in d

    def test_node_instance_to_dict_has_init_seed(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        ni = NodeInstance(node=factory, init_seed=42)
        d = ni.to_dict()
        assert d["init_seed"] == 42

    def test_node_instance_from_dict_policy_is_none(self):
        """NodeInstance.from_dict always returns policy=None."""
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        ni = NodeInstance(node=factory, init_seed=1, policy=_NoopFactoryPolicy())
        d = ni.to_dict()
        ni2 = NodeInstance.from_dict(d)
        assert ni2.policy is None


# ---------------------------------------------------------------------------
# Scenario.nodes_df() and edges_df()
# ---------------------------------------------------------------------------


class TestScenarioDataFrames:
    def test_nodes_df_has_correct_row_count(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.nodes_df()
        assert len(df) == 3

    def test_nodes_df_has_node_id_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.nodes_df()
        assert "node_id" in df.columns

    def test_nodes_df_has_node_type_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.nodes_df()
        assert "node_type" in df.columns

    def test_nodes_df_node_types_correct(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.nodes_df()
        types = list(df["node_type"])
        assert types == ["FactoryNode", "IntermediateNode", "DemandSinkNode"]

    def test_nodes_df_has_init_seed_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.nodes_df()
        assert "init_seed" in df.columns

    def test_edges_df_has_correct_row_count(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.edges_df()
        assert len(df) == 2

    def test_edges_df_has_supplier_id_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.edges_df()
        assert "supplier_id" in df.columns

    def test_edges_df_has_buyer_id_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.edges_df()
        assert "buyer_id" in df.columns

    def test_edges_df_has_default_lead_time_column(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.edges_df()
        assert "default_lead_time" in df.columns

    def test_edges_df_values_correct(self):
        pytest.importorskip("pandas")
        scenario = _build_graph_scenario()
        df = scenario.edges_df()
        rows = list(df[["supplier_id", "buyer_id"]].itertuples(index=False))
        assert ("factory-1", "intermediate-1") in rows
        assert ("intermediate-1", "sink-1") in rows


# ---------------------------------------------------------------------------
# make_nodes helper
# ---------------------------------------------------------------------------


class TestMakeNodes:
    def test_make_nodes_returns_node_instances(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        instances = make_nodes([(factory, 1, None)])
        assert len(instances) == 1
        assert isinstance(instances[0], NodeInstance)

    def test_make_nodes_empty_raises(self):
        with pytest.raises(ValueError):
            make_nodes([])

    def test_make_nodes_policy_attached(self):
        factory = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id="P0000", unit_cost=5.0,
            capacity_per_tick=100, inventory=0, list_price=5.0,
        )
        policy = _NoopFactoryPolicy()
        instances = make_nodes([(factory, 1, policy)])
        assert instances[0].policy is policy
