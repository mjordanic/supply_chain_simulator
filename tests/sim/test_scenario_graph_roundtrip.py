"""Tests for Scenario graph extension: nodes/edges round-trip, is_graph, NodeInstance.

Key contracts verified:
- Scenario with nodes/edges survives to_dict/from_dict round-trip
- NodeInstance.policy is intentionally omitted from JSON (mirrors StoreInstance)
- is_graph property: True when nodes list is non-empty, False for legacy stores-only
- nodes_df() and edges_df() return DataFrames with correct shape
- Existing StoreInstance/stores path is unchanged
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
    StoreInstance,
    StoreTemplate,
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
        stores=[],
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
        """NodeInstance.from_dict always returns policy=None (mirrors StoreInstance)."""
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
# Scenario.is_graph property
# ---------------------------------------------------------------------------


class TestIsGraph:
    def test_is_graph_true_when_nodes_present(self):
        scenario = _build_graph_scenario()
        assert scenario.is_graph is True

    def test_is_graph_false_when_only_stores(self, make_scenario):
        """Legacy stores-only scenario has is_graph == False."""
        template = StoreTemplate(
            id="t", region="US",
            capacity=1000, init_balance=10000,
            init_stock_pct=0.2, delivery_lag=3,
            holding_rate=0.005, order_fee=100,
            init_active_count=2,
        )
        scenario = make_scenario(
            stores=[StoreInstance(template=template, init_seed=1)],
            n_steps=5, world_seed=1,
        )
        assert scenario.is_graph is False

    def test_is_graph_false_when_nodes_empty(self, make_scenario):
        """Scenario with empty nodes list has is_graph == False."""
        template = StoreTemplate(
            id="t", region="US",
            capacity=1000, init_balance=10000,
            init_stock_pct=0.2, delivery_lag=3,
            holding_rate=0.005, order_fee=100,
            init_active_count=2,
        )
        scenario = make_scenario(
            stores=[StoreInstance(template=template, init_seed=1)],
            n_steps=5, world_seed=1,
        )
        assert scenario.is_graph is False


# ---------------------------------------------------------------------------
# Scenario graph round-trip (to_dict / from_dict)
# ---------------------------------------------------------------------------


class TestScenarioGraphRoundtrip:
    def test_roundtrip_preserves_node_count(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        assert len(restored.nodes) == len(scenario.nodes)

    def test_roundtrip_preserves_edge_count(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        assert len(restored.edges) == len(scenario.edges)

    def test_roundtrip_node_ids_preserved(self):
        scenario = _build_graph_scenario()
        original_ids = [ni.node.id for ni in scenario.nodes]
        restored = Scenario.from_dict(scenario.to_dict())
        restored_ids = [ni.node.id for ni in restored.nodes]
        assert original_ids == restored_ids

    def test_roundtrip_node_types_preserved(self):
        scenario = _build_graph_scenario()
        original_types = [type(ni.node).__name__ for ni in scenario.nodes]
        restored = Scenario.from_dict(scenario.to_dict())
        restored_types = [type(ni.node).__name__ for ni in restored.nodes]
        assert original_types == restored_types

    def test_roundtrip_factory_node_fields(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        original_factory = scenario.nodes[0].node
        restored_factory = restored.nodes[0].node
        assert isinstance(restored_factory, FactoryNode)
        assert restored_factory.id == original_factory.id
        assert restored_factory.region == original_factory.region
        assert restored_factory.produces_product_id == original_factory.produces_product_id
        assert restored_factory.unit_cost == original_factory.unit_cost
        assert restored_factory.inventory == original_factory.inventory
        assert restored_factory.list_price == original_factory.list_price

    def test_roundtrip_intermediate_node_fields(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        original = scenario.nodes[1].node
        restored_node = restored.nodes[1].node
        assert isinstance(restored_node, IntermediateNode)
        assert restored_node.id == original.id
        assert restored_node.carried_products == original.carried_products
        assert restored_node.tags == original.tags
        assert restored_node.inventory == original.inventory
        assert restored_node.list_prices == original.list_prices
        assert restored_node.min_order_imposed == original.min_order_imposed

    def test_roundtrip_sink_node_fields(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        original = scenario.nodes[2].node
        restored_node = restored.nodes[2].node
        assert isinstance(restored_node, DemandSinkNode)
        assert restored_node.id == original.id
        assert restored_node.product_id == original.product_id
        assert restored_node.income_rate == original.income_rate
        assert restored_node.cash == original.cash
        assert restored_node.activation_tick == original.activation_tick

    def test_roundtrip_policy_omitted(self):
        """After round-trip, all NodeInstance.policy values are None."""
        scenario = _build_graph_scenario(with_policy=True)
        # Confirm we actually had policies before serialisation
        assert scenario.nodes[0].policy is not None
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        for ni in restored.nodes:
            assert ni.policy is None, f"Expected policy=None on {ni.node.id}, got {ni.policy!r}"

    def test_roundtrip_edge_supplier_buyer(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        original_edges = [(e.supplier_id, e.buyer_id) for e in scenario.edges]
        restored_edges = [(e.supplier_id, e.buyer_id) for e in restored.edges]
        assert original_edges == restored_edges

    def test_roundtrip_edge_lead_times(self):
        scenario = _build_graph_scenario()
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        original_lts = [e.default_lead_time for e in scenario.edges]
        restored_lts = [e.default_lead_time for e in restored.edges]
        assert original_lts == restored_lts

    def test_json_roundtrip(self):
        """Full JSON string round-trip (to_json / from_json)."""
        scenario = _build_graph_scenario()
        json_str = scenario.to_json()
        restored = Scenario.from_json(json_str)
        assert len(restored.nodes) == 3
        assert len(restored.edges) == 2
        assert restored.is_graph is True

    def test_roundtrip_stores_unchanged(self):
        """Existing stores field survives round-trip unaffected."""
        scenario = _build_graph_scenario()
        assert scenario.stores == []
        d = scenario.to_dict()
        restored = Scenario.from_dict(d)
        assert restored.stores == []


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
