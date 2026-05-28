"""Tests for the Graph deep module — DAG validation, level computation, topology queries."""

from __future__ import annotations

import pytest

from src.sim.graph import EdgeSpec, Graph, build_graph, compute_levels, validate_dag


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def simple_chain_edges() -> list[EdgeSpec]:
    """factory → warehouse → shop: three-node linear chain."""
    return [
        EdgeSpec(supplier_id="factory", buyer_id="warehouse", default_lead_time=2),
        EdgeSpec(supplier_id="warehouse", buyer_id="shop", default_lead_time=1),
    ]


def simple_chain_nodes() -> list[str]:
    return ["factory", "warehouse", "shop"]


# ---------------------------------------------------------------------------
# Tracer bullet: build a valid chain and query topology
# ---------------------------------------------------------------------------

def test_build_graph_returns_graph_instance():
    g = build_graph(simple_chain_nodes(), simple_chain_edges())
    assert isinstance(g, Graph)


def test_suppliers_of_returns_correct_nodes():
    g = build_graph(simple_chain_nodes(), simple_chain_edges())
    assert g.suppliers_of("shop") == {"warehouse"}
    assert g.suppliers_of("warehouse") == {"factory"}
    assert g.suppliers_of("factory") == set()


def test_buyers_of_returns_correct_nodes():
    g = build_graph(simple_chain_nodes(), simple_chain_edges())
    assert g.buyers_of("factory") == {"warehouse"}
    assert g.buyers_of("warehouse") == {"shop"}
    assert g.buyers_of("shop") == set()


# ---------------------------------------------------------------------------
# lead_time: default and per-product overrides
# ---------------------------------------------------------------------------

def test_lead_time_default():
    g = build_graph(simple_chain_nodes(), simple_chain_edges())
    assert g.lead_time("factory", "warehouse", "pidA") == 2
    assert g.lead_time("warehouse", "shop", "pidA") == 1


def test_lead_time_per_product_override():
    edges = [
        EdgeSpec(
            supplier_id="factory",
            buyer_id="shop",
            default_lead_time=3,
            per_product_lead_time={"pidX": 7},
        )
    ]
    g = build_graph(["factory", "shop"], edges)
    assert g.lead_time("factory", "shop", "pidX") == 7   # override
    assert g.lead_time("factory", "shop", "pidY") == 3   # fallback to default


def test_lead_time_no_per_product_map_uses_default():
    edges = [EdgeSpec(supplier_id="a", buyer_id="b", default_lead_time=5)]
    g = build_graph(["a", "b"], edges)
    assert g.lead_time("a", "b", "any_pid") == 5


# ---------------------------------------------------------------------------
# compute_levels
# ---------------------------------------------------------------------------

def test_compute_levels_linear_chain():
    g = build_graph(simple_chain_nodes(), simple_chain_edges())
    levels = compute_levels(g)
    # factory (no suppliers) is level 0; warehouse is level 1; shop is level 2
    assert levels["factory"] == 0
    assert levels["warehouse"] == 1
    assert levels["shop"] == 2


def test_compute_levels_multi_tier_longest_path():
    # Two factories feed one warehouse; warehouse feeds one shop.
    # There is also a second warehouse fed by f2 that also feeds shop.
    # longest path to shop = f1 → wh1 → shop = length 2
    nodes = ["f1", "f2", "wh1", "wh2", "shop"]
    edges = [
        EdgeSpec(supplier_id="f1", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="f2", buyer_id="wh2", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="shop", default_lead_time=1),
        EdgeSpec(supplier_id="wh2", buyer_id="shop", default_lead_time=1),
    ]
    g = build_graph(nodes, edges)
    levels = compute_levels(g)
    assert levels["f1"] == 0
    assert levels["f2"] == 0
    assert levels["wh1"] == 1
    assert levels["wh2"] == 1
    assert levels["shop"] == 2  # longest path: f1 → wh1 → shop


# ---------------------------------------------------------------------------
# validate_dag: cycle detection
# ---------------------------------------------------------------------------

def test_validate_dag_rejects_direct_cycle():
    nodes = ["a", "b"]
    edges = [
        EdgeSpec(supplier_id="a", buyer_id="b", default_lead_time=1),
        EdgeSpec(supplier_id="b", buyer_id="a", default_lead_time=1),
    ]
    with pytest.raises(ValueError, match="[Cc]ycle"):
        validate_dag(nodes, edges)


def test_validate_dag_rejects_self_loop():
    nodes = ["a"]
    edges = [EdgeSpec(supplier_id="a", buyer_id="a", default_lead_time=1)]
    with pytest.raises(ValueError, match="[Cc]ycle"):
        validate_dag(nodes, edges)


def test_validate_dag_rejects_indirect_cycle():
    nodes = ["a", "b", "c"]
    edges = [
        EdgeSpec(supplier_id="a", buyer_id="b", default_lead_time=1),
        EdgeSpec(supplier_id="b", buyer_id="c", default_lead_time=1),
        EdgeSpec(supplier_id="c", buyer_id="a", default_lead_time=1),
    ]
    with pytest.raises(ValueError, match="[Cc]ycle"):
        validate_dag(nodes, edges)


# ---------------------------------------------------------------------------
# validate_dag: unreachable nodes
# ---------------------------------------------------------------------------

def test_validate_dag_rejects_isolated_node():
    nodes = ["factory", "shop", "orphan"]
    edges = [EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=1)]
    with pytest.raises(ValueError, match="[Uu]nreachable"):
        validate_dag(nodes, edges)


def test_validate_dag_accepts_graph_without_orphans():
    """Baseline: valid graph with all nodes connected should not raise."""
    validate_dag(simple_chain_nodes(), simple_chain_edges())  # no exception


# ---------------------------------------------------------------------------
# validate_dag: same-level supplier link
# ---------------------------------------------------------------------------

def test_validate_dag_rejects_same_level_link():
    # Two shops at the same level (both fed by a warehouse); one supplying the
    # other is a peer link and is forbidden.
    # Without the shop1→shop2 edge: both shops are at level 2 (wh is level 1).
    nodes = ["factory", "wh", "shop1", "shop2"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="wh", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="shop1", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="shop2", default_lead_time=1),
        EdgeSpec(supplier_id="shop1", buyer_id="shop2", default_lead_time=1),  # peer link!
    ]
    with pytest.raises(ValueError, match="[Ss]ame.level"):
        validate_dag(nodes, edges)


def test_validate_dag_accepts_cross_level_links():
    """Nodes at different levels connected correctly: no error."""
    nodes = ["factory", "wh", "shop"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="wh", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="shop", default_lead_time=1),
    ]
    validate_dag(nodes, edges)  # no exception


# ---------------------------------------------------------------------------
# build_graph propagates validation errors
# ---------------------------------------------------------------------------

def test_build_graph_raises_on_cycle():
    nodes = ["a", "b"]
    edges = [
        EdgeSpec(supplier_id="a", buyer_id="b", default_lead_time=1),
        EdgeSpec(supplier_id="b", buyer_id="a", default_lead_time=1),
    ]
    with pytest.raises(ValueError):
        build_graph(nodes, edges)
