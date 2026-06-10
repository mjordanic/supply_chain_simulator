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


def simple_chain_node_types() -> dict[str, str]:
    return {"factory": "factory", "warehouse": "intermediate", "shop": "demand_sink"}


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
# validate_dag: same-level supplier link (BFS check removed; lateral is now legal)
# ---------------------------------------------------------------------------

def test_validate_dag_accepts_lateral_link_without_node_types():
    # Previously the BFS same-level check would reject shop1→shop2.
    # After dropping the BFS check, this topology is accepted in backward-compat
    # mode (node_types=None). Type-based lateral acceptance is tested separately
    # in the type-based section.
    nodes = ["factory", "wh", "shop1", "shop2"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="wh", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="shop1", default_lead_time=1),
        EdgeSpec(supplier_id="wh", buyer_id="shop2", default_lead_time=1),
        EdgeSpec(supplier_id="shop1", buyer_id="shop2", default_lead_time=1),  # lateral — now legal
    ]
    validate_dag(nodes, edges)  # no exception (BFS check removed)


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


# ---------------------------------------------------------------------------
# Type-based validation: node_types parameter accepted by validate_dag / build_graph
# ---------------------------------------------------------------------------

def test_validate_dag_accepts_node_types_parameter():
    """validate_dag must accept the node_types keyword argument without error."""
    validate_dag(
        simple_chain_nodes(),
        simple_chain_edges(),
        node_types=simple_chain_node_types(),
    )  # no exception


def test_build_graph_accepts_node_types_parameter():
    """build_graph must accept the node_types keyword argument without error."""
    g = build_graph(
        simple_chain_nodes(),
        simple_chain_edges(),
        node_types=simple_chain_node_types(),
    )
    assert isinstance(g, Graph)


# ---------------------------------------------------------------------------
# Type-based validation: accepted edge types
# ---------------------------------------------------------------------------

def test_validate_dag_accepts_factory_to_intermediate():
    """factory→intermediate is a legal edge."""
    nodes = ["f", "w", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
    ]
    node_types = {"f": "factory", "w": "intermediate", "s": "demand_sink"}
    validate_dag(nodes, edges, node_types=node_types)  # no exception


def test_validate_dag_accepts_intermediate_to_sink():
    """intermediate→demand_sink is a legal edge."""
    nodes = ["f", "w", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
    ]
    node_types = {"f": "factory", "w": "intermediate", "s": "demand_sink"}
    validate_dag(nodes, edges, node_types=node_types)  # no exception


def test_validate_dag_accepts_lateral_intermediate_to_intermediate():
    """intermediate→intermediate (lateral) edge is legal under type-based rules."""
    # wh1 supplies wh2 laterally; both are fed by factory and wh2 feeds a sink.
    nodes = ["f", "wh1", "wh2", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="wh2", default_lead_time=1),  # lateral
        EdgeSpec(supplier_id="wh2", buyer_id="s", default_lead_time=1),
    ]
    node_types = {
        "f": "factory",
        "wh1": "intermediate",
        "wh2": "intermediate",
        "s": "demand_sink",
    }
    validate_dag(nodes, edges, node_types=node_types)  # no exception


def test_build_graph_accepts_lateral_edge():
    """build_graph returns a Graph for a topology with a lateral intermediate→intermediate edge."""
    nodes = ["f", "wh1", "wh2", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="wh2", default_lead_time=1),
        EdgeSpec(supplier_id="wh2", buyer_id="s", default_lead_time=1),
    ]
    node_types = {
        "f": "factory",
        "wh1": "intermediate",
        "wh2": "intermediate",
        "s": "demand_sink",
    }
    g = build_graph(nodes, edges, node_types=node_types)
    assert isinstance(g, Graph)
    assert g.suppliers_of("wh2") == {"wh1"}


# ---------------------------------------------------------------------------
# Type-based validation: rejected edge types
# ---------------------------------------------------------------------------

def test_validate_dag_rejects_factory_to_sink():
    """factory→demand_sink must be rejected: flow must pass through ≥1 intermediate."""
    nodes = ["f", "s"]
    edges = [EdgeSpec(supplier_id="f", buyer_id="s", default_lead_time=1)]
    node_types = {"f": "factory", "s": "demand_sink"}
    with pytest.raises(ValueError, match="factory.*sink|rule"):
        validate_dag(nodes, edges, node_types=node_types)


def test_validate_dag_rejects_edge_into_factory():
    """Any edge whose buyer is a factory must be rejected (factories don't buy)."""
    # w→f: intermediate supplying a factory is illegal; no cycle here since f has no outgoing edge.
    nodes = ["f", "w", "s"]
    edges = [
        EdgeSpec(supplier_id="w", buyer_id="f", default_lead_time=1),  # illegal: buying factory
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
        # f has no outgoing edges so we need another source for w; add a second factory
    ]
    # Simpler: use a standalone "src" node to feed w so we avoid orphan/cycle issues
    nodes = ["src", "w", "f", "s"]
    edges = [
        EdgeSpec(supplier_id="src", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="f", default_lead_time=1),  # illegal
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
    ]
    node_types = {
        "src": "factory",
        "w": "intermediate",
        "f": "factory",
        "s": "demand_sink",
    }
    with pytest.raises(ValueError, match="factory|rule"):
        validate_dag(nodes, edges, node_types=node_types)


def test_validate_dag_rejects_sink_as_supplier():
    """An edge whose supplier is a demand_sink must be rejected (sinks don't sell)."""
    nodes = ["f", "w", "s1", "s2"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="s1", default_lead_time=1),
        EdgeSpec(supplier_id="s1", buyer_id="s2", default_lead_time=1),  # illegal: sink selling
    ]
    node_types = {
        "f": "factory",
        "w": "intermediate",
        "s1": "demand_sink",
        "s2": "demand_sink",
    }
    with pytest.raises(ValueError, match="sink|rule"):
        validate_dag(nodes, edges, node_types=node_types)


# ---------------------------------------------------------------------------
# Type-based validation: existing checks still enforced when node_types present
# ---------------------------------------------------------------------------

def test_validate_dag_rejects_cycle_with_node_types():
    """Cycle detection is still active when node_types is supplied."""
    nodes = ["f", "wh1", "wh2", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="wh2", default_lead_time=1),
        EdgeSpec(supplier_id="wh2", buyer_id="wh1", default_lead_time=1),  # cycle
        EdgeSpec(supplier_id="wh2", buyer_id="s", default_lead_time=1),
    ]
    node_types = {
        "f": "factory",
        "wh1": "intermediate",
        "wh2": "intermediate",
        "s": "demand_sink",
    }
    with pytest.raises(ValueError, match="[Cc]ycle"):
        validate_dag(nodes, edges, node_types=node_types)


def test_validate_dag_rejects_self_loop_with_node_types():
    """Self-loop detection is still active when node_types is supplied."""
    nodes = ["f", "w", "s"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="w", default_lead_time=1),  # self-loop
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
    ]
    node_types = {"f": "factory", "w": "intermediate", "s": "demand_sink"}
    with pytest.raises(ValueError, match="[Cc]ycle"):
        validate_dag(nodes, edges, node_types=node_types)


def test_validate_dag_rejects_unreachable_node_with_node_types():
    """Unreachable-node detection is still active when node_types is supplied."""
    nodes = ["f", "w", "s", "orphan"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="s", default_lead_time=1),
    ]
    node_types = {
        "f": "factory",
        "w": "intermediate",
        "s": "demand_sink",
        "orphan": "intermediate",
    }
    with pytest.raises(ValueError, match="[Uu]nreachable"):
        validate_dag(nodes, edges, node_types=node_types)


# ---------------------------------------------------------------------------
# Error messages name the violated rule
# ---------------------------------------------------------------------------

def test_validate_dag_factory_to_sink_error_names_rule():
    """Error for factory→sink should name the violated rule clearly."""
    nodes = ["f", "s"]
    edges = [EdgeSpec(supplier_id="f", buyer_id="s", default_lead_time=1)]
    node_types = {"f": "factory", "s": "demand_sink"}
    with pytest.raises(ValueError) as exc_info:
        validate_dag(nodes, edges, node_types=node_types)
    msg = str(exc_info.value).lower()
    # Must mention factory and sink so modeller knows what went wrong
    assert "factory" in msg
    assert "sink" in msg


def test_validate_dag_sink_supplier_error_names_rule():
    """Error for sink→* should name the violated rule clearly."""
    nodes = ["f", "w", "s1", "s2"]
    edges = [
        EdgeSpec(supplier_id="f", buyer_id="w", default_lead_time=1),
        EdgeSpec(supplier_id="w", buyer_id="s1", default_lead_time=1),
        EdgeSpec(supplier_id="s1", buyer_id="s2", default_lead_time=1),
    ]
    node_types = {
        "f": "factory",
        "w": "intermediate",
        "s1": "demand_sink",
        "s2": "demand_sink",
    }
    with pytest.raises(ValueError) as exc_info:
        validate_dag(nodes, edges, node_types=node_types)
    msg = str(exc_info.value).lower()
    assert "sink" in msg


# ---------------------------------------------------------------------------
# Backward compatibility: node_types=None (default) keeps old behavior
# ---------------------------------------------------------------------------

def test_validate_dag_backward_compat_no_node_types_allows_factory_to_sink():
    """When node_types is omitted (None), old permissive behavior applies (no type check)."""
    # factory→sink was only rejected by the same-level BFS check, not type check.
    # With node_types absent the type check is skipped entirely.
    # In the OLD code, factory(level 0)→sink(level 1) would be different levels → accepted.
    nodes = ["f", "s"]
    edges = [EdgeSpec(supplier_id="f", buyer_id="s", default_lead_time=1)]
    # No node_types: type check is skipped.  DAG checks pass (no cycle, both connected).
    validate_dag(nodes, edges)  # no exception
