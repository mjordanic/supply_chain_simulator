"""Tests for topology_scaffolder.scaffold_topology.

Acceptance criteria:
- scaffold_topology(catalog, spec) returns a {nodes, edges} structure that
  build_graph accepts as a valid DAG.
- Sink-per-product / shop-count / sink_density expansion is correct.
- Output is deterministic (same catalog + spec => identical result).
- Result is re-loadable via load_setup when merged into a setup directory.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.sim.scenario import Ware
from src.sim.topology_scaffolder import ScaffoldSpec, scaffold_topology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(n: int = 3) -> list[Ware]:
    """Build a minimal catalog of ``n`` products."""
    return [
        Ware(
            product_id=f"P{i:04d}",
            name=f"Product {i}",
            category="widgets",
            related_products=[],
            base_price=10.0 + i,
            unit_cost=4.0 + i,
            seasonality="peak",
            init_stage=None,
            stage_change_probs=None,
            freshness_alpha=None,
            freshness_decay=None,
            init_stock_share=None,
        )
        for i in range(n)
    ]


_YAML_PREAMBLE = textwrap.dedent("""\
run:
  n_steps: 5
  start_date: "2024-01-01"
  world_seed: 42

market:
  cycle_len: 365
  cycle_amp: 0.0
  init_demand: 1.0
  init_supply: 1.0
  peak_factor: 1.0
  off_factor: 1.0
  season_months: {}
  regions: [US]
  correlation: 0.0
  trend_update_interval: 100
  min_value: 0.5
  max_value: 2.0
  stage_multipliers: {}
  price_elasticity: 0.0
  promo_multiplier: 1.0
  demand_factor_min: 0.1
  supply_factor_min: 0.01
  cross_inv_lo: 0.3
  cross_inv_hi: 0.7
  cross_factor_range: [0.5, 1.5]
  trend: {kind: constant, value: 1.0}
  demand_shock: {kind: constant, value: 0.0}
  supply_shock: {kind: constant, value: 0.0}
  base_demand: {kind: constant, value: 5.0}

disruption:
  event_prob: 0.0
  types: [natural_disaster]
  regions: [US]
  severity: {kind: constant, value: 0.0}
  duration: {kind: constant, value: 1}
""")

_CATALOG_CSV = (
    "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
)


def _make_catalog_csv(catalog: list[Ware]) -> str:
    lines = [_CATALOG_CSV.rstrip()]
    for w in catalog:
        lines.append(
            f"{w.product_id},{w.name},{w.category},"
            f"{w.base_price},{w.unit_cost},{w.seasonality},,1.0"
        )
    return "\n".join(lines) + "\n"


def _write_setup_with_topology(
    tmp_path: Path, catalog: list[Ware], topology: dict
) -> Path:
    """Write a minimal setup directory using the given topology and return path."""
    setup_dir = tmp_path / "scaffolded"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(_make_catalog_csv(catalog))
    doc = yaml.safe_load(_YAML_PREAMBLE)
    doc["nodes"] = topology["nodes"]
    doc["edges"] = topology["edges"]
    (setup_dir / "setup.yaml").write_text(yaml.safe_dump(doc, default_flow_style=False))
    return setup_dir


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_scaffold_returns_nodes_and_edges():
    """scaffold_topology returns a dict with 'nodes' and 'edges' keys."""
    catalog = _make_catalog(1)
    result = scaffold_topology(catalog)
    assert "nodes" in result
    assert "edges" in result
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)


def test_scaffold_single_product_three_nodes():
    """Single-product catalog with default spec yields factory + shop + sink."""
    catalog = _make_catalog(1)
    result = scaffold_topology(catalog)
    node_types = [n["type"] for n in result["nodes"]]
    assert node_types.count("factory") == 1
    assert node_types.count("intermediate") == 1
    assert node_types.count("demand_sink") == 1


def test_scaffold_produces_valid_dag():
    """scaffold_topology output is accepted by build_graph without errors."""
    from src.sim.graph import EdgeSpec, build_graph

    catalog = _make_catalog(3)
    result = scaffold_topology(catalog)
    node_ids = [n["id"] for n in result["nodes"]]
    edges = [
        EdgeSpec(
            supplier_id=e["supplier"],
            buyer_id=e["buyer"],
            default_lead_time=e["lead_time"],
        )
        for e in result["edges"]
    ]
    graph = build_graph(node_ids, edges)
    assert len(graph.nodes) == len(node_ids)


def test_scaffold_three_products_default_spec():
    """3-product catalog: 3 factories, 1 shop, 3 sinks by default."""
    catalog = _make_catalog(3)
    result = scaffold_topology(catalog)
    node_types = [n["type"] for n in result["nodes"]]
    assert node_types.count("factory") == 3
    assert node_types.count("intermediate") == 1
    assert node_types.count("demand_sink") == 3


# ---------------------------------------------------------------------------
# shop_count expansion
# ---------------------------------------------------------------------------


def test_scaffold_two_shops():
    """spec.shop_count=2 produces 2 intermediate nodes."""
    catalog = _make_catalog(4)
    spec = ScaffoldSpec(shop_count=2)
    result = scaffold_topology(catalog, spec)
    node_types = [n["type"] for n in result["nodes"]]
    assert node_types.count("intermediate") == 2


def test_scaffold_products_spread_across_shops():
    """Products are spread evenly across shops."""
    catalog = _make_catalog(4)
    spec = ScaffoldSpec(shop_count=2)
    result = scaffold_topology(catalog, spec)
    shops = [n for n in result["nodes"] if n["type"] == "intermediate"]
    # 4 products across 2 shops => 2 products per shop
    total_carried = sum(len(s["carried_products"]) for s in shops)
    assert total_carried == 4
    for shop in shops:
        assert len(shop["carried_products"]) == 2


# ---------------------------------------------------------------------------
# sink_density
# ---------------------------------------------------------------------------


def test_scaffold_sink_density_half():
    """sink_density=0.5 halves the number of sinks."""
    catalog = _make_catalog(4)
    spec = ScaffoldSpec(sink_density=0.5)
    result = scaffold_topology(catalog, spec)
    node_types = [n["type"] for n in result["nodes"]]
    # 4 products × 1 shop × 0.5 = 2 sinks
    assert node_types.count("demand_sink") == 2


def test_scaffold_sink_density_one_is_default():
    """sink_density=1.0 (default) produces one sink per (product, shop)."""
    catalog = _make_catalog(3)
    spec = ScaffoldSpec(sink_density=1.0)
    result = scaffold_topology(catalog, spec)
    node_types = [n["type"] for n in result["nodes"]]
    assert node_types.count("demand_sink") == 3


def test_scaffold_minimum_one_sink():
    """Very low sink_density still produces at least one sink."""
    catalog = _make_catalog(2)
    spec = ScaffoldSpec(sink_density=0.01)
    result = scaffold_topology(catalog, spec)
    node_types = [n["type"] for n in result["nodes"]]
    assert node_types.count("demand_sink") >= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_scaffold_is_deterministic():
    """Same catalog + spec always produces identical output."""
    catalog = _make_catalog(5)
    spec = ScaffoldSpec(shop_count=2, sink_density=0.8)
    r1 = scaffold_topology(catalog, spec)
    r2 = scaffold_topology(catalog, spec)
    assert r1 == r2


def test_scaffold_different_catalogs_differ():
    """Different catalogs produce different node ids."""
    catalog_a = _make_catalog(2)
    catalog_b = _make_catalog(3)
    r_a = scaffold_topology(catalog_a)
    r_b = scaffold_topology(catalog_b)
    ids_a = {n["id"] for n in r_a["nodes"]}
    ids_b = {n["id"] for n in r_b["nodes"]}
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Re-loadable via load_setup
# ---------------------------------------------------------------------------


def test_scaffold_reloadable_via_load_setup(tmp_path):
    """Topology from scaffold_topology is re-loadable via load_setup."""
    from src.sim.setup_io import load_setup

    catalog = _make_catalog(2)
    spec = ScaffoldSpec()
    topology = scaffold_topology(catalog, spec)
    setup_dir = _write_setup_with_topology(tmp_path, catalog, topology)
    scenario = load_setup(setup_dir)

    assert len(scenario.catalog) == 2
    assert len(scenario.nodes) == len(topology["nodes"])
    assert len(scenario.edges) == len(topology["edges"])


def test_scaffold_multi_shop_reloadable(tmp_path):
    """Multi-shop topology from scaffold_topology loads cleanly."""
    from src.sim.setup_io import load_setup

    catalog = _make_catalog(4)
    spec = ScaffoldSpec(shop_count=2)
    topology = scaffold_topology(catalog, spec)
    setup_dir = _write_setup_with_topology(tmp_path, catalog, topology)
    scenario = load_setup(setup_dir)

    assert len(scenario.nodes) == len(topology["nodes"])


# ---------------------------------------------------------------------------
# Node content
# ---------------------------------------------------------------------------


def test_scaffold_factory_fields():
    """Factory nodes carry expected fields."""
    catalog = _make_catalog(1)
    result = scaffold_topology(catalog)
    factory = next(n for n in result["nodes"] if n["type"] == "factory")
    assert factory["produces_product_id"] == "P0000"
    assert factory["unit_cost"] == catalog[0].unit_cost
    assert factory["list_price"] == catalog[0].unit_cost
    assert factory["capacity_per_tick"] == 100  # default


def test_scaffold_shop_carried_products_set_correctly():
    """Shop nodes' carried_products match the catalog products assigned to them."""
    catalog = _make_catalog(3)
    result = scaffold_topology(catalog)
    shop = next(n for n in result["nodes"] if n["type"] == "intermediate")
    assert set(shop["carried_products"]) == {"P0000", "P0001", "P0002"}


def test_scaffold_sink_has_demand_dist():
    """Sink nodes carry a demand_dist dict with a 'kind' key."""
    catalog = _make_catalog(1)
    result = scaffold_topology(catalog)
    sink = next(n for n in result["nodes"] if n["type"] == "demand_sink")
    assert "demand_dist" in sink
    assert "kind" in sink["demand_dist"]


# ---------------------------------------------------------------------------
# ScaffoldSpec validation
# ---------------------------------------------------------------------------


def test_scaffold_spec_invalid_shop_count():
    """ScaffoldSpec with shop_count < 1 raises ValueError."""
    with pytest.raises(ValueError, match="shop_count"):
        ScaffoldSpec(shop_count=0)


def test_scaffold_spec_invalid_sink_density():
    """ScaffoldSpec with sink_density <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="sink_density"):
        ScaffoldSpec(sink_density=0.0)


def test_scaffold_empty_catalog_raises():
    """Empty catalog raises ValueError."""
    with pytest.raises(ValueError, match="catalog"):
        scaffold_topology([])


# ---------------------------------------------------------------------------
# Custom region and sizes
# ---------------------------------------------------------------------------


def test_scaffold_custom_region():
    """spec.region is applied to all nodes."""
    catalog = _make_catalog(2)
    spec = ScaffoldSpec(region="EU")
    result = scaffold_topology(catalog, spec)
    for node in result["nodes"]:
        assert node["region"] == "EU"


def test_scaffold_custom_capacities():
    """spec.factory_capacity and shop_capacity are reflected in nodes."""
    catalog = _make_catalog(1)
    spec = ScaffoldSpec(factory_capacity=200, shop_capacity=1000)
    result = scaffold_topology(catalog, spec)
    factory = next(n for n in result["nodes"] if n["type"] == "factory")
    shop = next(n for n in result["nodes"] if n["type"] == "intermediate")
    assert factory["capacity_per_tick"] == 200
    assert shop["capacity"] == 1000


def test_scaffold_custom_lead_times():
    """spec.factory_lead_time and shop_lead_time are used in edges."""
    catalog = _make_catalog(1)
    spec = ScaffoldSpec(factory_lead_time=5, shop_lead_time=3)
    result = scaffold_topology(catalog, spec)
    node_ids = {n["id"] for n in result["nodes"]}
    shop_id = next(n["id"] for n in result["nodes"] if n["type"] == "intermediate")
    factory_shop_edge = next(
        e for e in result["edges"] if e["buyer"] == shop_id
    )
    shop_sink_edge = next(
        e for e in result["edges"] if e["supplier"] == shop_id
    )
    assert factory_shop_edge["lead_time"] == 5
    assert shop_sink_edge["lead_time"] == 3
