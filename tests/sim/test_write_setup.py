"""Tests for setup_io.write_setup, write_catalog_and_market, and load_or_build_setup.

Acceptance criteria (issue 04):
- write_setup(scenario, dir) followed by load_setup(dir) is a lossless round-trip
  for all setup data, policies included.
- write_catalog_and_market writes catalog.csv and market: block only (no nodes/edges/policy).
- Re-running against an existing setup directory loads without calling build_fn.
- A round-trip test asserts losslessness.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.sim.setup_io import (
    load_setup,
    load_or_build_setup,
    write_catalog_and_market,
    write_setup,
)


# ---------------------------------------------------------------------------
# Helpers: write a minimal setup, load it, use it as round-trip base
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent("""\
run:
  n_steps: 5
  start_date: "2024-01-01"
  world_seed: 7

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
  base_demand: {kind: constant, value: 8.0}

disruption:
  event_prob: 0.0
  types: [natural_disaster]
  regions: [US]
  severity: {kind: constant, value: 0.0}
  duration: {kind: constant, value: 1}

nodes:
  - id: factory-1
    type: factory
    region: US
    produces_product_id: P0001
    unit_cost: 4.0
    capacity_per_tick: 30
    inventory: 50
    list_price: 4.0
    cash: 0.0

  - id: shop-1
    type: intermediate
    region: US
    carried_products: [P0001]
    capacity: 200
    inventory:
      P0001: 10
    list_prices:
      P0001: 7.0
    min_order_imposed:
      P0001: 0
    cash: 500.0

  - id: sink-1
    type: demand_sink
    region: US
    product_id: P0001
    demand_dist: {kind: normal, mean: 5.0, std: 1.0}
    income_rate: 100.0
    cash: 1000.0

edges:
  - supplier: factory-1
    buyer: shop-1
    lead_time: 2

  - supplier: shop-1
    buyer: sink-1
    lead_time: 1
""")

_MINIMAL_CATALOG = (
    "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
    "P0001,Widget,widgets,10.0,4.0,peak,,1.0\n"
)


def _write_minimal_setup(tmp_path: Path) -> Path:
    """Write the minimal setup and return the directory path."""
    setup_dir = tmp_path / "base_setup"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(_MINIMAL_CATALOG)
    (setup_dir / "setup.yaml").write_text(_MINIMAL_YAML)
    return setup_dir


# ---------------------------------------------------------------------------
# write_setup round-trip
# ---------------------------------------------------------------------------


def test_write_setup_round_trip_catalog(tmp_path):
    """write_setup + load_setup produces identical catalog."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    assert len(reloaded.catalog) == len(scenario.catalog)
    for orig, rt in zip(scenario.catalog, reloaded.catalog):
        assert rt.product_id == orig.product_id
        assert rt.name == orig.name
        assert rt.category == orig.category
        assert rt.base_price == orig.base_price
        assert rt.unit_cost == orig.unit_cost
        assert rt.seasonality == orig.seasonality


def test_write_setup_round_trip_market(tmp_path):
    """write_setup + load_setup produces identical market params."""
    from dataclasses import fields

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written_market"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    for f in fields(scenario.market):
        orig_val = getattr(scenario.market, f.name)
        rt_val = getattr(reloaded.market, f.name)
        assert orig_val == rt_val, f"market.{f.name} mismatch: {orig_val!r} != {rt_val!r}"


def test_write_setup_round_trip_disruption(tmp_path):
    """write_setup + load_setup produces identical disruption params."""
    from dataclasses import fields

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written_disruption"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    for f in fields(scenario.disruption):
        orig_val = getattr(scenario.disruption, f.name)
        rt_val = getattr(reloaded.disruption, f.name)
        assert orig_val == rt_val, f"disruption.{f.name} mismatch"


def test_write_setup_round_trip_nodes(tmp_path):
    """write_setup + load_setup produces same node count and types."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written_nodes"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    assert len(reloaded.nodes) == len(scenario.nodes)
    for orig_ni, rt_ni in zip(scenario.nodes, reloaded.nodes):
        assert rt_ni.node.id == orig_ni.node.id
        assert type(rt_ni.node) is type(orig_ni.node)
        assert rt_ni.node.region == orig_ni.node.region


def test_write_setup_round_trip_edges(tmp_path):
    """write_setup + load_setup produces identical edges."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written_edges"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    assert len(reloaded.edges) == len(scenario.edges)
    for orig_e, rt_e in zip(scenario.edges, reloaded.edges):
        assert rt_e.supplier_id == orig_e.supplier_id
        assert rt_e.buyer_id == orig_e.buyer_id
        assert rt_e.default_lead_time == orig_e.default_lead_time


def test_write_setup_round_trip_run_params(tmp_path):
    """write_setup + load_setup preserves n_steps, world_seed, start_date."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "written_run"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    assert reloaded.n_steps == scenario.n_steps
    assert reloaded.world_seed == scenario.world_seed
    assert reloaded.start_date == scenario.start_date


def test_write_setup_creates_directory(tmp_path):
    """write_setup creates the directory if it doesn't exist."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "new" / "nested" / "dir"
    assert not out_dir.exists()
    write_setup(scenario, out_dir)
    assert (out_dir / "catalog.csv").is_file()
    assert (out_dir / "setup.yaml").is_file()


def test_write_setup_related_products_round_trip(tmp_path):
    """related_products round-trips correctly."""
    catalog_csv = (
        "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
        "P0001,Widget A,widgets,12.0,4.0,peak,P0002:0.5,1.0\n"
        "P0002,Widget B,widgets,10.0,3.0,peak,,1.0\n"
    )
    yaml_content = textwrap.dedent("""\
    run:
      n_steps: 3
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

    nodes:
      - id: factory-1
        type: factory
        region: US
        produces_product_id: P0001
        unit_cost: 4.0
        capacity_per_tick: 20
        inventory: 10
        list_price: 4.0
        cash: 0.0

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory: {P0001: 5}
        list_prices: {P0001: 7.0}
        cash: 500.0

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: constant, value: 5.0}
        income_rate: 50.0
        cash: 500.0

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 1
      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
    """)
    setup_dir = tmp_path / "related"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(catalog_csv)
    (setup_dir / "setup.yaml").write_text(yaml_content)

    scenario = load_setup(setup_dir)
    out_dir = tmp_path / "related_out"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    ware_a = reloaded.catalog[0]
    assert ware_a.product_id == "P0001"
    assert len(ware_a.related_products) == 1
    assert ware_a.related_products[0][0] == "P0002"
    assert abs(ware_a.related_products[0][1] - 0.5) < 1e-9


def test_write_setup_with_policies_round_trip(tmp_path):
    """Policies survive write_setup + load_setup round-trip."""
    yaml_with_policies = textwrap.dedent("""\
    run:
      n_steps: 5
      start_date: "2024-01-01"
      world_seed: 7

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

    nodes:
      - id: factory-1
        type: factory
        region: US
        produces_product_id: P0001
        unit_cost: 4.0
        capacity_per_tick: 30
        inventory: 50
        list_price: 4.0
        cash: 0.0
        policy:
          name: static_factory
          params:
            target_inventory: 100

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory:
          P0001: 10
        list_prices:
          P0001: 7.0
        min_order_imposed:
          P0001: 0
        cash: 500.0
        policy:
          name: order_up_to
          params:
            cover_horizon_ticks: 14

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: constant, value: 5.0}
        income_rate: 100.0
        cash: 1000.0
        policy:
          name: default_demand_sink
          params: {}

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 2
      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
    """)
    setup_dir = tmp_path / "with_policies"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(_MINIMAL_CATALOG)
    (setup_dir / "setup.yaml").write_text(yaml_with_policies)

    scenario = load_setup(setup_dir)
    # Policies must be attached after load
    factory_ni = next(ni for ni in scenario.nodes if ni.node.id == "factory-1")
    assert factory_ni.policy is not None

    out_dir = tmp_path / "with_policies_out"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    # Policies must survive round-trip
    factory_rt = next(ni for ni in reloaded.nodes if ni.node.id == "factory-1")
    shop_rt = next(ni for ni in reloaded.nodes if ni.node.id == "shop-1")
    sink_rt = next(ni for ni in reloaded.nodes if ni.node.id == "sink-1")

    assert factory_rt.policy is not None
    assert shop_rt.policy is not None
    assert sink_rt.policy is not None


def test_write_setup_determinism_run(tmp_path):
    """Round-tripped scenario produces bit-identical run logs."""
    from src.sim.runner import Runner

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "det_written"
    write_setup(scenario, out_dir)
    rt_scenario = load_setup(out_dir)

    log1 = Runner(scenario).run()
    log2 = Runner(rt_scenario).run()

    assert log1["n_steps"] == log2["n_steps"]
    for t in range(log1["n_steps"]):
        assert log1["ticks"][t]["node_cash"] == log2["ticks"][t]["node_cash"]
        assert log1["ticks"][t]["node_inventory"] == log2["ticks"][t]["node_inventory"]


# ---------------------------------------------------------------------------
# write_catalog_and_market — LLM generator output (data only)
# ---------------------------------------------------------------------------


def test_write_catalog_and_market_creates_catalog_csv(tmp_path):
    """write_catalog_and_market writes catalog.csv."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "llm_out"
    write_catalog_and_market(scenario.catalog, scenario.market, out_dir)

    assert (out_dir / "catalog.csv").is_file()


def test_write_catalog_and_market_creates_market_block(tmp_path):
    """write_catalog_and_market writes market: block to setup.yaml."""
    import yaml

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "llm_market"
    write_catalog_and_market(scenario.catalog, scenario.market, out_dir)

    yaml_path = out_dir / "setup.yaml"
    assert yaml_path.is_file()
    doc = yaml.safe_load(yaml_path.read_text())
    assert "market" in doc


def test_write_catalog_and_market_no_nodes_or_edges(tmp_path):
    """write_catalog_and_market does NOT write nodes or edges."""
    import yaml

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "llm_no_topo"
    write_catalog_and_market(scenario.catalog, scenario.market, out_dir)

    doc = yaml.safe_load((out_dir / "setup.yaml").read_text())
    assert "nodes" not in doc
    assert "edges" not in doc


def test_write_catalog_and_market_merges_existing_yaml(tmp_path):
    """write_catalog_and_market updates market: in an existing setup.yaml without destroying other keys."""
    import yaml

    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "llm_merge"
    out_dir.mkdir()
    existing_yaml = {"run": {"n_steps": 10, "world_seed": 99, "start_date": "2024-06-01"}}
    (out_dir / "setup.yaml").write_text(yaml.safe_dump(existing_yaml))

    write_catalog_and_market(scenario.catalog, scenario.market, out_dir)

    doc = yaml.safe_load((out_dir / "setup.yaml").read_text())
    assert doc["run"]["n_steps"] == 10  # preserved
    assert "market" in doc             # added


def test_write_catalog_and_market_catalog_round_trip(tmp_path):
    """Catalog written by write_catalog_and_market loads correctly."""
    src = _write_minimal_setup(tmp_path)
    scenario = load_setup(src)

    out_dir = tmp_path / "llm_catalog_rt"
    write_catalog_and_market(scenario.catalog, scenario.market, out_dir)

    # Add minimal nodes/edges/run/disruption so load_setup can fully load it
    import yaml
    doc = yaml.safe_load((out_dir / "setup.yaml").read_text())
    doc["run"] = {"n_steps": 3, "start_date": "2024-01-01", "world_seed": 1}
    doc["disruption"] = {
        "event_prob": 0.0, "types": ["natural_disaster"], "regions": ["US"],
        "severity": {"kind": "constant", "value": 0.0},
        "duration": {"kind": "constant", "value": 1},
    }
    doc["nodes"] = [
        {"id": "factory-1", "type": "factory", "region": "US",
         "produces_product_id": "P0001", "unit_cost": 4.0, "capacity_per_tick": 10,
         "inventory": 0, "list_price": 4.0, "cash": 0.0},
        {"id": "shop-1", "type": "intermediate", "region": "US",
         "carried_products": ["P0001"], "capacity": 100,
         "inventory": {"P0001": 0}, "list_prices": {"P0001": 6.0}, "cash": 500.0},
        {"id": "sink-1", "type": "demand_sink", "region": "US",
         "product_id": "P0001", "demand_dist": {"kind": "constant", "value": 5.0},
         "income_rate": 50.0, "cash": 500.0},
    ]
    doc["edges"] = [
        {"supplier": "factory-1", "buyer": "shop-1", "lead_time": 1},
        {"supplier": "shop-1", "buyer": "sink-1", "lead_time": 1},
    ]
    (out_dir / "setup.yaml").write_text(yaml.safe_dump(doc))

    reloaded = load_setup(out_dir)
    assert len(reloaded.catalog) == len(scenario.catalog)
    assert reloaded.catalog[0].product_id == "P0001"


# ---------------------------------------------------------------------------
# load_or_build_setup — dir-as-cache
# ---------------------------------------------------------------------------


def test_load_or_build_setup_calls_build_fn_on_miss(tmp_path):
    """load_or_build_setup calls build_fn when the setup dir is empty."""
    src = _write_minimal_setup(tmp_path)
    base_scenario = load_setup(src)

    call_count = {"n": 0}

    def build_fn():
        call_count["n"] += 1
        return base_scenario

    out_dir = tmp_path / "cache_miss"
    out_dir.mkdir()  # empty — no catalog.csv yet

    result = load_or_build_setup(out_dir, build_fn)
    assert call_count["n"] == 1
    assert result is not None


def test_load_or_build_setup_skips_build_fn_on_hit(tmp_path):
    """load_or_build_setup does NOT call build_fn when setup dir is populated."""
    src = _write_minimal_setup(tmp_path)
    base_scenario = load_setup(src)

    out_dir = tmp_path / "cache_hit"
    write_setup(base_scenario, out_dir)

    call_count = {"n": 0}

    def build_fn():
        call_count["n"] += 1
        return base_scenario

    result = load_or_build_setup(out_dir, build_fn)
    assert call_count["n"] == 0
    assert result is not None


def test_load_or_build_setup_force_rebuild(tmp_path):
    """force_rebuild=True calls build_fn even when setup dir exists."""
    src = _write_minimal_setup(tmp_path)
    base_scenario = load_setup(src)

    out_dir = tmp_path / "force_rebuild"
    write_setup(base_scenario, out_dir)

    call_count = {"n": 0}

    def build_fn():
        call_count["n"] += 1
        return base_scenario

    result = load_or_build_setup(out_dir, build_fn, force_rebuild=True)
    assert call_count["n"] == 1
    assert result is not None


def test_load_or_build_setup_writes_to_dir(tmp_path):
    """load_or_build_setup persists the built scenario on a cache miss."""
    src = _write_minimal_setup(tmp_path)
    base_scenario = load_setup(src)

    out_dir = tmp_path / "persisted"
    out_dir.mkdir()

    load_or_build_setup(out_dir, lambda: base_scenario)

    assert (out_dir / "catalog.csv").is_file()
    assert (out_dir / "setup.yaml").is_file()
