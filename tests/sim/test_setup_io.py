"""Tests for setup_io.load_setup and _derive_node_seed.

Acceptance criteria covered:
- catalog.csv + setup.yaml load into a typed Scenario
- related_products parsing
- build_graph accepts the resulting DAG
- All seeds derive from a single world_seed
- Running identical setup dir twice produces bit-identical run logs
- Malformed setups raise single clear errors
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.sim.setup_io import _derive_node_seed, load_setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_setup(tmp_path: Path, **overrides) -> Path:
    """Write a minimal valid 3-node chain setup directory and return its path."""
    catalog_content = overrides.get(
        "catalog",
        "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
        "P0001,Widget,widgets,10.0,4.0,peak,,1.0\n",
    )
    yaml_content = overrides.get(
        "yaml",
        textwrap.dedent("""\
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

          - id: shop-1
            type: intermediate
            region: US
            carried_products: [P0001]
            capacity: 200
            inventory:
              P0001: 10
            list_prices:
              P0001: 7.0

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {kind: constant, value: 5.0}
            income_rate: 100.0

        edges:
          - supplier: factory-1
            buyer: shop-1
            lead_time: 2

          - supplier: shop-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    setup_dir = tmp_path / "test_setup"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(catalog_content)
    (setup_dir / "setup.yaml").write_text(yaml_content)
    return setup_dir


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_setup_returns_scenario(tmp_path):
    """load_setup produces a valid Scenario from a minimal setup directory."""
    setup_dir = _write_minimal_setup(tmp_path)
    scenario = load_setup(setup_dir)

    assert len(scenario.catalog) == 1
    assert scenario.catalog[0].product_id == "P0001"
    assert scenario.n_steps == 5
    assert scenario.world_seed == 7
    assert len(scenario.nodes) == 3
    assert len(scenario.edges) == 2
    assert len(scenario.nodes) > 0  # graph-mode: nodes present


def test_load_setup_builds_valid_dag(tmp_path):
    """The loaded Scenario's nodes/edges form a valid DAG (build_graph succeeds)."""
    from src.sim.graph import EdgeSpec, build_graph

    setup_dir = _write_minimal_setup(tmp_path)
    scenario = load_setup(setup_dir)
    node_ids = [ni.node.id for ni in scenario.nodes]
    edges: list[EdgeSpec] = list(scenario.edges)
    # If graph is invalid, build_graph raises — this test verifies it doesn't.
    graph = build_graph(node_ids, edges)
    assert len(graph.nodes) == 3


def test_load_setup_seeds_derive_from_world_seed(tmp_path):
    """Each node's init_seed matches _derive_node_seed(world_seed, node_id)."""
    setup_dir = _write_minimal_setup(tmp_path)
    scenario = load_setup(setup_dir)

    world_seed = scenario.world_seed
    for ni in scenario.nodes:
        expected = _derive_node_seed(world_seed, ni.node.id)
        assert ni.node.init_seed == expected, (
            f"node {ni.node.id}: expected init_seed {expected}, got {ni.node.init_seed}"
        )


def test_load_setup_related_products_parsing(tmp_path):
    """related_products column is correctly parsed into (pid, weight) tuples."""
    catalog = (
        "product_id,name,category,base_price,unit_cost,seasonality,related_products,init_stock_share\n"
        "P0001,Widget A,widgets,10.0,4.0,peak,P0002:0.5;P0003:0.3,1.0\n"
        "P0002,Widget B,widgets,10.0,4.0,peak,,1.0\n"
        "P0003,Widget C,widgets,10.0,4.0,peak,,1.0\n"
    )
    yaml_base = textwrap.dedent("""\
    run:
      n_steps: 2
      start_date: "2024-01-01"
      world_seed: 99

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
        inventory: 20
        list_price: 4.0

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001, P0002, P0003]
        capacity: 300
        inventory: {P0001: 5, P0002: 5, P0003: 5}
        list_prices: {P0001: 7.0, P0002: 7.0, P0003: 7.0}

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: constant, value: 3.0}
        income_rate: 50.0

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 1

      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
    """)

    setup_dir = tmp_path / "rp_test"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text(catalog)
    (setup_dir / "setup.yaml").write_text(yaml_base)

    scenario = load_setup(setup_dir)
    ware_a = scenario.catalog[0]
    assert ware_a.product_id == "P0001"
    assert len(ware_a.related_products) == 2
    rel_ids = {p for p, _ in ware_a.related_products}
    assert "P0002" in rel_ids
    assert "P0003" in rel_ids


def test_load_setup_distribution_demand_dist(tmp_path):
    """demand_dist with a normal distribution tag is parsed into a Normal instance."""
    from src.sim.distributions import Normal

    setup_dir = _write_minimal_setup(tmp_path)
    scenario = load_setup(setup_dir)

    # In the minimal setup, demand_dist is a Constant.
    # Replace with a Normal by overriding the sink.
    yaml_content = textwrap.dedent("""\
    run:
      n_steps: 2
      start_date: "2024-01-01"
      world_seed: 1

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
        inventory: 20
        list_price: 4.0

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory: {P0001: 5}
        list_prices: {P0001: 7.0}

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: normal, mean: 8.0, std: 2.0}
        income_rate: 100.0

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 1
      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
    """)
    setup_dir2 = tmp_path / "normal_demand"
    setup_dir2.mkdir()
    (setup_dir2 / "catalog.csv").write_text(
        "product_id,name,category,base_price,unit_cost,seasonality\n"
        "P0001,Widget,widgets,10.0,4.0,peak\n"
    )
    (setup_dir2 / "setup.yaml").write_text(yaml_content)
    scenario2 = load_setup(setup_dir2)
    from src.sim.node import DemandSinkNode
    sink = next(
        ni.node for ni in scenario2.nodes if isinstance(ni.node, DemandSinkNode)
    )
    assert isinstance(sink.demand_dist, Normal)
    assert sink.demand_dist.mean == 8.0


# ---------------------------------------------------------------------------
# IntermediateNode economic parameters (holding_rate / order_fee)
# ---------------------------------------------------------------------------


def test_intermediate_economic_params_default_when_omitted(tmp_path):
    """An intermediate node that omits holding_rate/order_fee gets the canonical default."""
    from src.sim.node import DEFAULT_HOLDING_RATE, DEFAULT_ORDER_FEE, IntermediateNode

    setup_dir = _write_minimal_setup(tmp_path)
    scenario = load_setup(setup_dir)
    shop = next(
        ni.node for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)
    )
    assert shop.holding_rate == DEFAULT_HOLDING_RATE
    assert shop.order_fee == DEFAULT_ORDER_FEE


def test_intermediate_economic_params_parsed(tmp_path):
    """Per-node holding_rate/order_fee in setup.yaml are parsed onto the node."""
    from src.sim.node import IntermediateNode

    yaml_content = textwrap.dedent("""\
    run:
      n_steps: 2
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

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory: {P0001: 10}
        list_prices: {P0001: 7.0}
        holding_rate: 0.05
        order_fee: 12.5

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: constant, value: 5.0}
        income_rate: 100.0

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 2
      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
    """)
    setup_dir = _write_minimal_setup(tmp_path, yaml=yaml_content)
    scenario = load_setup(setup_dir)
    shop = next(
        ni.node for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)
    )
    assert shop.holding_rate == 0.05
    assert shop.order_fee == 12.5


def test_intermediate_economic_params_roundtrip(tmp_path):
    """holding_rate/order_fee survive a load -> write_setup -> load round-trip per node."""
    from src.sim.node import IntermediateNode
    from src.sim.setup_io import write_setup

    yaml_content = textwrap.dedent("""\
    run:
      n_steps: 2
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

      - id: shop-1
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory: {P0001: 10}
        list_prices: {P0001: 7.0}
        holding_rate: 0.05
        order_fee: 12.5

      - id: shop-2
        type: intermediate
        region: US
        carried_products: [P0001]
        capacity: 200
        inventory: {P0001: 10}
        list_prices: {P0001: 7.0}

      - id: sink-1
        type: demand_sink
        region: US
        product_id: P0001
        demand_dist: {kind: constant, value: 5.0}
        income_rate: 100.0

    edges:
      - supplier: factory-1
        buyer: shop-1
        lead_time: 2
      - supplier: factory-1
        buyer: shop-2
        lead_time: 2
      - supplier: shop-1
        buyer: sink-1
        lead_time: 1
      - supplier: shop-2
        buyer: sink-1
        lead_time: 1
    """)
    setup_dir = _write_minimal_setup(tmp_path, yaml=yaml_content)

    scenario = load_setup(setup_dir)
    out_dir = tmp_path / "roundtrip"
    write_setup(scenario, out_dir)
    reloaded = load_setup(out_dir)

    def _econ(scen):
        return {
            ni.node.id: (ni.node.holding_rate, ni.node.order_fee)
            for ni in scen.nodes
            if isinstance(ni.node, IntermediateNode)
        }

    expected = {"shop-1": (0.05, 12.5), "shop-2": (0.01, 50.0)}
    assert _econ(scenario) == expected
    assert _econ(reloaded) == expected


# ---------------------------------------------------------------------------
# Determinism: identical setup dir → bit-identical run logs
# ---------------------------------------------------------------------------


def test_load_setup_determinism(tmp_path):
    """Running an identical setup directory twice produces bit-identical run logs."""
    from src.sim.runner import Runner

    setup_dir = _write_minimal_setup(tmp_path)

    log1 = Runner(load_setup(setup_dir)).run()
    log2 = Runner(load_setup(setup_dir)).run()

    # Compare tick-by-tick cash and inventory — the observable run-log keys.
    assert log1["n_steps"] == log2["n_steps"]
    for t in range(log1["n_steps"]):
        tick1 = log1["ticks"][t]
        tick2 = log2["ticks"][t]
        assert tick1["node_cash"] == tick2["node_cash"], f"tick {t} node_cash differs"
        assert tick1["node_inventory"] == tick2["node_inventory"], (
            f"tick {t} node_inventory differs"
        )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_catalog_raises(tmp_path):
    """Missing catalog.csv raises FileNotFoundError."""
    setup_dir = tmp_path / "no_catalog"
    setup_dir.mkdir()
    (setup_dir / "setup.yaml").write_text("run: {n_steps: 1, start_date: '2024-01-01', world_seed: 1}\n")
    with pytest.raises(FileNotFoundError, match="catalog.csv"):
        load_setup(setup_dir)


def test_missing_setup_yaml_raises(tmp_path):
    """Missing setup.yaml raises FileNotFoundError."""
    setup_dir = tmp_path / "no_yaml"
    setup_dir.mkdir()
    (setup_dir / "catalog.csv").write_text("product_id,name\n")
    with pytest.raises(FileNotFoundError, match="setup.yaml"):
        load_setup(setup_dir)


def test_missing_required_node_field_raises(tmp_path):
    """A node missing 'produces_product_id' raises a clear ValueError."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
            # produces_product_id intentionally omitted
            unit_cost: 4.0
            capacity_per_tick: 20
            inventory: 20
            list_price: 4.0

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {kind: constant, value: 5.0}
            income_rate: 50.0

        edges:
          - supplier: factory-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="produces_product_id"):
        load_setup(setup_dir)


def test_malformed_distribution_tag_raises(tmp_path):
    """A distribution dict missing 'kind' raises a clear ValueError."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
            inventory: 20
            list_price: 4.0

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {mean: 5.0, std: 1.0}  # missing 'kind'
            income_rate: 50.0

        edges:
          - supplier: factory-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="kind"):
        load_setup(setup_dir)


def test_unknown_distribution_kind_raises(tmp_path):
    """A distribution dict with an unknown 'kind' raises a clear ValueError."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
            inventory: 20
            list_price: 4.0

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {kind: poisson, lambda: 5.0}  # unknown kind
            income_rate: 50.0

        edges:
          - supplier: factory-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="poisson"):
        load_setup(setup_dir)


def test_duplicate_node_id_raises(tmp_path):
    """Duplicate node ids raise a single clear error mentioning the duplicated id."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
            inventory: 20
            list_price: 4.0

          - id: factory-1
            type: factory
            region: US
            produces_product_id: P0001
            unit_cost: 4.0
            capacity_per_tick: 10
            inventory: 10
            list_price: 4.0

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {kind: constant, value: 5.0}
            income_rate: 50.0

        edges:
          - supplier: factory-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_setup(setup_dir)


def test_cyclic_graph_raises(tmp_path):
    """A cyclic graph in setup.yaml raises a clear ValueError mentioning the cycle."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
          - id: node-a
            type: factory
            region: US
            produces_product_id: P0001
            unit_cost: 4.0
            capacity_per_tick: 20
            inventory: 20
            list_price: 4.0

          - id: node-b
            type: intermediate
            region: US
            carried_products: [P0001]
            capacity: 100
            inventory: {P0001: 5}
            list_prices: {P0001: 7.0}

        edges:
          - supplier: node-a
            buyer: node-b
            lead_time: 1

          - supplier: node-b
            buyer: node-a
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="[Cc]ycle|cycle"):
        load_setup(setup_dir)


def test_unknown_policy_name_raises(tmp_path):
    """An unknown policy name raises a clear ValueError."""
    setup_dir = _write_minimal_setup(
        tmp_path,
        yaml=textwrap.dedent("""\
        run:
          n_steps: 2
          start_date: "2024-01-01"
          world_seed: 1

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
            inventory: 20
            list_price: 4.0
            policy:
              name: unknown_policy_xyz
              params: {}

          - id: wh-1
            type: intermediate
            region: US
            carried_products: [P0001]
            capacity: 200
            inventory: {P0001: 0}
            list_prices: {P0001: 6.0}
            policy:
              name: base_stock
              params: {S: 10}

          - id: sink-1
            type: demand_sink
            region: US
            product_id: P0001
            demand_dist: {kind: constant, value: 5.0}
            income_rate: 50.0

        edges:
          - supplier: factory-1
            buyer: wh-1
            lead_time: 1
          - supplier: wh-1
            buyer: sink-1
            lead_time: 1
        """),
    )
    with pytest.raises(ValueError, match="unknown_policy_xyz"):
        load_setup(setup_dir)


# ---------------------------------------------------------------------------
# _derive_node_seed unit tests
# ---------------------------------------------------------------------------


def test_derive_node_seed_is_deterministic():
    """Same inputs always produce the same seed."""
    s1 = _derive_node_seed(42, "factory-1")
    s2 = _derive_node_seed(42, "factory-1")
    assert s1 == s2


def test_derive_node_seed_is_32_bit():
    """Derived seed fits in a 32-bit unsigned integer."""
    seed = _derive_node_seed(999999, "shop-1", "policy")
    assert 0 <= seed <= 0xFFFF_FFFF


def test_derive_node_seed_different_nodes_differ():
    """Different node IDs produce different seeds."""
    s1 = _derive_node_seed(42, "factory-1")
    s2 = _derive_node_seed(42, "shop-1")
    assert s1 != s2


def test_derive_node_seed_different_world_seeds_differ():
    """Different world seeds produce different node seeds."""
    s1 = _derive_node_seed(42, "factory-1")
    s2 = _derive_node_seed(43, "factory-1")
    assert s1 != s2


def test_derive_node_seed_purpose_suffix_differs():
    """Adding a purpose suffix produces a different seed from no suffix."""
    s1 = _derive_node_seed(42, "factory-1")
    s2 = _derive_node_seed(42, "factory-1", "policy")
    assert s1 != s2
