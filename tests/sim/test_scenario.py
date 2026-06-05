"""Tier-3 tests: scenario authoring helpers.

Covers the contract described in PRD § Tier coverage T3:
- ``load_catalog`` assigns sequential P{i:04d} ids and normalises related_products
- ``make_nodes`` produces ``NodeInstance``s (graph-mode) with CRN seeding
- ``load_scenario_from_path`` loads a scenario from a Python file
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.sim.distributions import Constant, Normal, Uniform
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    Ware,
    load_catalog,
    load_scenario_from_path,
)


def _market() -> MarketParams:
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.001,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.2,
        off_factor=0.7,
        season_months={"all_season": list(range(1, 13))},
        regions=["US", "EU"],
        correlation=0.7,
        trend_update_interval=50,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.7,
            "growth": Uniform(1.2, 2.0),
            "maturity": 1.0,
            "decline": 0.2,
        },
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.02),
        supply_shock=Normal(0.0, 0.02),
        base_demand=Constant(50),
    )


def _disruption() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US", "EU"],
        severity=Constant(0.01),
        duration=Constant(5),
    )


def _item_lifecycle() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _catalog() -> list[Ware]:
    return load_catalog(
        [
            {
                "name": "Widget A",
                "category": "Widgets",
                "related_products": [],
                "base_price": 20.0,
                "unit_cost": 12.0,
                "seasonality": "all_season",
            },
            {
                "name": "Widget B",
                "category": "Widgets",
                "related_products": [["Widget A", 0.5]],
                "base_price": 30.0,
                "unit_cost": 18.0,
                "seasonality": "all_season",
            },
        ]
    )


# ---------------------------------------------------------------- catalog


def test_load_catalog_assigns_sequential_product_ids() -> None:
    cat = _catalog()
    assert [w.product_id for w in cat] == ["P0000", "P0001"]


def test_load_catalog_returns_ware_namedtuples() -> None:
    cat = _catalog()
    assert all(isinstance(w, Ware) for w in cat)
    assert cat[0]._fields == (
        "product_id",
        "name",
        "category",
        "related_products",
        "base_price",
        "unit_cost",
        "seasonality",
        "init_stage",
        "stage_change_probs",
        "freshness_alpha",
        "freshness_decay",
        "init_stock_share",
    )
    # New per-Ware override fields default to ``None`` when unset.
    assert cat[0].init_stage is None
    assert cat[0].stage_change_probs is None
    assert cat[0].freshness_alpha is None
    assert cat[0].freshness_decay is None
    assert cat[0].init_stock_share is None


def test_load_catalog_normalises_related_products_to_tuples() -> None:
    cat = _catalog()
    rels = cat[1].related_products
    assert all(isinstance(p, tuple) for p in rels)
    assert rels == [("Widget A", 0.5)]


def test_load_catalog_drops_caller_supplied_product_id() -> None:
    cat = load_catalog(
        [
            {
                "product_id": "ZZZZ",
                "name": "Foo",
                "category": "Bar",
                "related_products": [],
                "base_price": 1.0,
                "unit_cost": 0.5,
                "seasonality": "all_season",
            }
        ]
    )
    assert cat[0].product_id == "P0000"


# ---------------------------------------------------------------- graph k-way


def test_graph_kway_three_nodes_sharing_init_seed_are_bit_identical_at_step0() -> None:
    """k=3 CRN comparison: three graph-mode nodes sharing the same init_seed
    start bit-identical at step 0 regardless of attached policy.

    Phase-4 (issue 11): migrated from the legacy Store engine to the
    graph engine. Three IntermediateNode shops with the same seed and
    inventory should have identical initial state.
    """
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import OrderUpToPolicy
    from src.sim.runner import build_world
    from src.sim.scenario import NodeInstance

    cat = _catalog()
    pid = cat[0].product_id

    # Three shops with the same seed — should start bit-identical.
    shops_and_policies = [
        ("shopA", OrderUpToPolicy(policy_seed=1)),
        ("shopB", OrderUpToPolicy(policy_seed=2)),
        ("shopC", OrderUpToPolicy(policy_seed=3)),
    ]

    node_instances = []
    edges = []
    for shop_id, pol in shops_and_policies:
        factory_id = f"{shop_id}-factory"
        factory = FactoryNode(
            id=factory_id, region="US", init_seed=99,
            produces_product_id=pid, unit_cost=12.0,
            capacity_per_tick=50, inventory=100,
            list_price=12.0, cash=0.0,
        )
        shop = IntermediateNode(
            id=shop_id, region="US", init_seed=99,
            carried_products={pid}, capacity=1000,
            tags=[], inventory={pid: 10}, pending={},
            list_prices={pid: 20.0}, min_order_imposed={pid: 0},
            cash=10000.0,
        )
        sink_id = f"{shop_id}-sink"
        sink = DemandSinkNode(
            id=sink_id, region="US", init_seed=99,
            product_id=pid, demand_dist=Constant(5.0),
            income_rate=100.0, cash=500.0, activation_tick={},
        )
        node_instances.extend([
            NodeInstance(node=factory, init_seed=99, policy=None),
            NodeInstance(node=shop, init_seed=99, policy=pol),
            NodeInstance(node=sink, init_seed=99, policy=None),
        ])
        edges.extend([
            EdgeSpec(supplier_id=factory_id, buyer_id=shop_id, default_lead_time=2),
            EdgeSpec(supplier_id=shop_id, buyer_id=sink_id, default_lead_time=1),
        ])

    scenario = Scenario(
        catalog=cat,
        market=_market(),
        disruption=_disruption(),
        item_lifecycle=_item_lifecycle(),
        nodes=node_instances,
        edges=edges,
        n_steps=1,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    sim = build_world(scenario)
    shop_a = sim.nodes["shopA"]
    shop_b = sim.nodes["shopB"]
    shop_c = sim.nodes["shopC"]
    # All three shops started with the same init_seed — same inventory and cash.
    assert shop_a.inventory == shop_b.inventory == shop_c.inventory
    assert shop_a.cash == shop_b.cash == shop_c.cash


# ---------------------------------------------------------------- load_scenario_from_path


# Minimal scenario module written inline for load_scenario_from_path tests.
_MINIMAL_SCENARIO_SRC = """\
from datetime import datetime
from src.sim.distributions import Constant, Normal
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import DefaultDemandSinkPolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.sim.scenario import (
    DisruptionParams, ItemLifecycleParams, MarketParams, NodeInstance, Scenario, load_catalog,
)

_catalog = load_catalog([
    {"name": "Widget", "category": "g", "related_products": [], "base_price": 10.0,
     "unit_cost": 4.0, "seasonality": "all_season"}
])
_pid = _catalog[0].product_id
_f = FactoryNode(id="f-1", region="US", init_seed=1, produces_product_id=_pid,
                 unit_cost=4.0, capacity_per_tick=50, inventory=100, list_price=4.0, cash=0.0)
_f.policy = StaticFactoryPolicy(capacity_per_tick=50, unit_cost=4.0, policy_seed=1)
_s = IntermediateNode(id="s-1", region="US", init_seed=2, carried_products={_pid},
                      capacity=500, tags=["shop"], inventory={_pid: 20}, pending={},
                      list_prices={_pid: 10.0}, min_order_imposed={_pid: 0}, cash=500.0)
_s.policy = OrderUpToPolicy(cover_horizon_ticks=14, policy_seed=2)
_d = DemandSinkNode(id="d-1", region="US", init_seed=3, product_id=_pid,
                    demand_dist=Normal(mean=10, std=2), income_rate=200.0, cash=1000.0)
_d.policy = DefaultDemandSinkPolicy(policy_seed=3)

_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
scenario = Scenario(
    catalog=_catalog,
    market=MarketParams(
        cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
        peak_factor=1.0, off_factor=1.0, season_months={}, regions=["US"],
        correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
        stage_multipliers={s: 1.0 for s in _STAGES},
        price_elasticity=0.0, promo_multiplier=1.0,
        demand_factor_min=0.1, supply_factor_min=0.01,
        cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
        base_demand=Constant(10),
    ),
    disruption=DisruptionParams(
        event_prob=0.0, types=["natural_disaster"], regions=["US"],
        severity=Constant(0.0), duration=Constant(1),
    ),
    item_lifecycle=ItemLifecycleParams(
        stages=_STAGES, init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in _STAGES},
    ),
    nodes=[NodeInstance(node=_f, init_seed=1, policy=_f.policy),
           NodeInstance(node=_s, init_seed=2, policy=_s.policy),
           NodeInstance(node=_d, init_seed=3, policy=_d.policy)],
    edges=[
        EdgeSpec(supplier_id="f-1", buyer_id="s-1", default_lead_time=2),
        EdgeSpec(supplier_id="s-1", buyer_id="d-1", default_lead_time=1),
    ],
    n_steps=5, start_date=datetime(2024, 1, 1), world_seed=42,
)
"""


def test_load_scenario_from_path_returns_scenario_with_catalog_and_nodes(
    tmp_path: Path,
) -> None:
    """load_scenario_from_path loads a graph-mode Scenario from a Python file."""
    sc_file = tmp_path / "minimal_scenario.py"
    sc_file.write_text(_MINIMAL_SCENARIO_SRC)
    sc = load_scenario_from_path(sc_file)
    assert isinstance(sc, Scenario)
    assert len(sc.catalog) > 0
    assert len(sc.nodes) > 0


def test_load_scenario_from_path_preserves_live_policies(tmp_path: Path) -> None:
    sc_file = tmp_path / "minimal_scenario.py"
    sc_file.write_text(_MINIMAL_SCENARIO_SRC)
    sc = load_scenario_from_path(sc_file)
    assert any(ni.policy is not None for ni in sc.nodes)


def test_load_scenario_from_path_accepts_string_path(tmp_path: Path) -> None:
    sc_file = tmp_path / "minimal_scenario.py"
    sc_file.write_text(_MINIMAL_SCENARIO_SRC)
    sc = load_scenario_from_path(str(sc_file))
    assert isinstance(sc, Scenario)


def test_load_scenario_from_path_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario_from_path("/nonexistent/path/scenario.py")


def test_load_scenario_from_path_missing_attribute_raises_attribute_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad_scenario.py"
    bad.write_text("x = 1\n")
    with pytest.raises(AttributeError, match="scenario"):
        load_scenario_from_path(bad)
