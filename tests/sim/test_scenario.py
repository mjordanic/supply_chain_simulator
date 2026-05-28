"""Tier-3 tests: scenario authoring helpers and JSON round-trip.

Covers the contract described in PRD § Tier coverage T3:
- ``make_stores`` produces ``StoreInstance``s from a literal triple list,
  expressing CRN/paired/k-way comparisons by repeated ``(template, init_seed)``
- ``Scenario`` round-trips via JSON with structural equality preserved
- malformed JSON is rejected at parse time
"""

from __future__ import annotations

import json
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
    StoreInstance,
    StoreTemplate,
    Ware,
    load_catalog,
    load_scenario_from_path,
    make_stores,
)


class StubPolicy:
    """Stand-in for a real ``Policy`` (the ABC lands in issue 06).

    The scenario module is policy-agnostic — it only needs an attachable object —
    so an inert stub is enough to drive these tests.
    """

    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StubPolicy) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("StubPolicy", self.name))


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


def _template() -> StoreTemplate:
    return StoreTemplate(
        id="small",
        region="US",
        capacity=1000,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
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


def _scenario(stores: list[StoreInstance] | None = None) -> Scenario:
    return Scenario(
        catalog=_catalog(),
        market=_market(),
        disruption=_disruption(),
        item_lifecycle=_item_lifecycle(),
        stores=stores
        if stores is not None
        else [StoreInstance(template=_template(), init_seed=42, policy=None)],
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=12345,
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


# -------------------------------------------------------------- make_stores


def test_make_stores_returns_one_instance_per_triple() -> None:
    template = _template()
    a, b, c = StubPolicy("a"), StubPolicy("b"), StubPolicy("c")
    out = make_stores([(template, 10, a), (template, 11, b), (template, 12, c)])
    assert len(out) == 3
    assert all(isinstance(s, StoreInstance) for s in out)


def test_make_stores_preserves_triple_order_and_attaches_each_field() -> None:
    template = _template()
    a, b = StubPolicy("a"), StubPolicy("b")
    triples = [(template, 7, a), (template, 7, b), (template, 9, a)]
    out = make_stores(triples)
    for instance, (tpl, seed, pol) in zip(out, triples):
        assert instance.template is tpl
        assert instance.init_seed == seed
        assert instance.policy is pol


def test_make_stores_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        make_stores([])


def test_make_stores_paired_two_policies_share_template_and_seed() -> None:
    """Two policies on one (template, init_seed) is the paired-CRN pattern."""
    template = _template()
    a, b = StubPolicy("a"), StubPolicy("b")
    out = make_stores(
        [
            (template, 1, a),
            (template, 1, b),
            (template, 2, a),
            (template, 2, b),
        ]
    )
    for i in range(0, 4, 2):
        s_a, s_b = out[i], out[i + 1]
        assert s_a.template is s_b.template
        assert s_a.init_seed == s_b.init_seed
        assert s_a.policy is not s_b.policy


def test_make_stores_kway_three_policies_share_template_and_seed_step0_identity() -> None:
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
        stores=[],
        nodes=node_instances,
        edges=edges,
        n_steps=1,
        start_date=__import__("datetime").datetime(2024, 1, 1),
        world_seed=42,
    )
    # Build the world and inspect initial state: all three shops should be identical.
    from src.sim.runner import build_world as _build_world
    sim = _build_world(scenario)
    shop_a = sim.nodes["shopA"]
    shop_b = sim.nodes["shopB"]
    shop_c = sim.nodes["shopC"]
    # All three shops started with the same init_seed — same inventory and cash.
    assert shop_a.inventory == shop_b.inventory == shop_c.inventory
    assert shop_a.cash == shop_b.cash == shop_c.cash


# ----------------------------------------------------------- JSON round-trip


def test_scenario_round_trip_preserves_equality() -> None:
    scenario = _scenario()
    restored = Scenario.from_json(scenario.to_json())
    assert restored == scenario


def test_scenario_round_trip_preserves_distribution_inside_dict() -> None:
    scenario = _scenario()
    restored = Scenario.from_json(scenario.to_json())
    assert restored.market.stage_multipliers["growth"] == Uniform(1.2, 2.0)
    assert restored.market.stage_multipliers["maturity"] == 1.0


def test_scenario_round_trip_preserves_tuple_typed_ranges() -> None:
    scenario = _scenario()
    restored = Scenario.from_json(scenario.to_json())
    assert restored.market.cross_factor_range == (0.3, 1.6)


def test_scenario_round_trip_preserves_catalog_related_products_as_tuples() -> None:
    scenario = _scenario()
    restored = Scenario.from_json(scenario.to_json())
    rels = restored.catalog[1].related_products
    assert all(isinstance(p, tuple) for p in rels)
    assert restored.catalog == scenario.catalog


def test_scenario_round_trip_preserves_distributions_in_disruption() -> None:
    scenario = _scenario()
    restored = Scenario.from_json(scenario.to_json())
    assert restored.disruption.severity == Constant(0.01)
    assert restored.disruption.duration == Constant(5)


def test_scenario_to_json_omits_policy_field() -> None:
    scenario = _scenario(
        stores=[StoreInstance(template=_template(), init_seed=42, policy=StubPolicy("p"))]
    )
    raw = json.loads(scenario.to_json())
    store_dict = raw["stores"][0]
    assert "policy" not in store_dict
    assert set(store_dict.keys()) == {"template", "init_seed"}


def test_scenario_from_json_assigns_none_policy() -> None:
    scenario = _scenario(
        stores=[StoreInstance(template=_template(), init_seed=42, policy=StubPolicy("p"))]
    )
    restored = Scenario.from_json(scenario.to_json())
    assert restored.stores[0].policy is None


def test_scenario_from_json_rejects_malformed_json() -> None:
    with pytest.raises(ValueError):
        Scenario.from_json("not valid json {{")


def test_scenario_from_json_rejects_missing_keys() -> None:
    with pytest.raises(ValueError):
        Scenario.from_json(json.dumps({"catalog": []}))


def test_scenario_from_json_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError):
        Scenario.from_json("[1, 2, 3]")


# -------------------------------------------- per-stage transition probs (issue 02)


def test_scenario_round_trip_preserves_default_stage_change_probs_dict() -> None:
    """The per-stage default dict survives a JSON round-trip."""
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="introduction",
        default_stage_change_probs={
            "introduction": 0.02,
            "growth": 0.005,
            "maturity": Uniform(0.0, 0.01),
            "decline": 0.005,
            "dead": 0.003,
        },
    )
    scenario = _scenario()
    scenario.item_lifecycle = lifecycle
    restored = Scenario.from_json(scenario.to_json())
    probs = restored.item_lifecycle.default_stage_change_probs
    assert probs["introduction"] == 0.02
    assert probs["dead"] == 0.003
    assert probs["maturity"] == Uniform(0.0, 0.01)


def test_scenario_round_trip_preserves_default_freshness_fields() -> None:
    """``default_freshness_alpha`` / ``default_freshness_decay`` round-trip
    through JSON (issue 03), including ``Distribution`` values."""
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="introduction",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_freshness_alpha=Uniform(0.1, 0.4),
        default_freshness_decay=30.0,
    )
    scenario = _scenario()
    scenario.item_lifecycle = lifecycle
    restored = Scenario.from_json(scenario.to_json())
    assert restored.item_lifecycle.default_freshness_alpha == Uniform(0.1, 0.4)
    assert restored.item_lifecycle.default_freshness_decay == 30.0


def test_scenario_round_trip_preserves_per_ware_stage_change_probs_override() -> None:
    """A per-``Ware`` override dict (with optional Distribution entries) round-trips."""
    cat = _catalog()
    overridden = cat[0]._replace(
        init_stage="growth",
        stage_change_probs={"introduction": 0.05, "growth": Uniform(0.0, 0.1)},
    )
    cat = [overridden, *cat[1:]]
    scenario = _scenario()
    scenario.catalog = cat
    restored = Scenario.from_json(scenario.to_json())
    w = restored.catalog[0]
    assert w.init_stage == "growth"
    assert w.stage_change_probs == {
        "introduction": 0.05,
        "growth": Uniform(0.0, 0.1),
    }
    # Wares without overrides keep their ``None`` defaults.
    assert restored.catalog[1].init_stage is None
    assert restored.catalog[1].stage_change_probs is None


# -------------------------------------------- per-Ware freshness overrides (issue 04)


def test_scenario_round_trip_preserves_init_active_products() -> None:
    """``StoreTemplate.init_active_products`` (issue 06) round-trips through JSON.

    Pins both branches: an explicit list survives, and ``None`` (default)
    on a sibling template is preserved as ``None``.
    """
    template_explicit = StoreTemplate(
        id="explicit",
        region="US",
        capacity=1000,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
        init_active_products=["P0001"],
    )
    template_default = _template()
    scenario = _scenario(
        stores=[
            StoreInstance(template=template_explicit, init_seed=1, policy=None),
            StoreInstance(template=template_default, init_seed=2, policy=None),
        ]
    )
    restored = Scenario.from_json(scenario.to_json())
    assert restored.stores[0].template.init_active_products == ["P0001"]
    assert restored.stores[1].template.init_active_products is None


def test_scenario_round_trip_preserves_init_freshness() -> None:
    """``StoreTemplate.init_freshness`` (issue 07) round-trips through
    JSON. Pins both branches: an explicit ``"fresh"`` survives, and a
    sibling default-``"baseline"`` template loads back with the same
    default."""
    template_fresh = StoreTemplate(
        id="fresh",
        region="US",
        capacity=1000,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
        init_freshness="fresh",
    )
    template_default = _template()
    scenario = _scenario(
        stores=[
            StoreInstance(template=template_fresh, init_seed=1, policy=None),
            StoreInstance(template=template_default, init_seed=2, policy=None),
        ]
    )
    restored = Scenario.from_json(scenario.to_json())
    assert restored.stores[0].template.init_freshness == "fresh"
    assert restored.stores[1].template.init_freshness == "baseline"


def test_scenario_from_json_rejects_invalid_init_freshness() -> None:
    """An unknown ``init_freshness`` value fails loudly at ``from_dict``."""
    template = _template()
    scenario = _scenario(
        stores=[StoreInstance(template=template, init_seed=1, policy=None)]
    )
    payload = json.loads(scenario.to_json())
    payload["stores"][0]["template"]["init_freshness"] = "bogus"
    with pytest.raises(ValueError, match="init_freshness"):
        Scenario.from_dict(payload)


def test_scenario_round_trip_preserves_per_ware_freshness_overrides() -> None:
    """Per-``Ware`` ``freshness_alpha`` / ``freshness_decay`` overrides
    round-trip with numerical equality, including ``Distribution`` values
    and the staple-style ``alpha = 0`` override."""
    cat = _catalog()
    fashion = cat[0]._replace(freshness_alpha=0.4, freshness_decay=Uniform(20.0, 40.0))
    staple = cat[1]._replace(freshness_alpha=0.0, freshness_decay=1.0)
    cat = [fashion, staple]
    scenario = _scenario()
    scenario.catalog = cat
    restored = Scenario.from_json(scenario.to_json())
    assert restored.catalog[0].freshness_alpha == 0.4
    assert restored.catalog[0].freshness_decay == Uniform(20.0, 40.0)
    assert restored.catalog[1].freshness_alpha == 0.0
    assert restored.catalog[1].freshness_decay == 1.0


# -------------------------------------- init_stock_share round-trip (issue 08)


def test_scenario_round_trip_preserves_default_init_stock_share() -> None:
    """``ItemLifecycleParams.default_init_stock_share`` (issue 08)
    round-trips through JSON, including ``Distribution`` values."""
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="introduction",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_init_stock_share=Uniform(0.5, 2.0),
    )
    scenario = _scenario()
    scenario.item_lifecycle = lifecycle
    restored = Scenario.from_json(scenario.to_json())
    assert restored.item_lifecycle.default_init_stock_share == Uniform(0.5, 2.0)


def test_scenario_round_trip_preserves_per_ware_init_stock_share() -> None:
    """Per-``Ware`` ``init_stock_share`` overrides (issue 08) round-trip
    through JSON. Pin both branches: a scalar override survives, and a
    sibling ``None`` Ware preserves the implicit-fallback default."""
    cat = _catalog()
    cat[0] = cat[0]._replace(init_stock_share=2.0)
    scenario = _scenario()
    scenario.catalog = cat
    restored = Scenario.from_json(scenario.to_json())
    assert restored.catalog[0].init_stock_share == 2.0
    assert restored.catalog[1].init_stock_share is None


# ---------------------------------------------------------------- load_scenario_from_path


_HOMOGENEOUS = (
    Path(__file__).parent.parent.parent / "scenarios" / "example_homogeneous.py"
)


def test_load_scenario_from_path_returns_scenario_with_catalog_and_nodes() -> None:
    """Phase-4: example_homogeneous uses graph-mode nodes, not stores."""
    sc = load_scenario_from_path(_HOMOGENEOUS)
    assert isinstance(sc, Scenario)
    assert len(sc.catalog) > 0
    # Graph-mode scenario: nodes are populated, stores are empty.
    assert sc.is_graph
    assert len(sc.nodes) > 0


def test_load_scenario_from_path_preserves_live_policies() -> None:
    sc = load_scenario_from_path(_HOMOGENEOUS)
    # Graph-mode: check node policies instead of store policies.
    assert any(ni.policy is not None for ni in sc.nodes)


def test_load_scenario_from_path_accepts_string_path() -> None:
    sc = load_scenario_from_path(str(_HOMOGENEOUS))
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
