"""Issue 06: DataExporter parquet parity regression test.

Loads committed fixtures (products.parquet / stores.parquet) produced by the
pre-refactor DataExporter against the deterministic offline CannedClient
scenario, then re-runs the same scenario through the refactored DataExporter
and asserts byte-identical output via pd.testing.assert_frame_equal.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pytest
from pydantic import BaseModel

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _build_offline_scenario():
    """Reconstruct the offline LLM scenario (previously in scenarios/example_llm_world_offline.py)."""
    from src.llm.schemas import (
        Catalog,
        CatalogItem,
        Correlations,
        FreshnessSet,
        ItemFreshness,
        ItemRelations,
        MarketDomain,
        RelatedRef,
        Seasonality,
        Taxonomy,
        TaxonomyCategory,
    )
    from src.llm.world_builder import WorldBuilder
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.policy import (
        DefaultDemandSinkPolicy,
        OrderUpToPolicy,
        StaticFactoryPolicy,
    )
    from src.sim.scenario import (
        DisruptionParams,
        ItemLifecycleParams,
        NodeInstance,
        Scenario,
    )

    T = TypeVar("T", bound=BaseModel)

    class CannedClient:
        def __init__(self, responses):
            self._responses = list(responses)

        def structured_completion(self, *, system, user, schema):
            if not self._responses:
                raise RuntimeError("CannedClient: out of canned responses")
            payload = self._responses.pop(0)
            if not isinstance(payload, schema):
                raise RuntimeError(
                    f"CannedClient: expected {schema.__name__}, got "
                    f"{type(payload).__name__}"
                )
            return payload

    _MARKET = MarketDomain(
        cycle_len=365,
        peak_factor=1.3,
        off_factor=0.6,
        season_months=[
            {"name": "spring/summer", "months": [3, 4, 5, 6, 7, 8]},
            {"name": "fall/winter", "months": [9, 10, 11, 12, 1, 2]},
            {"name": "all_season", "months": list(range(1, 13))},
        ],
        regions=["US"],
        price_elasticity=-1.4,
    )
    _TAXONOMY = Taxonomy(
        archetype="fashion_retail",
        categories=[
            TaxonomyCategory(name="Apparel", description="", target_share=0.5),
            TaxonomyCategory(name="Accessories", description="", target_share=0.3),
            TaxonomyCategory(name="Footwear", description="", target_share=0.2),
        ],
    )
    _CATALOG = Catalog(
        items=[
            CatalogItem(name="Linen Shirt", category="Apparel", base_price=60.0,
                        unit_cost=24.0, seasonality=Seasonality.SPRING_SUMMER),
            CatalogItem(name="Wool Coat", category="Apparel", base_price=220.0,
                        unit_cost=110.0, seasonality=Seasonality.FALL_WINTER),
            CatalogItem(name="Denim Jeans", category="Apparel", base_price=80.0,
                        unit_cost=32.0, seasonality=Seasonality.ALL_SEASON),
            CatalogItem(name="Leather Belt", category="Accessories", base_price=45.0,
                        unit_cost=18.0, seasonality=Seasonality.ALL_SEASON),
            CatalogItem(name="Silk Scarf", category="Accessories", base_price=70.0,
                        unit_cost=28.0, seasonality=Seasonality.FALL_WINTER),
            CatalogItem(name="Running Sneakers", category="Footwear", base_price=110.0,
                        unit_cost=50.0, seasonality=Seasonality.ALL_SEASON),
        ]
    )
    _CORRELATIONS = Correlations(
        items=[
            ItemRelations(name="Linen Shirt", related=[]),
            ItemRelations(name="Wool Coat", related=[]),
            ItemRelations(
                name="Denim Jeans",
                related=[RelatedRef(name="Leather Belt", correlation=0.6)],
            ),
            ItemRelations(
                name="Leather Belt",
                related=[RelatedRef(name="Denim Jeans", correlation=0.6)],
            ),
            ItemRelations(name="Silk Scarf", related=[]),
            ItemRelations(name="Running Sneakers", related=[]),
        ]
    )
    _FRESHNESS = FreshnessSet(
        items=[
            ItemFreshness(name="Linen Shirt", alpha=0.2, decay=30.0),
            ItemFreshness(name="Wool Coat", alpha=0.3, decay=40.0),
            ItemFreshness(name="Denim Jeans", alpha=0.0, decay=20.0),
            ItemFreshness(name="Leather Belt", alpha=0.0, decay=20.0),
            ItemFreshness(name="Silk Scarf", alpha=0.25, decay=35.0),
            ItemFreshness(name="Running Sneakers", alpha=0.15, decay=25.0),
        ]
    )

    client = CannedClient([_MARKET, _TAXONOMY, _CATALOG, _CORRELATIONS, _FRESHNESS])
    builder = WorldBuilder(archetype="fashion_retail", client=client)
    market_params = builder.build_market_domain_params()
    wares = builder.sample_catalog(n=len(_CATALOG.items))

    pids = [w.product_id for w in wares]
    pid_set = set(pids)

    factory = FactoryNode(
        id="factory-1", region="US", init_seed=1,
        produces_product_id=pids[0], unit_cost=wares[0].unit_cost,
        capacity_per_tick=50, inventory=200, list_price=wares[0].unit_cost, cash=0.0,
    )
    factory.policy = StaticFactoryPolicy(
        capacity_per_tick=50, unit_cost=wares[0].unit_cost, policy_seed=10,
    )

    shop = IntermediateNode(
        id="shop-1", region="US", init_seed=2, carried_products=pid_set,
        capacity=500, tags=["shop"], inventory={pid: 20 for pid in pids}, pending={},
        list_prices={w.product_id: w.base_price for w in wares},
        min_order_imposed={pid: 0 for pid in pids}, cash=20_000.0,
    )
    shop.policy = OrderUpToPolicy(
        cover_horizon_ticks=14, safety_lead_pct_of_lag=1/3, delivery_lag=2,
        unit_cost=wares[0].unit_cost, list_price_out=wares[0].base_price, policy_seed=20,
    )

    sink = DemandSinkNode(
        id="sink-1", region="US", init_seed=3, product_id=pids[0],
        demand_dist=Constant(5), income_rate=300.0, cash=5_000.0,
    )
    sink.policy = DefaultDemandSinkPolicy(policy_seed=30)

    nodes = [
        NodeInstance(node=factory, init_seed=1),
        NodeInstance(node=shop, init_seed=2),
        NodeInstance(node=sink, init_seed=3),
    ]
    edges = [
        EdgeSpec(supplier_id="factory-1", buyer_id="shop-1", default_lead_time=2),
        EdgeSpec(supplier_id="shop-1", buyer_id="sink-1", default_lead_time=1),
    ]

    return Scenario(
        catalog=wares,
        market=market_params,
        disruption=DisruptionParams(
            event_prob=0.05,
            types=["natural_disaster", "economic_crisis"],
            regions=market_params.regions,
            severity=Constant(0.01),
            duration=Constant(3),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=["introduction", "growth", "maturity", "decline", "dead"],
            init_stage="maturity",
            default_stage_change_probs={
                s: 0.0
                for s in ["introduction", "growth", "maturity", "decline", "dead"]
            },
        ),
        nodes=nodes,
        edges=edges,
        n_steps=50,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


@pytest.fixture(scope="module")
def _offline_run():
    """Run the offline scenario through DataExporter once per module."""
    from src.sim.data_exporter import DataExporter
    from src.sim.runner import Runner

    scenario = _build_offline_scenario()
    run_log = Runner(scenario).run()
    tmp = tempfile.mkdtemp()
    DataExporter(scenario, run_log).export_all(tmp)
    return Path(tmp) / "data"


def test_products_parquet_matches_fixture(_offline_run):
    actual = pd.read_parquet(_offline_run / "products.parquet")
    fixture = pd.read_parquet(FIXTURES / "products.parquet")
    pd.testing.assert_frame_equal(actual, fixture, check_dtype=True, check_exact=True)


def test_stores_parquet_matches_fixture(_offline_run):
    actual = pd.read_parquet(_offline_run / "stores.parquet")
    fixture = pd.read_parquet(FIXTURES / "stores.parquet")
    pd.testing.assert_frame_equal(actual, fixture, check_dtype=True, check_exact=True)
