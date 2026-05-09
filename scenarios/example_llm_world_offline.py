"""Example: WorldBuilder wired to a canned (offline) LLM client.

Demonstrates the five-call pipeline without needing ``OPENAI_API_KEY``.
``CannedClient`` implements the ``LLMClient`` Protocol and pops a
pre-built Pydantic payload per ``structured_completion`` call. This is
the same test seam used by ``tests/llm/test_world_builder.py``.

Use it to:

- inspect what each stage's payload looks like end-to-end
- run a full simulation deterministically without billing the OpenAI API
- prototype prompt/schema changes against fixed inputs

Run it directly::

    uv run python scenarios/example_llm_world_offline.py

or via the CLI shim::

    uv run python main.py scenarios/example_llm_world_offline.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.schemas import (
    Catalog,
    CatalogItem,
    Correlations,
    FreshnessSet,
    InitFreshness,
    ItemFreshness,
    ItemRelations,
    MarketDomain,
    RelatedRef,
    Seasonality,
    StoreTemplateList,
    StoreTemplateSpec,
    Taxonomy,
    TaxonomyCategory,
)
from src.llm.world_builder import WorldBuilder
from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant
from src.sim.policy import BaselinePolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    Scenario,
    make_stores,
)


T = TypeVar("T", bound=BaseModel)


class CannedClient:
    """Implements ``LLMClient`` Protocol with a fixed response queue.

    Order matches ``WorldBuilder.build``: market → taxonomy → catalog →
    correlations → freshness → templates. Each ``structured_completion``
    call pops the head.
    """

    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)

    def structured_completion(
        self, *, system: str, user: str, schema: type[T]
    ) -> T:
        if not self._responses:
            raise RuntimeError("CannedClient: out of canned responses")
        payload = self._responses.pop(0)
        if not isinstance(payload, schema):
            raise RuntimeError(
                f"CannedClient: expected {schema.__name__}, got "
                f"{type(payload).__name__}"
            )
        return payload  # type: ignore[return-value]


_MARKET = MarketDomain(
    cycle_len=365,
    peak_factor=1.3,
    off_factor=0.6,
    init_demand=120.0,
    init_supply=120.0,
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
        CatalogItem(
            name="Linen Shirt",
            category="Apparel",
            base_price=60.0,
            unit_cost=24.0,
            seasonality=Seasonality.SPRING_SUMMER,
        ),
        CatalogItem(
            name="Wool Coat",
            category="Apparel",
            base_price=220.0,
            unit_cost=110.0,
            seasonality=Seasonality.FALL_WINTER,
        ),
        CatalogItem(
            name="Denim Jeans",
            category="Apparel",
            base_price=80.0,
            unit_cost=32.0,
            seasonality=Seasonality.ALL_SEASON,
        ),
        CatalogItem(
            name="Leather Belt",
            category="Accessories",
            base_price=45.0,
            unit_cost=18.0,
            seasonality=Seasonality.ALL_SEASON,
        ),
        CatalogItem(
            name="Silk Scarf",
            category="Accessories",
            base_price=70.0,
            unit_cost=28.0,
            seasonality=Seasonality.FALL_WINTER,
        ),
        CatalogItem(
            name="Running Sneakers",
            category="Footwear",
            base_price=110.0,
            unit_cost=50.0,
            seasonality=Seasonality.ALL_SEASON,
        ),
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


_TEMPLATES = StoreTemplateList(
    templates=[
        StoreTemplateSpec(
            id="flagship",
            region="US",
            capacity=500.0,
            init_balance=20_000.0,
            init_stock_pct=0.5,
            delivery_lag=1.0,
            holding_rate=0.01,
            order_fee=20.0,
            init_active_count=6,
            init_freshness=InitFreshness.BASELINE,
        ),
        StoreTemplateSpec(
            id="standard",
            region="US",
            capacity=200.0,
            init_balance=10_000.0,
            init_stock_pct=0.4,
            delivery_lag=2.0,
            holding_rate=0.005,
            order_fee=10.0,
            init_active_count=4,
            init_freshness=InitFreshness.BASELINE,
        ),
    ]
)


_client = CannedClient(
    [_MARKET, _TAXONOMY, _CATALOG, _CORRELATIONS, _FRESHNESS, _TEMPLATES]
)
_builder = WorldBuilder(archetype="fashion_retail", client=_client)
_world = _builder.build(n_items=len(_CATALOG.items))


_template = _world.store_templates["standard"]


_policy = BaselinePolicy(
    policy_seed=1000,
    min_qty=1,
    init_qty_factor=0.3,
    min_promo_len=3,
    max_promo_len=5,
    promo_cd_len=5,
    review_interval=10,
    promo_threshold=0.4,
    target_active_count=4,
    slow_sales_limit=2,
    history_window=4,
    max_history=50,
    promo_discount=0.7,
)


scenario = Scenario.from_world(
    _world,
    disruption=DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=_world.market.regions,
        severity=Constant(1.0),
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
    stores=make_stores(
        [
            (_template, 1, _policy),
            (_template, 2, _policy),
            (_template, 3, _policy),
        ]
    ),
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "example_llm_world_offline"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Catalog ({len(_world.catalog)} items):")
    for w in _world.catalog:
        print(f"  {w.product_id}  {w.category:<14}  {w.name}")
    print(f"Templates: {list(_world.store_templates.keys())}")
    print(f"Regions: {_world.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
