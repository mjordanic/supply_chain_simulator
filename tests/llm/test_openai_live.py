"""Live OpenAI integration tests for ``WorldBuilder``.

Vertical end-to-end coverage: every schema is round-tripped through a real
``OpenAIClient`` against a cheap production model. Skipped when no
``OPENAI_API_KEY`` is available so CI without secrets stays green.

Loads ``.env`` lazily via ``python-dotenv`` so a developer with the key in
``.env`` can ``uv run pytest tests/llm/test_openai_live.py`` without exporting
it manually.

Cost control:
- Uses ``OPENAI_TEST_MODEL`` (default ``gpt-4o-mini``) — cheapest tier with
  reliable structured-output support. Override with e.g. ``gpt-5-nano`` if
  available in your account.
- Catalogs are kept tiny (``n_items=3``).
- One full ``builder.build()`` plus per-schema spot tests; total of a few
  thousand input/output tokens per run.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import (
    catalog_prompt,
    correlations_prompt,
    freshness_prompt,
    market_domain_prompt,
    store_templates_prompt,
    taxonomy_prompt,
)
from src.llm.schemas import (
    Catalog,
    Correlations,
    FreshnessSet,
    InitFreshness,
    MarketDomain,
    Seasonality,
    StoreTemplateList,
    Taxonomy,
)
from src.llm.world_builder import WorldBuilder, allocate_skeletons
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
)
from src.sim.distributions import Constant


load_dotenv()


_ALLOWED_SEASON_VALUES = {s.value for s in Seasonality}
_ALLOWED_FRESHNESS_VALUES = {f.value for f in InitFreshness}


pytestmark = [
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set; skipping live OpenAI tests",
    ),
    pytest.mark.live,
]


@pytest.fixture(scope="module")
def client() -> OpenAIClient:
    model = os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini")
    return OpenAIClient(model=model)


def test_market_domain_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    """Regression test for the season_months bug: if MarketDomain ever
    drifts back to a free-form ``dict[str, list[int]]`` shape, this call
    fails with a 400 from OpenAI's strict-schema enforcement."""
    system, user = market_domain_prompt("luxury")
    domain = client.structured_completion(
        system=system, user=user, schema=MarketDomain
    )
    assert isinstance(domain, MarketDomain)
    assert domain.price_elasticity < 0
    assert domain.cycle_len > 0
    assert len(domain.season_months) >= 1
    # Validator: season names unique
    names = [w.name for w in domain.season_months]
    assert len(names) == len(set(names))
    # Each month is 1..12 (validator already enforces; spot-check anyway)
    for window in domain.season_months:
        assert all(1 <= m <= 12 for m in window.months)
    # Adapter shape used by WorldBuilder.build_market_domain_params:
    assert isinstance(domain.season_months_dict(), dict)


def test_taxonomy_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    system, user = taxonomy_prompt("athletic-wear")
    taxonomy = client.structured_completion(
        system=system, user=user, schema=Taxonomy
    )
    assert isinstance(taxonomy, Taxonomy)
    assert taxonomy.archetype  # any non-empty string
    assert 3 <= len(taxonomy.categories) <= 8
    assert all(0 < c.target_share <= 1 for c in taxonomy.categories)


def test_store_templates_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    system, user = store_templates_prompt("luxury", regions=["US", "EU"])
    payload = client.structured_completion(
        system=system, user=user, schema=StoreTemplateList
    )
    assert isinstance(payload, StoreTemplateList)
    assert 1 <= len(payload.templates) <= 4
    ids = [t.id for t in payload.templates]
    assert len(ids) == len(set(ids)), "store template ids must be unique"
    for t in payload.templates:
        assert t.region in {"US", "EU"}
        assert t.init_freshness.value in _ALLOWED_FRESHNESS_VALUES
        assert t.init_active_count >= 0


def test_catalog_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    """Spot-check: feed a pre-built taxonomy and skeleton list, confirm
    OpenAI returns a parseable Catalog. The catalog stage no longer
    authors cross-product correlations, so the prompt is small enough
    that even cheap models stay reliable."""
    taxonomy = Taxonomy.model_validate(
        {
            "archetype": "luxury",
            "categories": [
                {"name": "Apparel", "description": "garments", "target_share": 0.6},
                {
                    "name": "Accessories",
                    "description": "small leather goods",
                    "target_share": 0.4,
                },
            ],
        }
    )
    skeletons = allocate_skeletons(8, taxonomy)
    system, user = catalog_prompt(
        "luxury", taxonomy.model_dump_json(), skeletons
    )
    catalog = client.structured_completion(
        system=system, user=user, schema=Catalog
    )
    assert isinstance(catalog, Catalog)
    assert len(catalog.items) == 8
    for it in catalog.items:
        assert it.base_price > it.unit_cost
        assert it.unit_cost >= 0
        assert it.seasonality.value in _ALLOWED_SEASON_VALUES


def test_correlations_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    """Spot-check the dedicated correlations stage: given an explicit
    item list, the LLM should produce one ``ItemRelations`` per item and
    only reference names from that list. Dangling refs are sanitised by
    ``WorldBuilder``, so this test only asserts schema-level correctness;
    it does not require the LLM to never hallucinate."""
    items = [
        ("Cashmere Sweater", "Apparel"),
        ("Leather Wallet", "Accessories"),
        ("Silk Scarf", "Accessories"),
        ("Wool Coat", "Apparel"),
    ]
    candidate_names = [name for name, _ in items]
    system, user = correlations_prompt("luxury", items, candidate_names)
    payload = client.structured_completion(
        system=system, user=user, schema=Correlations
    )
    assert isinstance(payload, Correlations)
    # The schema requires at least one entry; we also expect roughly one
    # per input item, but the LLM occasionally skips items — we don't
    # bind that strictly.
    assert len(payload.items) >= 1
    for entry in payload.items:
        for ref in entry.related:
            assert 0.0 <= ref.correlation <= 1.0


def test_freshness_schema_accepted_by_openai_strict_mode(
    client: OpenAIClient,
) -> None:
    """Spot-check the dedicated freshness stage: given an explicit
    item list, the LLM should produce one ``ItemFreshness`` per item
    with non-negative ``alpha`` and strictly-positive ``decay``. Mixed
    staple/trend items exercise both ends of the prompt heuristic."""
    items = [
        ("Cashmere Sweater", "Apparel"),
        ("Hand Soap", "Household"),
        ("Statement Necklace", "Accessories"),
        ("All-Purpose Flour", "Pantry"),
    ]
    system, user = freshness_prompt("luxury", items)
    payload = client.structured_completion(
        system=system, user=user, schema=FreshnessSet
    )
    assert isinstance(payload, FreshnessSet)
    assert len(payload.items) >= 1
    for entry in payload.items:
        assert entry.alpha >= 0.0
        assert entry.decay > 0.0


def test_full_builder_pipeline_against_live_openai(client: OpenAIClient) -> None:
    """End-to-end smoke: ``WorldBuilder`` exercises all LLM calls
    (market, taxonomy, catalog, correlations, freshness) plus the
    deterministic skeleton sampler. Validates the assembled catalog
    and market are internally consistent."""
    builder = WorldBuilder("boutique-grocer", client, max_retries=3)
    market = builder.build_market_domain_params()
    catalog = builder.sample_catalog(n=8)

    assert isinstance(market, MarketParams)

    # Catalog is the right size and ids are stable.
    assert len(catalog) == 8
    assert [w.product_id for w in catalog] == [
        f"P{i:04d}" for i in range(8)
    ]

    # Market regions are populated.
    assert len(market.regions) >= 1

    # Correlations stage: every related-product reference resolves to a
    # name in the catalog (sanitisation guarantees this) and is not a
    # self-reference. Passing this on a live cheap model is the structural
    # invariant the simplification was meant to deliver.
    catalog_names = {w.name for w in world.catalog}
    for w in world.catalog:
        for ref_name, corr in w.related_products:
            assert ref_name in catalog_names, (
                f"{w.name!r} → {ref_name!r} dangling after sanitisation"
            )
            assert ref_name != w.name
            assert 0.0 <= corr <= 1.0

    # Freshness stage: the LLM should author at least one item even if
    # some get pruned. Tighter "all items" would risk flake on cheap
    # models even after sanitisation; one-or-more is the right floor.
    assert any(w.freshness_alpha is not None for w in world.catalog)
    for w in world.catalog:
        if w.freshness_alpha is not None:
            assert w.freshness_alpha >= 0.0
            assert w.freshness_decay is not None
            assert w.freshness_decay > 0.0


def test_full_builder_world_runs_through_runner(client: OpenAIClient) -> None:
    """Drive the produced world through the Runner for a few steps and
    assert a non-empty run log. Catches drift in the wire-up between
    WorldBuilder output and Scenario consumption (e.g., template values
    survive Store construction, init_active_count drives the random
    sampler in init_store_state, etc.)."""
    n_items = 5
    builder = WorldBuilder("boutique-grocer", client, max_retries=3)
    world = builder.build(n_items=n_items)

    canonical_stages = ["introduction", "growth", "maturity", "decline", "dead"]
    template = next(iter(world.store_templates.values()))
    scenario = Scenario(
        catalog=world.catalog,
        market=world.market,
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=world.market.regions,
            severity=Constant(1.0),
            duration=Constant(1),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=canonical_stages,
            init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in canonical_stages},
        ),
        stores=[StoreInstance(template=template, init_seed=1, policy=None)],
        n_steps=5,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    run_log = Runner(scenario).run()
    assert run_log, "Runner produced an empty run log"
