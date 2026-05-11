"""``WorldBuilder`` end-to-end against a mocked ``LLMClient``.

Covers:
- deterministic skeleton allocator (category mix preserved)
- end-to-end builder against canned mock responses (no retries)
- retry path: validation failure on attempt 1 routes the error into the
  attempt 2 prompt
- ``OpenAIClient`` conforms to the ``LLMClient`` Protocol shape
- correlations stage: dangling / self / duplicate references are
  sanitised at the boundary instead of triggering retries

No live API calls are made — every test uses a ``MockClient`` whose
``structured_completion`` pops pre-canned responses (or raises
``ValidationError``) per call.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.llm.openai_client import LLMClient, OpenAIClient
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
from src.llm.world_builder import World, WorldBuilder, allocate_skeletons
from src.sim.scenario import MarketParams, StoreTemplate


class MockClient:
    """Test double for ``LLMClient``.

    ``responses`` is a list of either ``BaseModel`` instances (returned
    in order on each call) or ``Exception`` instances (raised in order).
    Records every call so tests can assert on the prompts the builder
    constructed.
    """

    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def structured_completion(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self._responses:
            raise AssertionError(
                f"MockClient: out of canned responses (call #{len(self.calls)})"
            )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _validation_error_for(schema: type[BaseModel]) -> ValidationError:
    """Construct a real ``ValidationError`` for ``schema`` by parsing junk."""
    try:
        schema.model_validate({})
    except ValidationError as e:
        return e
    raise AssertionError("expected schema.model_validate({}) to fail")


def _market_response(regions=("US",)) -> MarketDomain:
    return MarketDomain(
        cycle_len=365,
        peak_factor=1.2,
        off_factor=0.7,
        season_months=[{"name": "all_season", "months": list(range(1, 13))}],
        regions=list(regions),
        price_elasticity=-1.5,
    )


def _templates_response() -> StoreTemplateList:
    return StoreTemplateList(
        templates=[
            StoreTemplateSpec(
                id="flagship",
                region="US",
                capacity=500.0,
                init_balance=20000.0,
                init_stock_pct=0.5,
                delivery_lag=1.0,
                holding_rate=0.01,
                order_fee=20.0,
                init_active_count=3,
                init_freshness=InitFreshness.FRESH,
            ),
            StoreTemplateSpec(
                id="standard",
                region="US",
                capacity=200.0,
                init_balance=10000.0,
                init_stock_pct=0.4,
                delivery_lag=2.0,
                holding_rate=0.005,
                order_fee=10.0,
                init_active_count=2,
                init_freshness=InitFreshness.BASELINE,
            ),
        ]
    )


def _taxonomy_response() -> Taxonomy:
    return Taxonomy(
        archetype="luxury",
        categories=[
            TaxonomyCategory(name="Apparel", description="", target_share=0.5),
            TaxonomyCategory(name="Accessories", description="", target_share=0.3),
            TaxonomyCategory(name="Footwear", description="", target_share=0.2),
        ],
    )


def _catalog_response(skeletons: list[str]) -> Catalog:
    items = []
    for i, cat in enumerate(skeletons):
        items.append(
            CatalogItem(
                name=f"Item-{i:03d}",
                category=cat,
                base_price=20.0 + i,
                unit_cost=10.0 + i,
                seasonality=Seasonality.ALL_SEASON,
            )
        )
    return Catalog(items=items)


def _empty_correlations(catalog: Catalog) -> Correlations:
    """Default correlations payload: every item has an empty related list."""
    return Correlations(
        items=[ItemRelations(name=it.name, related=[]) for it in catalog.items]
    )


def _full_freshness(
    catalog: Catalog, alpha: float = 0.2, decay: float = 30.0
) -> FreshnessSet:
    """Default freshness payload: every catalog item gets the same curve."""
    return FreshnessSet(
        items=[
            ItemFreshness(name=it.name, alpha=alpha, decay=decay)
            for it in catalog.items
        ]
    )


def test_openai_client_conforms_to_protocol() -> None:
    # Static-shape conformance: the class implements the one method the
    # Protocol requires. We don't instantiate (would need an API key).
    assert hasattr(OpenAIClient, "structured_completion")
    # runtime_checkable Protocol — class itself isn't an instance, but a
    # MockClient stand-in is.
    mock = MockClient([])
    assert isinstance(mock, LLMClient)


def test_allocate_skeletons_respects_category_mix() -> None:
    taxonomy = _taxonomy_response()  # 0.5 / 0.3 / 0.2
    skeletons = allocate_skeletons(10, taxonomy)
    assert len(skeletons) == 10
    counts = Counter(skeletons)
    assert counts["Apparel"] == 5
    assert counts["Accessories"] == 3
    assert counts["Footwear"] == 2


def test_allocate_skeletons_handles_rounding_remainder() -> None:
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(7, taxonomy)
    assert len(skeletons) == 7
    counts = Counter(skeletons)
    # Every category gets at least one slot when n >= len(categories).
    assert all(counts[c.name] >= 1 for c in taxonomy.categories)


def test_allocate_skeletons_n_smaller_than_categories() -> None:
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(2, taxonomy)
    # The two highest-share categories absorb the slots.
    assert sorted(skeletons) == sorted(["Apparel", "Accessories"])


def test_allocate_skeletons_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        allocate_skeletons(0, _taxonomy_response())
    with pytest.raises(ValueError):
        allocate_skeletons(-3, _taxonomy_response())


def test_build_taxonomy_caches_result() -> None:
    taxonomy = _taxonomy_response()
    client = MockClient([taxonomy])
    builder = WorldBuilder("luxury", client)
    a = builder.build_taxonomy()
    b = builder.build_taxonomy()
    assert a is b
    assert len(client.calls) == 1
    assert client.calls[0]["schema"] is Taxonomy


def test_sample_catalog_four_stage_flow() -> None:
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(6, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    out = builder.sample_catalog(6)

    assert len(out) == 6
    assert all(w.product_id == f"P{i:04d}" for i, w in enumerate(out))
    # Four LLM calls in order: taxonomy, catalog, correlations, freshness.
    assert client.calls[0]["schema"] is Taxonomy
    assert client.calls[1]["schema"] is Catalog
    assert client.calls[2]["schema"] is Correlations
    assert client.calls[3]["schema"] is FreshnessSet
    # The catalog call sees the taxonomy in its prompt.
    assert "Apparel" in client.calls[1]["user"]
    # The correlations call sees the catalog item names verbatim.
    assert "Item-000" in client.calls[2]["user"]
    # The freshness call sees the catalog item names verbatim.
    assert "Item-000" in client.calls[3]["user"]
    # Category mix preserved through to the Ware list.
    cats = Counter(w.category for w in out)
    assert cats == Counter(skeletons)
    # No correlations authored → every Ware has an empty related list.
    assert all(w.related_products == [] for w in out)


def test_sample_catalog_attaches_correlations_to_wares() -> None:
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(3, taxonomy)
    catalog = _catalog_response(skeletons)  # names: Item-000, Item-001, Item-002
    correlations = Correlations(
        items=[
            ItemRelations(
                name="Item-000",
                related=[RelatedRef(name="Item-001", correlation=0.6)],
            ),
            ItemRelations(name="Item-001", related=[]),
            ItemRelations(
                name="Item-002",
                related=[RelatedRef(name="Item-000", correlation=0.4)],
            ),
        ]
    )
    freshness = _full_freshness(catalog)
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    wares = builder.sample_catalog(3)

    assert wares[0].related_products == [("Item-001", 0.6)]
    assert wares[1].related_products == []
    assert wares[2].related_products == [("Item-000", 0.4)]


def test_sample_catalog_chunks_correlations_call() -> None:
    """A single correlations call over many items causes the model to
    emit ``related: []`` for everything to fit its output budget. The
    builder chunks the catalog so each call stays small. With
    ``chunk_size=2`` over 5 items we expect 3 correlations calls
    (chunks of 2, 2, 1) and per-chunk results to merge into one
    per-Ware list."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(5, taxonomy)
    catalog = _catalog_response(skeletons)  # Item-000..Item-004

    chunk1 = Correlations(
        items=[
            ItemRelations(
                name="Item-000",
                related=[RelatedRef(name="Item-001", correlation=0.5)],
            ),
            ItemRelations(
                name="Item-001",
                related=[RelatedRef(name="Item-000", correlation=0.5)],
            ),
        ]
    )
    chunk2 = Correlations(
        items=[
            ItemRelations(
                name="Item-002",
                related=[RelatedRef(name="Item-004", correlation=0.6)],
            ),
            ItemRelations(name="Item-003", related=[]),
        ]
    )
    chunk3 = Correlations(
        items=[
            ItemRelations(
                name="Item-004",
                related=[RelatedRef(name="Item-002", correlation=0.6)],
            ),
        ]
    )

    freshness = _full_freshness(catalog)
    client = MockClient([taxonomy, catalog, chunk1, chunk2, chunk3, freshness])
    builder = WorldBuilder("luxury", client, correlations_chunk_size=2)
    wares = builder.sample_catalog(5)

    # 1 taxonomy + 1 catalog + 3 correlations chunks + 1 freshness call
    # (freshness chunk size defaults to 50, so 5 items fit in one call) = 6.
    assert len(client.calls) == 6
    schemas = [c["schema"] for c in client.calls]
    assert schemas == [
        Taxonomy,
        Catalog,
        Correlations,
        Correlations,
        Correlations,
        FreshnessSet,
    ]

    # Each chunk's correlations are merged into the per-Ware list.
    assert wares[0].related_products == [("Item-001", 0.5)]
    assert wares[1].related_products == [("Item-000", 0.5)]
    assert wares[2].related_products == [("Item-004", 0.6)]
    assert wares[3].related_products == []
    assert wares[4].related_products == [("Item-002", 0.6)]

    # Each correlations chunk's prompt sees only its own items, but the
    # full candidate-name list is included so cross-chunk references
    # remain valid.
    chunk_call_users = [c["user"] for c in client.calls if c["schema"] is Correlations]
    assert "Item-000" in chunk_call_users[0]
    assert "Item-001" in chunk_call_users[0]
    # Last chunk only annotates Item-004 but still lists every catalog
    # name as a candidate, so a cross-chunk ref like Item-002 is allowed.
    assert "Item-002" in chunk_call_users[2]  # candidate list
    assert "Item-004" in chunk_call_users[2]  # annotated item


def test_sample_catalog_rejects_non_positive_chunk_size() -> None:
    client = MockClient([])
    with pytest.raises(ValueError, match="correlations_chunk_size"):
        WorldBuilder("luxury", client, correlations_chunk_size=0)
    with pytest.raises(ValueError, match="correlations_chunk_size"):
        WorldBuilder("luxury", client, correlations_chunk_size=-5)


def test_sample_catalog_rejects_non_positive_freshness_chunk_size() -> None:
    client = MockClient([])
    with pytest.raises(ValueError, match="freshness_chunk_size"):
        WorldBuilder("luxury", client, freshness_chunk_size=0)
    with pytest.raises(ValueError, match="freshness_chunk_size"):
        WorldBuilder("luxury", client, freshness_chunk_size=-5)


def test_sample_catalog_drops_dangling_self_and_duplicate_refs() -> None:
    """Sanitisation at the boundary: dangling references, self-references,
    and within-item duplicates are silently pruned rather than triggering
    a retry. The candidate list is bounded in the prompt so this should
    rarely fire — but doing it here keeps the pipeline robust on cheap
    models."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(3, taxonomy)
    catalog = _catalog_response(skeletons)  # names: Item-000, Item-001, Item-002
    correlations = Correlations(
        items=[
            ItemRelations(
                name="Item-000",
                related=[
                    RelatedRef(name="Phantom", correlation=0.5),  # dangling
                    RelatedRef(name="Item-000", correlation=0.5),  # self-ref
                    RelatedRef(name="Item-001", correlation=0.5),  # ok
                    RelatedRef(name="Item-001", correlation=0.7),  # dup
                ],
            ),
            # The LLM emitted a name that isn't a catalog item — the whole
            # entry should be ignored.
            ItemRelations(
                name="Phantom",
                related=[RelatedRef(name="Item-002", correlation=0.5)],
            ),
            ItemRelations(name="Item-002", related=[]),
        ]
    )
    freshness = _full_freshness(catalog)
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    wares = builder.sample_catalog(3)

    # Item-000 keeps only the single valid, non-self, non-duplicate ref.
    assert wares[0].related_products == [("Item-001", 0.5)]
    # Item-001 had no entry → empty list (default initialised).
    assert wares[1].related_products == []
    # Item-002 had an empty related list.
    assert wares[2].related_products == []


def test_sample_catalog_does_not_set_lifecycle_or_stock_share_overrides() -> None:
    """The catalog prompt no longer authors per-Ware lifecycle or
    stock-share fields. Wares should have ``None`` in those slots so
    ``ItemRegistry`` falls back to ``ItemLifecycleParams`` defaults.
    Freshness has its own dedicated stage and is asserted separately."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(2, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    wares = builder.sample_catalog(2)

    for w in wares:
        assert w.init_stage is None
        assert w.stage_change_probs is None
        assert w.init_stock_share is None


def test_sample_catalog_propagates_freshness_overrides_from_llm() -> None:
    """LLM-authored ``alpha`` / ``decay`` flow through ``load_catalog``
    onto each ``Ware``. Items the LLM authored end up with non-None
    overrides; ``ItemRegistry`` resolves them ahead of params defaults."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(3, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = FreshnessSet(
        items=[
            ItemFreshness(name="Item-000", alpha=0.0, decay=20.0),
            ItemFreshness(name="Item-001", alpha=0.3, decay=35.0),
            ItemFreshness(name="Item-002", alpha=0.15, decay=25.0),
        ]
    )
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    wares = builder.sample_catalog(3)

    assert wares[0].freshness_alpha == 0.0
    assert wares[0].freshness_decay == 20.0
    assert wares[1].freshness_alpha == 0.3
    assert wares[1].freshness_decay == 35.0
    assert wares[2].freshness_alpha == 0.15
    assert wares[2].freshness_decay == 25.0


def test_sample_catalog_chunks_freshness_call() -> None:
    """``freshness_chunk_size=2`` over 5 items → 3 freshness calls.
    ``correlations_chunk_size`` keeps its 50-item default → 1 correlations
    call. Asserts the exact schema sequence so a regression on either
    chunk size reads cleanly."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(5, taxonomy)
    catalog = _catalog_response(skeletons)  # Item-000..Item-004
    correlations = _empty_correlations(catalog)

    chunk1 = FreshnessSet(
        items=[
            ItemFreshness(name="Item-000", alpha=0.1, decay=20.0),
            ItemFreshness(name="Item-001", alpha=0.2, decay=25.0),
        ]
    )
    chunk2 = FreshnessSet(
        items=[
            ItemFreshness(name="Item-002", alpha=0.3, decay=30.0),
            ItemFreshness(name="Item-003", alpha=0.0, decay=15.0),
        ]
    )
    chunk3 = FreshnessSet(
        items=[ItemFreshness(name="Item-004", alpha=0.4, decay=40.0)]
    )

    client = MockClient(
        [taxonomy, catalog, correlations, chunk1, chunk2, chunk3]
    )
    builder = WorldBuilder("luxury", client, freshness_chunk_size=2)
    wares = builder.sample_catalog(5)

    # 1 taxonomy + 1 catalog + 1 correlations + 3 freshness chunks = 6.
    assert len(client.calls) == 6
    schemas = [c["schema"] for c in client.calls]
    assert schemas == [
        Taxonomy,
        Catalog,
        Correlations,
        FreshnessSet,
        FreshnessSet,
        FreshnessSet,
    ]

    # Each chunk's authored values land on the right Ware.
    assert wares[0].freshness_alpha == 0.1
    assert wares[1].freshness_alpha == 0.2
    assert wares[2].freshness_alpha == 0.3
    assert wares[3].freshness_alpha == 0.0
    assert wares[4].freshness_alpha == 0.4

    # Each freshness chunk's prompt only mentions its own items.
    chunk_users = [c["user"] for c in client.calls if c["schema"] is FreshnessSet]
    assert "Item-000" in chunk_users[0]
    assert "Item-001" in chunk_users[0]
    assert "Item-002" not in chunk_users[0]
    assert "Item-004" in chunk_users[2]


def test_sample_catalog_drops_unknown_freshness_entries() -> None:
    """Names the LLM emitted that don't resolve against the catalog are
    silently dropped. The corresponding ``Ware`` keeps its ``None`` default
    so ``ItemRegistry`` falls back to ``ItemLifecycleParams`` defaults."""
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(3, taxonomy)
    catalog = _catalog_response(skeletons)  # Item-000..Item-002
    correlations = _empty_correlations(catalog)
    freshness = FreshnessSet(
        items=[
            ItemFreshness(name="Phantom", alpha=0.5, decay=99.0),  # dropped
            ItemFreshness(name="Item-001", alpha=0.3, decay=35.0),
            ItemFreshness(name="Item-002", alpha=0.0, decay=20.0),
        ]
    )
    client = MockClient([taxonomy, catalog, correlations, freshness])
    builder = WorldBuilder("luxury", client)
    wares = builder.sample_catalog(3)

    # Item-000 was not authored → falls back to None.
    assert wares[0].freshness_alpha is None
    assert wares[0].freshness_decay is None
    # Items the LLM did author propagate verbatim.
    assert wares[1].freshness_alpha == 0.3
    assert wares[1].freshness_decay == 35.0
    assert wares[2].freshness_alpha == 0.0
    assert wares[2].freshness_decay == 20.0


def test_build_store_templates_returns_dict_keyed_by_id() -> None:
    market = _market_response()
    client = MockClient([market, _templates_response()])
    builder = WorldBuilder("luxury", client)
    builder.build_market_domain_params()
    out = builder.build_store_templates()
    assert set(out.keys()) == {"flagship", "standard"}
    assert all(isinstance(v, StoreTemplate) for v in out.values())
    assert out["flagship"].capacity == 500.0


def test_build_store_templates_preserves_init_freshness() -> None:
    """``init_freshness`` is still per-template; rest of the roster comes
    from the random sampler at store-construction time."""
    market = _market_response()
    client = MockClient([market, _templates_response()])
    builder = WorldBuilder("luxury", client)
    builder.build_market_domain_params()
    out = builder.build_store_templates()
    assert out["flagship"].init_freshness == "fresh"
    assert out["standard"].init_freshness == "baseline"
    # No LLM-authored roster anymore — the template falls back to None so
    # ``init_store_state`` does the random sample at construction time.
    assert out["flagship"].init_active_products is None
    assert out["standard"].init_active_products is None


def test_build_store_templates_does_not_require_catalog() -> None:
    """The store-templates stage is independent of the catalog stage now
    that ``init_active_products`` is no longer authored. ``regions``
    comes from the market stage; everything else is self-contained."""
    market = _market_response()
    client = MockClient([market, _templates_response()])
    builder = WorldBuilder("luxury", client)
    builder.build_market_domain_params()
    out = builder.build_store_templates()
    assert set(out.keys()) == {"flagship", "standard"}


def test_build_market_domain_params_merges_handset_defaults() -> None:
    market = _market_response()
    client = MockClient([market])
    builder = WorldBuilder("luxury", client)
    params = builder.build_market_domain_params()
    assert isinstance(params, MarketParams)
    # LLM-supplied fields:
    assert params.cycle_len == 365
    assert params.price_elasticity == -1.5
    assert params.regions == ["US"]
    # Hand-set math defaults:
    assert params.cycle_amp == 0.0
    assert params.demand_factor_min == 0.1
    assert params.cross_inv_lo == 0.3
    # Init demand/supply default to midpoint of the clamp band.
    assert params.init_demand == (params.max_value - params.min_value) / 2
    assert params.init_supply == (params.max_value - params.min_value) / 2
    # Distribution defaults are present and correctly typed:
    from src.sim.distributions import Distribution
    assert isinstance(params.trend, Distribution)
    assert isinstance(params.demand_shock, Distribution)
    assert isinstance(params.base_demand, Distribution)


def test_build_end_to_end_against_canned_responses_no_retries() -> None:
    market = _market_response(regions=("US",))
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(8, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    templates = _templates_response()
    client = MockClient(
        [market, taxonomy, catalog, correlations, freshness, templates]
    )
    builder = WorldBuilder("luxury", client)

    world = builder.build(n_items=8)

    assert isinstance(world, World)
    assert isinstance(world.market, MarketParams)
    assert len(world.catalog) == 8
    assert set(world.store_templates.keys()) == {"flagship", "standard"}
    # Exactly six LLM calls — no retries triggered.
    assert len(client.calls) == 6
    # Build order: market → taxonomy → catalog → correlations → freshness
    # → templates.
    assert client.calls[0]["schema"] is MarketDomain
    assert client.calls[1]["schema"] is Taxonomy
    assert client.calls[2]["schema"] is Catalog
    assert client.calls[3]["schema"] is Correlations
    assert client.calls[4]["schema"] is FreshnessSet
    assert client.calls[5]["schema"] is StoreTemplateList


def test_validation_failure_retries_with_error_in_next_prompt() -> None:
    err = _validation_error_for(Taxonomy)
    valid = _taxonomy_response()
    client = MockClient([err, valid])
    builder = WorldBuilder("luxury", client, max_retries=3)

    out = builder.build_taxonomy()

    assert out is valid
    assert len(client.calls) == 2
    # The retry's user prompt embeds the validation error verbatim.
    second_user = client.calls[1]["user"]
    assert "failed schema validation" in second_user.lower()
    assert "Taxonomy" in second_user or "taxonomy" in second_user.lower()


def test_validation_failure_exhausts_retries_and_raises() -> None:
    err1 = _validation_error_for(Taxonomy)
    err2 = _validation_error_for(Taxonomy)
    err3 = _validation_error_for(Taxonomy)
    client = MockClient([err1, err2, err3])
    builder = WorldBuilder("luxury", client, max_retries=3)
    with pytest.raises(RuntimeError, match="failed validation"):
        builder.build_taxonomy()
    assert len(client.calls) == 3


def test_sample_catalog_rejects_non_positive_n() -> None:
    client = MockClient([])
    builder = WorldBuilder("luxury", client)
    with pytest.raises(ValueError):
        builder.sample_catalog(0)


def test_world_artifact_round_trips_through_scenario_json() -> None:
    """The W from WorldBuilder must be persistable via Scenario.to_json/from_json."""
    from datetime import datetime

    from src.sim.scenario import (
        DisruptionParams,
        ItemLifecycleParams,
        Scenario,
        StoreInstance,
    )
    from src.sim.distributions import Constant

    market = _market_response()
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(5, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    templates = _templates_response()
    client = MockClient(
        [market, taxonomy, catalog, correlations, freshness, templates]
    )
    builder = WorldBuilder("luxury", client)
    world = builder.build(n_items=5)

    scenario = Scenario(
        catalog=world.catalog,
        market=world.market,
        disruption=DisruptionParams(
            event_prob=0.05,
            types=["x"],
            regions=["US"],
            severity=Constant(1.0),
            duration=Constant(3),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=["growth"],
            init_stage="growth",
            default_stage_change_probs={"growth": 0.0},
        ),
        stores=[
            StoreInstance(template=world.store_templates["standard"], init_seed=1)
        ],
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    encoded = scenario.to_json()
    decoded = Scenario.from_json(encoded)
    assert len(decoded.catalog) == 5
    assert decoded.market.price_elasticity == -1.5
    assert decoded.stores[0].template.id == "standard"
    # Per-template fields survive round-trip.
    assert decoded.stores[0].template.init_freshness == "baseline"
    # Roster falls back to the random sampler at construction time, so the
    # template carries no explicit list.
    assert decoded.stores[0].template.init_active_products is None
    # Per-Ware lifecycle / stock-share are no longer authored, so they
    # fall back to ``ItemLifecycleParams`` defaults.
    assert decoded.catalog[0].init_stage is None
    assert decoded.catalog[0].init_stock_share is None
    # Freshness is now authored; ``_full_freshness`` paints every Ware
    # with the same curve. Round-trip preserves the override.
    assert decoded.catalog[0].freshness_alpha == 0.2
    assert decoded.catalog[0].freshness_decay == 30.0
