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
    ItemFreshness,
    ItemRelations,
    MarketDomain,
    RelatedRef,
    Seasonality,
    Taxonomy,
    TaxonomyCategory,
)
from src.llm.world_builder import WorldBuilder, allocate_skeletons
from src.sim.scenario import MarketParams


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
    assert params.cycle_amp == 0.0065
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
    """End-to-end: market + catalog pipeline (no store-template stage)."""
    market = _market_response(regions=("US",))
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(8, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    client = MockClient(
        [market, taxonomy, catalog, correlations, freshness]
    )
    builder = WorldBuilder("luxury", client)

    market_params = builder.build_market_domain_params()
    wares = builder.sample_catalog(8)

    assert isinstance(market_params, MarketParams)
    assert len(wares) == 8
    # Exactly five LLM calls — no retries triggered.
    assert len(client.calls) == 5
    # Build order: market → taxonomy → catalog → correlations → freshness.
    assert client.calls[0]["schema"] is MarketDomain
    assert client.calls[1]["schema"] is Taxonomy
    assert client.calls[2]["schema"] is Catalog
    assert client.calls[3]["schema"] is Correlations
    assert client.calls[4]["schema"] is FreshnessSet


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


# ---------------------------------------------------------------------------
# WorldBuilder.build_setup — dir-as-cache
# ---------------------------------------------------------------------------


def _build_setup_responses() -> tuple[list, "Catalog"]:
    """Return the minimal MockClient responses for build_setup (market + catalog)."""
    market = _market_response()
    taxonomy = _taxonomy_response()
    skeletons = allocate_skeletons(3, taxonomy)
    catalog = _catalog_response(skeletons)
    correlations = _empty_correlations(catalog)
    freshness = _full_freshness(catalog)
    responses = [market, taxonomy, catalog, correlations, freshness]
    return responses, catalog


def test_build_setup_writes_catalog_csv(tmp_path):
    """build_setup writes catalog.csv to the target directory."""
    responses, _ = _build_setup_responses()
    client = MockClient(responses)
    builder = WorldBuilder("luxury", client)

    setup_dir = tmp_path / "my_setup"
    catalog, market = builder.build_setup(n_items=3, setup_dir=setup_dir)

    assert (setup_dir / "catalog.csv").is_file()
    assert len(catalog) == 3


def test_build_setup_writes_market_block_only(tmp_path):
    """build_setup writes market: to setup.yaml but NOT nodes, edges, or run."""
    import yaml

    responses, _ = _build_setup_responses()
    client = MockClient(responses)
    builder = WorldBuilder("luxury", client)

    setup_dir = tmp_path / "data_only"
    builder.build_setup(n_items=3, setup_dir=setup_dir)

    doc = yaml.safe_load((setup_dir / "setup.yaml").read_text())
    assert "market" in doc
    assert "nodes" not in doc
    assert "edges" not in doc
    assert "run" not in doc


def test_build_setup_cache_hit_skips_llm(tmp_path):
    """build_setup on a cache hit (catalog.csv present) makes zero LLM calls."""
    responses, catalog_resp = _build_setup_responses()
    client = MockClient(responses)
    builder = WorldBuilder("luxury", client)

    setup_dir = tmp_path / "cached"
    # First call: builds and writes
    builder.build_setup(n_items=3, setup_dir=setup_dir)
    first_call_count = len(client.calls)

    # Second call: cache hit — must not call LLM
    fresh_client = MockClient([])  # empty — would error on any call
    builder2 = WorldBuilder("luxury", fresh_client)
    catalog2, market2 = builder2.build_setup(n_items=3, setup_dir=setup_dir)

    assert len(fresh_client.calls) == 0  # no LLM calls on cache hit
    assert len(catalog2) == 3


def test_build_setup_force_rebuild_calls_llm_again(tmp_path):
    """force_rebuild=True causes the LLM to be called even when cache exists."""
    responses1, _ = _build_setup_responses()
    client1 = MockClient(responses1)
    builder1 = WorldBuilder("luxury", client1)
    setup_dir = tmp_path / "rebuild"
    builder1.build_setup(n_items=3, setup_dir=setup_dir)

    responses2, _ = _build_setup_responses()
    client2 = MockClient(responses2)
    builder2 = WorldBuilder("luxury", client2)
    builder2.build_setup(n_items=3, setup_dir=setup_dir, force_rebuild=True)

    # LLM was called again (market + catalog pipeline)
    assert len(client2.calls) > 0
