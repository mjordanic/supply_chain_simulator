"""Schema validators for the world-builder LLM stages.

Each acceptance-criterion validator gets a dedicated rejection test plus
a happy-path check confirming a valid payload parses cleanly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def _valid_item(**overrides):
    base = dict(
        name="Widget A",
        category="Widgets",
        base_price=10.0,
        unit_cost=5.0,
        seasonality="all_season",
    )
    base.update(overrides)
    return base


def test_catalog_item_accepts_valid_payload() -> None:
    it = CatalogItem(**_valid_item())
    assert it.name == "Widget A"
    assert it.seasonality is Seasonality.ALL_SEASON


def test_catalog_item_rejects_base_price_equal_to_unit_cost() -> None:
    with pytest.raises(ValidationError, match="base_price"):
        CatalogItem(**_valid_item(base_price=5.0, unit_cost=5.0))


def test_catalog_item_rejects_base_price_below_unit_cost() -> None:
    with pytest.raises(ValidationError, match="base_price"):
        CatalogItem(**_valid_item(base_price=3.0, unit_cost=5.0))


def test_catalog_item_rejects_unknown_seasonality() -> None:
    with pytest.raises(ValidationError):
        CatalogItem(**_valid_item(seasonality="monsoon"))


def test_catalog_item_rejects_negative_unit_cost() -> None:
    with pytest.raises(ValidationError):
        CatalogItem(**_valid_item(base_price=10.0, unit_cost=-1.0))


def test_catalog_item_rejects_extra_legacy_field() -> None:
    """``CatalogItem`` no longer carries lifecycle / freshness / stock-share
    fields. ``extra='forbid'`` should reject any LLM that drifts back to
    the old shape — surfaces drift loudly rather than silently dropping."""
    with pytest.raises(ValidationError):
        CatalogItem(**_valid_item(init_stage="maturity"))


def test_catalog_accepts_valid_payload() -> None:
    catalog = Catalog(
        items=[
            CatalogItem(**_valid_item(name="A")),
            CatalogItem(**_valid_item(name="B")),
        ]
    )
    assert len(catalog.items) == 2


def test_catalog_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        Catalog(items=[])


def test_related_ref_rejects_correlation_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        RelatedRef(name="X", correlation=1.5)
    with pytest.raises(ValidationError):
        RelatedRef(name="X", correlation=-0.1)


def test_item_relations_accepts_empty_related_list() -> None:
    """An item with no obvious cross-correlations is fine."""
    rel = ItemRelations(name="A", related=[])
    assert rel.name == "A"
    assert rel.related == []


def test_correlations_accepts_valid_payload() -> None:
    payload = Correlations(
        items=[
            ItemRelations(
                name="A",
                related=[RelatedRef(name="B", correlation=0.5)],
            ),
            ItemRelations(name="B", related=[]),
        ]
    )
    assert len(payload.items) == 2


def test_correlations_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        Correlations(items=[])


def test_item_freshness_accepts_valid_payload() -> None:
    f = ItemFreshness(name="A", alpha=0.2, decay=30.0)
    assert f.name == "A"
    assert f.alpha == 0.2
    assert f.decay == 30.0


def test_item_freshness_accepts_zero_alpha_staple() -> None:
    """``alpha=0`` is the staple override — curve identically 1. Decay
    must still be positive even though the curve doesn't depend on it."""
    f = ItemFreshness(name="Bread", alpha=0.0, decay=20.0)
    assert f.alpha == 0.0


def test_item_freshness_rejects_negative_alpha() -> None:
    with pytest.raises(ValidationError):
        ItemFreshness(name="A", alpha=-0.1, decay=20.0)


def test_item_freshness_rejects_zero_decay() -> None:
    """Open-interval ``decay > 0`` keeps the consumer free of
    division-by-zero pitfalls."""
    with pytest.raises(ValidationError):
        ItemFreshness(name="A", alpha=0.2, decay=0.0)


def test_item_freshness_rejects_negative_decay() -> None:
    with pytest.raises(ValidationError):
        ItemFreshness(name="A", alpha=0.2, decay=-5.0)


def test_item_freshness_rejects_extra_field() -> None:
    """``extra='forbid'`` regression guard — the LLM must not slip in
    fields the schema doesn't declare."""
    with pytest.raises(ValidationError):
        ItemFreshness(name="A", alpha=0.2, decay=20.0, novelty=0.5)


def test_freshness_set_accepts_valid_payload() -> None:
    payload = FreshnessSet(
        items=[
            ItemFreshness(name="A", alpha=0.2, decay=30.0),
            ItemFreshness(name="B", alpha=0.0, decay=20.0),
        ]
    )
    assert len(payload.items) == 2


def test_freshness_set_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        FreshnessSet(items=[])


def _valid_template(**overrides):
    base = dict(
        id="standard",
        region="US",
        capacity=200.0,
        init_balance=10000.0,
        init_stock_pct=0.4,
        delivery_lag=2.0,
        holding_rate=0.005,
        order_fee=10.0,
        init_active_count=3,
        init_freshness="baseline",
    )
    base.update(overrides)
    return base


def test_store_template_accepts_valid_payload() -> None:
    StoreTemplateSpec(**_valid_template())


def test_store_template_rejects_negative_capacity() -> None:
    with pytest.raises(ValidationError):
        StoreTemplateSpec(**_valid_template(capacity=-1.0))


def test_store_template_rejects_negative_balance() -> None:
    with pytest.raises(ValidationError):
        StoreTemplateSpec(**_valid_template(init_balance=-100.0))


def test_store_template_rejects_negative_lead_time() -> None:
    with pytest.raises(ValidationError):
        StoreTemplateSpec(**_valid_template(delivery_lag=-1.0))


def test_store_template_rejects_extra_legacy_field() -> None:
    """``init_active_products`` is no longer authored — the simulator
    falls back to a random sampler driven by ``init_active_count``.
    ``extra='forbid'`` should reject any payload that tries to slip the
    legacy field back in."""
    with pytest.raises(ValidationError):
        StoreTemplateSpec(**_valid_template(init_active_products=["P0000"]))


def test_store_template_accepts_both_init_freshness_modes() -> None:
    """Closed enum: ``"baseline"`` and ``"fresh"`` parse, anything else fails."""
    for mode in ("baseline", "fresh"):
        spec = StoreTemplateSpec(**_valid_template(init_freshness=mode))
        assert spec.init_freshness == InitFreshness(mode)
    with pytest.raises(ValidationError):
        StoreTemplateSpec(**_valid_template(init_freshness="grand-opening"))


def test_store_template_list_rejects_duplicate_ids() -> None:
    a = StoreTemplateSpec(**_valid_template(id="dup"))
    b = StoreTemplateSpec(**_valid_template(id="dup"))
    with pytest.raises(ValidationError, match="unique"):
        StoreTemplateList(templates=[a, b])


def _valid_market(**overrides):
    base = dict(
        cycle_len=365,
        peak_factor=1.2,
        off_factor=0.7,
        init_demand=100.0,
        init_supply=100.0,
        season_months=[
            {
                "name": "all_season",
                "months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        ],
        regions=["US"],
        price_elasticity=-1.5,
    )
    base.update(overrides)
    return base


def test_market_domain_accepts_valid_payload() -> None:
    MarketDomain(**_valid_market())


def test_market_domain_rejects_non_negative_elasticity() -> None:
    with pytest.raises(ValidationError, match="price_elasticity"):
        MarketDomain(**_valid_market(price_elasticity=0.0))
    with pytest.raises(ValidationError, match="price_elasticity"):
        MarketDomain(**_valid_market(price_elasticity=1.2))


def test_market_domain_rejects_invalid_month() -> None:
    with pytest.raises(ValidationError):
        MarketDomain(
            **_valid_market(season_months=[{"name": "summer", "months": [13]}])
        )


def test_market_domain_rejects_duplicate_season_names() -> None:
    with pytest.raises(ValidationError, match="duplicate season names"):
        MarketDomain(
            **_valid_market(
                season_months=[
                    {"name": "summer", "months": [6, 7, 8]},
                    {"name": "summer", "months": [9]},
                ]
            )
        )


def test_market_domain_season_months_dict_round_trip() -> None:
    """``season_months_dict`` is the adapter the builder relies on to merge
    LLM payload into ``MarketParams.season_months: dict[str, list[int]]``.
    Regression-guard that contract."""
    md = MarketDomain(
        **_valid_market(
            season_months=[
                {"name": "spring", "months": [3, 4, 5]},
                {"name": "fall", "months": [9, 10, 11]},
            ]
        )
    )
    assert md.season_months_dict() == {
        "spring": [3, 4, 5],
        "fall": [9, 10, 11],
    }


def test_taxonomy_accepts_valid_payload() -> None:
    t = Taxonomy(
        archetype="luxury",
        categories=[
            TaxonomyCategory(name="Apparel", description="", target_share=0.6),
            TaxonomyCategory(name="Accessories", description="", target_share=0.4),
        ],
    )
    assert len(t.categories) == 2


def test_taxonomy_rejects_empty_categories() -> None:
    with pytest.raises(ValidationError):
        Taxonomy(archetype="x", categories=[])


def test_taxonomy_category_rejects_share_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        TaxonomyCategory(name="X", description="", target_share=0.0)
    with pytest.raises(ValidationError):
        TaxonomyCategory(name="X", description="", target_share=1.5)
