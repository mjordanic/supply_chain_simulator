"""Tests for World serialisation and meta block (issue 01).

Covers:
- full round-trip with non-trivial catalog/market/store_templates/meta
- distribution-typed Ware override round-trip
- meta=None round-trip (canonical: None stays None)
- WorldBuilder.build() populates the five meta keys
- BUILDER_VERSION constant is "1"
- to_json returns string when path=None
- to_json writes file when path is given
- from_json accepts a filesystem Path
- from_json accepts a JSON string body
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.llm.world_builder import BUILDER_VERSION, World, WorldBuilder
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import MarketParams, StoreTemplate, Ware, _ware_from_dict


# ---------------------------------------------------------------------------
# Helpers shared with test_world_builder.py (duplicated here to keep tests
# self-contained — no cross-module fixture import needed).
# ---------------------------------------------------------------------------

def _minimal_market() -> MarketParams:
    from src.sim.distributions import Constant, Normal, Uniform
    return MarketParams(
        cycle_len=12,
        cycle_amp=0.001,
        init_demand=0.5,
        init_supply=0.5,
        peak_factor=1.2,
        off_factor=0.8,
        season_months={"all": list(range(1, 13))},
        regions=["EU"],
        correlation=0.5,
        trend_update_interval=10,
        min_value=0.1,
        max_value=1.0,
        stage_multipliers={"maturity": 1.0},
        price_elasticity=-1.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.05),
        supply_shock=Normal(0.0, 0.05),
        base_demand=Uniform(2, 8),
    )


def _minimal_template(tid: str = "t1") -> StoreTemplate:
    return StoreTemplate(
        id=tid,
        region="EU",
        capacity=200.0,
        init_balance=5000.0,
        init_stock_pct=0.5,
        delivery_lag=1.0,
        holding_rate=0.01,
        order_fee=10.0,
        init_active_count=2,
    )


def _minimal_ware(pid: str = "P0000") -> Ware:
    return Ware(
        product_id=pid,
        name="Widget",
        category="Gadgets",
        related_products=[],
        base_price=20.0,
        unit_cost=10.0,
        seasonality="all_season",
    )


def _world_with_meta(meta: dict[str, Any] | None = None) -> World:
    return World(
        catalog=[_minimal_ware("P0000"), _minimal_ware("P0001")],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template("t1")},
        meta=meta,
    )


# ---------------------------------------------------------------------------
# BUILDER_VERSION constant
# ---------------------------------------------------------------------------

def test_builder_version_is_one() -> None:
    assert BUILDER_VERSION == "1"


# ---------------------------------------------------------------------------
# Full round-trip: catalog / market / store_templates / meta
# ---------------------------------------------------------------------------

def test_world_round_trip_with_all_fields() -> None:
    meta = {"archetype": "luxury", "n_items": 2, "model": None, "builder_version": "1", "built_at": "2024-01-01T00:00:00+00:00"}
    world = _world_with_meta(meta)

    recovered = World.from_dict(world.to_dict())

    assert len(recovered.catalog) == 2
    assert recovered.catalog[0].product_id == "P0000"
    assert recovered.catalog[1].product_id == "P0001"
    assert recovered.market.price_elasticity == -1.0
    assert recovered.market.regions == ["EU"]
    assert set(recovered.store_templates.keys()) == {"t1"}
    assert recovered.store_templates["t1"].capacity == 200.0
    assert recovered.meta == meta


# ---------------------------------------------------------------------------
# Distribution-typed Ware override round-trip
# ---------------------------------------------------------------------------

def test_world_round_trip_distribution_typed_ware_override() -> None:
    ware_with_dist = Ware(
        product_id="P0000",
        name="Widget",
        category="Gadgets",
        related_products=[],
        base_price=20.0,
        unit_cost=10.0,
        seasonality="all_season",
        freshness_alpha=Uniform(0.1, 0.3),
        freshness_decay=Normal(30.0, 5.0),
    )
    world = World(
        catalog=[ware_with_dist],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template()},
    )

    recovered = World.from_dict(world.to_dict())

    alpha = recovered.catalog[0].freshness_alpha
    decay = recovered.catalog[0].freshness_decay
    assert isinstance(alpha, Uniform)
    assert alpha.low == 0.1
    assert alpha.high == 0.3
    assert isinstance(decay, Normal)
    assert decay.mean == 30.0
    assert decay.std == 5.0


# ---------------------------------------------------------------------------
# meta=None round-trip — canonical: None stays None
# ---------------------------------------------------------------------------

def test_world_round_trip_meta_none() -> None:
    world = _world_with_meta(meta=None)
    recovered = World.from_dict(world.to_dict())
    assert recovered.meta is None


# ---------------------------------------------------------------------------
# to_json returns string when path=None
# ---------------------------------------------------------------------------

def test_to_json_returns_string_when_no_path() -> None:
    world = _world_with_meta({"key": "val"})
    result = world.to_json()
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["meta"] == {"key": "val"}
    assert "catalog" in parsed
    assert "market" in parsed
    assert "store_templates" in parsed


# ---------------------------------------------------------------------------
# to_json writes file and returns string when path is given
# ---------------------------------------------------------------------------

def test_to_json_writes_file_and_returns_string(tmp_path: Path) -> None:
    world = _world_with_meta({"x": 1})
    out_path = tmp_path / "nested" / "world.json"

    result = world.to_json(path=out_path)

    assert isinstance(result, str)
    assert out_path.exists()
    file_content = json.loads(out_path.read_text())
    assert file_content["meta"] == {"x": 1}
    # Return value matches file content exactly.
    assert result == out_path.read_text()


# ---------------------------------------------------------------------------
# from_json accepts a filesystem Path
# ---------------------------------------------------------------------------

def test_from_json_accepts_file_path(tmp_path: Path) -> None:
    world = _world_with_meta({"source": "file"})
    out_path = tmp_path / "world.json"
    world.to_json(path=out_path)

    recovered = World.from_json(out_path)

    assert recovered.meta == {"source": "file"}
    assert len(recovered.catalog) == 2


# ---------------------------------------------------------------------------
# from_json accepts a JSON string body
# ---------------------------------------------------------------------------

def test_from_json_accepts_json_string() -> None:
    world = _world_with_meta({"source": "string"})
    json_str = world.to_json()

    recovered = World.from_json(json_str)

    assert recovered.meta == {"source": "string"}
    assert len(recovered.catalog) == 2


# ---------------------------------------------------------------------------
# from_json accepts a str filesystem path
# ---------------------------------------------------------------------------

def test_from_json_accepts_str_path(tmp_path: Path) -> None:
    world = _world_with_meta({"source": "str_path"})
    out_path = tmp_path / "world.json"
    world.to_json(path=out_path)

    recovered = World.from_json(str(out_path))

    assert recovered.meta == {"source": "str_path"}


# ---------------------------------------------------------------------------
# WorldBuilder.build() populates meta with five keys
# ---------------------------------------------------------------------------

class _MockClient:
    """Minimal mock matching LLMClient protocol for meta tests."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)

    def structured_completion(self, *, system: str, user: str, schema: type) -> Any:
        return self._responses.pop(0)


def _make_builder_responses() -> list:
    """Six canned responses for WorldBuilder.build(n_items=3)."""
    from src.llm.schemas import (
        Catalog, CatalogItem, Correlations, FreshnessSet, InitFreshness,
        ItemFreshness, ItemRelations, MarketDomain, SeasonWindow,
        StoreTemplateList, StoreTemplateSpec, Taxonomy, TaxonomyCategory,
        Seasonality,
    )

    market = MarketDomain(
        cycle_len=12,
        peak_factor=1.2,
        off_factor=0.8,
        season_months=[SeasonWindow(name="all", months=list(range(1, 13)))],
        regions=["EU"],
        price_elasticity=-1.0,
    )
    taxonomy = Taxonomy(
        archetype="test",
        categories=[TaxonomyCategory(name="Cat", description="", target_share=1.0)],
    )
    catalog = Catalog(
        items=[
            CatalogItem(name=f"Item-{i:03d}", category="Cat", base_price=20.0 + i, unit_cost=10.0 + i, seasonality=Seasonality.ALL_SEASON)
            for i in range(3)
        ]
    )
    correlations = Correlations(
        items=[ItemRelations(name=it.name, related=[]) for it in catalog.items]
    )
    freshness = FreshnessSet(
        items=[ItemFreshness(name=it.name, alpha=0.1, decay=20.0) for it in catalog.items]
    )
    templates = StoreTemplateList(
        templates=[
            StoreTemplateSpec(
                id="t1", region="EU", capacity=200.0, init_balance=5000.0,
                init_stock_pct=0.5, delivery_lag=1.0, holding_rate=0.01,
                order_fee=10.0, init_active_count=2,
                init_freshness=InitFreshness.BASELINE,
            )
        ]
    )
    return [market, taxonomy, catalog, correlations, freshness, templates]


def test_world_builder_build_populates_five_meta_keys() -> None:
    client = _MockClient(_make_builder_responses())
    builder = WorldBuilder("luxury", client)

    world = builder.build(n_items=3)

    assert world.meta is not None
    assert world.meta["archetype"] == "luxury"
    assert world.meta["n_items"] == 3
    assert "model" in world.meta
    assert world.meta["builder_version"] == BUILDER_VERSION
    assert "built_at" in world.meta
    # built_at is an ISO-8601 UTC timestamp parseable by datetime.
    dt = datetime.fromisoformat(world.meta["built_at"])
    assert dt.tzinfo is not None


def test_world_builder_build_model_none_when_client_has_no_model_id() -> None:
    """MockClient has no model_id attribute → meta["model"] is None."""
    client = _MockClient(_make_builder_responses())
    assert not hasattr(client, "model_id")
    builder = WorldBuilder("luxury", client)
    world = builder.build(n_items=3)
    assert world.meta["model"] is None


def test_world_builder_build_model_from_client_model_id() -> None:
    """Client exposes model_id → propagated into meta."""
    client = _MockClient(_make_builder_responses())
    client.model_id = "gpt-4o"  # type: ignore[attr-defined]
    builder = WorldBuilder("luxury", client)
    world = builder.build(n_items=3)
    assert world.meta["model"] == "gpt-4o"
