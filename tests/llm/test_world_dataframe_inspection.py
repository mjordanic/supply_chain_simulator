"""Tests for World DataFrame inspection methods (issue 03)."""

from __future__ import annotations

from typing import Any

import pytest

from src.sim.world import World
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import MarketParams, StoreTemplate, Ware


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_market() -> MarketParams:
    return MarketParams(
        cycle_len=12,
        cycle_amp=0.001,
        init_demand=0.5,
        init_supply=0.5,
        peak_factor=1.2,
        off_factor=0.8,
        season_months={"all": list(range(1, 13))},
        regions=["EU", "US"],
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


def _minimal_ware(pid: str = "P0000", base_price: float = 20.0, unit_cost: float = 10.0) -> Ware:
    return Ware(
        product_id=pid,
        name="Widget",
        category="Gadgets",
        related_products=[],
        base_price=base_price,
        unit_cost=unit_cost,
        seasonality="all_season",
    )


def _fixture_world(meta: dict[str, Any] | None = None) -> World:
    return World(
        catalog=[
            _minimal_ware("P0000", base_price=20.0, unit_cost=10.0),
            _minimal_ware("P0001", base_price=30.0, unit_cost=15.0),
        ],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template("t1"), "t2": _minimal_template("t2")},
        meta=meta,
    )


_EXPECTED_META = {
    "archetype": "luxury",
    "n_items": 2,
    "model": "gpt-4o",
    "builder_version": "1",
    "built_at": "2024-01-01T00:00:00+00:00",
}

_META_COLUMNS = {"archetype", "n_items", "model", "builder_version", "built_at"}

_CATALOG_COLUMNS = {
    "product_id", "name", "category", "base_price", "unit_cost", "margin",
    "seasonality", "freshness_alpha", "freshness_decay",
    "init_stage", "stage_change_probs", "init_stock_share", "related_products",
}

_STORE_TEMPLATE_COLUMNS = {
    "id", "region", "capacity", "init_balance", "init_stock_pct",
    "delivery_lag", "holding_rate", "order_fee", "init_active_count",
    "init_active_products", "init_freshness",
}

_MARKET_COLUMNS = {
    "cycle_len", "cycle_amp", "init_demand", "init_supply", "peak_factor",
    "off_factor", "season_months", "regions", "correlation",
    "trend_update_interval", "min_value", "max_value", "stage_multipliers",
    "price_elasticity", "promo_multiplier", "demand_factor_min",
    "supply_factor_min",
    "cross_inv_lo", "cross_inv_hi", "cross_factor_range",
    "trend", "demand_shock", "supply_shock", "base_demand",
}


# ---------------------------------------------------------------------------
# catalog_df
# ---------------------------------------------------------------------------

def test_catalog_df_row_count():
    world = _fixture_world()
    df = world.catalog_df()
    assert len(df) == 2


def test_catalog_df_column_set():
    world = _fixture_world()
    df = world.catalog_df()
    assert set(df.columns) == _CATALOG_COLUMNS


def test_catalog_df_margin_computed():
    world = _fixture_world()
    df = world.catalog_df()
    row0 = df[df["product_id"] == "P0000"].iloc[0]
    row1 = df[df["product_id"] == "P0001"].iloc[0]
    assert row0["margin"] == pytest.approx(10.0)   # 20.0 - 10.0
    assert row1["margin"] == pytest.approx(15.0)   # 30.0 - 15.0


def test_catalog_df_returns_dataframe():
    import pandas as pd
    world = _fixture_world()
    assert isinstance(world.catalog_df(), pd.DataFrame)


# ---------------------------------------------------------------------------
# store_templates_df
# ---------------------------------------------------------------------------

def test_store_templates_df_row_count():
    world = _fixture_world()
    df = world.store_templates_df()
    assert len(df) == 2


def test_store_templates_df_column_set():
    world = _fixture_world()
    df = world.store_templates_df()
    assert set(df.columns) == _STORE_TEMPLATE_COLUMNS


def test_store_templates_df_returns_dataframe():
    import pandas as pd
    world = _fixture_world()
    assert isinstance(world.store_templates_df(), pd.DataFrame)


# ---------------------------------------------------------------------------
# market_df
# ---------------------------------------------------------------------------

def test_market_df_single_row():
    world = _fixture_world()
    df = world.market_df()
    assert len(df) == 1


def test_market_df_column_set():
    world = _fixture_world()
    df = world.market_df()
    assert set(df.columns) == _MARKET_COLUMNS


def test_market_df_regions_is_list():
    world = _fixture_world()
    df = world.market_df()
    regions = df.iloc[0]["regions"]
    assert isinstance(regions, list)
    assert regions == ["EU", "US"]


def test_market_df_season_months_is_dict():
    world = _fixture_world()
    df = world.market_df()
    sm = df.iloc[0]["season_months"]
    assert isinstance(sm, dict)


def test_market_df_returns_dataframe():
    import pandas as pd
    world = _fixture_world()
    assert isinstance(world.market_df(), pd.DataFrame)


# ---------------------------------------------------------------------------
# meta_df
# ---------------------------------------------------------------------------

def test_meta_df_single_row_when_populated():
    world = _fixture_world(meta=_EXPECTED_META)
    df = world.meta_df()
    assert len(df) == 1


def test_meta_df_column_set():
    world = _fixture_world(meta=_EXPECTED_META)
    df = world.meta_df()
    assert set(df.columns) == _META_COLUMNS


def test_meta_df_values():
    world = _fixture_world(meta=_EXPECTED_META)
    df = world.meta_df()
    row = df.iloc[0]
    assert row["archetype"] == "luxury"
    assert row["n_items"] == 2
    assert row["model"] == "gpt-4o"
    assert row["builder_version"] == "1"
    assert row["built_at"] == "2024-01-01T00:00:00+00:00"


def test_meta_df_empty_when_meta_none():
    world = _fixture_world(meta=None)
    df = world.meta_df()
    assert len(df) == 0
    assert set(df.columns) == _META_COLUMNS


def test_meta_df_returns_dataframe():
    import pandas as pd
    world = _fixture_world(meta=_EXPECTED_META)
    assert isinstance(world.meta_df(), pd.DataFrame)
