"""Tests for Scenario.from_world() and the six DataFrame view methods.

Covers the contract described in issue 04:
- from_world wires catalog/market from World and forwards the six required kwargs
- catalog_df / stores_df / market_df / disruption_df / lifecycle_df / summary_df
  return DataFrames with expected columns and row counts
- stores_df.policy_class populated with class name or None
- all methods work on a from_json round-trip (policy_class = None expected)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.llm.world_builder import World
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    Ware,
    load_catalog,
    make_stores,
)


# ------------------------------------------------------------------ fixtures


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


def _lifecycle() -> ItemLifecycleParams:
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


def _world() -> World:
    return World(catalog=_catalog(), market=_market(), store_templates={}, meta=None)


class StubPolicy:
    pass


def _stores_with_policy() -> list[StoreInstance]:
    return make_stores([(_template(), 1, StubPolicy()), (_template(), 2, StubPolicy())])


def _stores_no_policy() -> list[StoreInstance]:
    return [StoreInstance(template=_template(), init_seed=1, policy=None)]


def _scenario(stores=None) -> Scenario:
    return Scenario(
        catalog=_catalog(),
        market=_market(),
        disruption=_disruption(),
        item_lifecycle=_lifecycle(),
        stores=stores if stores is not None else _stores_no_policy(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


# --------------------------------------------------------- Scenario.from_world


def test_from_world_catalog_is_world_catalog() -> None:
    world = _world()
    scenario = Scenario.from_world(
        world,
        disruption=_disruption(),
        item_lifecycle=_lifecycle(),
        stores=_stores_no_policy(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    assert scenario.catalog is world.catalog


def test_from_world_market_is_world_market() -> None:
    world = _world()
    scenario = Scenario.from_world(
        world,
        disruption=_disruption(),
        item_lifecycle=_lifecycle(),
        stores=_stores_no_policy(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    assert scenario.market is world.market


def test_from_world_forwards_all_kwargs() -> None:
    world = _world()
    disruption = _disruption()
    lifecycle = _lifecycle()
    stores = _stores_no_policy()
    start = datetime(2025, 6, 1)
    scenario = Scenario.from_world(
        world,
        disruption=disruption,
        item_lifecycle=lifecycle,
        stores=stores,
        n_steps=99,
        start_date=start,
        world_seed=7,
    )
    assert scenario.disruption is disruption
    assert scenario.item_lifecycle is lifecycle
    assert scenario.stores is stores
    assert scenario.n_steps == 99
    assert scenario.start_date == start
    assert scenario.world_seed == 7


# ------------------------------------------------------------ catalog_df


_CATALOG_DF_COLUMNS = {
    "product_id",
    "name",
    "category",
    "base_price",
    "unit_cost",
    "margin",
    "seasonality",
    "freshness_alpha",
    "freshness_decay",
    "init_stage",
    "stage_change_probs",
    "init_stock_share",
    "related_products",
}


def test_catalog_df_columns_and_row_count() -> None:
    df = _scenario().catalog_df()
    assert set(df.columns) == _CATALOG_DF_COLUMNS
    assert len(df) == 2


def test_catalog_df_margin_is_derived() -> None:
    df = _scenario().catalog_df()
    for _, row in df.iterrows():
        assert row["margin"] == pytest.approx(row["base_price"] - row["unit_cost"])


def test_catalog_df_columns_match_world_catalog_df() -> None:
    world = _world()
    scenario = Scenario.from_world(
        world,
        disruption=_disruption(),
        item_lifecycle=_lifecycle(),
        stores=_stores_no_policy(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    assert set(scenario.catalog_df().columns) == set(world.catalog_df().columns)


# ------------------------------------------------------------ stores_df


_STORES_DF_COLUMNS = {"store_id", "template_id", "region", "init_seed", "policy_class"}


def test_stores_df_columns_and_row_count() -> None:
    df = _scenario(stores=_stores_with_policy()).stores_df()
    assert set(df.columns) == _STORES_DF_COLUMNS
    assert len(df) == 2


def test_stores_df_policy_class_populated_when_policy_attached() -> None:
    df = _scenario(stores=_stores_with_policy()).stores_df()
    assert list(df["policy_class"]) == ["StubPolicy", "StubPolicy"]


def test_stores_df_policy_class_none_when_no_policy() -> None:
    df = _scenario(stores=_stores_no_policy()).stores_df()
    assert df["policy_class"].iloc[0] is None


def test_stores_df_policy_class_none_after_from_json_round_trip() -> None:
    scenario = _scenario(stores=_stores_with_policy())
    restored = Scenario.from_json(scenario.to_json())
    df = restored.stores_df()
    assert all(v is None for v in df["policy_class"])


# ------------------------------------------------------------ market_df


def test_market_df_single_row_and_columns_match_world() -> None:
    world = _world()
    scenario = Scenario.from_world(
        world,
        disruption=_disruption(),
        item_lifecycle=_lifecycle(),
        stores=_stores_no_policy(),
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )
    mdf = scenario.market_df()
    assert len(mdf) == 1
    assert set(mdf.columns) == set(world.market_df().columns)


# --------------------------------------------------------- disruption_df


def test_disruption_df_single_row_with_required_columns() -> None:
    df = _scenario().disruption_df()
    assert len(df) == 1
    assert {"event_prob", "types", "regions", "severity", "duration"}.issubset(
        set(df.columns)
    )


# ---------------------------------------------------------- lifecycle_df


def test_lifecycle_df_single_row_with_required_columns() -> None:
    df = _scenario().lifecycle_df()
    assert len(df) == 1
    assert {
        "stages",
        "init_stage",
        "default_stage_change_probs",
        "default_freshness_alpha",
        "default_freshness_decay",
        "default_init_stock_share",
    }.issubset(set(df.columns))


# ------------------------------------------------------------ summary_df


def test_summary_df_single_row_five_columns() -> None:
    scenario = _scenario(stores=_stores_with_policy())
    df = scenario.summary_df()
    assert len(df) == 1
    assert set(df.columns) == {
        "n_steps",
        "start_date",
        "world_seed",
        "n_stores",
        "n_products",
    }


def test_summary_df_values_match_scenario() -> None:
    stores = _stores_with_policy()
    scenario = _scenario(stores=stores)
    df = scenario.summary_df()
    row = df.iloc[0]
    assert row["n_steps"] == scenario.n_steps
    assert row["start_date"] == scenario.start_date
    assert row["world_seed"] == scenario.world_seed
    assert row["n_stores"] == len(scenario.stores)
    assert row["n_products"] == len(scenario.catalog)


# ----------------------------------------- all methods work after from_json


def test_all_df_methods_work_after_from_json_round_trip() -> None:
    scenario = _scenario(stores=_stores_with_policy())
    restored = Scenario.from_json(scenario.to_json())

    catalog_df = restored.catalog_df()
    assert len(catalog_df) == 2
    assert "margin" in catalog_df.columns

    stores_df = restored.stores_df()
    assert len(stores_df) == 2
    assert all(v is None for v in stores_df["policy_class"])

    market_df = restored.market_df()
    assert len(market_df) == 1

    disruption_df = restored.disruption_df()
    assert len(disruption_df) == 1

    lifecycle_df = restored.lifecycle_df()
    assert len(lifecycle_df) == 1

    summary_df = restored.summary_df()
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["n_products"] == 2
