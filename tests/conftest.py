"""Shared fixtures for the new ``Scenario``-shaped test suite.

The old ``MINI_*`` dict fixtures (mirrors of the deleted six-section config
system) are gone; the rest of the tiered tests in ``tests/sim/`` author
``Scenario`` instances against this builder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pytest

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
)


def _default_market() -> MarketParams:
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
            "growth": 1.5,
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


def _default_disruption() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US", "EU"],
        severity=Constant(0.01),
        duration=Constant(5),
    )


def _default_item_lifecycle() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _default_template() -> StoreTemplate:
    return StoreTemplate(
        id="small",
        region="US",
        capacity=Uniform(800, 1200),
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
    )


def _default_catalog() -> list[Ware]:
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
            {
                "name": "Widget C",
                "category": "Widgets",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 8.0,
                "seasonality": "all_season",
            },
        ]
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--run-live`` so live OpenAI tests stay opt-in.

    Tests marked with ``@pytest.mark.live`` are skipped by default to keep
    ``uv run pytest`` fast, free, and runnable without an API key. Pass
    ``--run-live`` (or set ``PYTEST_ADDOPTS=--run-live``) to include them.
    """
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (hits real OpenAI API).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def make_scenario() -> Callable[..., Scenario]:
    """Tiny ``Scenario`` builder.

    Override any field by keyword. Defaults: 3-Ware mini catalog, 2 stores
    each carrying ``policy=None``, 10 steps, ``world_seed=12345``.
    """

    def _build(**overrides: Any) -> Scenario:
        template = overrides.pop("template", _default_template())
        defaults: dict[str, Any] = {
            "catalog": _default_catalog(),
            "market": _default_market(),
            "disruption": _default_disruption(),
            "item_lifecycle": _default_item_lifecycle(),
            "stores": [
                StoreInstance(template=template, init_seed=1, policy=None),
                StoreInstance(template=template, init_seed=2, policy=None),
            ],
            "n_steps": 10,
            "start_date": datetime(2024, 1, 1),
            "world_seed": 12345,
        }
        defaults.update(overrides)
        return Scenario(**defaults)

    return _build
