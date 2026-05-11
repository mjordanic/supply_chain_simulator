"""T6: Market invariants (issue 04).

Pins the math-fidelity contract for ``Market``:

- ``market_demand`` and ``market_supply`` stay in
  ``[min_value, max_value]`` even under extreme shock distributions.
- ``trend`` updates only at ``trend_update_interval`` boundaries.
- ``season_factor`` returns ``peak_factor`` iff
  ``month ∈ season_months[season]``.
- ``cross_demand_factor`` is clamped to ``cross_factor_range``.
"""

from __future__ import annotations

from datetime import datetime
from random import Random
from types import SimpleNamespace

import pytest

from src.sim.distributions import Constant, Normal
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.scenario import (
    ItemLifecycleParams,
    MarketParams,
    Ware,
    load_catalog,
)


def _market_params(**overrides) -> MarketParams:
    defaults: dict = dict(
        cycle_len=365,
        cycle_amp=0.001,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.5,
        off_factor=0.7,
        season_months={"summer": [6, 7, 8], "winter": [12, 1, 2]},
        regions=["US", "EU"],
        correlation=0.5,
        trend_update_interval=10,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={"introduction": 0.7, "growth": 1.5, "maturity": 1.0, "decline": 0.2},
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.01),
        supply_shock=Normal(0.0, 0.01),
        base_demand=Constant(50),
    )
    defaults.update(overrides)
    return MarketParams(**defaults)


def _build_market(**overrides) -> Market:
    params = _market_params(**overrides)
    return Market(params, Random(123), datetime(2024, 1, 1))


def test_market_demand_supply_clamped_under_extreme_shocks():
    """Even with shock distributions far outside ``[min_value, max_value]``,
    the clamp keeps both series inside bounds at every step."""
    params = _market_params(
        # Wildly bipolar shocks: standard deviation 10 will pull values to
        # ±thousands of the 0–2 percent scale without the clamp.
        demand_shock=Normal(0.0, 10.0),
        supply_shock=Normal(0.0, 10.0),
        min_value=0.2,
        max_value=2.0,
    )
    market = Market(params, Random(7), datetime(2024, 1, 1))

    for _ in range(200):
        market.tick()
        for region in market.regions:
            assert market.min_value <= market.market_state[region]["market_demand"] <= market.max_value
            assert market.min_value <= market.market_state[region]["market_supply"] <= market.max_value


def test_trend_updates_only_at_interval():
    """``trend`` is sampled at construction, then re-sampled only when
    ``step % trend_update_interval == 0`` inside ``update_market``."""

    class _CountingTrend:
        """Distribution-shaped fake; counts ``sample`` calls and emits 1.0 + n."""

        def __init__(self) -> None:
            self.calls = 0

        def sample(self, rng: Random) -> float:
            self.calls += 1
            return 1.0 + 0.001 * self.calls  # tiny so clamp doesn't fire

    counting = _CountingTrend()
    params = _market_params(trend=counting, trend_update_interval=10)
    market = Market(params, Random(1), datetime(2024, 1, 1))

    # Construction draws once.
    assert counting.calls == 1
    assert market.trend == pytest.approx(1.001)

    trend_after_construction = market.trend
    # Tick 1 → step is 0 going into update_market; 0 % 10 == 0 → resample.
    market.tick()
    assert counting.calls == 2
    new_trend_step0 = market.trend
    assert new_trend_step0 != trend_after_construction

    # Ticks 2..10 → step 1..9 entering update_market; none divisible by 10.
    snapshot = market.trend
    for _ in range(9):
        market.tick()
        assert market.trend == snapshot
    assert counting.calls == 2

    # Tick 11 → entering update_market with self.step == 10 → resample.
    market.tick()
    assert counting.calls == 3
    assert market.trend != snapshot


def test_season_factor_peak_iff_month_in_season_months():
    market = _build_market()
    # season_months: summer = {6,7,8}; winter = {12,1,2}
    for month in range(1, 13):
        assert market.season_factor("summer", month) == (
            market.peak_factor if month in (6, 7, 8) else market.off_factor
        )
        assert market.season_factor("winter", month) == (
            market.peak_factor if month in (12, 1, 2) else market.off_factor
        )
    # Unknown season → empty list → always off_factor.
    for month in range(1, 13):
        assert market.season_factor("never_listed", month) == market.off_factor
    # season=None falls through ``dict.get(None, []) → []`` → always off.
    for month in range(1, 13):
        assert market.season_factor(None, month) == market.off_factor


def _store_stub(active_items: list[str], inventory: dict[str, int], capacity: float):
    return SimpleNamespace(
        active_items=active_items,
        inventory=inventory,
        capacity=capacity,
        region="US",
    )


def _registry_with_related(related_map: dict[str, list[tuple[str, float]]]) -> ItemRegistry:
    """Build an ``ItemRegistry`` whose items carry the given related_products."""
    catalog = load_catalog(
        [
            {
                "name": pid,
                "category": "test",
                "related_products": related_map.get(f"P{i:04d}", []),
                "base_price": 10.0,
                "unit_cost": 5.0,
                "seasonality": "all_season",
            }
            for i, pid in enumerate(related_map)
        ]
    )
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )
    return ItemRegistry(lifecycle, catalog, Random(0))


def test_cross_demand_factor_clamped_to_range():
    """A pathological ``corr`` should not push the factor outside
    ``cross_factor_range``; clamp is the last operation."""
    # P0000 has one related (P0001) with a huge +corr — empty stock should
    # push the raw factor toward +∞, the clamp should pull it to the upper
    # bound. Same product run with full stock should hit the lower bound.
    related_map = {
        "P0000": [("P0001", 100.0)],
        "P0001": [],
    }
    registry = _registry_with_related(related_map)
    params = _market_params(cross_factor_range=(0.3, 1.6))
    market = Market(params, Random(1), datetime(2024, 1, 1), registry=registry)

    # Empty related stock → ratio=0, raw factor jumps massively; clamp upper.
    store_empty = _store_stub(
        active_items=["P0000", "P0001"],
        inventory={"P0000": 0, "P0001": 0},
        capacity=200.0,
    )
    factor_low_stock = market.cross_demand_factor("P0000", store_empty)
    assert market.params.cross_factor_range[0] <= factor_low_stock <= market.params.cross_factor_range[1]
    assert factor_low_stock == market.params.cross_factor_range[1]

    # Saturated related stock → ratio=1.0, raw factor sinks; clamp lower.
    store_full = _store_stub(
        active_items=["P0000", "P0001"],
        inventory={"P0000": 100, "P0001": 100},
        capacity=200.0,
    )
    factor_high_stock = market.cross_demand_factor("P0000", store_full)
    assert market.params.cross_factor_range[0] <= factor_high_stock <= market.params.cross_factor_range[1]
    assert factor_high_stock == market.params.cross_factor_range[0]


def test_cross_demand_factor_no_related_returns_one():
    """No related products → factor stays at the neutral 1.0."""
    registry = _registry_with_related({"P0000": []})
    params = _market_params(cross_factor_range=(0.3, 1.6))
    market = Market(params, Random(1), datetime(2024, 1, 1), registry=registry)

    store = _store_stub(active_items=["P0000"], inventory={"P0000": 50}, capacity=100.0)
    assert market.cross_demand_factor("P0000", store) == 1.0
