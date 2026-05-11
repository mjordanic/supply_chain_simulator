"""T6: EventEngine invariants (issue 04)."""

from __future__ import annotations

from datetime import datetime
from random import Random
from types import SimpleNamespace

from src.sim.distributions import Constant
from src.sim.event_engine import EventEngine, WorldEvent
from src.sim.market import Market
from src.sim.scenario import DisruptionParams, MarketParams


def _market() -> Market:
    params = MarketParams(
        cycle_len=365,
        cycle_amp=0.001,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.2,
        off_factor=0.7,
        season_months={"all": list(range(1, 13))},
        regions=["US", "EU"],
        correlation=0.5,
        trend_update_interval=10,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={"maturity": 1.0},
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(50),
    )
    return Market(params, Random(1), datetime(2024, 1, 1))


def test_no_event_when_event_prob_zero():
    """``event_prob = 0`` → ``tick`` returns an empty active list every
    step and ``active`` stays empty."""
    params = DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US", "EU"],
        severity=Constant(1.0),
        duration=Constant(3),
    )
    engine = EventEngine(params, Random(42))
    market = _market()

    for _ in range(100):
        market.tick()
        result = engine.tick(market)
        assert result == []
    assert engine.active == []


def test_active_event_applied_each_tick_until_expiry():
    """An active event is applied to its affected regions until duration runs out."""
    market = _market()
    initial_demand = market.market_state["US"]["market_demand"]

    engine = EventEngine(
        DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=["US"],
            severity=Constant(1.0),
            duration=Constant(3),
        ),
        Random(0),
    )
    engine.active.append(
        WorldEvent(
            event_type="natural_disaster",
            severity=0.1,
            affected_regions=["US"],
            duration=3,
        )
    )

    # First three ticks apply the event; the fourth finds it expired.
    for _ in range(3):
        market.tick()
        engine.tick(market)
        assert engine.active and engine.active[0].duration > 0 or len(engine.active) == 0

    # After 3 applications the event is filtered out.
    assert engine.active == []
    # Demand was decremented by 0.1 three times, then capped at min_value if needed.
    assert market.market_state["US"]["market_demand"] < initial_demand


def test_scheduled_callback_fires_when_step_reaches_delay():
    """``schedule(callback, delay=k)`` fires when ``market.current_step() >= k``."""
    market = _market()
    engine = EventEngine(
        DisruptionParams(
            event_prob=0.0,
            types=["x"],
            regions=["US"],
            severity=Constant(1.0),
            duration=Constant(1),
        ),
        Random(0),
    )

    fired_at: list[int] = []

    def cb() -> None:
        fired_at.append(market.current_step())

    engine.schedule("delivery", delay=3, callback=cb)

    for _ in range(5):
        market.tick()
        engine.tick(market)

    assert fired_at == [3]
    assert engine.queued == []
