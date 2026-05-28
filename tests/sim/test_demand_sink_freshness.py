"""Tests for DemandSinkNode.demand_target — lifecycle/freshness composition.

Covers:
- CRN invariant: world_rng consumed for every catalog pid every tick in order
- Freshness composition: never-activated pid → multiplier 1.0; τ=0 → 1+α
- Stage multiplier composition: lifecycle stage factor applied correctly
- Market multiplier composition: demand_multiplier folded in
- Visible hype curve: freshness_alpha=2.0 vs 0.0 produces different series
"""

from __future__ import annotations

from datetime import datetime
from random import Random

import pytest

from src.sim.distributions import Constant, Uniform
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.node import DemandSinkNode
from src.sim.scenario import ItemLifecycleParams, MarketParams, Ware, load_catalog


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _catalog_3() -> list[Ware]:
    return load_catalog(
        [
            {"name": "Alpha", "category": "A", "related_products": [], "base_price": 10.0, "unit_cost": 5.0, "seasonality": "all_season"},
            {"name": "Beta",  "category": "B", "related_products": [], "base_price": 20.0, "unit_cost": 8.0, "seasonality": "all_season"},
            {"name": "Gamma", "category": "C", "related_products": [], "base_price": 15.0, "unit_cost": 6.0, "seasonality": "all_season"},
        ]
    )


def _lifecycle(alpha: float = 0.0, decay: float = 30.0) -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_freshness_alpha=alpha,
        default_freshness_decay=decay,
    )


def _market_params() -> MarketParams:
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=50,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.5,
            "growth": 1.5,
            "maturity": 1.0,
            "decline": 0.3,
            "dead": 0.0,
        },
        price_elasticity=0.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(100),
    )


def _make_sink(product_id: str = "P0000", demand_base: float = 100.0, *, crn_safe: bool = False) -> DemandSinkNode:
    """Build a DemandSinkNode.

    ``crn_safe=True`` uses ``Uniform(demand_base, demand_base)`` which
    consumes exactly one world_rng draw per sample — required for CRN tests.
    Default uses ``Constant(demand_base)`` which skips the draw (faster for
    determinism tests that don't care about draw count).
    """
    dist = Uniform(demand_base, demand_base) if crn_safe else Constant(demand_base)
    return DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id=product_id,
        demand_dist=dist,
        income_rate=0.0,
    )


class _DrawCountingRng(Random):
    """Counts every random() draw."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return super().random()


# ---------------------------------------------------------------------------
# CRN invariant: one world_rng draw per catalog pid per tick
# ---------------------------------------------------------------------------

def test_demand_target_consumes_one_rng_draw_per_catalog_pid():
    """demand_target draws world_rng exactly once per catalog pid per call,
    regardless of which pid is the sink's product_id.

    This is the load-bearing CRN invariant: the RNG stream position after
    demand_target must depend only on the catalog size and the number of ticks,
    never on which product the sink is bound to.

    Uses Uniform(100, 100) because Constant skips the draw (documented on
    Constant.sample and in test_freshness_integration).
    """
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=0.0)
    rng = _DrawCountingRng(42)
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)
    market.tick()  # advance to tick 1

    sink = _make_sink("P0001", crn_safe=True)
    draws_before = rng.draws
    sink.demand_target(tick=1, market=market, registry=registry, world_rng=rng)
    # Must consume exactly len(catalog) = 3 draws
    assert rng.draws - draws_before == len(catalog)


def test_crn_draw_count_independent_of_bound_product():
    """A sink bound to P0000 and a sink bound to P0002 both consume exactly
    len(catalog) draws — the bound product_id does not change draw count."""
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=0.0)

    rng_a = _DrawCountingRng(99)
    registry_a = ItemRegistry(lifecycle, catalog, rng_a)
    market_a = Market(_market_params(), rng_a, datetime(2024, 1, 1), registry=registry_a)
    market_a.tick()
    sink_a = _make_sink("P0000", crn_safe=True)
    draws_before_a = rng_a.draws
    sink_a.demand_target(tick=1, market=market_a, registry=registry_a, world_rng=rng_a)

    rng_b = _DrawCountingRng(99)
    registry_b = ItemRegistry(lifecycle, catalog, rng_b)
    market_b = Market(_market_params(), rng_b, datetime(2024, 1, 1), registry=registry_b)
    market_b.tick()
    sink_b = _make_sink("P0002", crn_safe=True)
    draws_before_b = rng_b.draws
    sink_b.demand_target(tick=1, market=market_b, registry=registry_b, world_rng=rng_b)

    assert rng_a.draws - draws_before_a == rng_b.draws - draws_before_b == len(catalog)


# ---------------------------------------------------------------------------
# Freshness composition
# ---------------------------------------------------------------------------

def test_freshness_multiplier_is_one_for_never_activated_product():
    """If activation_tick is not set for the sink's product, freshness = 1.0.

    With base_demand=100, stage_multiplier=1.0 (maturity), market_factor=1.0,
    freshness=1.0 → demand_target returns 100.
    """
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=2.0, decay=30.0)  # large α, but never activated
    rng = Random(0)
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)
    market.tick()

    sink = _make_sink("P0000", demand_base=100.0)
    # activation_tick empty → no freshness boost
    result = sink.demand_target(tick=1, market=market, registry=registry, world_rng=rng)
    assert result == 100


def test_freshness_multiplier_peaks_at_one_plus_alpha_on_activation():
    """Right after activation (τ=0), multiplier = 1 + α.

    With α=1.0, base_demand=100, maturity stage (×1.0), market factor 1.0:
    demand_target = int(100 * 1.0 * (1+1.0) * 1.0) = 200.
    """
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=1.0, decay=30.0)
    rng = Random(0)
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)
    market.tick()

    sink = _make_sink("P0000", demand_base=100.0)
    sink.activation_tick["P0000"] = 1  # activated at tick=1 → τ=0
    result = sink.demand_target(tick=1, market=market, registry=registry, world_rng=rng)
    assert result == 200  # 100 * 2.0 = 200


def test_freshness_decays_with_time():
    """Freshness multiplier decays as τ increases."""
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=1.0, decay=30.0)
    rng_a = Random(0)
    registry_a = ItemRegistry(lifecycle, catalog, rng_a)
    market_a = Market(_market_params(), rng_a, datetime(2024, 1, 1), registry=registry_a)
    for _ in range(5):
        market_a.tick()  # advance to tick 5

    rng_b = Random(0)
    registry_b = ItemRegistry(lifecycle, catalog, rng_b)
    market_b = Market(_market_params(), rng_b, datetime(2024, 1, 1), registry=registry_b)
    for _ in range(31):
        market_b.tick()  # advance to tick 31

    sink_a = _make_sink("P0000", demand_base=100.0)
    sink_a.activation_tick["P0000"] = 0  # τ = 5 at tick 5

    sink_b = _make_sink("P0000", demand_base=100.0)
    sink_b.activation_tick["P0000"] = 0  # τ = 31 at tick 31

    demand_early = sink_a.demand_target(tick=5, market=market_a, registry=registry_a, world_rng=rng_a)
    demand_late = sink_b.demand_target(tick=31, market=market_b, registry=registry_b, world_rng=rng_b)

    assert demand_early > demand_late  # more hype early than late


# ---------------------------------------------------------------------------
# Stage multiplier composition
# ---------------------------------------------------------------------------

def test_stage_multiplier_growth_greater_than_maturity():
    """Growth stage (×1.5) produces higher demand_target than maturity (×1.0).

    All other factors identical; we use α=0 to isolate lifecycle from freshness.
    """
    catalog = _catalog_3()
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    # Maturity lifecycle
    lifecycle_m = ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_freshness_alpha=0.0,
        default_freshness_decay=30.0,
    )
    # Growth lifecycle
    lifecycle_g = ItemLifecycleParams(
        stages=stages,
        init_stage="growth",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_freshness_alpha=0.0,
        default_freshness_decay=30.0,
    )

    rng_m = Random(7)
    registry_m = ItemRegistry(lifecycle_m, catalog, rng_m)
    market_m = Market(_market_params(), rng_m, datetime(2024, 1, 1), registry=registry_m)
    market_m.tick()

    rng_g = Random(7)
    registry_g = ItemRegistry(lifecycle_g, catalog, rng_g)
    market_g = Market(_market_params(), rng_g, datetime(2024, 1, 1), registry=registry_g)
    market_g.tick()

    sink_m = _make_sink("P0000", demand_base=100.0)
    sink_g = _make_sink("P0000", demand_base=100.0)

    # Use independent RNGs so draws don't interfere.
    demand_maturity = sink_m.demand_target(tick=1, market=market_m, registry=registry_m, world_rng=Random(1))
    demand_growth = sink_g.demand_target(tick=1, market=market_g, registry=registry_g, world_rng=Random(1))

    assert demand_growth > demand_maturity  # 1.5 > 1.0 stage factor


# ---------------------------------------------------------------------------
# Market multiplier composition
# ---------------------------------------------------------------------------

def test_market_multiplier_scales_demand_target():
    """Higher regional demand_multiplier produces proportionally higher demand_target."""
    catalog = _catalog_3()
    lifecycle = _lifecycle(alpha=0.0)

    # Low market demand
    mp_low = _market_params()
    mp_low.init_demand = 0.3

    # High market demand
    mp_high = _market_params()
    mp_high.init_demand = 1.8

    rng_low = Random(5)
    registry_low = ItemRegistry(lifecycle, catalog, rng_low)
    market_low = Market(mp_low, rng_low, datetime(2024, 1, 1), registry=registry_low)
    market_low.tick()

    rng_high = Random(5)
    registry_high = ItemRegistry(lifecycle, catalog, rng_high)
    market_high = Market(mp_high, rng_high, datetime(2024, 1, 1), registry=registry_high)
    market_high.tick()

    sink_low = _make_sink("P0000", demand_base=100.0)
    sink_high = _make_sink("P0000", demand_base=100.0)

    # Use same RNG seed so stochastic samples match.
    demand_low = sink_low.demand_target(tick=1, market=market_low, registry=registry_low, world_rng=Random(1))
    demand_high = sink_high.demand_target(tick=1, market=market_high, registry=registry_high, world_rng=Random(1))

    assert demand_high > demand_low


# ---------------------------------------------------------------------------
# Visible hype curve
# ---------------------------------------------------------------------------

def test_freshness_alpha_two_vs_zero_produces_different_series():
    """Running a sink with freshness_alpha=2.0 vs 0.0, with activation at tick 0,
    produces visibly different demand target series over multiple ticks.

    This is the visible-hype verification from the acceptance criteria.
    """
    catalog = _catalog_3()
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    n_ticks = 20

    def _run_series(alpha: float) -> list[int]:
        lifecycle = ItemLifecycleParams(
            stages=stages,
            init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in stages},
            default_freshness_alpha=alpha,
            default_freshness_decay=10.0,
        )
        rng = Random(42)
        registry = ItemRegistry(lifecycle, catalog, rng)
        market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)

        sink = _make_sink("P0000", demand_base=100.0)
        sink.activation_tick["P0000"] = 0  # activate at tick 0

        series: list[int] = []
        for tick in range(1, n_ticks + 1):
            market.tick()
            world_rng = Random(tick * 1000)  # independent per tick for isolation
            result = sink.demand_target(tick=tick, market=market, registry=registry, world_rng=world_rng)
            series.append(result)
        return series

    series_hype = _run_series(alpha=2.0)
    series_no_hype = _run_series(alpha=0.0)

    # Hype series should start significantly higher and decay toward baseline.
    assert series_hype[0] > series_no_hype[0]
    # Early ticks have hype; no-hype series should be flat at 100.
    assert all(d == 100 for d in series_no_hype)
    # Hype series decays over time.
    assert series_hype[0] > series_hype[-1]
