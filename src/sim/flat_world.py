"""Flat-world authoring helper (issue 01).

Produces ``MarketParams`` and ``ItemLifecycleParams`` where the full ADR 0015
demand-multiplier chain is exactly 1.0 on every tick for every product and
region, enabling pure demand replay:

    demand_target = demand_dist.sample(world_rng) * market.demand_multiplier(pid, region)

``market.demand_multiplier`` = ``seasonal * regional`` where:

- ``seasonal`` = ``peak_factor`` or ``off_factor`` (whichever season applies).
  Set both to 1.0 so seasonal is always 1.0.
- ``regional`` = ``max(demand_factor_min, market_state[region]["market_demand"])``.
  Set ``init_demand = 1.0``, ``cycle_amp = 0``, ``demand_shock = Constant(0)``,
  ``trend = Constant(1.0)`` (multiplicative) so ``market_demand`` stays at 1.0
  every tick. Set ``demand_factor_min = 0.0`` so ``max(0.0, 1.0) = 1.0``.

Result: ``market.demand_multiplier(pid, region) == 1.0`` — bit-exact — for all
(pid, region) on every tick.

Usage::

    market_params, lifecycle_params = flat_world(regions=["US"])
"""

from __future__ import annotations

from src.sim.distributions import Constant
from src.sim.scenario import DisruptionParams, ItemLifecycleParams, MarketParams


def flat_world(
    regions: list[str],
) -> tuple[MarketParams, ItemLifecycleParams]:
    """Return ``(MarketParams, ItemLifecycleParams)`` whose multiplier chain is identically 1.0.

    Parameters
    ----------
    regions:
        Region keys — must match the regions used by the scenario nodes.

    Returns
    -------
    tuple[MarketParams, ItemLifecycleParams]
        Both objects are ordinary, immutable parameter bags.  Wire them
        into a ``Scenario`` exactly as you would any other params.
    """
    market = MarketParams(
        # Seasonal: peak and off both 1.0 → seasonal factor always 1.0.
        peak_factor=1.0,
        off_factor=1.0,
        season_months={},  # empty → every month is "off-season" (1.0)
        # Regional demand: init at 1.0, no drift, no shocks, no cycle.
        init_demand=1.0,
        init_supply=1.0,
        cycle_len=365,  # value irrelevant when amplitude is 0
        cycle_amp=0.0,
        trend=Constant(1.0),     # multiplicative; keeps demand at 1.0
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        # demand_factor_min = 0 so max(0, 1.0) = 1.0 (no clamping away from 1)
        demand_factor_min=0.0,
        supply_factor_min=0.0,
        # Clamp range wide enough that the stable 1.0 value never hits them.
        min_value=0.0,
        max_value=10.0,
        correlation=0.0,
        trend_update_interval=1,
        regions=list(regions),
        # Price/promo/cross effects: neutral/identity values.
        price_elasticity=0.0,
        promo_multiplier=1.0,
        cross_inv_lo=0.0,
        cross_inv_hi=1.0,
        cross_factor_range=(1.0, 1.0),
        # Stage multipliers: all 1.0 (currently not applied by the engine,
        # set for completeness and forward-compatibility).
        stage_multipliers={
            "introduction": 1.0,
            "growth": 1.0,
            "maturity": 1.0,
            "decline": 1.0,
            "dead": 1.0,
        },
        # base_demand is not part of the multiplier chain; set a stable default.
        base_demand=Constant(1.0),
    )

    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
        # freshness_alpha = 0 → freshness curve is the no-op identity (ADR 0015).
        default_freshness_alpha=0.0,
        default_freshness_decay=1.0,
        default_init_stock_share=1.0,
    )

    return market, lifecycle


__all__ = ["flat_world"]
