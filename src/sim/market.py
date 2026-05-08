"""Typed ``Market`` (issue 04).

Replaces the previous ``src/environment/environment.py``. The
demand/supply update math, seasonal multiplier, cross-product factor,
and ``sample_demand`` calculation are preserved verbatim. Two
construction-shape changes vs. the old module:

1. Randomness flows through an injected ``world_rng`` instead of the
   global ``random`` module.
2. ``MarketParams`` (typed dataclass with ``Distribution`` fields) plus
   ``start_date`` replace the old ``init_params`` / ``live_params``
   dict pair.

``trend`` is sampled once at ``__init__`` (preserving the original
construction-time draw) and then re-sampled in ``update_market`` when
``step % trend_update_interval == 0`` — including step 0 of the first
tick, matching the original behaviour.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from random import Random
from typing import Any

from src.sim.distributions import Distribution
from src.sim.item_registry import ItemRegistry
from src.sim.scenario import MarketParams


def _maybe_sample(value: Any, rng: Random) -> Any:
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


class Market:
    """Regional demand/supply state driven by ``MarketParams`` and ``world_rng``."""

    def __init__(
        self,
        params: MarketParams,
        world_rng: Random,
        start_date: datetime,
        registry: ItemRegistry | None = None,
    ) -> None:
        self.params = params
        self.rng = world_rng
        self.registry = registry
        self.stores: list[Any] = []
        self.step = 0
        self.date = start_date

        self.regions: list[str] = list(params.regions)
        self.market_state: dict[str, dict[str, float]] = {
            r: {
                "market_demand": float(params.init_demand),
                "market_supply": float(params.init_supply),
            }
            for r in self.regions
        }

        self.wave_len = float(params.cycle_len)
        self.wave_amp = float(params.cycle_amp)
        self.correlation = float(params.correlation)
        self.trend_update_interval = int(params.trend_update_interval)
        self.min_value = float(params.min_value)
        self.max_value = float(params.max_value)

        self.stage_multipliers = dict(params.stage_multipliers)
        self.price_elasticity = float(params.price_elasticity)
        self.promo_multiplier = float(params.promo_multiplier)
        self.cross_inv_lo = float(params.cross_inv_lo)
        self.cross_inv_hi = float(params.cross_inv_hi)
        self.cross_factor_range = tuple(params.cross_factor_range)
        self.peak_factor = float(_maybe_sample(params.peak_factor, world_rng)) \
            if isinstance(params.peak_factor, Distribution) else float(params.peak_factor)
        self.off_factor = float(_maybe_sample(params.off_factor, world_rng)) \
            if isinstance(params.off_factor, Distribution) else float(params.off_factor)
        self.season_months = dict(params.season_months)

        # One trend draw at construction, mirroring the original.
        self.trend = float(params.trend.sample(world_rng))

    def add_store(self, store) -> None:
        self.stores.append(store)

    def tick(self) -> None:
        """Advance the market one step; mutates state, step counter, and date."""
        self.update_market()
        self.step += 1
        self.date += timedelta(days=1)

    def update_market(self) -> None:
        """Update each region's demand/supply with trend, shocks, and cycle.

        RNG draw order per step (preserved verbatim): for each region,
        ``demand_shock`` then ``supply_shock``; then, if
        ``step % trend_update_interval == 0``, one ``trend`` draw at the
        end.
        """
        cycle = math.sin(2 * math.pi * self.step / self.wave_len)
        for region in self.regions:
            d_shock = float(self.params.demand_shock.sample(self.rng))
            cycle_effect = self.wave_amp * cycle
            trend_effect = self.trend
            new_demand = (
                self.market_state[region]["market_demand"] * trend_effect
                + d_shock
                + cycle_effect
            )
            self.market_state[region]["market_demand"] = max(
                self.min_value, min(self.max_value, new_demand)
            )

            s_shock = float(self.params.supply_shock.sample(self.rng))
            corr_effect = self.correlation * d_shock
            new_supply = (
                self.market_state[region]["market_supply"] + s_shock + corr_effect
            )
            self.market_state[region]["market_supply"] = max(
                self.min_value, min(self.max_value, new_supply)
            )

        if self.step % self.trend_update_interval == 0:
            self.trend = float(self.params.trend.sample(self.rng))

    def state(self) -> dict[str, dict[str, float]]:
        return self.market_state

    def current_step(self) -> int:
        return self.step

    def current_date(self) -> datetime:
        return self.date

    def sample_base_demand(self) -> float:
        """Draw one raw ``base_demand`` value (used by the runner skeleton).

        Issue 07 replaces the runner's per-step base draw with full
        ``sample_demand(product_id, store, price)`` calls; until then the
        skeletal demand trace consumes ``world_rng`` here.
        """
        return float(self.params.base_demand.sample(self.rng))

    def sample_demand(
        self,
        product_id: str,
        store,
        price: float,
        current_step: int | None = None,
    ) -> int:
        """Sample realised demand for one product in one store at the given price.

        Composes the demand multiplier in the order pinned by the CRN
        contract: ``stage * freshness * season * promo * cross``. The
        order is load-bearing for floating-point identity in
        regression tests — do not reshuffle.

        ``current_step`` defaults to ``self.step`` so call sites that
        haven't been migrated stay correct; the Runner passes it
        explicitly to keep the freshness clock aligned with the demand
        draw's tick.
        """
        if self.registry is None:
            raise RuntimeError(
                "Market.sample_demand requires an attached ItemRegistry"
            )
        if current_step is None:
            current_step = self.step

        base = self.sample_base_demand()
        stage = self.registry.stage(product_id)

        demand_factor = max(
            self.params.demand_factor_min,
            self.market_state[store.region]["market_demand"] / self.params.demand_divisor,
        )
        multiplier = self.stage_multipliers[stage]

        # Per-(store, product) freshness factor. ``Store`` returns 1.0
        # for products that have never been activated, so the canonical
        # CRN contract (one world_rng draw per inventory key) is
        # unaffected by which products are active.
        multiplier *= store.freshness_multiplier(product_id, current_step)

        month = self.date.month
        season = self.registry.seasonality(product_id)
        multiplier *= self.season_factor(season, month)

        if hasattr(store, "promotions") and product_id in store.promotions:
            multiplier *= 1 + store.promotions[product_id]["discount"] * self.promo_multiplier

        multiplier *= self.cross_demand_factor(product_id, store)

        base_price = self.registry.items[product_id].base_price
        price_factor = (price / base_price) ** self.price_elasticity

        return max(0, int(base * multiplier * demand_factor * price_factor))

    def cross_demand_factor(self, product_id: str, store) -> float:
        """Verbatim port of ``Market.cross_demand_factor``."""
        if self.registry is None:
            return 1.0
        related = self.registry.related(product_id)
        factor = 1.0
        for rel_id, corr in related:
            if rel_id in store.active_items:
                rel_stock = store.inventory.get(rel_id, 0)
                rel_cap = store.capacity / len(store.active_items)
                ratio = rel_stock / rel_cap

                if ratio < self.cross_inv_lo:
                    factor += corr * (1 - ratio / self.cross_inv_lo)
                elif ratio > self.cross_inv_hi:
                    factor -= corr * (ratio - self.cross_inv_hi) / (1 - self.cross_inv_hi)

        return max(self.cross_factor_range[0], min(factor, self.cross_factor_range[1]))

    def season_factor(self, season: str | None, month: int) -> float:
        """Return ``peak_factor`` iff month ∈ ``season_months[season]``, else ``off_factor``."""
        if month in self.season_months.get(season, []):
            return self.peak_factor
        return self.off_factor


__all__ = ["Market"]
