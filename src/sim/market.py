"""Typed ``Market`` (issue 04).

Owns the shared regional environment: per-region demand and supply
state, seasonal cycle, and trend drift. ``tick()`` mutates that state
once per simulation step; ``sample_demand(product_id, store, price)``
draws realised demand for one product in one store at the given price.

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
    """Sample a ``Distribution`` value once, or pass through a scalar."""
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
        # Keep the typed params for downstream sampling (shocks, trend).
        self.params = params
        # Shared world RNG. Every shock / trend draw goes through this.
        self.rng = world_rng
        # Optional item registry. ``sample_demand`` requires one;
        # ``cross_demand_factor`` short-circuits to 1.0 without one.
        self.registry = registry
        # Late-bound list of stores (filled in via ``add_store`` if a
        # caller wants Market-side store iteration; the Runner doesn't
        # use this hook today).
        self.stores: list[Any] = []
        # Simulation step counter, advanced once per ``tick``.
        self.step = 0
        # Wall-clock date — advances one day per tick.
        self.date = start_date

        # Region keys (copied so external mutation of params doesn't
        # leak into Market behaviour).
        self.regions: list[str] = list(params.regions)
        # Per-region mutable state — the demand/supply update mutates
        # these dicts each tick.
        self.market_state: dict[str, dict[str, float]] = {
            r: {
                "market_demand": float(params.init_demand),
                "market_supply": float(params.init_supply),
            }
            for r in self.regions
        }

        # Cycle length / amplitude for the deterministic seasonal sin
        # wave layered on top of the stochastic demand path.
        self.wave_len = float(params.cycle_len)
        self.wave_amp = float(params.cycle_amp)
        # Correlation between demand and supply shocks (used in
        # ``update_market`` to make supply track demand partially).
        self.correlation = float(params.correlation)
        # How often the trend factor is re-sampled.
        self.trend_update_interval = int(params.trend_update_interval)
        # Clamp range for demand/supply state.
        self.min_value = float(params.min_value)
        self.max_value = float(params.max_value)

        # Lifecycle-stage demand multipliers (copy so external edits
        # don't change Market behaviour mid-run).
        self.stage_multipliers = dict(params.stage_multipliers)
        # Price-elasticity exponent (expected to be negative).
        self.price_elasticity = float(params.price_elasticity)
        # Demand multiplier applied when a product is on promotion.
        self.promo_multiplier = float(params.promo_multiplier)
        # Lower/upper inventory-ratio thresholds for the cross-product
        # demand adjustment (see ``cross_demand_factor``).
        self.cross_inv_lo = float(params.cross_inv_lo)
        self.cross_inv_hi = float(params.cross_inv_hi)
        # Bounds for the final cross-product factor — clamps prevent
        # cascading correlations from producing degenerate values.
        self.cross_factor_range = tuple(params.cross_factor_range)
        # Peak/off-season multipliers. ``Distribution``-valued
        # peak/off factors are sampled once at construction (preserving
        # the original behaviour); scalars passthrough.
        self.peak_factor = float(_maybe_sample(params.peak_factor, world_rng)) \
            if isinstance(params.peak_factor, Distribution) else float(params.peak_factor)
        self.off_factor = float(_maybe_sample(params.off_factor, world_rng)) \
            if isinstance(params.off_factor, Distribution) else float(params.off_factor)
        # Per-season month map: ``{label: [month1, month2, …]}``.
        self.season_months = dict(params.season_months)

        # One trend draw at construction, mirroring the original. The
        # subsequent re-samples happen inside ``update_market``.
        self.trend = float(params.trend.sample(world_rng))

    def add_store(self, store) -> None:
        """Register a ``Store`` for Market-side iteration (currently unused)."""
        self.stores.append(store)

    def tick(self) -> None:
        """Advance the market one step; mutates state, step counter, and date."""
        self.update_market()
        # Step / date increment AFTER the update so ``current_step()``
        # returns ``0`` during the very first ``update_market`` call —
        # matching the verbatim port from the deleted module.
        self.step += 1
        self.date += timedelta(days=1)

    def update_market(self) -> None:
        """Update each region's demand/supply with trend, shocks, and cycle.

        RNG draw order per step (preserved verbatim): for each region,
        ``demand_shock`` then ``supply_shock``; then, if
        ``step % trend_update_interval == 0``, one ``trend`` draw at the
        end. Reordering breaks CRN identity for fixed ``world_seed`` runs.
        """
        # Deterministic seasonal component, shared across all regions.
        cycle = math.sin(2 * math.pi * self.step / self.wave_len)
        for region in self.regions:
            # Stochastic demand shock for this region.
            d_shock = float(self.params.demand_shock.sample(self.rng))
            cycle_effect = self.wave_amp * cycle
            trend_effect = self.trend
            new_demand = (
                self.market_state[region]["market_demand"] * trend_effect
                + d_shock
                + cycle_effect
            )
            # Clamp to [min_value, max_value] so successive shocks
            # can't drive the demand series to extremes.
            self.market_state[region]["market_demand"] = max(
                self.min_value, min(self.max_value, new_demand)
            )

            # Supply shock + correlation pickup from the demand shock —
            # this is what keeps supply broadly tracking demand.
            s_shock = float(self.params.supply_shock.sample(self.rng))
            corr_effect = self.correlation * d_shock
            new_supply = (
                self.market_state[region]["market_supply"] + s_shock + corr_effect
            )
            self.market_state[region]["market_supply"] = max(
                self.min_value, min(self.max_value, new_supply)
            )

        # Trend is re-drawn at fixed intervals, including step 0 of the
        # first tick. Keep this *after* the per-region updates so its
        # draw lands in the same position in the RNG stream as before.
        if self.step % self.trend_update_interval == 0:
            self.trend = float(self.params.trend.sample(self.rng))

    def state(self) -> dict[str, dict[str, float]]:
        """Return the (mutable) per-region demand/supply state dict."""
        return self.market_state

    def current_step(self) -> int:
        """Return the current simulation step (0-based)."""
        return self.step

    def current_date(self) -> datetime:
        """Return the current wall-clock datetime."""
        return self.date

    def sample_base_demand(self) -> float:
        """Draw one raw ``base_demand`` value.

        Used inside ``sample_demand`` to seed the multiplier stack.
        Each call consumes exactly one ``world_rng`` draw — load-bearing
        for the "one demand draw per (store, product) per tick"
        CRN contract.
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

        # Base draw — one ``world_rng`` consumption per (store, product, tick).
        base = self.sample_base_demand()
        # Global PLC stage for this product.
        stage = self.registry.stage(product_id)

        # Region-level demand factor — ``market_demand`` is already on
        # the 0–2 scale, so it's consumed directly as a factor.
        # The clamp prevents a depressed market from collapsing demand
        # entirely.
        demand_factor = max(
            self.params.demand_factor_min,
            self.market_state[store.region]["market_demand"],
        )
        # 1. Lifecycle stage multiplier (e.g. growth ≫ decline).
        multiplier = self.stage_multipliers[stage]

        # 2. Per-(store, product) freshness factor. ``Store`` returns 1.0
        # for products that have never been activated, so the canonical
        # CRN contract (one world_rng draw per inventory key) is
        # unaffected by which products are active.
        multiplier *= store.freshness_multiplier(product_id, current_step)

        # 3. Seasonal factor — peak vs. off depending on month / season.
        month = self.date.month
        season = self.registry.seasonality(product_id)
        multiplier *= self.season_factor(season, month)

        # 4. Promotion boost (only if the store has an active promo on this pid).
        if hasattr(store, "promotions") and product_id in store.promotions:
            multiplier *= 1 + store.promotions[product_id]["discount"] * self.promo_multiplier

        # 5. Cross-product factor — driven by stock levels of related items.
        multiplier *= self.cross_demand_factor(product_id, store)

        # Price elasticity: ``(price / base_price) ** elasticity`` where
        # ``elasticity`` is negative ⇒ higher price ⇒ lower factor.
        base_price = self.registry.items[product_id].base_price
        price_factor = (price / base_price) ** self.price_elasticity

        # Floor at zero (the multiplier stack can produce small negatives
        # under extreme shocks) and integer-round for the unit count.
        return max(0, int(base * multiplier * demand_factor * price_factor))

    def cross_demand_factor(self, product_id: str, store) -> float:
        """Verbatim port of ``Market.cross_demand_factor``.

        Boosts demand when *related* products are under-stocked and
        suppresses it when they are over-stocked, on the heuristic that
        an unbalanced complementary assortment shifts shopper attention.
        """
        if self.registry is None:
            return 1.0
        related = self.registry.related(product_id)
        # Multiplicative running factor (starts at 1 ⇒ no effect).
        factor = 1.0
        for rel_id, corr in related:
            if rel_id in store.active_items:
                # Per-related-product inventory share of the store's
                # *per-active-SKU* capacity slice.
                rel_stock = store.inventory.get(rel_id, 0)
                rel_cap = store.capacity / len(store.active_items)
                ratio = rel_stock / rel_cap

                if ratio < self.cross_inv_lo:
                    # Related product under-stocked ⇒ boost demand for ``product_id``.
                    factor += corr * (1 - ratio / self.cross_inv_lo)
                elif ratio > self.cross_inv_hi:
                    # Related product over-stocked ⇒ suppress demand for ``product_id``.
                    factor -= corr * (ratio - self.cross_inv_hi) / (1 - self.cross_inv_hi)

        # Clamp inside the authored range to prevent cascade explosions.
        return max(self.cross_factor_range[0], min(factor, self.cross_factor_range[1]))

    def season_factor(self, season: str | None, month: int) -> float:
        """Return ``peak_factor`` iff month ∈ ``season_months[season]``, else ``off_factor``."""
        # ``season_months.get(season, [])`` keeps unknown labels safe —
        # they collapse to off-season instead of raising.
        if month in self.season_months.get(season, []):
            return self.peak_factor
        return self.off_factor

    def demand_multiplier(
        self,
        pid: str,
        region: str,
        tick: int,
    ) -> float:
        """Return the combined demand multiplier for a product in a region at a tick.

        Bundles the three market-level factors without drawing from ``world_rng``:

        1. **Seasonal factor** — ``peak_factor`` or ``off_factor`` depending on
           the product's season and the current month.  Derived from the internal
           date (already advanced by ``tick_world`` before this is called).
        2. **Regional demand factor** — the region's current ``market_demand``
           state, floored at ``demand_factor_min``.
        3. **Active disruption multiplier** — disruption shocks are applied by
           ``EventEngine`` directly to ``market_state`` so they are already
           folded into the regional demand factor above.

        This method is the ``Market`` surface called by ``DemandSinkNode``
        demand-target composition (ADR 0015).  It does *not* consume
        ``world_rng``; all stochastic state was already advanced by
        ``market.tick()``.

        Parameters
        ----------
        pid:
            Product ID — used to look up the seasonal label from
            ``ItemRegistry``.
        region:
            Region key; must be present in ``self.market_state``.
        tick:
            Current simulation tick.  Not consumed in the current
            implementation but accepted for forward compatibility.

        Returns
        -------
        float
            A ≥ 0 multiplier.  Values < 1 indicate suppressed demand;
            values > 1 indicate elevated demand.
        """
        if self.registry is None:
            raise RuntimeError(
                "Market.demand_multiplier requires an attached ItemRegistry"
            )

        # 1. Seasonal factor driven by the current wall-clock month.
        season = self.registry.seasonality(pid)
        month = self.date.month
        seasonal = self.season_factor(season, month)

        # 2. Regional demand factor — current market_demand for this region.
        regional = max(
            self.params.demand_factor_min,
            self.market_state[region]["market_demand"],
        )

        return seasonal * regional


__all__ = ["Market"]
