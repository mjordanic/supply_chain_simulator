"""Typed ``ItemRegistry``.

Per-item lifecycle state. Distribution-typed fields on
``ItemLifecycleParams`` (and per-``Ware`` overrides) are resolved against
``world_rng`` once at construction time; the cached scalar values feed
``LifecycleClock.advance_stage`` on every tick.

Per-``Ware`` overrides (issue 02): ``Ware.init_stage`` and
``Ware.stage_change_probs`` win over the corresponding
``ItemLifecycleParams`` defaults when set; ``None`` falls back. The
override dict may be partial — stages absent from the dict default to
``0.0`` per ``LifecycleClock``'s contract.

Per-``Ware`` freshness overrides (issue 04): ``Ware.freshness_alpha``
and ``Ware.freshness_decay`` win over the catalog-wide
``default_freshness_*`` values when set; ``None`` falls back to the
already-resolved scalar default. Resolution happens once at
construction so ``Store.freshness_multiplier`` can look up per-product
``(α, β)`` without further sampling.

Per-``Ware`` initial-stock weights (issue 08): ``Ware.init_stock_share``
wins over ``ItemLifecycleParams.default_init_stock_share`` when set;
``None`` falls back to the already-resolved scalar default. Same
resolve-once-at-construction discipline as freshness — the weights
feed ``StoreInitializer.init_store_state`` so every store sharing the
catalog allocates from the same per-product weights.
"""

from __future__ import annotations

from random import Random
from typing import Any

from src.sim import lifecycle_clock
from src.sim.distributions import Distribution
from src.sim.scenario import ItemLifecycleParams, Ware


def _maybe_sample(value: Any, rng: Random) -> Any:
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


def _resolve_stage_change_probs(
    override: dict[str, Any] | None,
    default: dict[str, Any],
    rng: Random,
) -> dict[str, float]:
    """Pick override-or-default and sample any Distribution-valued entries."""
    source = override if override is not None else default
    return {stage: float(_maybe_sample(prob, rng)) for stage, prob in source.items()}


class Item:
    """One product's mutable lifecycle state."""

    def __init__(
        self,
        product_id: str,
        name: str,
        category: str,
        related_products: list[tuple[str, float]],
        base_price: float,
        unit_cost: float,
        lifecycle_stage: str,
        stage_change_probs: dict[str, float],
        freshness_alpha: float,
        freshness_decay: float,
        init_stock_share: float,
        seasonality: str = "all_season",
    ) -> None:
        self.product_id = product_id
        self.name = name
        self.category = category
        self.related_products = related_products
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.lifecycle_stage = lifecycle_stage
        self.stage_change_probs = stage_change_probs
        self.freshness_alpha = freshness_alpha
        self.freshness_decay = freshness_decay
        self.init_stock_share = init_stock_share
        self.seasonality = seasonality


class ItemRegistry:
    """Catalog of ``Item``s indexed by ``product_id``."""

    def __init__(
        self,
        params: ItemLifecycleParams,
        catalog: list[Ware],
        world_rng: Random,
    ) -> None:
        self.params = params
        self.rng = world_rng
        # Catalog-wide freshness defaults are resolved once. Sampling
        # order: before the per-Ware loop, so downstream world_rng
        # consumption is the same whether or not a Ware sets a per-item
        # override (issue 04).
        self.default_freshness_alpha = float(
            _maybe_sample(params.default_freshness_alpha, world_rng)
        )
        self.default_freshness_decay = float(
            _maybe_sample(params.default_freshness_decay, world_rng)
        )
        # Catalog-wide initial-stock weight resolved alongside the
        # freshness defaults (issue 08). Sampled before the per-Ware loop
        # so per-Ware overrides resolve against the already-cached scalar
        # default. ``world_rng`` consumes a draw iff the value is a
        # ``Distribution``; scalars (the default 1.0 included) are no-op.
        self.default_init_stock_share = float(
            _maybe_sample(params.default_init_stock_share, world_rng)
        )
        self.items: dict[str, Item] = {}
        for w in catalog:
            init_stage_source = (
                w.init_stage if w.init_stage is not None else params.init_stage
            )
            init_stage = _maybe_sample(init_stage_source, world_rng)
            stage_change_probs = _resolve_stage_change_probs(
                w.stage_change_probs,
                params.default_stage_change_probs,
                world_rng,
            )
            # Per-Ware freshness override or already-resolved catalog
            # scalar. ``None`` ⇒ no extra world_rng draw is consumed for
            # this Ware (the catalog default is reused as a scalar);
            # ``Distribution`` ⇒ one draw per overriding Ware.
            alpha_source = (
                w.freshness_alpha
                if w.freshness_alpha is not None
                else self.default_freshness_alpha
            )
            freshness_alpha = float(_maybe_sample(alpha_source, world_rng))
            decay_source = (
                w.freshness_decay
                if w.freshness_decay is not None
                else self.default_freshness_decay
            )
            freshness_decay = float(_maybe_sample(decay_source, world_rng))
            # Per-Ware init_stock_share override (issue 08) or already-
            # resolved catalog scalar. ``None`` ⇒ no extra world_rng draw
            # is consumed for this Ware (the catalog default is reused as
            # a scalar); ``Distribution`` ⇒ one draw per overriding Ware.
            share_source = (
                w.init_stock_share
                if w.init_stock_share is not None
                else self.default_init_stock_share
            )
            init_stock_share = float(_maybe_sample(share_source, world_rng))
            self.items[w.product_id] = Item(
                product_id=w.product_id,
                name=w.name,
                category=w.category,
                related_products=list(w.related_products),
                base_price=w.base_price,
                unit_cost=w.unit_cost,
                lifecycle_stage=init_stage,
                stage_change_probs=stage_change_probs,
                freshness_alpha=freshness_alpha,
                freshness_decay=freshness_decay,
                init_stock_share=init_stock_share,
                seasonality=w.seasonality,
            )

    def tick(self) -> None:
        """Advance every item one cyclic step via ``LifecycleClock``.

        One ``world_rng`` draw per item per tick — load-bearing for CRN.
        """
        for item in self.items.values():
            item.lifecycle_stage = lifecycle_clock.advance_stage(
                self.rng, item.lifecycle_stage, item.stage_change_probs
            )

    def stage(self, product_id: str) -> str | None:
        item = self.items.get(product_id)
        return item.lifecycle_stage if item else None

    def seasonality(self, product_id: str) -> str | None:
        item = self.items.get(product_id)
        return item.seasonality if item else None

    def related(self, product_id: str) -> list[tuple[str, float]]:
        item = self.items.get(product_id)
        return list(item.related_products) if item else []

    def freshness_alpha(self, product_id: str) -> float:
        """Resolved per-product hype peak ``α``. Falls back to the catalog default."""
        item = self.items.get(product_id)
        return item.freshness_alpha if item else self.default_freshness_alpha

    def freshness_decay(self, product_id: str) -> float:
        """Resolved per-product hype decay constant ``β``."""
        item = self.items.get(product_id)
        return item.freshness_decay if item else self.default_freshness_decay

    def init_stock_share(self, product_id: str) -> float:
        """Resolved per-product initial-stock weight (issue 08).

        ``StoreInitializer.init_store_state`` normalises these weights
        across the active SKU set to allocate the
        ``capacity * init_stock_pct`` budget proportionally.
        """
        item = self.items.get(product_id)
        return item.init_stock_share if item else self.default_init_stock_share


__all__ = ["Item", "ItemRegistry"]
