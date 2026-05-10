"""Typed ``ItemRegistry``.

Holds one live ``Item`` per catalog ``Ware`` plus the global lifecycle
stage for each. Distribution-typed fields on ``ItemLifecycleParams``
(and per-``Ware`` overrides) are resolved against ``world_rng`` once at
construction time; the cached scalar values feed
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
    """If ``value`` is a ``Distribution``, draw one sample; otherwise pass through."""
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


def _resolve_stage_change_probs(
    override: dict[str, Any] | None,
    default: dict[str, Any],
    rng: Random,
) -> dict[str, float]:
    """Pick override-or-default and sample any Distribution-valued entries."""
    # ``override is None`` ⇒ fall back to the catalog-wide default table.
    source = override if override is not None else default
    # ``_maybe_sample`` on each entry, in iteration order — so any
    # ``Distribution``-valued probability consumes a deterministic
    # ``world_rng`` draw at construction time.
    return {stage: float(_maybe_sample(prob, rng)) for stage, prob in source.items()}


class Item:
    """One product's mutable lifecycle state.

    Constructed once per catalog entry by ``ItemRegistry.__init__``. The
    only field that mutates after construction is ``lifecycle_stage``
    (advanced by ``ItemRegistry.tick``); everything else is fixed.
    """

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
        # Stable id assigned by ``load_catalog`` (``P0000``, ``P0001``, …).
        self.product_id = product_id
        # Display name (free-form string).
        self.name = name
        # Taxonomy category for grouping in dashboards / reports.
        self.category = category
        # Cross-product correlation graph: ``[(other_pid, weight∈[0,1])]``.
        self.related_products = related_products
        # Catalog-authored base price; ``Market`` uses it to compute the
        # price-elasticity factor in ``sample_demand``.
        self.base_price = base_price
        # Authoring cost per unit; ``Store`` charges this for orders and
        # holding, and uses it as a price floor in the policy layer.
        self.unit_cost = unit_cost
        # Current PLC stage — mutates each tick via ``advance_stage``.
        self.lifecycle_stage = lifecycle_stage
        # Per-current-stage transition probability table (already
        # resolved — no ``Distribution`` objects survive past here).
        self.stage_change_probs = stage_change_probs
        # Hype amplitude ``α`` in the freshness curve. ``0`` ⇒ staple.
        self.freshness_alpha = freshness_alpha
        # Hype decay length ``β`` in the freshness curve.
        self.freshness_decay = freshness_decay
        # Weight used by ``StoreInitializer.init_store_state`` to
        # allocate the initial-stock budget across the active set.
        self.init_stock_share = init_stock_share
        # Seasonality label (closed enum on ``Ware``; used by
        # ``Market.season_factor`` to pick peak vs. off multiplier).
        self.seasonality = seasonality


class ItemRegistry:
    """Catalog of ``Item``s indexed by ``product_id``.

    Owns the global lifecycle stage for every catalog SKU (one stage
    across all stores) and its stochastic transitions. ``tick()``
    advances every item exactly once per call using ``world_rng``.
    """

    def __init__(
        self,
        params: ItemLifecycleParams,
        catalog: list[Ware],
        world_rng: Random,
    ) -> None:
        # Keep the params dataclass for downstream lookups (mostly tests).
        self.params = params
        # The shared world RNG — every per-item Distribution sample
        # draws from this stream so paired Scenarios stay CRN-aligned.
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
        # ``product_id → Item`` index. Preserves catalog declaration
        # order because Python dicts are insertion-ordered since 3.7.
        self.items: dict[str, Item] = {}
        for w in catalog:
            # Resolve init_stage: per-Ware override beats catalog default.
            init_stage_source = (
                w.init_stage if w.init_stage is not None else params.init_stage
            )
            init_stage = _maybe_sample(init_stage_source, world_rng)
            # Resolve the per-stage transition table. ``override`` wins
            # iff explicitly set on the Ware; otherwise reuse the
            # catalog-wide default. Distribution-valued entries get
            # sampled at this point.
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
        Iteration order is catalog declaration order (dict
        insertion-ordered) so two runs with the same ``world_seed`` see
        the same draw sequence.
        """
        for item in self.items.values():
            item.lifecycle_stage = lifecycle_clock.advance_stage(
                self.rng, item.lifecycle_stage, item.stage_change_probs
            )

    def stage(self, product_id: str) -> str | None:
        """Current lifecycle stage of ``product_id``, or ``None`` if unknown."""
        item = self.items.get(product_id)
        return item.lifecycle_stage if item else None

    def seasonality(self, product_id: str) -> str | None:
        """Seasonality label of ``product_id``, or ``None`` if unknown."""
        item = self.items.get(product_id)
        return item.seasonality if item else None

    def related(self, product_id: str) -> list[tuple[str, float]]:
        """Cross-product correlation graph for ``product_id`` (copy)."""
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
