"""Pydantic schemas for every LLM structured-output payload.

These models are the parse-time contract between the LLM and the rest of
the simulator. Every validator in the issue's acceptance criteria lives
here so a malformed completion fails immediately and the failure can be
fed back into the next prompt by ``WorldBuilder``.

The catalog flow is split across three LLM calls so each prompt stays
narrow: ``Catalog`` carries only the per-item basics (name, category,
price, cost, seasonality); cross-product correlations live in a separate
``Correlations`` payload that takes the explicit name list from the
catalog stage as input; per-``Ware`` freshness curve params live in a
``FreshnessSet`` payload annotating each ``(name, category)`` pair
independently. Per-``Ware`` lifecycle and stock-share fields are not
authored by the LLM — ``ItemRegistry`` falls back to
``ItemLifecycleParams`` defaults — keeping each prompt small enough that
cheap models stay reliable.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Seasonality(str, Enum):
    """Closed enum for ``CatalogItem.seasonality``."""

    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    SPRING_SUMMER = "spring/summer"
    FALL_WINTER = "fall/winter"
    ALL_SEASON = "all_season"


class InitFreshness(str, Enum):
    """Closed enum for ``StoreTemplateSpec.init_freshness``."""

    BASELINE = "baseline"
    FRESH = "fresh"


class RelatedRef(BaseModel):
    """One ``related_products`` entry inside an ``ItemRelations``.

    ``correlation`` is in [0, 1] — it scales cross-product demand effects in
    ``Market.cross_demand_factor`` and is not signed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    correlation: float = Field(ge=0.0, le=1.0)


class CatalogItem(BaseModel):
    """One concrete catalog SKU before assignment of a ``P{i:04d}`` id.

    Validators:
    - ``base_price > unit_cost`` (margin must be positive)
    - ``unit_cost >= 0`` (no negative input cost)
    - ``seasonality`` is a member of the ``Seasonality`` enum
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    base_price: float = Field(gt=0)
    unit_cost: float = Field(ge=0)
    seasonality: Seasonality

    @model_validator(mode="after")
    def _price_strictly_above_cost(self) -> "CatalogItem":
        if self.base_price <= self.unit_cost:
            raise ValueError(
                f"CatalogItem {self.name!r}: base_price ({self.base_price}) "
                f"must be strictly greater than unit_cost ({self.unit_cost})"
            )
        return self


class Catalog(BaseModel):
    """A complete catalog of named SKUs (without correlations).

    Cross-product correlations live in the separate ``Correlations`` payload
    so the catalog prompt stays narrow. The empty-catalog edge case is
    blocked by ``Field(min_length=1)``.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[CatalogItem] = Field(min_length=1)


class ItemRelations(BaseModel):
    """Per-item bag of ``related_products`` references.

    ``name`` should match a ``CatalogItem.name`` from the same world; bad
    references are pruned by ``WorldBuilder.sample_catalog`` after parsing
    rather than failing the schema. The schema-side strictness was the
    sticking point that made cheap models retry-loop; the prompt-bounded
    name list plus a Python sanitiser is more robust.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    related: list[RelatedRef]


class Correlations(BaseModel):
    """Output of the dedicated correlations LLM call.

    One ``ItemRelations`` per catalog item. The candidate-name set is
    fixed before the call (see ``correlations_prompt``); the
    ``WorldBuilder`` drops any reference that doesn't resolve, plus
    self-references and duplicates.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[ItemRelations] = Field(min_length=1)


class ItemFreshness(BaseModel):
    """Per-``Ware`` freshness curve parameters authored by the LLM.

    Distinct from ``StoreTemplateSpec.init_freshness`` (which is a closed
    enum picking a per-store starting freshness mode). This model
    parameterises the freshness/hype curve evaluated by
    ``freshness_curve.multiplier`` for each item:

    - ``alpha = 0`` is the staple override — the curve is identically
      ``1`` regardless of decay (no novelty premium). Use for
      essentials, basics, pantry, consumables.
    - ``alpha > 0`` drives a hype boost that decays over time. Typical
      range for trend-driven items is ``[0.1, 0.4]``.
    - ``decay`` is the hype decay length in simulation steps; strictly
      positive (open interval) so the consumer never divides by zero.
      Typical range ``[15, 45]``.

    Names that don't resolve against the catalog are pruned by
    ``WorldBuilder._sanitise_freshness`` after parsing.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    alpha: float = Field(ge=0.0)
    decay: float = Field(gt=0.0)


class FreshnessSet(BaseModel):
    """Output of the dedicated freshness LLM call.

    One ``ItemFreshness`` per item in the input chunk, in the same order.
    Items the LLM omits or misnames are pruned at the boundary; the
    corresponding ``Ware`` retains its ``None`` default so
    ``ItemRegistry`` falls back to ``ItemLifecycleParams`` defaults.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[ItemFreshness] = Field(min_length=1)


class TaxonomyCategory(BaseModel):
    """One bucket in a catalog taxonomy.

    ``target_share`` weights the deterministic skeleton sampler in
    ``WorldBuilder.sample_catalog``. Shares need not sum to exactly 1; the
    sampler renormalises.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    target_share: float = Field(gt=0.0, le=1.0)


class Taxonomy(BaseModel):
    """LLM-authored taxonomy: the category set + per-category target shares."""

    model_config = ConfigDict(extra="forbid")

    archetype: str = Field(min_length=1)
    categories: list[TaxonomyCategory] = Field(min_length=1)


class StoreTemplateSpec(BaseModel):
    """Pydantic model for one ``StoreTemplate``.

    The starting roster (``init_active_products``) is no longer LLM-authored
    — ``init_store_state`` falls back to ``init_rng.sample(catalog,
    init_active_count)`` when the template's roster is ``None``. Cuts a
    cross-payload validation invariant out of the prompt.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    capacity: float = Field(ge=0)
    init_balance: float = Field(ge=0)
    init_stock_pct: float = Field(ge=0, le=1)
    delivery_lag: float = Field(ge=0)
    holding_rate: float = Field(ge=0)
    order_fee: float = Field(ge=0)
    init_active_count: int = Field(ge=0)
    init_freshness: InitFreshness


class StoreTemplateList(BaseModel):
    """Wrapper around the LLM's store-templates payload."""

    model_config = ConfigDict(extra="forbid")

    templates: list[StoreTemplateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "StoreTemplateList":
        ids = [t.id for t in self.templates]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"StoreTemplateList: template ids must be unique; got {ids}"
            )
        return self


class SeasonWindow(BaseModel):
    """One ``(season_label, months)`` entry inside ``MarketDomain``.

    Modeled as a list of records rather than ``dict[str, list[int]]`` because
    OpenAI's structured-output strict mode rejects free-form object keys
    (no support for ``additionalProperties: <subschema>``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    months: list[Annotated[int, Field(ge=1, le=12)]]


class MarketDomain(BaseModel):
    """Domain-meaningful subset of ``MarketParams`` the LLM owns.

    The remaining ``MarketParams`` fields (math knobs, distributions,
    stage multipliers) are hand-set defaults applied by ``WorldBuilder``.
    Validator: ``price_elasticity < 0`` (demand decreases with price).
    """

    model_config = ConfigDict(extra="forbid")

    cycle_len: int = Field(gt=0)
    peak_factor: float = Field(gt=0)
    off_factor: float = Field(gt=0)
    init_demand: float = Field(ge=0)
    init_supply: float = Field(ge=0)
    season_months: list[SeasonWindow] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    price_elasticity: float

    @model_validator(mode="after")
    def _elasticity_negative(self) -> "MarketDomain":
        if self.price_elasticity >= 0:
            raise ValueError(
                f"MarketDomain.price_elasticity must be negative, got "
                f"{self.price_elasticity}"
            )
        return self

    @model_validator(mode="after")
    def _season_names_unique(self) -> "MarketDomain":
        names = [w.name for w in self.season_months]
        if len(names) != len(set(names)):
            raise ValueError(
                f"MarketDomain.season_months: duplicate season names {names}"
            )
        return self

    def season_months_dict(self) -> dict[str, list[int]]:
        """Convert to the ``MarketParams.season_months`` shape."""
        return {w.name: list(w.months) for w in self.season_months}


__all__ = [
    "Catalog",
    "CatalogItem",
    "Correlations",
    "FreshnessSet",
    "InitFreshness",
    "ItemFreshness",
    "ItemRelations",
    "MarketDomain",
    "RelatedRef",
    "SeasonWindow",
    "Seasonality",
    "StoreTemplateList",
    "StoreTemplateSpec",
    "Taxonomy",
    "TaxonomyCategory",
]
