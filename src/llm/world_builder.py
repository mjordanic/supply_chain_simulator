"""``WorldBuilder``: six-stage catalog + store/market builds.

Orchestrates the LLM stages plus a deterministic Python sampler:

1. ``build_market_domain_params()`` — one LLM call producing the
   domain-meaningful slice; ``MarketParams`` is assembled by merging the
   slice with hand-set math defaults. Sets ``regions`` for downstream.
2. ``build_taxonomy()`` — one LLM call, returns ``Taxonomy``.
3. ``sample_catalog(n)`` — deterministic skeleton allocator distributes
   ``n`` slots across taxonomy categories proportional to ``target_share``;
   one LLM call names+prices the slots into a ``Catalog``; a chunked LLM
   call authors cross-product ``related_products`` against the explicit
   name list; a second chunked LLM call authors per-item freshness
   curve params (``alpha``, ``decay``). Refs that don't resolve (or
   self-references / duplicates) are dropped at the boundary rather
   than triggering retries — the prompt-bounded candidate sets keep
   this near-zero in practice. Output is converted to ``list[Ware]``
   via ``load_catalog``; items the LLM didn't author for freshness
   keep ``Ware`` defaults of ``None`` so ``ItemRegistry`` falls back
   to ``ItemLifecycleParams`` defaults.
4. ``build_store_templates()`` — one LLM call, returns
   ``dict[str, StoreTemplate]``. The starting roster is sampled at
   store-construction time from ``init_active_count`` so this prompt no
   longer depends on the catalog.

Schema-failure retry feeds the previous validation error back into the
next prompt (see ``llm.prompts._with_retry``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.llm.openai_client import LLMClient
from src.llm.prompts import (
    catalog_prompt,
    correlations_prompt,
    freshness_prompt,
    market_domain_prompt,
    store_templates_prompt,
    taxonomy_prompt,
)
from src.llm.schemas import (
    Catalog,
    Correlations,
    FreshnessSet,
    MarketDomain,
    StoreTemplateList,
    Taxonomy,
)
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import (
    MarketParams,
    StoreTemplate,
    Ware,
    load_catalog,
)


_DEFAULT_MAX_RETRIES = 3

# Correlations are authored in chunks so each call stays within the model's
# output budget. With a single call over 100+ items the model tends to emit
# ``related: []`` for every entry to fit; chunking keeps each response small
# enough that ``related`` lists actually get populated.
_DEFAULT_CORRELATIONS_CHUNK_SIZE = 50

# Freshness is also chunked. Per-item output is small (one ``alpha`` and
# one ``decay`` float), so the same chunk size as correlations comfortably
# fits the response budget while keeping the schema-failure retry blast
# radius bounded to a single chunk.
_DEFAULT_FRESHNESS_CHUNK_SIZE = 50


# Hand-set math defaults applied to every ``MarketParams`` produced by
# ``build_market_domain_params``. The LLM does not author these.
_MARKET_MATH_DEFAULTS: dict[str, Any] = {
    "cycle_amp": 0.1,
    "correlation": 0.7,
    "trend_update_interval": 20,
    "min_value": 20.0,
    "max_value": 200.0,
    "stage_multipliers": {
        "introduction": 0.7,
        "growth": 1.5,
        "maturity": 1.0,
        "decline": 0.2,
        "dead": 0.05,
    },
    "promo_multiplier": 1.0,
    "demand_factor_min": 0.1,
    "demand_divisor": 100.0,
    "supply_factor_min": 0.01,
    "supply_divisor": 100.0,
    "demand_range": (80.0, 120.0),
    "cross_inv_lo": 0.3,
    "cross_inv_hi": 0.7,
    "cross_factor_range": (0.3, 1.6),
    "trend": Constant(1.0),
    "demand_shock": Normal(0.0, 5.0),
    "supply_shock": Normal(0.0, 5.0),
    "base_demand": Uniform(2, 8),
}


@dataclass
class World:
    """LLM-generated world artifact consumed by ``Scenario`` authoring."""

    catalog: list[Ware]
    market: MarketParams
    store_templates: dict[str, StoreTemplate]


def allocate_skeletons(n: int, taxonomy: Taxonomy) -> list[str]:
    """Distribute ``n`` catalog slots across taxonomy categories.

    Deterministic, proportional to ``target_share``. Each category gets at
    least one slot when ``n >= len(categories)``; the last category absorbs
    rounding remainder so ``len(out) == n`` exactly. Output is the flat
    list of category names in taxonomy order.
    """
    if n <= 0:
        raise ValueError("allocate_skeletons: n must be positive")
    cats = taxonomy.categories
    if not cats:
        raise ValueError("allocate_skeletons: taxonomy has no categories")

    total = sum(c.target_share for c in cats)
    counts: list[int] = []
    if n >= len(cats):
        # Floor each share, top up the largest-shortfall category until total == n.
        floors = [int((n * c.target_share) // total) for c in cats]
        floors = [max(1, f) for f in floors]  # every category at least one
        deficit = n - sum(floors)
        if deficit > 0:
            order = sorted(
                range(len(cats)),
                key=lambda i: cats[i].target_share / total,
                reverse=True,
            )
            for i in order:
                if deficit == 0:
                    break
                floors[i] += 1
                deficit -= 1
        elif deficit < 0:
            order = sorted(
                range(len(cats)),
                key=lambda i: cats[i].target_share / total,
            )
            for i in order:
                if deficit == 0:
                    break
                if floors[i] > 1:
                    floors[i] -= 1
                    deficit += 1
        counts = floors
    else:
        # n < number of categories: take the n highest-share categories.
        order = sorted(
            range(len(cats)),
            key=lambda i: cats[i].target_share / total,
            reverse=True,
        )
        keep = set(order[:n])
        counts = [1 if i in keep else 0 for i in range(len(cats))]

    return [
        cat.name for cat, count in zip(cats, counts) for _ in range(count)
    ]


def _sanitise_correlations(
    payload: Correlations, valid_names: set[str]
) -> dict[str, list[tuple[str, float]]]:
    """Extract authored ``{item_name: [(related_name, c), ...]}`` from one chunk.

    Drops references the LLM emitted that don't resolve, self-references,
    and within-item duplicates. Returns only entries the LLM actually
    authored — the caller merges chunk dicts into the final per-catalog
    map (initialised to empty lists).
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for entry in payload.items:
        if entry.name not in valid_names:
            continue  # LLM hallucinated an item that's not in the catalog
        seen: set[str] = set()
        bag: list[tuple[str, float]] = []
        for ref in entry.related:
            if ref.name not in valid_names:
                continue  # dangling reference
            if ref.name == entry.name:
                continue  # self-reference
            if ref.name in seen:
                continue  # duplicate within the same item
            seen.add(ref.name)
            bag.append((ref.name, ref.correlation))
        out[entry.name] = bag
    return out


def _sanitise_freshness(
    payload: FreshnessSet, valid_names: set[str]
) -> dict[str, tuple[float, float]]:
    """Extract authored ``{item_name: (alpha, decay)}`` from one chunk.

    Drops entries with names that don't resolve against the catalog.
    Returns only entries the LLM actually authored — items missing from
    the merged map keep ``Ware`` default ``None`` so ``ItemRegistry``
    falls back to ``ItemLifecycleParams`` defaults.
    """
    return {
        e.name: (e.alpha, e.decay)
        for e in payload.items
        if e.name in valid_names
    }


class WorldBuilder:
    """Orchestrate the LLM stages plus deterministic skeleton sampling.

    Mutating the same instance across calls is fine; ``build_taxonomy`` and
    later builders cache their last result so ``build()`` does not re-call
    the LLM redundantly. Pass a fresh ``WorldBuilder`` to start over.
    """

    def __init__(
        self,
        archetype: str,
        client: LLMClient,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        correlations_chunk_size: int = _DEFAULT_CORRELATIONS_CHUNK_SIZE,
        freshness_chunk_size: int = _DEFAULT_FRESHNESS_CHUNK_SIZE,
    ) -> None:
        if correlations_chunk_size <= 0:
            raise ValueError("correlations_chunk_size must be positive")
        if freshness_chunk_size <= 0:
            raise ValueError("freshness_chunk_size must be positive")
        self.archetype = archetype
        self.client = client
        self.max_retries = max_retries
        self.correlations_chunk_size = correlations_chunk_size
        self.freshness_chunk_size = freshness_chunk_size
        self._taxonomy: Taxonomy | None = None
        self._catalog: list[Ware] | None = None
        self._templates: dict[str, StoreTemplate] | None = None
        self._market: MarketParams | None = None
        self._regions: list[str] | None = None

    def build_taxonomy(self) -> Taxonomy:
        if self._taxonomy is not None:
            return self._taxonomy
        result = self._call_with_retry(
            schema=Taxonomy,
            prompt_builder=lambda err: taxonomy_prompt(self.archetype, err),
        )
        self._taxonomy = result
        return result

    def sample_catalog(self, n: int) -> list[Ware]:
        """Four-stage catalog flow.

        1. (Python) call ``build_taxonomy`` if not cached, allocate ``n``
           skeletons across categories proportionally to ``target_share``.
        2. (LLM)    name+price the skeletons into a ``Catalog`` (no
           cross-product correlations or freshness params in this prompt).
        3. (LLM)    author ``related_products`` against the explicit name
           list from step 2, in chunks of ``correlations_chunk_size`` so
           each call's response stays within the model's output budget;
           dangling/self/duplicate refs are sanitised post-parse. Total
           correlations LLM calls = ``ceil(n / correlations_chunk_size)``.
        4. (LLM)    author per-item ``freshness_alpha`` / ``freshness_decay``
           in chunks of ``freshness_chunk_size``. Items the LLM omits or
           misnames keep ``Ware`` default ``None`` so ``ItemRegistry``
           falls back to ``ItemLifecycleParams`` defaults. Total
           freshness LLM calls = ``ceil(n / freshness_chunk_size)``.

        Returns ``list[Ware]`` with ``P{i:04d}`` ids assigned by
        ``load_catalog``. Per-``Ware`` lifecycle and stock-share fields
        are not authored by the LLM — ``ItemRegistry`` falls back to
        ``ItemLifecycleParams`` defaults for those.
        """
        if n <= 0:
            raise ValueError("sample_catalog: n must be positive")
        taxonomy = self.build_taxonomy()
        skeletons = allocate_skeletons(n, taxonomy)
        taxonomy_json = taxonomy.model_dump_json()

        catalog_payload = self._call_with_retry(
            schema=Catalog,
            prompt_builder=lambda err: catalog_prompt(
                self.archetype, taxonomy_json, skeletons, err
            ),
        )

        catalog_names = [it.name for it in catalog_payload.items]
        valid_names = set(catalog_names)
        item_pairs = [(it.name, it.category) for it in catalog_payload.items]

        related_by_name: dict[str, list[tuple[str, float]]] = {
            n: [] for n in catalog_names
        }
        corr_chunk_size = self.correlations_chunk_size
        for start in range(0, len(item_pairs), corr_chunk_size):
            chunk = item_pairs[start : start + corr_chunk_size]
            payload = self._call_with_retry(
                schema=Correlations,
                prompt_builder=lambda err, _chunk=chunk: correlations_prompt(
                    self.archetype, _chunk, catalog_names, err
                ),
            )
            related_by_name.update(
                _sanitise_correlations(payload, valid_names)
            )

        freshness_by_name: dict[str, tuple[float, float]] = {}
        fresh_chunk_size = self.freshness_chunk_size
        for start in range(0, len(item_pairs), fresh_chunk_size):
            chunk = item_pairs[start : start + fresh_chunk_size]
            payload = self._call_with_retry(
                schema=FreshnessSet,
                prompt_builder=lambda err, _chunk=chunk: freshness_prompt(
                    self.archetype, _chunk, err
                ),
            )
            freshness_by_name.update(
                _sanitise_freshness(payload, valid_names)
            )

        items: list[dict[str, Any]] = []
        for it in catalog_payload.items:
            d: dict[str, Any] = {
                "name": it.name,
                "category": it.category,
                "related_products": [
                    [rel_name, corr]
                    for rel_name, corr in related_by_name[it.name]
                ],
                "base_price": it.base_price,
                "unit_cost": it.unit_cost,
                "seasonality": it.seasonality.value,
            }
            if it.name in freshness_by_name:
                alpha, decay = freshness_by_name[it.name]
                d["freshness_alpha"] = alpha
                d["freshness_decay"] = decay
            items.append(d)
        wares = load_catalog(items)
        self._catalog = wares
        return wares

    def build_store_templates(self) -> dict[str, StoreTemplate]:
        """One LLM call producing the store-templates payload.

        The starting roster is sampled at store-construction time from
        ``init_active_count`` so this prompt is independent of the catalog
        stage. ``init_freshness`` is still authored per template.
        """
        if self._templates is not None:
            return self._templates
        regions = self._regions_from_market()

        payload = self._call_with_retry(
            schema=StoreTemplateList,
            prompt_builder=lambda err: store_templates_prompt(
                self.archetype, regions, err
            ),
        )
        templates: dict[str, StoreTemplate] = {}
        for spec in payload.templates:
            templates[spec.id] = StoreTemplate(
                id=spec.id,
                region=spec.region,
                capacity=spec.capacity,
                init_balance=spec.init_balance,
                init_stock_pct=spec.init_stock_pct,
                delivery_lag=spec.delivery_lag,
                holding_rate=spec.holding_rate,
                order_fee=spec.order_fee,
                init_active_count=spec.init_active_count,
                init_freshness=spec.init_freshness.value,
            )
        self._templates = templates
        return templates

    def build_market_domain_params(self) -> MarketParams:
        if self._market is not None:
            return self._market
        domain = self._call_with_retry(
            schema=MarketDomain,
            prompt_builder=lambda err: market_domain_prompt(self.archetype, err),
        )
        merged: dict[str, Any] = {
            "cycle_len": domain.cycle_len,
            "peak_factor": domain.peak_factor,
            "off_factor": domain.off_factor,
            "init_demand": domain.init_demand,
            "init_supply": domain.init_supply,
            "season_months": domain.season_months_dict(),
            "regions": list(domain.regions),
            "price_elasticity": domain.price_elasticity,
        }
        merged.update(_MARKET_MATH_DEFAULTS)
        self._market = MarketParams(**merged)
        self._regions = list(domain.regions)
        return self._market

    def build(self, n_items: int) -> World:
        """Run all stages and return the merged ``World``.

        Order: market_domain (sets regions) → taxonomy → catalog (three
        sub-calls: catalog → correlations → freshness) → store_templates.
        Cached results are reused on re-entry.
        """
        market = self.build_market_domain_params()
        catalog = self.sample_catalog(n_items)
        templates = self.build_store_templates()
        return World(catalog=catalog, market=market, store_templates=templates)

    def _regions_from_market(self) -> list[str]:
        """Best-effort regions list for the store-templates prompt.

        Calls ``build_market_domain_params`` if regions are not yet
        available, so callers can invoke ``build_store_templates`` first
        without it failing.
        """
        if self._regions is not None:
            return list(self._regions)
        self.build_market_domain_params()
        assert self._regions is not None
        return list(self._regions)

    def _call_with_retry(
        self,
        *,
        schema: type[BaseModel],
        prompt_builder: Callable[[str | None], tuple[str, str]],
        post_validate: Callable[[Any], str | None] | None = None,
    ) -> Any:
        """Call ``client.structured_completion`` with up to ``max_retries`` attempts.

        On ``ValidationError``, the next attempt's user prompt prepends a
        block describing the failure (see ``llm.prompts._with_retry``).
        Other exceptions propagate immediately.

        ``post_validate`` is an optional callback invoked on the parsed
        payload. It returns ``None`` to accept or an error string that
        is fed back into the next prompt (same channel as Pydantic
        ``ValidationError``). Use it for cross-payload invariants the
        per-call schema cannot express.
        """
        last_error: str | None = None
        last_exc: ValidationError | None = None
        for _ in range(self.max_retries):
            system, user = prompt_builder(last_error)
            try:
                parsed = self.client.structured_completion(
                    system=system, user=user, schema=schema
                )
            except ValidationError as e:
                last_error = str(e)
                last_exc = e
                continue
            if post_validate is not None:
                err_msg = post_validate(parsed)
                if err_msg is not None:
                    last_error = err_msg
                    continue
            return parsed
        raise RuntimeError(
            f"WorldBuilder: schema {schema.__name__} failed validation "
            f"after {self.max_retries} attempts; last error:\n{last_error}"
        ) from last_exc


__all__ = ["World", "WorldBuilder", "allocate_skeletons"]
