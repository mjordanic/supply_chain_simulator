"""``WorldBuilder``: LLM-driven world generator.

Given a single archetype string (``"fashion_retail"``, ``"grocery"``,
…), ``WorldBuilder.build_setup(n_items, setup_dir)`` generates a
catalog + market and persists them as ``catalog.csv`` + ``setup.yaml``
in the given directory. Topology, policies, disruption, and run
parameters are the modeller's domain.

Pipeline (each stage is cached per-builder instance):

1. ``build_market_domain_params()`` — one LLM call producing the
   domain-meaningful slice; ``MarketParams`` is assembled by merging the
   slice with hand-set math defaults.
2. ``build_taxonomy()`` — one LLM call, returns ``Taxonomy``.
3. ``sample_catalog(n)`` — four sub-steps. A deterministic Python
   skeleton allocator distributes ``n`` slots across taxonomy
   categories proportional to ``target_share``; one LLM call names and
   prices the slots; a chunked LLM call authors cross-product
   ``related_products`` against the explicit name list; a chunked LLM
   call authors per-``Ware`` freshness curve params (``alpha``,
   ``decay``). Refs that don't resolve (or self-references /
   duplicates) are dropped at the boundary rather than triggering
   retries — the prompt-bounded candidate sets keep this near-zero in
   practice. Output is converted to ``list[Ware]`` via ``load_catalog``;
   items the LLM didn't author for freshness keep ``Ware`` defaults of
   ``None`` so ``ItemRegistry`` falls back to ``ItemLifecycleParams``
   defaults.

Total LLM calls for ``build_setup(n_items, setup_dir)``:
``2 + ceil(n/correlations_chunk_size) + ceil(n/freshness_chunk_size)``
(market, taxonomy, catalog, chunked correlations, chunked freshness).

Schema-failure retry feeds the previous validation error back into the
next prompt (see ``llm.prompts._with_retry``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError
from pydantic import BaseModel, ValidationError

from src.llm.openai_client import LLMClient


# Module-scoped logger; surfaced via ``logging.basicConfig`` in
# scenarios.
logger = logging.getLogger(__name__)
from src.llm.prompts import (
    catalog_prompt,
    correlations_prompt,
    freshness_prompt,
    market_domain_prompt,
    taxonomy_prompt,
)
from src.llm.schemas import (
    Catalog,
    Correlations,
    FreshnessSet,
    MarketDomain,
    Taxonomy,
)
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import (
    MarketParams,
    Ware,
    _ware_from_dict,
    _ware_to_dict,
    load_catalog,
)


# Bumped manually when the builder/output shape changes; written into
# ``World.meta`` so old cached worlds can be detected on load.
BUILDER_VERSION = "1"

# Default retry budget per LLM stage when schema validation fails.
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
# ``build_market_domain_params``. The LLM does not author these. Keeping
# them centralised here means a downstream tweak (e.g. a new
# ``stage_multipliers["dead"]`` value) propagates to every generated
# world automatically.
_MARKET_MATH_DEFAULTS: dict[str, Any] = {
    "cycle_amp": 0.0065,
    "correlation": 0.7,
    "trend_update_interval": 20,
    "min_value": 0.2,
    "max_value": 2.0,
    "stage_multipliers": {
        "introduction": 0.7,
        "growth": 1.5,
        "maturity": 1.0,
        "decline": 0.2,
        "dead": 0.05,
    },
    "promo_multiplier": 1.0,
    "demand_factor_min": 0.1,
    "supply_factor_min": 0.01,
    "cross_inv_lo": 0.3,
    "cross_inv_hi": 0.7,
    "cross_factor_range": (0.3, 1.6),
    "trend": Constant(1.001),
    "demand_shock": Normal(0.0, 0.01),
    "supply_shock": Normal(0.0, 0.01),
    "base_demand": Uniform(5, 20),
}


def allocate_skeletons(n: int, taxonomy: Taxonomy) -> list[str]:
    """Distribute ``n`` catalog slots across taxonomy categories.

    Deterministic, proportional to ``target_share``. Each category gets at
    least one slot when ``n >= len(categories)``; the last category absorbs
    rounding remainder so ``len(out) == n`` exactly. Output is the flat
    list of category names in taxonomy order.
    """
    if n <= 0:
        raise ValueError("allocate_skeletons: n must be positive")
    # Pull the category list once for repeated indexing below.
    cats = taxonomy.categories
    if not cats:
        raise ValueError("allocate_skeletons: taxonomy has no categories")

    # Normalise denominators in case shares don't sum to exactly 1.
    total = sum(c.target_share for c in cats)
    # Per-category slot count — populated by one of two branches below.
    counts: list[int] = []
    if n >= len(cats):
        # Floor each share, top up the largest-shortfall category until total == n.
        floors = [int((n * c.target_share) // total) for c in cats]
        floors = [max(1, f) for f in floors]  # every category at least one
        # Rounding remainder. Positive ⇒ we need to add slots; negative
        # (rare, when the floor + min-1 stretches past n) ⇒ we need to
        # subtract.
        deficit = n - sum(floors)
        if deficit > 0:
            # Add to highest-share categories first.
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
            # Remove from lowest-share categories first; never below 1.
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

    # Expand counts → flat list of category names in taxonomy order.
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
    # Output map for this chunk.
    out: dict[str, list[tuple[str, float]]] = {}
    for entry in payload.items:
        if entry.name not in valid_names:
            continue  # LLM hallucinated an item that's not in the catalog
        # Per-item dedup set so duplicate refs collapse to one entry.
        seen: set[str] = set()
        # Per-item related list (preserves first-seen order).
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


def _load_catalog_csv(catalog_path: "Path") -> "list[Any]":
    """Load a catalog.csv file into a list of Ware namedtuples.

    Thin wrapper around ``setup_io._parse_catalog`` so WorldBuilder can
    reload catalog from a cached setup directory without importing the
    full load_setup pipeline.
    """
    from src.sim.setup_io import _parse_catalog
    return _parse_catalog(catalog_path)


class WorldBuilder:
    """Orchestrate the LLM stages plus deterministic skeleton sampling.

    Mutating the same instance across calls is fine; each stage method
    caches its last result so ``build()`` does not re-call the LLM
    redundantly. Pass a fresh ``WorldBuilder`` to start over.
    """

    def __init__(
        self,
        archetype: str,
        client: LLMClient,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        correlations_chunk_size: int = _DEFAULT_CORRELATIONS_CHUNK_SIZE,
        freshness_chunk_size: int = _DEFAULT_FRESHNESS_CHUNK_SIZE,
    ) -> None:
        # Validate chunk sizes upfront — zero/negative would silently
        # produce no calls and an empty result.
        if correlations_chunk_size <= 0:
            raise ValueError("correlations_chunk_size must be positive")
        if freshness_chunk_size <= 0:
            raise ValueError("freshness_chunk_size must be positive")
        # Archetype string passed verbatim into every prompt.
        self.archetype = archetype
        # ``LLMClient`` implementation — production OpenAI client or a test fake.
        self.client = client
        # Per-call retry budget when Pydantic validation fails.
        self.max_retries = max_retries
        # Per-chunk size for the chunked correlations / freshness calls.
        self.correlations_chunk_size = correlations_chunk_size
        self.freshness_chunk_size = freshness_chunk_size
        # Per-stage caches. ``None`` ⇒ "not yet built"; subsequent calls
        # reuse the cached result so ``build_setup()`` is idempotent.
        self._taxonomy: Taxonomy | None = None
        self._catalog: list[Ware] | None = None
        self._market: MarketParams | None = None

    def build_taxonomy(self) -> Taxonomy:
        """Return (and cache) the LLM-authored ``Taxonomy``."""
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
        # Stage 2: deterministic skeleton allocation (uses cached taxonomy).
        taxonomy = self.build_taxonomy()
        skeletons = allocate_skeletons(n, taxonomy)
        # Encode taxonomy as JSON for the catalog prompt — keeps the
        # category context tight without bloating the user message.
        taxonomy_json = taxonomy.model_dump_json()

        # Stage 3a: catalog naming.
        catalog_payload = self._call_with_retry(
            schema=Catalog,
            prompt_builder=lambda err: catalog_prompt(
                self.archetype, taxonomy_json, skeletons, err
            ),
        )

        # Catalog name list & lookup set for downstream sanitisation.
        catalog_names = [it.name for it in catalog_payload.items]
        valid_names = set(catalog_names)
        # (name, category) pairs used as the chunked input for the
        # correlations & freshness prompts.
        item_pairs = [(it.name, it.category) for it in catalog_payload.items]

        # Stage 3b: chunked correlations. Pre-seed every catalog name
        # with an empty related list so items the LLM omits stay valid.
        related_by_name: dict[str, list[tuple[str, float]]] = {
            n: [] for n in catalog_names
        }
        corr_chunk_size = self.correlations_chunk_size
        for start in range(0, len(item_pairs), corr_chunk_size):
            chunk = item_pairs[start : start + corr_chunk_size]
            # Bind ``_chunk=chunk`` in the lambda to dodge the
            # "late-binding loop variable" footgun.
            payload = self._call_with_retry(
                schema=Correlations,
                prompt_builder=lambda err, _chunk=chunk: correlations_prompt(
                    self.archetype, _chunk, catalog_names, err
                ),
            )
            related_by_name.update(
                _sanitise_correlations(payload, valid_names)
            )

        # Stage 3c: chunked freshness. Authored entries override the
        # ``Ware`` default; missing entries stay ``None``.
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

        # Stitch the per-item dicts together into kwargs for ``load_catalog``.
        items: list[dict[str, Any]] = []
        for it in catalog_payload.items:
            d: dict[str, Any] = {
                "name": it.name,
                "category": it.category,
                # Convert the sanitised correlation tuples into the
                # ``[[name, corr], …]`` list shape ``load_catalog`` expects.
                "related_products": [
                    [rel_name, corr]
                    for rel_name, corr in related_by_name[it.name]
                ],
                "base_price": it.base_price,
                "unit_cost": it.unit_cost,
                "seasonality": it.seasonality.value,
            }
            # Only set freshness fields when the LLM actually authored them.
            if it.name in freshness_by_name:
                alpha, decay = freshness_by_name[it.name]
                d["freshness_alpha"] = alpha
                d["freshness_decay"] = decay
            items.append(d)
        wares = load_catalog(items)
        self._catalog = wares
        return wares

    def build_market_domain_params(self) -> MarketParams:
        """One LLM call producing the domain slice; merged with math defaults."""
        if self._market is not None:
            return self._market
        # Domain-meaningful slice authored by the LLM.
        domain = self._call_with_retry(
            schema=MarketDomain,
            prompt_builder=lambda err: market_domain_prompt(self.archetype, err),
        )
        # Build the merged params: LLM slice ∪ hand-set math defaults.
        merged: dict[str, Any] = {
            "cycle_len": domain.cycle_len,
            "peak_factor": domain.peak_factor,
            "off_factor": domain.off_factor,
            "season_months": domain.season_months_dict(),
            "regions": list(domain.regions),
            "price_elasticity": domain.price_elasticity,
        }
        merged.update(_MARKET_MATH_DEFAULTS)
        # Default starting demand/supply: midpoint of the clamp band.
        init_default = (merged["max_value"] - merged["min_value"]) / 2
        merged["init_demand"] = init_default
        merged["init_supply"] = init_default
        self._market = MarketParams(**merged)
        return self._market

    def build_setup(
        self,
        n_items: int,
        setup_dir: "str | Path",
        *,
        force_rebuild: bool = False,
    ) -> tuple["list[Any]", "Any"]:
        """Build and persist catalog + market to a setup directory (dir-as-cache).

        If ``setup_dir/catalog.csv`` already exists (and ``force_rebuild``
        is False), loads and returns the existing catalog + market without
        making any LLM calls.  Otherwise runs the LLM market + catalog
        stages and writes ``catalog.csv`` and the ``market:`` block of
        ``setup.yaml`` to ``setup_dir``.

        Topology, policies, disruption, and run parameters are intentionally
        NOT written — those are the modeller's domain, not the generator's.

        Parameters
        ----------
        n_items:
            Number of catalog items to generate on a cache miss.
        setup_dir:
            Target directory.  Used as both the cache key and write target.
        force_rebuild:
            Ignore an existing setup directory and always call the LLM.

        Returns
        -------
        (catalog, market) — list[Ware], MarketParams
        """
        from pathlib import Path as _Path

        from src.sim.setup_io import write_catalog_and_market

        setup_dir = _Path(setup_dir)
        catalog_path = setup_dir / "catalog.csv"

        if not force_rebuild and catalog_path.is_file():
            # Cache hit: load catalog from CSV and market from setup.yaml.
            catalog = _load_catalog_csv(catalog_path)
            yaml_path = setup_dir / "setup.yaml"
            if yaml_path.is_file():
                import yaml as _yaml
                with yaml_path.open(encoding="utf-8") as f:
                    doc = _yaml.safe_load(f) or {}
                if "market" in doc:
                    from src.sim.setup_io import _parse_market
                    market = _parse_market(doc["market"], "setup.yaml.market")
                else:
                    # Fall through to generate market (catalog is cached but market block missing).
                    market = self.build_market_domain_params()
            else:
                market = self.build_market_domain_params()
            self._catalog = catalog
            return catalog, market

        # Cache miss: run LLM pipeline and persist data only.
        market = self.build_market_domain_params()
        catalog = self.sample_catalog(n_items)
        write_catalog_and_market(catalog, market, setup_dir)
        return catalog, market

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
        # Last seen error string — drives the retry preamble next iteration.
        last_error: str | None = None
        # Original exception for the eventual ``raise … from`` chain.
        last_exc: ValidationError | None = None
        for attempt in range(self.max_retries):
            system, user = prompt_builder(last_error)
            logger.info(
                "WorldBuilder LLM request schema=%s attempt=%s/%s",
                schema.__name__,
                attempt + 1,
                self.max_retries,
            )
            try:
                parsed = self.client.structured_completion(
                    system=system, user=user, schema=schema
                )
            except APIConnectionError:
                # Network/proxy errors aren't going to fix themselves
                # via prompt retries — propagate so the caller can act.
                logger.error(
                    "WorldBuilder: connection error during schema=%s "
                    "(not retried; fix network/base URL)",
                    schema.__name__,
                )
                raise
            except ValidationError as e:
                # Stash the error string for the next attempt's preamble.
                last_error = str(e)
                last_exc = e
                continue
            if post_validate is not None:
                # Cross-payload invariant check; same retry channel.
                err_msg = post_validate(parsed)
                if err_msg is not None:
                    last_error = err_msg
                    continue
            return parsed
        # Budget exhausted — raise with full context for the operator.
        raise RuntimeError(
            f"WorldBuilder: schema {schema.__name__} failed validation "
            f"after {self.max_retries} attempts; last error:\n{last_error}"
        ) from last_exc


__all__ = [
    "BUILDER_VERSION",
    "WorldBuilder",
    "allocate_skeletons",
]
