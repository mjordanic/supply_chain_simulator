"""LLM prompts for the world-builder pipeline.

One function per LLM stage. Each returns a ``(system, user)`` tuple
suitable for the ``OpenAIClient.structured_completion`` call. The
``last_error`` argument prepends a validation-failure block to the user
message so the LLM sees what was wrong on the previous attempt; pass
``None`` on the first attempt to omit it.

The catalog flow is split across ``catalog_prompt`` (per-item basics:
name, category, price, cost, seasonality), ``correlations_prompt``
(cross-product references), and ``freshness_prompt`` (per-item
freshness curve params). Splitting keeps each prompt narrow enough
that cheap models stay reliable. ``correlations`` and ``freshness``
are also *chunked* by the caller — see ``world_builder.WorldBuilder``.
"""

from __future__ import annotations


# Templated preamble prepended to a user prompt on retry. ``{error}``
# is the stringified Pydantic ``ValidationError`` from the previous
# attempt; surfacing it verbatim has been the most reliable way to
# coax cheap models into producing a schema-compliant response.
_RETRY_PREAMBLE = (
    "Your previous response failed schema validation. "
    "The exact validation error was:\n\n{error}\n\n"
    "Fix the issues and produce a response that satisfies the schema.\n\n"
)


def _with_retry(user: str, last_error: str | None) -> str:
    """If ``last_error`` is non-empty, prepend the retry preamble; else passthrough."""
    if last_error:
        return _RETRY_PREAMBLE.format(error=last_error) + user
    return user


def taxonomy_prompt(archetype: str, last_error: str | None = None) -> tuple[str, str]:
    """Stage 1: ask the LLM for a category taxonomy."""
    # System message — sets the role and the schema-strictness ground rule.
    system = (
        "You are a retail merchandising expert. You design product taxonomies "
        "for retail simulations. Respond strictly in the requested schema."
    )
    # User message — the actual task description.
    user = (
        f"Design a product taxonomy for a retail business of archetype "
        f"'{archetype}'. Produce 3 to 8 distinct top-level categories. For "
        "each category give a short description and a target_share in "
        "(0, 1] reflecting its slice of the catalog. Shares should sum to "
        "approximately 1 but the schema does not strictly require it.\n\n"
        "Set the archetype field on the taxonomy to the input archetype "
        "string verbatim."
    )
    return system, _with_retry(user, last_error)


def catalog_prompt(
    archetype: str,
    taxonomy_json: str,
    skeletons: list[str],
    last_error: str | None = None,
) -> tuple[str, str]:
    """Stage 3a: name and refine N catalog skeletons.

    ``skeletons`` is a list of category labels — one per item slot,
    pre-allocated by the deterministic Python sampler in stage 2.
    Cross-product correlations are authored separately by
    ``correlations_prompt`` so this prompt stays narrow.
    """
    system = (
        "You are a retail merchandising expert. Produce a concrete catalog "
        "of SKUs for a simulation. Respect the provided category for each "
        "slot. Respond strictly in the requested schema."
    )
    # Pre-rendered bullet list of skeleton slots (one per output item).
    bulleted = "\n".join(
        f"- slot {i}: category={cat!r}" for i, cat in enumerate(skeletons)
    )
    user = (
        f"Archetype: '{archetype}'.\n\n"
        f"Taxonomy:\n{taxonomy_json}\n\n"
        f"Catalog has exactly {len(skeletons)} slots, each pre-assigned a "
        f"category by an upstream sampler:\n{bulleted}\n\n"
        "For each slot produce a CatalogItem with:\n"
        "- name: distinctive product name (must be unique across the catalog)\n"
        "- category: matches the slot's pre-assigned category\n"
        "- base_price: must be strictly greater than unit_cost (positive margin)\n"
        "- unit_cost: non-negative\n"
        "- seasonality: one of "
        "'spring', 'summer', 'fall', 'winter', 'spring/summer', "
        "'fall/winter', 'all_season'.\n\n"
        "Produce exactly the requested number of items in slot order."
    )
    return system, _with_retry(user, last_error)


def correlations_prompt(
    archetype: str,
    items: list[tuple[str, str]],
    candidate_names: list[str],
    last_error: str | None = None,
) -> tuple[str, str]:
    """Stage 3b: author per-item ``related_products`` references.

    ``items`` is the chunk of ``(name, category)`` pairs the model should
    annotate in this call — kept small (~50) so the response stays within
    the model's output budget and ``related`` lists actually get
    populated. ``candidate_names`` is the full catalog name list so
    cross-chunk references are still possible. Any reference that still
    fails to resolve is pruned by ``WorldBuilder.sample_catalog``.
    """
    system = (
        "You are a retail merchandising expert. You annotate cross-product "
        "demand correlations. Respond strictly in the requested schema."
    )
    # Pre-rendered chunk list — one bullet per item to annotate.
    bulleted = "\n".join(f"- {name!r} (category: {cat})" for name, cat in items)
    user = (
        f"Archetype: '{archetype}'.\n\n"
        f"Items to annotate ({len(items)}):\n{bulleted}\n\n"
        "For each item above, list 1 to 4 related products that an "
        "average shopper would plausibly buy together. Output one "
        "ItemRelations per item with:\n"
        "- name: the item's name verbatim from the list above\n"
        "- related: list of {name, correlation in [0, 1]} entries. Each "
        "'name' MUST be drawn from the candidate list below (verbatim). "
        "Use 'related: []' only when no plausible cross-correlation "
        "exists (rare for typical retail items). Do not reference an "
        "item to itself. Do not invent external products.\n\n"
        f"Candidate names (use only these, verbatim): {candidate_names}\n\n"
        "Produce exactly one ItemRelations entry per item to annotate, "
        "in the same order. Aim for 2-3 related entries per item where "
        "reasonable; correlation values typically fall in [0.3, 0.8]."
    )
    return system, _with_retry(user, last_error)


def freshness_prompt(
    archetype: str,
    items: list[tuple[str, str]],
    last_error: str | None = None,
) -> tuple[str, str]:
    """Stage 3c: author per-item freshness curve params.

    ``items`` is the chunk of ``(name, category)`` pairs the model should
    annotate in this call — kept small (~50) so the response stays within
    the model's output budget. There are no cross-item references, so no
    candidate-name list is needed; each item is annotated independently.
    Names the LLM emits that don't resolve against the catalog are
    pruned by ``WorldBuilder._sanitise_freshness``.
    """
    system = (
        "You are a retail merchandising expert. You annotate per-product "
        "freshness curves for a demand simulation. Respond strictly in "
        "the requested schema."
    )
    # Pre-rendered chunk list.
    bulleted = "\n".join(f"- {name!r} (category: {cat})" for name, cat in items)
    user = (
        f"Archetype: '{archetype}'.\n\n"
        f"Items to annotate ({len(items)}):\n{bulleted}\n\n"
        "For each item produce one ItemFreshness with:\n"
        "- name: the item's name verbatim from the list above\n"
        "- alpha: hype amplitude. Use 0 for staples (essentials, basics, "
        "pantry, consumables, replenishables — items shoppers buy on "
        "habit, with no novelty premium). Use 0.1 to 0.4 for "
        "trend-driven or fashion-like items where the launch boost "
        "matters.\n"
        "- decay: hype decay length in simulation steps; typically 15 "
        "to 45. Strictly positive even when alpha is 0 (the curve "
        "still expects a finite decay constant).\n\n"
        "Produce exactly one ItemFreshness entry per input item, in "
        "the same order."
    )
    return system, _with_retry(user, last_error)


def store_templates_prompt(
    archetype: str,
    regions: list[str],
    last_error: str | None = None,
) -> tuple[str, str]:
    """Build prompt for store-template generation.

    The starting roster (``init_active_products``) is no longer authored —
    ``init_store_state`` random-samples ``init_active_count`` SKUs from
    the catalog at construction time. Keeps this prompt independent of
    the catalog stage.
    """
    system = (
        "You are a retail operations expert. Design realistic store templates "
        "for a retail simulation. Respond strictly in the requested schema."
    )
    user = (
        f"Archetype: '{archetype}'. Regions in scope: {regions}.\n\n"
        "Produce 1 to 4 store templates (e.g., flagship, standard, outlet) "
        "that fit this archetype. Each template needs:\n"
        "- id: short identifier (unique across the list)\n"
        "- region: one of the listed regions\n"
        "- capacity: total inventory units, non-negative\n"
        "- init_balance: opening cash balance, non-negative\n"
        "- init_stock_pct: fraction of capacity to fill at step 0, in [0, 1]\n"
        "- delivery_lag: lead time in steps, non-negative\n"
        "- holding_rate: per-step holding cost as a fraction of unit cost\n"
        "- order_fee: fixed fee per replenishment order, non-negative\n"
        "- init_active_count: number of SKUs the store carries at step 0 "
        "(sampled randomly from the catalog at store-construction time)\n"
        "- init_freshness: 'baseline' (established store, initial SKUs skip "
        "the hype window) or 'fresh' (grand-opening, initial SKUs enter at "
        "full hype). Default to 'baseline' for established formats; pick "
        "'fresh' only if the template is explicitly modelling a launch."
    )
    return system, _with_retry(user, last_error)


def market_domain_prompt(
    archetype: str,
    last_error: str | None = None,
) -> tuple[str, str]:
    """Build prompt for the LLM-owned slice of ``MarketParams``."""
    system = (
        "You are a retail demand-modelling expert. Pick domain-meaningful "
        "market parameters for a simulation. Respond strictly in the "
        "requested schema."
    )
    user = (
        f"Archetype: '{archetype}'.\n\n"
        "Produce the LLM-owned slice of the market parameters:\n"
        "- cycle_len: period of seasonal cycle in simulation steps "
        "(positive integer; typically 90 to 730 for daily steps)\n"
        "- peak_factor: in-season demand multiplier (positive, typically > 1)\n"
        "- off_factor: out-of-season demand multiplier (positive, typically < 1)\n"
        "- init_demand: starting demand level (non-negative)\n"
        "- init_supply: starting supply level (non-negative)\n"
        "- season_months: list of {name, months} entries. 'name' is a "
        "seasonality label drawn from 'spring', 'summer', 'fall', 'winter', "
        "'spring/summer', 'fall/winter', 'all_season'. 'months' is a list of "
        "month numbers (1-12). Names must be unique. Include only labels that "
        "are relevant to the archetype.\n"
        "- regions: list of region identifiers the simulation runs over\n"
        "- price_elasticity: STRICTLY NEGATIVE (e.g., -1.2 for typical "
        "retail; demand falls as price rises)."
    )
    return system, _with_retry(user, last_error)


__all__ = [
    "catalog_prompt",
    "correlations_prompt",
    "freshness_prompt",
    "market_domain_prompt",
    "store_templates_prompt",
    "taxonomy_prompt",
]
