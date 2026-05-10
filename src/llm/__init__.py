"""LLM world-builder pipeline.

This sub-package generates a runnable ``World(catalog, market,
store_templates)`` from a single archetype string (``"fashion_retail"``,
``"grocery"``, …). It is the "give me a thematically coherent
catalog/market without hand-coding one" entry point — the rest of the
scenario (policies, disruption, lifecycle, seeds) is still authored by
the human/script.

Exposes:

- ``LLMClient`` — Protocol consumed by ``WorldBuilder`` for test injection.
- ``OpenAIClient`` — concrete wrapper around ``openai.OpenAI`` doing one
  structured-output completion per call.
- Pydantic schemas for every structured-output payload (Taxonomy,
  Catalog, Correlations, FreshnessSet, StoreTemplateList, MarketDomain).
- ``World`` / ``WorldBuilder`` — the artifact + the orchestrator that
  produces it.

The OpenAI client is imported eagerly here so plain ``from src.llm
import OpenAIClient`` works without diving into the sub-module.
"""

# Concrete OpenAI client + the Protocol seam consumed by WorldBuilder.
from src.llm.openai_client import LLMClient, OpenAIClient
# Pydantic schemas — one per structured-output payload.
from src.llm.schemas import (
    Catalog,
    CatalogItem,
    Correlations,
    FreshnessSet,
    InitFreshness,
    ItemFreshness,
    ItemRelations,
    MarketDomain,
    RelatedRef,
    Seasonality,
    StoreTemplateList,
    StoreTemplateSpec,
    Taxonomy,
    TaxonomyCategory,
)
# Top-level pipeline.
from src.llm.world_builder import World, WorldBuilder

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "Catalog",
    "CatalogItem",
    "Correlations",
    "FreshnessSet",
    "InitFreshness",
    "ItemFreshness",
    "ItemRelations",
    "MarketDomain",
    "RelatedRef",
    "Seasonality",
    "StoreTemplateList",
    "StoreTemplateSpec",
    "Taxonomy",
    "TaxonomyCategory",
    "World",
    "WorldBuilder",
]
