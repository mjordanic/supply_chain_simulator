"""LLM world-builder pipeline.

This sub-package generates a catalog + market from a single archetype
string (``"fashion_retail"``, ``"grocery"``, …) and persists them as
``catalog.csv`` + ``setup.yaml``. It is the "give me a thematically
coherent catalog/market without hand-coding one" entry point — topology,
policies, disruption, and run parameters are still authored by the
human/script.

Exposes:

- ``LLMClient`` — Protocol consumed by ``WorldBuilder`` for test injection.
- ``OpenAIClient`` — concrete wrapper around ``openai.OpenAI`` doing one
  structured-output completion per call.
- Pydantic schemas for every structured-output payload (Taxonomy,
  Catalog, Correlations, FreshnessSet, StoreTemplateList, MarketDomain).
- ``WorldBuilder`` — orchestrates LLM stages and writes setup directories.

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
from src.llm.world_builder import WorldBuilder

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
    "WorldBuilder",
]
