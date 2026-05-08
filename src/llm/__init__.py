"""LLM world-builder pipeline.

Exposes:

- ``LLMClient`` — Protocol consumed by ``WorldBuilder`` for test injection.
- ``OpenAIClient`` — concrete wrapper around ``openai.OpenAI`` doing one
  structured-output completion per call.
- Pydantic schemas for every structured-output payload.
- ``WorldBuilder`` — produces a ``World`` (catalog, ``MarketParams``,
  ``dict[str, StoreTemplate]``) from one ``archetype`` string.
"""

from src.llm.openai_client import LLMClient, OpenAIClient
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
