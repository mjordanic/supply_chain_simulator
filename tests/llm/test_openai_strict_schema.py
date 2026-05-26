"""Verify every LLM payload schema satisfies OpenAI structured-output strict mode.

This is the regression test for the ``MarketDomain.season_months: dict[str, list[int]]``
bug: the schema parsed fine in Pydantic but OpenAI's strict-mode JSON Schema
validator rejected it because ``additionalProperties`` may not be a subschema —
free-form object values aren't supported.

OpenAI's own ``to_strict_json_schema`` does NOT raise on this violation client-side
(it only auto-promotes default fields into ``required``); the request only fails
at the server. So we walk the strict-converted schema ourselves and assert the
two rules that actually bite:

1. ``additionalProperties`` must be ``false`` on every object (never a dict).
2. ``required`` must list every key in ``properties``.

Running this test on the pre-fix ``MarketDomain`` (where ``season_months`` was
``dict[str, list[int]]``) fails on rule 1; that's the bug we want to keep out.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from src.llm.schemas import (
    Catalog,
    CatalogItem,
    Correlations,
    FreshnessSet,
    ItemFreshness,
    ItemRelations,
    MarketDomain,
    RelatedRef,
    SeasonWindow,
    StoreTemplateList,
    StoreTemplateSpec,
    Taxonomy,
    TaxonomyCategory,
)


# Every Pydantic model used as ``response_format`` (or transitively reachable
# from one) must satisfy strict mode. Top-level ones are what ``WorldBuilder``
# actually passes to ``client.beta.chat.completions.parse``; the rest are
# nested but still get walked when we recurse.
_TOP_LEVEL_SCHEMAS: list[type[BaseModel]] = [
    Catalog,
    Correlations,
    FreshnessSet,
    MarketDomain,
    StoreTemplateList,
    Taxonomy,
]

_NESTED_SCHEMAS: list[type[BaseModel]] = [
    CatalogItem,
    ItemFreshness,
    ItemRelations,
    RelatedRef,
    SeasonWindow,
    StoreTemplateSpec,
    TaxonomyCategory,
]


def _walk_object_nodes(node: Any, path: str = "$"):
    """Yield every JSON Schema object node in ``node`` with its dotted path."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for k, v in node.items():
            yield from _walk_object_nodes(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_object_nodes(v, f"{path}[{i}]")


def _assert_openai_strict_compatible(schema_cls: type[BaseModel]) -> None:
    schema = to_strict_json_schema(schema_cls)
    violations: list[str] = []
    for path, obj in _walk_object_nodes(schema):
        ap = obj.get("additionalProperties")
        if ap is not False:
            violations.append(
                f"{path}: additionalProperties must be `false`, got {ap!r}. "
                f"Free-form `dict[str, X]` is not allowed in OpenAI strict mode."
            )
        props = obj.get("properties", {})
        required = set(obj.get("required", []))
        missing = set(props.keys()) - required
        if missing:
            violations.append(
                f"{path}: every property must be in `required`; missing {sorted(missing)}. "
                f"OpenAI strict mode forbids optional fields."
            )
    assert not violations, (
        f"\n{schema_cls.__name__} violates OpenAI strict mode:\n  - "
        + "\n  - ".join(violations)
    )


@pytest.mark.parametrize(
    "schema_cls",
    _TOP_LEVEL_SCHEMAS + _NESTED_SCHEMAS,
    ids=lambda c: c.__name__,
)
def test_schema_satisfies_openai_strict_mode(schema_cls: type[BaseModel]) -> None:
    _assert_openai_strict_compatible(schema_cls)


def test_walker_catches_free_form_dict_regression() -> None:
    """Sanity-check the walker itself: a model with the original bug shape
    must trip rule 1. Without this, a silent walker bug would let the
    real schemas drift back into a broken state undetected.
    """

    class Regressed(BaseModel):
        season_months: dict[str, list[int]]

    with pytest.raises(AssertionError, match="additionalProperties must be"):
        _assert_openai_strict_compatible(Regressed)


def test_walker_catches_optional_field_regression() -> None:
    """Synthetic schema with a property absent from ``required`` must trip
    rule 2. Sanity-check that the walker fires — needed because Pydantic's
    default-field path is auto-promoted by ``to_strict_json_schema``, so we
    can't reach this rule via a real model."""
    bogus_schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    found = [
        path
        for path, obj in _walk_object_nodes(bogus_schema)
        if set(obj.get("properties", {}).keys()) - set(obj.get("required", []))
    ]
    assert found, "walker failed to flag missing-required property"
