## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Promote `World` from a transient dataclass to a persistable artifact. Add an optional `meta: dict[str, Any] | None` field to `World` and four serialisation methods: `to_dict`, `to_json`, `from_dict`, `from_json`. `to_json` accepts an optional path and writes through; `from_json` accepts either a path or a JSON string. Reuse the existing `_ware_to_dict` / `_ware_from_dict` and `MarketParams.to_dict` / `StoreTemplate.to_dict` round-trips — no new schema work for the contained types.

`WorldBuilder.build()` populates `meta` automatically with `{archetype, n_items, model, builder_version, built_at}`. `model` is `client.model_id` if the LLM client exposes it, else `None`. `builder_version` is a module-level constant (`"1"` initially) that is bumped manually when `WorldBuilder` behaviour changes meaningfully. `built_at` is an ISO-8601 UTC timestamp at build time. The `_meta` block is for human inspection only — never validated on load.

## Acceptance criteria

- [ ] `World` has `meta: dict[str, Any] | None = None` field.
- [ ] `World.to_dict()` and `World.from_dict()` round-trip every field including `catalog`, `market`, `store_templates`, and `meta`.
- [ ] `World.to_json(path=None)` returns a JSON string when `path` is `None`, otherwise writes to `path` (creating parent directories) and returns the string.
- [ ] `World.from_json(source)` accepts either a `Path` / `str` filesystem path that exists, or a JSON string body.
- [ ] Distribution-typed `Ware` overrides survive a `to_json` → `from_json` round-trip (uses existing `Distribution.to_dict` / `distribution_from_dict`).
- [ ] A `World` with `meta=None` round-trips to a `World` with `meta=None` (or `{}` — pin whichever is canonical and document the choice in the test name).
- [ ] `WorldBuilder.build()` populates `meta` with the five keys (`archetype`, `n_items`, `model`, `builder_version`, `built_at`).
- [ ] `BUILDER_VERSION` is a module-level constant in `world_builder.py` initialised to `"1"`.
- [ ] Unit tests in `tests/llm/test_world_builder.py` (or a new `test_world_serialisation.py`) cover: full round-trip with non-trivial catalog/market/store_templates/meta; distribution-typed override round-trip; `meta=None` round-trip; `WorldBuilder.build()` populates the five meta keys.

## Blocked by

None - can start immediately.
