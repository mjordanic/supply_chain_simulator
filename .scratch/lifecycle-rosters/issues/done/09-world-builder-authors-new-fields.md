# `WorldBuilder` authors per-`Ware` and per-template fields end-to-end

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Extend the LLM `WorldBuilder` so it authors every new field introduced by the lifecycle / freshness / roster work.

Update Pydantic schemas in `src/llm/schemas.py`:

- catalog-stage schema gains per-`Ware` `init_stage`, `stage_change_probs` (per-stage dict keyed by `introduction`, `growth`, `maturity`, `decline`, `dead`), `freshness_alpha`, `freshness_decay`, `init_stock_share`
- store-templates-stage schema gains `init_active_products` (list of product IDs from the catalog) and `init_freshness` (`"baseline"` or `"fresh"`)

Update `src/llm/prompts.py` to prompt the LLM to author per-category values from realistic ranges:

- `freshness_alpha ∈ [0.1, 0.4]` for fashion-like categories; `freshness_alpha = 0` for staples
- `freshness_decay ∈ [15, 45]` ticks
- per-stage transition probs roughly `intro→growth: 0.02`, `growth→maturity: 0.005`, `maturity→decline: 0.001`, `decline→dead: 0.005`, `dead→intro: 0.003` — LLM may override per category

Update offline LLM fixtures so `scenarios/example_llm_world_offline.py` runs cleanly with the new schemas.

## Acceptance criteria

- [ ] `src/llm/schemas.py` catalog-stage schema includes `init_stage`, `stage_change_probs`, `freshness_alpha`, `freshness_decay`, `init_stock_share`
- [ ] `src/llm/schemas.py` store-templates-stage schema includes `init_active_products` and `init_freshness`
- [ ] `src/llm/prompts.py` prompts elicit per-category values for the new fields with the PRD-suggested ranges
- [ ] `scenarios/example_llm_world_offline.py` runs end-to-end and produces a non-empty run log
- [ ] `tests/llm/test_world_builder.py` extended with mocked-`LLMClient` coverage for the new schema fields and the catalog/store-templates retry path
- [ ] **`tests/llm/test_openai_live.py` extended (live OpenAI API)** with a test that:
  - Skips when `OPENAI_API_KEY` is missing (existing pattern)
  - Uses a tiny catalog (`n_items ≤ 5`) and `gpt-4o-mini` (or the model named by `OPENAI_TEST_MODEL`) for cost control (existing pattern)
  - Calls a real `WorldBuilder.build()` end-to-end
  - Pins that every catalog item has `init_stage`, `stage_change_probs` (with all 5 stages keyed), `freshness_alpha`, `freshness_decay`, and `init_stock_share` populated
  - Pins that every store template has `init_active_products` (non-empty list of valid product IDs from the catalog) and `init_freshness ∈ {"baseline", "fresh"}`
  - Pins that staples-category items have `freshness_alpha == 0`
  - Runs the produced `Scenario` through `Runner` for a small number of steps and asserts a non-empty run log
- [ ] `tests/llm/test_openai_strict_schema.py` and `tests/llm/test_schemas.py` updated to cover the new fields

## Testing notes

The live-API test is the load-bearing one — it catches schema/prompt mismatches that mocked tests cannot. A Pydantic schema can pass mock validation but still produce structured-output errors against the real OpenAI API, or the prompt can fail to elicit a field that the schema accepts as optional.

Drive the schema extension test-first: add a single field, write the live test for that field, watch it fail, fix the prompt, watch it pass — repeat per field. Do **not** bulk-add all new fields to the schema and prompt before the first live test runs. Mocked tests in `test_world_builder.py` are for fast CI coverage of the deterministic skeleton allocator and retry path; they are not a substitute for the live-API contract.

## Blocked by

Issues 02, 04, 06, 07, 08
