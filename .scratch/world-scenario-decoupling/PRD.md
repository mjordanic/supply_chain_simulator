# World as a persistable artifact, with notebook-friendly inspection

Status: ready-for-agent

## Problem Statement

I'm authoring scenarios for the simulator and the path from `WorldBuilder` to `Runner` has two sharp edges that hurt me daily.

1. **Every scenario re-runs the LLM.** `WorldBuilder.build()` returns a `World(catalog, market, store_templates)` that lives only in process memory. Existing scenario scripts (`scenarios/example_llm_world.py:46–48`, `scenarios/llm_world_100.py`) call `_world = _builder.build(...)` at module import time, so every `uv run python main.py scenarios/example_llm_world.py` issues 5+ OpenAI calls. There's no `World.to_json` / `from_json`, so the only persistence path is to stuff the world into a Scenario and serialise that — which conflates LLM-authored content (catalog, market, store templates) with hand-authored experiment knobs (disruption, lifecycle, policies, seeds).

2. **There's no first-class way to inspect a World or a Scenario in a notebook.** Both are typed dataclasses with no DataFrame views. `notebooks/01-openai_world_builder.ipynb` hand-rolls `pd.DataFrame([w._asdict() for w in world.catalog])` for catalog and templates; nothing equivalent exists for stores or for the full Scenario. Inspecting a Scenario means reading raw JSON or running the simulation and reading `data/<run>/data/*.parquet` afterwards.

Together: I can't build a world once and reuse it across many policy/disruption/store-roster experiments without paying the LLM bill again, and I can't open a notebook on a world or a scenario and see "all info as a DataFrame" before committing to a run.

## Solution

Promote `World` from a transient dataclass to a saved artifact at `data/worlds/<name>/world.json`, and add tidy DataFrame inspection methods to both `World` and `Scenario`.

1. **World gains `to_json` / `from_json` / `to_dict` / `from_dict` plus a `meta` block** (`archetype`, `n_items`, `model`, `builder_version`, `built_at`). `WorldBuilder.build()` populates `meta` automatically. The block is recorded for human inspection only — never validated on load.

2. **`load_or_build_world(name, build_fn)` is the canonical script entry point.** Cache hit → load. Cache miss → print a warning, prompt interactively (`Press Enter to proceed, anything else to abort`), then build and save. Non-TTY contexts (CI, piped stdin) raise `LLMBuildAbortedError` unless `auto_confirm=True`.

3. **`Scenario.from_world(world, ...)` makes the World→Scenario relationship explicit at the call site** and removes the `catalog=world.catalog, market=world.market` boilerplate from every scenario script. The remaining six kwargs (`disruption`, `item_lifecycle`, `stores`, `n_steps`, `start_date`, `world_seed`) stay required — silent defaults would be footguns.

4. **DataFrame inspection methods on both objects** (`world.catalog_df()`, `scenario.stores_df()`, etc.). Discoverable via tab-completion; one method per logical view.

5. **`load_scenario_from_path()` is lifted out of `main.py:_load_scenario`** into `scenario.py` so notebooks and the CLI share one entry point. The Scenario-inspection notebook imports the live module (policies present) rather than the saved JSON snapshot (policies stripped).

6. **`DataExporter.save_products_parquet` / `save_stores_parquet` delegate to `Scenario.catalog_df()` / `stores_df()`**, layering run-log enrichment (`freshness_alpha`, `init_stock_share`) on top. Pure refactor; parquet schemas remain byte-identical.

The CRN demand-draw contract (ADR 0003) is untouched. The two-layer lifecycle (ADR 0001) is untouched. The `make_stores(triples)` helper is untouched.

## User Stories

1. As a scenario author, I want my LLM-built world to be saved to disk on first build and reused on subsequent runs, so that re-running a scenario script doesn't burn fresh OpenAI tokens every time.

2. As a scenario author, I want one canonical helper (`load_or_build_world`) for "load the world or build it", so that I don't have to write cache-then-build glue in every scenario script.

3. As a scenario author, I want saved worlds to live at a predictable path (`data/worlds/<name>/world.json`), so that I can find and load them from any notebook or script without configuration.

4. As a scenario author who built a 12-item world six weeks ago and now wants 100 items, I want to bump the world name (`fashion_v1` → `fashion_v1_100`) and have a fresh build go to a new path, so that my old artifact is preserved.

5. As a scenario author, I want every saved `world.json` to carry a `_meta` block recording archetype, n_items, model, builder_version, and build timestamp, so that opening a file later tells me what produced it.

6. As a scenario author who realises the world they're loading isn't cached, I want the program to print a clear warning naming the missing path, so that I have a chance to abort instead of silently kicking off an LLM build.

7. As a scenario author who realises the cache is missing, I want the program to prompt me interactively (`Press Enter to proceed`), so that the LLM build is consensual and I can hit anything-but-Enter to abort.

8. As a scenario author running a script under cron / CI / `< /dev/null`, I want a non-TTY cache miss to raise `LLMBuildAbortedError` by default, so that scheduled jobs never silently rack up an OpenAI bill.

9. As a scenario author who has decided to rebuild without supervision, I want to pass `auto_confirm=True` to `load_or_build_world`, so that the prompt is bypassed at one explicit call site.

10. As a scenario author who wants to force a rebuild even when the cache is fresh (e.g. the LLM has improved), I want a `force_rebuild=True` kwarg orthogonal to `auto_confirm`, so that I still get the prompt before re-issuing the calls.

11. As a scenario author building a fresh experiment, I want to write `Scenario.from_world(world, disruption=…, item_lifecycle=…, stores=make_stores([…]), n_steps=…, start_date=…, world_seed=…)`, so that the catalog and market come from the world automatically and the experiment knobs stay explicit.

12. As a scenario author, I want to reuse one saved world across many Scenarios with different disruption parameters, lifecycle settings, store rosters, and seeds, so that comparing those knobs doesn't require rebuilding the world.

13. As a scenario author who wants to swap only the store roster (different `init_seed`s, different policies, different counts), I want to load the saved world and pass a fresh `make_stores([…])` list to `Scenario.from_world`, so that no World-level edit is needed.

14. As a scenario author who wants to edit a single store template (e.g. flagship capacity 500 → 800), I want to load the world, mutate via `dataclasses.replace`, and save under a new world name, so that the original world is preserved and the variant is its own artifact.

15. As a scenario author who wants entirely fresh store templates against the same catalog and market, I want the same load → replace `world.store_templates` → save-under-new-name pattern as (14), so that there's exactly one way to derive a variant world.

16. As a notebook user inspecting a World, I want `world.catalog_df()` to return one row per `Ware` with product_id, name, category, base_price, unit_cost, margin, seasonality, freshness params, and related products, so that I can sanity-check the LLM's output in a tidy table.

17. As a notebook user, I want `world.store_templates_df()` to return one row per `StoreTemplate`, so that I can compare templates side by side.

18. As a notebook user, I want `world.market_df()` to return a single-row summary of `MarketParams`, with regions and seasonality unfolded into list-typed cells, so that I can see all the market knobs at once.

19. As a notebook user, I want `world.meta_df()` to surface the `_meta` block (archetype, n_items, model, builder_version, built_at), so that I know what built this world without opening the JSON.

20. As a notebook user inspecting a Scenario, I want `scenario.summary_df()`, `catalog_df()`, `stores_df()`, `market_df()`, `disruption_df()`, and `lifecycle_df()` methods, so that I can see every part of an experiment as a DataFrame without re-implementing row construction.

21. As a notebook user, I want `scenario.stores_df()` to include `policy_class` (when policies are attached), so that I can verify my roster's policy assignment matches my intent.

22. As a notebook user inspecting a *historical* run via `Scenario.from_json("data/<run>/config/scenario.json")`, I want the same DataFrame methods to work (with `policy_class = None` since policies aren't serialised), so that I can audit any past run with the same surface.

23. As a notebook user, I want to load a scenario module by path with `load_scenario_from_path("scenarios/<file>.py")` and get the live Scenario back including policies, so that I can inspect the exact object the Runner would consume.

24. As a notebook user opening `notebooks/03-inspect_scenario.ipynb` against a scenario whose world isn't cached, I want the import to surface the `load_or_build_world` prompt (or fail with `LLMBuildAbortedError` in non-TTY), so that I never accidentally trigger an LLM build by clicking "Run All".

25. As a `notebooks/01-openai_world_builder.ipynb` reader, I want the existing ad-hoc `pd.DataFrame([w._asdict() ...])` cells replaced with the new methods, so that the canonical example uses the canonical API.

26. As a `notebooks/01-openai_world_builder.ipynb` user, I want the final cell to write `world.to_json("data/worlds/<name>/world.json")`, so that running the notebook produces an artifact subsequent scenario scripts will load.

27. As a researcher reading `notebooks/02-inspect_world.ipynb`, I want a worked example of the (b)/(c) "load → mutate → save under new name" pattern, so that deriving a variant world is something I copy-paste rather than invent.

28. As a downstream analyst loading `data/<run>/data/products.parquet` after the refactor, I want the schema and values to be byte-identical to the pre-refactor output, so that my existing analysis notebooks don't break.

29. As a developer, I want `World.to_json` → `from_json` to round-trip every field including `meta`, distribution-typed catalog overrides, and store templates, so that load-bearing serialisation invariants don't sit as emergent properties of the dataclass field set.

30. As a developer, I want the Scenario DataFrame methods and `DataExporter`'s parquet writers to share row-construction code in one place (`Scenario.catalog_df()` / `stores_df()`), so that "what columns does the scenario expose" has exactly one answer.

31. As a `CONTEXT.md` reader, I want the WorldBuilder and Scenario glossary entries updated to mention the saved-world path and the canonical `load_or_build_world` / `Scenario.from_world` entry points, so that the glossary stays the single source of truth for the seam.

## Implementation Decisions

### Modules modified in place

- **`world_builder.py`** — `World` gains an optional `meta: dict[str, Any] | None` field and four serialisation methods (`to_dict`, `to_json`, `from_dict`, `from_json`). `to_json` accepts an optional path and writes through; `from_json` accepts either a path or a JSON string. Reuses the existing `_ware_to_dict` / `_ware_from_dict` and `MarketParams.to_dict` / `StoreTemplate.to_dict` round-trips — no new schema work for the contained types. `WorldBuilder.build()` populates `meta` with `{archetype, n_items, model, builder_version, built_at}` (`builder_version` is a module constant bumped manually when WorldBuilder behaviour changes). Adds four DataFrame methods (`catalog_df`, `store_templates_df`, `market_df`, `meta_df`). Adds `LLMBuildAbortedError` and `load_or_build_world(name, build_fn, *, base_dir, force_rebuild, auto_confirm)`.

- **`scenario.py`** — `Scenario` gains `from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed)` (the `world` param is typed under `TYPE_CHECKING` to avoid an import cycle). Six DataFrame methods on `Scenario`: `catalog_df`, `stores_df`, `market_df`, `disruption_df`, `lifecycle_df`, `summary_df`. New free function `load_scenario_from_path(path) -> Scenario` housing the import-by-path logic currently in `main.py:_load_scenario`.

- **`main.py`** — `_load_scenario` becomes a thin call into `load_scenario_from_path`. No CLI behaviour change.

- **`data_exporter.py`** — `save_products_parquet` and `save_stores_parquet` delegate to `Scenario.catalog_df()` / `stores_df()`. The exporter's run-log enrichment (`freshness_alpha`, `freshness_decay`, `init_stock_share` columns from `run_log["global"]["products"][pid]`) is layered on top of the Scenario DataFrame, not duplicated. Parquet schema and values must be byte-identical to the pre-refactor output for a fixed scenario.

- **Existing scenario scripts** — `scenarios/example_llm_world.py` and `scenarios/llm_world_100.py` switch to `load_or_build_world(...)` and `Scenario.from_world(...)`. `scenarios/example_llm_world_offline.py` keeps its `CannedClient` build path (the offline determinism is the point) but switches to `Scenario.from_world(_world, ...)` to demonstrate the helper. No top-of-file consent boilerplate in any script — consent happens per-call inside `load_or_build_world`.

- **Notebooks** — `notebooks/01-openai_world_builder.ipynb` is updated to use the new DataFrame methods and ends with `world.to_json("data/worlds/<name>/world.json")`. `notebooks/02-inspect_world.ipynb` is new: loads any saved world, walks the DataFrame views, demonstrates the (b)/(c) copy-and-mutate-and-save pattern. `notebooks/03-inspect_scenario.ipynb` is new: calls `load_scenario_from_path("scenarios/<file>.py")` and displays all six Scenario DataFrame views; opens with a header note about the cache-miss prompt.

- **`CONTEXT.md`** — WorldBuilder entry mentions the saved-world path, the `_meta` block, and `load_or_build_world` as the canonical entry point with interactive consent. Scenario entry references `Scenario.from_world` and the new DataFrame methods. Cross-link the new notebooks.

### Cache-then-build semantics

`load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False)`:

- `path = Path(base_dir) / name / "world.json"`.
- If `path.exists()` and not `force_rebuild` → return `World.from_json(path)`. No prompt, no warning.
- Otherwise: print a two-line warning to `stderr` naming the path and the LLM cost (`5+ OpenAI calls`).
- If `auto_confirm=False`, call `input("  Press Enter to proceed, anything else to abort: ")`. Empty/whitespace response → proceed. Anything else → raise `LLMBuildAbortedError` with the user input quoted in the message. `KeyboardInterrupt` / `EOFError` → raise `LLMBuildAbortedError` noting the exception type and that `auto_confirm=True` exists for non-interactive runs.
- Then call `build_fn()`, write via `world.to_json(path)`, return.

`force_rebuild` and `auto_confirm` are orthogonal: `force_rebuild=True` triggers the prompt; `auto_confirm=True, force_rebuild=False` proceeds silently on cache miss but still returns the cached file when present.

### Cache-key conventions (no validation, by design)

The cache key is the user-provided `name`. There is no hashing of build params and no validation of the `_meta` block on load — the meta is a human-readable record only. The discipline of bumping the name when build params (archetype, n_items, model) change is the user's. A stale-cache footgun is mitigated by:

1. The `_meta` block makes "what produced this file?" answerable in one read.
2. The interactive prompt makes every build a deliberate act, so renaming is in the user's flow.

This is a deliberate trade-off. Hashing was considered (file-per-hash under `<name>/`) and rejected — the orphan-file management cost outweighs the silent-stale risk in a personal-project setting.

### "Change stores" coverage

Three flavours of "change stores" are supported, all without API additions beyond what's listed above:

- **(a) Roster only** (different `init_seed`s, policies, counts) — edit the Scenario's `make_stores([…])` list. World untouched. Already supported by today's `Scenario` — `Scenario.from_world` just makes it slightly less verbose.
- **(b) Edit a store template** (capacity, balance, lead time) — load world, mutate via `dataclasses.replace(world, store_templates={**world.store_templates, "flagship": replace(world.store_templates["flagship"], capacity=800)})`, `world.to_json("data/worlds/<new_name>/world.json")`.
- **(c) Replace templates wholesale** — same load → replace `store_templates` → save-under-new-name pattern as (b).

There is no "swap parts across worlds" combinatorial mode (e.g. world-A's catalog + world-B's templates). Copy-and-edit is the only path.

### Scenario.from_world

Six required kwargs. Defaults considered (`event_prob=0`, no-transition lifecycle) and rejected — silent defaults that change simulation behaviour are footgunny. The value of `from_world` is symbolic (it makes the World→Scenario relationship explicit at the call site) plus boilerplate elimination for `catalog` and `market` only. The remaining knobs stay author-supplied.

### Scenario inspection: import-module, not load-JSON

`load_scenario_from_path` is the canonical inspection entry point because:

- It returns the live Scenario including `Policy` instances on every `StoreInstance` (so `scenario.stores_df().policy_class` is populated).
- It works *before* a run (no need to wait for `data/<run>/config/scenario.json` to be written).
- Under `load_or_build_world`, importing a scenario module is cheap on cache hit (no LLM) and prompts on cache miss (consensual).

The JSON path (`Scenario.from_json(open("data/<run>/config/scenario.json").read())`) remains supported for inspecting *historical* runs — same DataFrame methods, with `policy_class = None`.

### DataExporter parity

The refactor moves row construction for `products.parquet` and `stores.parquet` out of `data_exporter.py` and into `Scenario.catalog_df()` / `stores_df()`. The exporter then layers run-log enrichment on top:

```
products_df = scenario.catalog_df().assign(
    freshness_alpha=…, freshness_decay=…, init_stock_share=…   # from run_log["global"]["products"][pid]
)
products_df.to_parquet(path)
```

Pre/post-refactor parquet outputs must be byte-identical for a fixed scenario. Verified via diff in the verification flow.

## Testing Decisions

A good test here pins the **external contract** — the round-trip a downstream caller depends on, the schema a notebook reader sees, the byte-equality of an artifact a future analysis depends on — not the internal structure (don't assert `_meta` is a `dict` rather than a `dataclass`; don't assert the order in which `to_dict` populates its keys; don't assert that `load_or_build_world` calls `Path.exists` exactly once).

### Modules that get tests

1. **`World.to_json` / `from_json` round-trip** (unit). Pin: a `World` with non-trivial `catalog`, `market`, `store_templates`, and `meta` survives `to_json` → `from_json` with structural and numerical equality. Distribution-typed `Ware` overrides are preserved through the round-trip (uses `Distribution.to_dict` / `distribution_from_dict`). A world with `meta=None` round-trips to a world with `meta=None` (or `{}` — pin whichever is canonical).

2. **`load_or_build_world` cache behaviour** (integration, with monkeypatched `build_fn` that returns a fixture world). Pin:
   - Cache hit (file exists, not `force_rebuild`) → returns the saved world; `build_fn` is not called.
   - Cache miss + `auto_confirm=True` → calls `build_fn`, writes the file, returns the new world.
   - Cache miss + `auto_confirm=False` + non-TTY stdin (`sys.stdin = StringIO("")` or similar) → raises `LLMBuildAbortedError`; `build_fn` is not called; no file is written.
   - Cache miss + `auto_confirm=False` + simulated Enter (mock `input` returning `""`) → calls `build_fn`, writes the file.
   - Cache miss + `auto_confirm=False` + simulated abort (mock `input` returning `"n"`) → raises `LLMBuildAbortedError`; `build_fn` is not called.
   - `force_rebuild=True` + existing cache + `auto_confirm=True` → calls `build_fn` and overwrites the file.

3. **`Scenario.from_world` + DataFrame views** (unit). Pin:
   - `Scenario.from_world(world, disruption=…, …)` produces a Scenario whose `catalog` is `world.catalog` and `market` is `world.market`.
   - `scenario.catalog_df()` returns the expected columns and one row per `Ware`.
   - `scenario.stores_df()` returns one row per `StoreInstance` with `store_id`, `template_id`, `region`, `init_seed`, `policy_class`. `policy_class` is the policy class name when policies are attached and `None` when they aren't.
   - `scenario.summary_df()` is one row with `n_steps`, `start_date`, `world_seed`, `n_stores`, `n_products`.
   - `scenario.market_df()`, `disruption_df()`, `lifecycle_df()` each return at least their non-trivial fields as columns.
   - `world.catalog_df()` and `scenario.catalog_df()` agree on the catalog columns shared between them.

4. **`DataExporter` parquet parity** (regression). Pin: for a fixed scenario (the offline `CannedClient` example is ideal), running `DataExporter.export_all` before and after the refactor produces byte-identical `products.parquet` and `stores.parquet`. Implementation: capture the pre-refactor parquet outputs as test fixtures (committed under `tests/fixtures/`), run the post-refactor `DataExporter` against the same scenario, assert `pd.testing.assert_frame_equal(loaded_actual, loaded_fixture, check_dtype=True, check_exact=True)`.

5. **`load_scenario_from_path` module-import** (integration). Pin:
   - Loading `scenarios/example_llm_world_offline.py` returns a `Scenario` with a non-empty `catalog`, a non-empty `stores` list, and `stores[i].policy is not None` (live `BaselinePolicy` instances preserved).
   - Loading a path that doesn't define a `scenario` module attr raises a clear error.
   - The notebook and `main.py` paths use the same helper (no divergent module-loading code).

### Prior art

The existing test suite (`tests/llm/test_world_builder.py`, `tests/sim/test_scenario.py`) already covers `Scenario.to_json` / `from_json` round-trips and `WorldBuilder` stage outputs against the `CannedClient` test seam. The new tests follow the same shape: build a small `World` or `Scenario` against canned LLM responses (or a hand-built fixture), exercise the new method, assert on the result. No new test-harness machinery is required.

The `DataExporter` parity test is the only new shape — it captures pre-refactor parquet output as a fixture so the post-refactor version can be diffed exactly. This pattern can be reused if `DataExporter` grows further write paths.

## Out of Scope

- Hashing the build params into the cache key, or validating `_meta` on load. The user-side discipline of bumping `<name>` when params change, plus the interactive consent prompt, is the chosen mitigation. A future PRD can revisit if it bites.
- Mix-and-match of world parts across saved worlds (e.g. world-A's catalog + world-B's templates). The (b)/(c) copy-and-mutate-and-save pattern is the only supported variant-derivation path.
- Auto-detection of "notebook context" to switch defaults. Consent is per-call inside `load_or_build_world`; there is no module-level `allow_llm_builds()` / `block_llm_builds()` toggle.
- A `Scenario.to_dataframes() -> dict[str, DataFrame]` bundle method. Individual methods only — discoverability via tab-completion is the priority.
- Re-running the LLM to refresh `store_templates` only against an existing catalog (a `WorldBuilder.build_store_templates_only(market)` re-entry point). Loading + mutating + saving under a new name covers the use case for now.
- Migrating `notebooks/00-check_simulated_data.ipynb` to the new DataFrame methods. That notebook reads parquet outputs only and is unaffected by this PRD beyond the parquet-parity guarantee.
- Backward-compatibility shims for the inline-build pattern in `scenarios/example_llm_world.py` / `scenarios/llm_world_100.py`. The scripts are migrated as part of this PRD.

## Further Notes

- The `_meta` block schema is `{archetype, n_items, model, builder_version, built_at}`. `model` is `client.model_id` if the LLM client exposes it, else `None`. `builder_version` is a module-level constant (`"1"` initially) that I bump manually in code when `WorldBuilder` behaviour changes meaningfully.
- The interactive prompt in `load_or_build_world` must use `print(..., file=sys.stderr)` for the warning lines and `input(...)` for the prompt. `input` writes its prompt to stdout by default; in Jupyter both render in the cell output, so this is fine.
- The `DataExporter` parity guarantee is load-bearing: any analysis notebook that already reads `data/<run>/data/products.parquet` or `stores.parquet` continues to work without change. The byte-equality regression test in fixture form is the safety net.
- The (b)/(c) copy-and-mutate pattern relies on `dataclasses.replace`, which works because `Ware`, `StoreTemplate`, `MarketParams`, and `World` are all frozen-or-mutable dataclasses (or namedtuples that already support `_replace`). The notebook example demonstrates the exact incantation so users don't have to derive it.
- The Scenario inspection notebook deliberately avoids loading `data/<run>/config/scenario.json` as the primary path because that loses policy classes. The same DataFrame methods work on a `Scenario.from_json` result for historical-run audit, but `policy_class` is `None` there — a documented limitation, not a bug.
