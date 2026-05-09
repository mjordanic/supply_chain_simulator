## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Migrate the three existing scenario scripts to the new canonical entry points. No top-of-file consent boilerplate — consent happens per-call inside `load_or_build_world`.

- `scenarios/example_llm_world.py` — replace inline `_world = _builder.build(...)` at module import with `_world = load_or_build_world("<name>", lambda: _builder.build(...))`. Switch the Scenario construction to `Scenario.from_world(_world, disruption=…, item_lifecycle=…, stores=make_stores([…]), n_steps=…, start_date=…, world_seed=…)`.
- `scenarios/llm_world_100.py` — same pattern with its own world `<name>` (e.g. matching the script's archetype/n_items).
- `scenarios/example_llm_world_offline.py` — keep the `CannedClient` build path (offline determinism is the point), but switch Scenario construction to `Scenario.from_world(_world, …)` to demonstrate the helper.

The script-side `name` choice should follow the implied convention from the PRD: bumping the name (`fashion_v1` → `fashion_v1_100`) when build params change is the user's discipline. Pick names that won't collide with each other or with future variants.

## Acceptance criteria

- [ ] `scenarios/example_llm_world.py` uses `load_or_build_world(...)` and `Scenario.from_world(...)`.
- [ ] `scenarios/llm_world_100.py` uses `load_or_build_world(...)` and `Scenario.from_world(...)`.
- [ ] `scenarios/example_llm_world_offline.py` keeps `CannedClient` build but uses `Scenario.from_world(_world, ...)`.
- [ ] No top-of-file consent boilerplate added in any script.
- [ ] `uv run python main.py scenarios/example_llm_world_offline.py` (deterministic, no LLM) still runs end-to-end with no behavioural change.
- [ ] First run of `uv run python main.py scenarios/example_llm_world.py` against an empty `data/worlds/` prints the consent warning and prompt; second run loads silently from cache.
- [ ] `tests/test_examples_and_cli.py` (or equivalent) still passes.

## Blocked by

- `02-load-or-build-world-helper.md`
- `04-scenario-from-world-and-dataframe-views.md`
