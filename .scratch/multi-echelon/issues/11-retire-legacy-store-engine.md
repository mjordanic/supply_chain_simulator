# Retire legacy Store engine: delete, rename, migrate scenarios, world_to_graph helper, rewrite legacy tests

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Delete the single-store path. Every scenario is now a graph; single-store cases are 1F→1Shop→1Sink graphs. One engine, one `Scenario` shape.

**Deletions.**
- `src/sim/store.py`
- `src/sim/store_initializer.py`
- `StoreInstance`, `StoreTemplate`, `stores`, `make_stores` from `scenario.py`
- `HeuristicPolicy` (retained only for parameter-demo scenarios, not the canonical anchor per ADR 0006)
- `RLPolicy` (replaced in issue 12 by `RLIntermediatePolicy`)
- `Market.sample_demand`

**Renames.** `GraphSimulation` → `Simulation`, `GraphRunner` → `Runner`. Public function names preserved (`build_world`, `Runner`, `Simulation`, `TickResult`). `Policy` ABC kept as one-release alias of `NodePolicy` for grep-friendly migration.

**Scenario migration.** Every file under `scenarios/` rewritten:
- `example_homogeneous.py`: each "store" expands to a 3-node sub-graph; `(template, init_seed)` shape preserved on the `IntermediateNode`
- `example_paired_comparison.py`: CRN-paired comparison = two identical 3-node graphs with different policies on the intermediate
- `example_llm_world*.py`, `llm_world_20/100/250/1000.py`: use new `world_to_graph(world, *, sink_density)` helper in `src/sim/world.py`; `World` artifact gains optional `default_graph_topology: dict | None` (LLM-authored hints, optional)

**Tests rewritten.** `test_full_run.py`, `test_runner_refactor.py`, `test_scenario.py`, `test_scenario_dataframe_views.py`. Delete `test_store_accounting.py`, `test_store_initializer.py`, `test_heuristic_policy.py`.

**Out of scope:** backward compatibility for legacy `Scenario` JSON — once this lands, old `stores`-shaped JSON cannot be loaded. Migration path is "rebuild from the scenario authoring script."

**Explicitly untouched per PRD §36:** `event_engine.py`, `item_registry.py`, `distributions.py`, `freshness_curve.py`, `lifecycle_clock.py`, `metrics.py`, `agents/ppo.py`, notebooks 04–07.

## Acceptance criteria

- [ ] `src/sim/store.py`, `src/sim/store_initializer.py` deleted
- [ ] `StoreInstance`, `StoreTemplate`, `stores`, `make_stores` removed from `scenario.py`
- [ ] `HeuristicPolicy`, `RLPolicy` deleted
- [ ] `Market.sample_demand` deleted
- [ ] `GraphSimulation` → `Simulation`, `GraphRunner` → `Runner` renamed; public function names preserved
- [ ] `Policy` kept as one-release alias of `NodePolicy`
- [ ] All `scenarios/*.py` files rewritten on graph topology
- [ ] `src/sim/world.py::world_to_graph(world, *, sink_density)` synthesises a default topology from `World.store_templates`
- [ ] `World` artifact gains optional `default_graph_topology: dict | None`
- [ ] Legacy tests `test_store_accounting.py`, `test_store_initializer.py`, `test_heuristic_policy.py` deleted
- [ ] `test_full_run.py`, `test_runner_refactor.py`, `test_scenario.py`, `test_scenario_dataframe_views.py` rewritten on graph engine
- [ ] `uv run pytest tests/sim` is green
- [ ] `uv run python main.py scenarios/example_homogeneous.py` runs on the graph engine (CLI muscle memory preserved)
- [ ] `uv run python scenarios/example_llm_world_offline.py` runs
- [ ] No file in `event_engine.py`, `item_registry.py`, `distributions.py`, `freshness_curve.py`, `lifecycle_clock.py`, `metrics.py`, `agents/ppo.py`, or notebooks 04–07 modified

## Blocked by

- `.scratch/multi-echelon/issues/10-lifecycle-freshness-sink-path.md`
