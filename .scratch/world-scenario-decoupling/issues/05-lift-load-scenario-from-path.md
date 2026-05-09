## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Lift the import-by-path logic currently in `main.py:_load_scenario` into a free function `load_scenario_from_path(path: str | Path) -> Scenario` in `scenario.py`. The function imports a Python module by filesystem path, fetches the module's `scenario` attribute, and returns it as a live `Scenario` (with `Policy` instances on every `StoreInstance` preserved).

`main.py:_load_scenario` becomes a thin wrapper that calls `load_scenario_from_path`. No CLI behaviour change.

The motivation: notebooks and the CLI share one entry point. The notebook path returns the live module (policies present), unlike `Scenario.from_json` which is for historical-run inspection (policies stripped).

## Acceptance criteria

- [ ] `load_scenario_from_path(path)` defined as a free function in `scenario.py`.
- [ ] `main.py:_load_scenario` is a thin call into `load_scenario_from_path` (no divergent module-loading code).
- [ ] Loading `scenarios/example_homogeneous.py` (cheap, no LLM) returns a `Scenario` with non-empty `catalog`, non-empty `stores` list, and `stores[i].policy is not None` for at least one store.
- [ ] Loading a path that doesn't define a module-level `scenario` attribute raises a clear, named error (e.g. `AttributeError` with a helpful message, or a domain-specific error — pin whichever is chosen).
- [ ] `uv run python main.py scenarios/example_homogeneous.py` (or equivalent existing CLI invocation) still succeeds end-to-end.
- [ ] Integration test in `tests/sim/test_scenario.py` exercises the live-module-import case and the missing-attribute error case.

## Blocked by

None - can start immediately.
