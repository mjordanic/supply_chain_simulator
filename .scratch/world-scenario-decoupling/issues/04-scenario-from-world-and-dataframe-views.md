## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Add `Scenario.from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed) -> Scenario` to `scenario.py`. The `world` parameter is typed under `TYPE_CHECKING` to avoid an import cycle. Six required kwargs — no defaults. Internally it sets `catalog=world.catalog, market=world.market` and forwards the rest. Symbolic value is making the World→Scenario relationship explicit at the call site and removing `catalog=`/`market=` boilerplate.

Add six DataFrame methods on `Scenario`:

- `catalog_df()` — one row per `Ware`; column set agrees with `World.catalog_df()` for shared columns.
- `stores_df()` — one row per `StoreInstance` with `store_id`, `template_id`, `region`, `init_seed`, `policy_class`. `policy_class` is the policy class name when policies are attached, `None` otherwise (e.g. when loaded from `Scenario.from_json`).
- `market_df()` — single-row summary, columns agree with `World.market_df()` shape.
- `disruption_df()` — at minimum the non-trivial fields of the disruption config as columns.
- `lifecycle_df()` — at minimum the non-trivial fields of the item lifecycle config as columns.
- `summary_df()` — one row with `n_steps`, `start_date`, `world_seed`, `n_stores`, `n_products`.

## Acceptance criteria

- [ ] `Scenario.from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed)` exists with all six kwargs required (no defaults).
- [ ] `Scenario.from_world(world, …).catalog is world.catalog` and `.market is world.market`.
- [ ] All six DataFrame methods exist on `Scenario` and return a `pandas.DataFrame`.
- [ ] `stores_df()` includes `policy_class` column; populated with class name when policies attached, `None` when not.
- [ ] `summary_df()` is a single row with the five named columns.
- [ ] `world.catalog_df()` and `scenario.catalog_df()` return the same set of columns for the columns they share.
- [ ] DataFrame methods work on a `Scenario.from_json(...)` result (historical-run inspection), with `policy_class = None` in `stores_df()` documented as expected.
- [ ] Unit tests in `tests/sim/test_scenario.py` (or new file) pin: `from_world` produces equivalent Scenario; all six DataFrame methods return expected columns and row counts; `stores_df().policy_class` populates correctly with and without policies.
- [ ] No `Scenario.to_dataframes()` bundle method (out of scope per PRD).

## Blocked by

None - can start immediately.
