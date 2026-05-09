## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Update `CONTEXT.md` so the glossary stays the single source of truth for the World→Scenario seam.

- **WorldBuilder entry** — mention the saved-world artifact path (`data/worlds/<name>/world.json`), the `_meta` block schema (`archetype`, `n_items`, `model`, `builder_version`, `built_at`), and `load_or_build_world(name, build_fn, ...)` as the canonical script entry point with interactive consent on cache miss.
- **Scenario entry** — reference `Scenario.from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed)` as the explicit World→Scenario derivation, and the six DataFrame methods (`catalog_df`, `stores_df`, `market_df`, `disruption_df`, `lifecycle_df`, `summary_df`) as the canonical inspection surface.
- **Cross-link** the new notebooks (`notebooks/02-inspect_world.ipynb`, `notebooks/03-inspect_scenario.ipynb`) from the relevant glossary entries.
- Note the documented limitation that `Scenario.from_json` (historical-run audit) returns `policy_class = None` in `stores_df()` — by design, not a bug.
- No ADR changes — ADR 0001 (two-layer lifecycle) and ADR 0003 (CRN demand draws) are untouched by this work.

## Acceptance criteria

- [ ] `CONTEXT.md` WorldBuilder entry mentions `data/worlds/<name>/world.json`, the `_meta` block, and `load_or_build_world` as the canonical entry point.
- [ ] `CONTEXT.md` Scenario entry references `Scenario.from_world` and lists the six DataFrame methods.
- [ ] Notebooks 02 and 03 are cross-linked from the relevant glossary entries.
- [ ] `policy_class = None` historical-audit limitation is documented next to the relevant entry.
- [ ] No ADR files are modified.

## Blocked by

- `02-load-or-build-world-helper.md`
- `04-scenario-from-world-and-dataframe-views.md`
- `08-update-notebook-01-and-add-notebook-02.md`
- `09-add-notebook-03-inspect-scenario.md`
