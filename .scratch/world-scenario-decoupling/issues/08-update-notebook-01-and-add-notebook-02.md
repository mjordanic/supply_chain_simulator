## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

**Update `notebooks/01-openai_world_builder.ipynb`** to use the new canonical APIs:

- Replace the existing ad-hoc `pd.DataFrame([w._asdict() for w in world.catalog])` cells (and any equivalents for templates) with calls to `world.catalog_df()`, `world.store_templates_df()`, `world.market_df()`, `world.meta_df()`.
- Add a final cell that writes `world.to_json("data/worlds/<name>/world.json")` so running the notebook produces a cached artifact subsequent scenario scripts will load.

**Add `notebooks/02-inspect_world.ipynb`** as the canonical "load any saved world and inspect it" notebook:

- Loads a world via `World.from_json("data/worlds/<name>/world.json")`.
- Walks each of the four DataFrame views with one cell per view and a header markdown cell explaining what's shown.
- Demonstrates the (b)/(c) "load → mutate → save under new name" pattern with worked examples:
  - **(b)** Edit a single store template, e.g. `dataclasses.replace(world, store_templates={**world.store_templates, "flagship": replace(world.store_templates["flagship"], capacity=800)})`, then `world.to_json("data/worlds/<new_name>/world.json")`.
  - **(c)** Replace `store_templates` wholesale via the same load → replace → save-under-new-name pattern.
- Uses the project's domain vocabulary (see `CONTEXT.md`).

## Acceptance criteria

- [ ] `notebooks/01-openai_world_builder.ipynb` no longer contains hand-rolled `pd.DataFrame([w._asdict() ...])` cells; uses `world.catalog_df()` / `store_templates_df()` / `market_df()` / `meta_df()` instead.
- [ ] `notebooks/01-openai_world_builder.ipynb` ends with `world.to_json("data/worlds/<name>/world.json")`.
- [ ] `notebooks/02-inspect_world.ipynb` exists and runs top-to-bottom against a saved world produced by notebook 01 (or an offline fixture).
- [ ] Notebook 02 covers all four `World` DataFrame views with explanatory markdown cells.
- [ ] Notebook 02 contains worked, runnable examples of the (b) edit-one-template and (c) replace-templates-wholesale patterns, each saving under a distinct new world `<name>`.
- [ ] Notebook 02 cleans up or marks demo artifacts so it can be re-run without leaking state across runs (or documents the intent if it doesn't).

## Blocked by

- `01-world-serialisation-and-meta-block.md`
- `03-world-dataframe-inspection-methods.md`
