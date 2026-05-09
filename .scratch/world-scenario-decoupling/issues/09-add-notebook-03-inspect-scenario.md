## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Add `notebooks/03-inspect_scenario.ipynb` — the canonical "open a scenario before running it and inspect every part" notebook.

- Opens with a markdown header note explaining the cache-miss behaviour: opening this notebook against a scenario whose world isn't cached will surface the `load_or_build_world` consent prompt (or fail with `LLMBuildAbortedError` in non-TTY contexts). Readers should never accidentally trigger an LLM build by clicking "Run All".
- Calls `load_scenario_from_path("scenarios/<file>.py")` to get the live `Scenario` (with policies attached). Use `scenarios/example_llm_world_offline.py` as the demo target so the notebook is reproducible without an OpenAI key.
- Displays each of the six Scenario DataFrame views with one cell per view and a markdown cell explaining what's shown:
  - `scenario.summary_df()`
  - `scenario.catalog_df()`
  - `scenario.stores_df()` (verify `policy_class` populates because policies are live)
  - `scenario.market_df()`
  - `scenario.disruption_df()`
  - `scenario.lifecycle_df()`
- Optional: include a final cell demonstrating the historical-run audit path with `Scenario.from_json("data/<run>/config/scenario.json")` + the same DataFrame methods, noting that `policy_class = None` there as documented.

Uses the project's domain vocabulary (see `CONTEXT.md`).

## Acceptance criteria

- [ ] `notebooks/03-inspect_scenario.ipynb` exists.
- [ ] Opens with a markdown cell explaining cache-miss / LLM-consent behaviour.
- [ ] Loads a scenario via `load_scenario_from_path("scenarios/example_llm_world_offline.py")` (or another non-LLM target).
- [ ] Displays all six Scenario DataFrame views, one cell each, with explanatory markdown.
- [ ] `stores_df()` cell visibly shows non-`None` `policy_class` values (proves live policies are preserved).
- [ ] Notebook runs top-to-bottom without an OpenAI key when pointed at the offline scenario.

## Blocked by

- `02-load-or-build-world-helper.md`
- `04-scenario-from-world-and-dataframe-views.md`
- `05-lift-load-scenario-from-path.md`
