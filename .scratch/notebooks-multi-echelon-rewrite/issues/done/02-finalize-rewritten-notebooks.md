# Finalize & re-verify the already-rewritten notebooks (00, 01, 02, 07)

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

A previous slice (commit `8c99e04`) already rewrote the **contents** of four notebooks for the
graph engine, but left them under stale names/numbers. This slice finalizes their names and numbers
to the new lineup and re-verifies each runs top-to-bottom.

Renames (content already correct, numbering/name does not match contents):
- old `01-openai_world_builder` → `00-build-or-load-world` — LLM world generation; the OpenAI client
  is constructed **lazily inside the `build_fn`** passed to `load_or_build_world`, so a warm
  `fashion_retail_20` cache runs with no API key, and a cold cache without a key raises a clear,
  actionable message.
- old `02-inspect_world` → `01-inspect-world` — walk every `World` inspection surface (catalog,
  market, store templates) and derive a graph scenario via `world_to_graph`.
- old `03-inspect_scenario` → `02-inspect-scenario` — inspect a graph-mode `Scenario`
  (`nodes_df()`, `edges_df()`, catalog/market/lifecycle/disruption frames) before running.
- old `08-tune_textbook_policy` → `07-tune-textbook-policy` — Optuna tuning tutorial + analysis;
  reads `runs/tuning/order_up_to_v2_retail`.

Each notebook uses `%matplotlib inline`, modest figure DPI, and is committed with executed outputs
so plots render on GitHub's static viewer.

## Acceptance criteria

- [ ] The four notebooks exist under their new names (`00-build-or-load-world`, `01-inspect-world`, `02-inspect-scenario`, `07-tune-textbook-policy`) and the old-named files are gone
- [ ] `00` constructs the OpenAI client lazily inside `build_fn`; executing against the warm `data/worlds/fashion_retail_20/` cache needs no API key; a cold cache + missing key produces a clear, actionable error message
- [ ] Each notebook uses `%matplotlib inline` and is committed with embedded executed outputs
- [ ] `jupyter nbconvert --to notebook --execute` (loading `.env`) runs each of the four with zero cell errors and a rendered figure for every plotting cell

## Blocked by

None — can start immediately
