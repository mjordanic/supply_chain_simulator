# Notebook `04-deep-dive-per-product`

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

Merge the old `04-deep_dive_per_product` and `04a-deep_dive_active_only` notebooks into a single
`04-deep-dive-per-product` on the graph engine: a per-product card grid (one card per SKU) showing
lifecycle stage, freshness multiplier, seasonal multiplier, demand, sales, inventory, and orders,
with an `ACTIVE_ONLY` flag at the top that switches between every product and active-only — removing
the need for the separate `04a`.

Runs live (`llm_world_20`), reads the in-memory `run_log`, and pulls per-`(tick, pid)` series via
`src/sim/inspect.py` (`per_product_df`), so the synthetic `_total` key is handled by the shared
parser rather than re-parsed here.

## Acceptance criteria

- [ ] `notebooks/04-deep-dive-per-product.ipynb` exists; old `04-deep_dive_per_product` and `04a-deep_dive_active_only` are removed
- [ ] Per-product card grid shows lifecycle stage, freshness multiplier, seasonal multiplier, demand, sales, inventory, and orders per SKU
- [ ] A single `ACTIVE_ONLY` flag toggles between all products and active-only
- [ ] Per-product series come from `src/sim/inspect.py` (`per_product_df`); the notebook runs live and reads the in-memory `run_log`
- [ ] `%matplotlib inline`; committed with embedded executed outputs; modest DPI
- [ ] `jupyter nbconvert --to notebook --execute` runs the notebook with zero cell errors and a rendered figure for every plotting cell

## Blocked by

- `.scratch/notebooks-multi-echelon-rewrite/issues/01-inspect-deep-module.md`
