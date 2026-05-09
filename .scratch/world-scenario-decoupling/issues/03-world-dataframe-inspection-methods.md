## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Add four DataFrame inspection methods to `World`, discoverable via tab-completion. One method per logical view; no bundle method.

- `world.catalog_df()` — one row per `Ware` with `product_id`, `name`, `category`, `base_price`, `unit_cost`, `margin`, `seasonality`, freshness params (`freshness_alpha`, `freshness_decay` etc. as the `Ware` exposes them), and related products.
- `world.store_templates_df()` — one row per `StoreTemplate` (template id, capacity, balance, lead time, region, and any other template fields).
- `world.market_df()` — single-row summary of `MarketParams`, with regions and seasonality unfolded into list-typed cells so all market knobs are visible at once.
- `world.meta_df()` — surfaces the `_meta` block as a single-row DataFrame with `archetype`, `n_items`, `model`, `builder_version`, `built_at`. When `meta is None`, return an empty DataFrame with the expected columns (or document the chosen behaviour in tests).

## Acceptance criteria

- [ ] All four methods exist on `World` and return a `pandas.DataFrame`.
- [ ] `catalog_df()` has one row per `Ware`; column set is the agreed list above.
- [ ] `store_templates_df()` has one row per `StoreTemplate`.
- [ ] `market_df()` is a single row with regions and seasonality as list-typed cells.
- [ ] `meta_df()` is a single row when `meta` is populated; behaviour for `meta=None` is documented in the test name.
- [ ] Unit tests pin column sets and row counts for each method against a fixture `World`.
- [ ] No `World.to_dataframes()` bundle method (out of scope per PRD).

## Blocked by

- `01-world-serialisation-and-meta-block.md`
