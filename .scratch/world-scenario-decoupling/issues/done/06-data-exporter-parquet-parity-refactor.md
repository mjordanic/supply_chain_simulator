## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Refactor `data_exporter.py` so `save_products_parquet` and `save_stores_parquet` delegate row construction to `Scenario.catalog_df()` / `stores_df()`. The exporter then layers run-log enrichment (`freshness_alpha`, `freshness_decay`, `init_stock_share` columns sourced from `run_log["global"]["products"][pid]`) on top of the Scenario DataFrame, e.g.:

```
products_df = scenario.catalog_df().assign(
    freshness_alpha=…, freshness_decay=…, init_stock_share=…
)
products_df.to_parquet(path)
```

Pure refactor — parquet schemas and values must remain byte-identical to the pre-refactor output for a fixed scenario.

Verification: capture pre-refactor parquet outputs as committed test fixtures under `tests/fixtures/`. The post-refactor `DataExporter` runs against the same scenario; the test asserts `pd.testing.assert_frame_equal(loaded_actual, loaded_fixture, check_dtype=True, check_exact=True)`.

The offline `CannedClient` example (`scenarios/example_llm_world_offline.py`) is the ideal fixture source because its determinism makes the byte-equality assertion meaningful.

## Acceptance criteria

- [ ] `DataExporter.save_products_parquet` calls `scenario.catalog_df()` and layers run-log columns on top.
- [ ] `DataExporter.save_stores_parquet` calls `scenario.stores_df()` and layers any required run-log columns on top.
- [ ] No row-construction code duplicated between `Scenario` DataFrame methods and `DataExporter`.
- [ ] `tests/fixtures/` holds committed `products.parquet` and `stores.parquet` produced by the pre-refactor `DataExporter` against a deterministic scenario (the offline `CannedClient` example).
- [ ] Regression test loads both fixtures and asserts `pd.testing.assert_frame_equal(actual, fixture, check_dtype=True, check_exact=True)` for both.
- [ ] All existing tests in `tests/sim/` (especially `test_regression_snapshot.py` and `test_full_run.py`) still pass.

## Blocked by

- `04-scenario-from-world-and-dataframe-views.md`
