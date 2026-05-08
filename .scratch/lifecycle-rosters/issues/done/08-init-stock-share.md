# `init_stock_share` per-`Ware` weighted stock allocation

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Add `init_stock_share: float | None` to `Ware` and `default_init_stock_share: float | Distribution` to `ItemLifecycleParams`. `StoreInitializer.init_store_state` distributes the initial stock budget (`capacity * init_stock_pct`) across initially active products by normalized weights — staples (high `init_stock_share`) receive more initial stock than fashion (low share) within the same budget.

Resolved per-`Ware` values are captured in the static products parquet.

## Acceptance criteria

- [ ] `Ware.init_stock_share: float | None` added (optional)
- [ ] `ItemLifecycleParams.default_init_stock_share: float | Distribution` added
- [ ] `init_store_state` allocates per-product initial stock proportional to normalized `init_stock_share` weights, summing to ≤ `capacity * init_stock_pct`
- [ ] Per-`Ware` value resolved first; falls back to `default_init_stock_share`
- [ ] `DataExporter.save_products_parquet` includes the resolved `init_stock_share` column
- [ ] `Scenario.to_json()` / `from_json()` round-trips both fields
- [ ] Test pins: weights `(2, 1, 1)` across three active SKUs of equal cost distributes stock 2:1:1 within budget
- [ ] Test pins: total allocated stock ≤ `capacity * init_stock_pct`
- [ ] Existing example scenarios pass unchanged (uniform default share preserves prior behavior)

## Testing notes

Pin the proportional-distribution and budget contracts via `Store.inventory` after construction — that's the public interface. Don't pin specific integer rounding behavior unless it's part of the documented contract.

## Blocked by

Issue 05
