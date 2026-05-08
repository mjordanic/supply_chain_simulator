# Per-`Ware` `freshness_alpha` / `freshness_decay` overrides

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Make `freshness_alpha` and `freshness_decay` authoring fields on `Ware`, with per-`Ware` values overriding the catalog-wide defaults from `ItemLifecycleParams`. `Store.freshness_multiplier(pid, current_step)` resolves the per-`Ware` value if present, otherwise falls back to the default. Both fields are optional — catalogs that don't set them inherit `default_*` and behave equivalently.

Resolved per-`Ware` values are captured in the static products parquet so downstream analysis can reproduce demand math without re-resolving overrides.

This unblocks "fashion ≈ 0.4 / 30, staples = 0" per-category authoring.

## Acceptance criteria

- [ ] `Ware.freshness_alpha: float | None` and `Ware.freshness_decay: float | None` added (both optional)
- [ ] `Store.freshness_multiplier(pid, current_step)` resolves per-`Ware` overrides first, falls back to `ItemLifecycleParams.default_*`
- [ ] `DataExporter.save_products_parquet` includes resolved `freshness_alpha` and `freshness_decay` columns for every catalog product
- [ ] `Scenario.to_json()` / `from_json()` round-trips the per-`Ware` freshness fields with numerical equality
- [ ] Test pins: a `Ware` with `freshness_alpha = 0` (staple) yields multiplier ≡ 1 for all τ regardless of catalog defaults
- [ ] Test pins: a `Ware` with overrides yields a multiplier matching its own `(α, β)`, not the defaults
- [ ] Test pins: a `Ware` with no overrides yields a multiplier matching `ItemLifecycleParams.default_*`
- [ ] Existing example scenarios pass unchanged

## Testing notes

Drive each behavior with one test at a time. The override / fallback resolution is the main contract — pin it through `Store.freshness_multiplier` (the public interface), not by inspecting internal lookup logic. Don't introduce a new test seam just to test the override resolution in isolation.

## Blocked by

Issue 03
