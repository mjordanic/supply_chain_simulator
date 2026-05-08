# `init_freshness` baseline / fresh mode

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Add `init_freshness: Literal["baseline", "fresh"]` to `StoreTemplate` (default `"baseline"`). `StoreInitializer.init_store_state` computes `activation_tick` per mode:

- `"baseline"` (established store) ⇒ `activation_tick[pid]` is set such that `Store.freshness_multiplier(pid, 0) == 1` for every initial active SKU — initial active SKUs skip the hype window
- `"fresh"` (grand-opening) ⇒ `activation_tick[pid] = 0` for all initial active SKUs — full hype at step 0

JSON round-trip covers the new field.

## Acceptance criteria

- [ ] `StoreTemplate.init_freshness: Literal["baseline", "fresh"]` added with default `"baseline"`
- [ ] `init_store_state` produces `activation_tick` such that `Store.freshness_multiplier(pid, 0) == 1` for all initial active SKUs when `init_freshness == "baseline"`
- [ ] `init_store_state` sets `activation_tick[pid] = 0` for all initial active SKUs when `init_freshness == "fresh"`
- [ ] `Scenario.to_json()` / `from_json()` round-trips `init_freshness`
- [ ] Test pins both modes via `Store.freshness_multiplier` after construction
- [ ] Existing example scenarios (default `"baseline"`) run with unchanged demand at step 0

## Testing notes

Pin observable behavior — the multiplier value at step 0 — not the `activation_tick` value itself (that's an internal mechanism). The contract a caller depends on is "at step 0, baseline gives multiplier 1; fresh gives multiplier 1 + α".

## Blocked by

Issues 03, 05
