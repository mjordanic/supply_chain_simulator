# `init_active_products` explicit per-template roster

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Add `init_active_products: list[str] | None` to `StoreTemplate`. `StoreInitializer.init_store_state` respects the explicit list when set ("this fashion specialist starts with these 30 SKUs"); falls back to `init_rng.sample(catalog, init_active_count)` when unset. JSON round-trip covers the new field.

## Acceptance criteria

- [ ] `StoreTemplate.init_active_products: list[str] | None` added (optional, default `None`)
- [ ] `init_store_state` activates exactly the listed product IDs when `init_active_products` is set
- [ ] `init_store_state` falls back to random sampling of size `init_active_count` when `init_active_products` is `None` (existing behavior preserved)
- [ ] `Scenario.to_json()` / `from_json()` round-trips `init_active_products`
- [ ] Test pins: explicit list ⇒ that exact set of SKUs active at step 0
- [ ] Test pins: unset ⇒ random sample of size `init_active_count`, deterministic given `init_seed`
- [ ] Existing example scenarios pass unchanged (they don't set `init_active_products`)

## Testing notes

Pin behavior through `Store` after construction (which products are active) — that's the public interface a caller depends on. Don't pin `StoreInitializer` internals. Deterministic-given-seed is part of the bit-identity contract — pin it.

## Blocked by

Issue 05
