# Extract `StoreInitializer` (pure refactor)

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Move the existing `Store.__init__` step-0 setup logic into a pure function `init_store_state(template, init_seed, catalog) -> InitialStoreState` in its own module. `Store.__init__` delegates to it. No new authoring fields are introduced in this slice — the goal is to make the deep module exist and convert the bit-identity contract from an emergent property of `Store.__init__` into an single explicit assertion against `init_store_state`. Subsequent slices (06, 07, 08) plug new authoring fields into this seam.

`InitialStoreState` is a small dataclass holding: which SKUs are active, per-product initial stock allocation, `activation_tick` map, and resolved scalar values for any `Distribution`-typed template fields.

This is a pure refactor — no observable behavior changes.

## Acceptance criteria

- [ ] `StoreInitializer.init_store_state(template, init_seed, catalog) -> InitialStoreState` exists as a pure function in its own module
- [ ] `InitialStoreState` dataclass holds the active SKU set, per-product initial stock, `activation_tick` map, and resolved `Distribution` values
- [ ] `Store.__init__` delegates step-0 setup to `init_store_state` — no behavioral change
- [ ] Bit-identity contract is pinned by one explicit test against `init_store_state`: same `(template, init_seed, catalog)` ⇒ identical `InitialStoreState`
- [ ] Bit-identity is preserved end-to-end: two `Store` instances built from `make_stores` with the same `(template, init_seed)` and different policies produce identical step-0 state (existing determinism tests pass)
- [ ] Existing CRN/regression tests pass

## Testing notes

Write the bit-identity test against `init_store_state` first (RED — function doesn't exist), then extract just enough logic to make it pass (GREEN). Do **not** test the order of attribute assignment in `Store.__init__` or any internal mechanics. The point of the extraction is to have one explicit assertion replace what was previously an emergent property of `Store.__init__`.

## Blocked by

Issue 01
