# `FreshnessCurve` + `activation_tick` + multiplier composition

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Add a per-(store, product) freshness multiplier `m(τ) = 1 + α · exp(−τ / β)` to demand math, where `τ` is ticks since the product was last activated in this store.

Extract `FreshnessCurve.multiplier(alpha, decay, ticks_since_activation) -> float` as a pure function in its own module — single swap-point for a future Bass-shaped diffusion or polynomial curve. Owns no state.

Add `activation_tick: dict[product_id, int]` to `Store`. `Store.activate_item(pid)` writes `activation_tick[pid] = current_step`; re-activation overwrites (curve resets). `deactivate_item` may leave the entry in place. Expose `Store.freshness_multiplier(pid, current_step) -> float` that delegates to `FreshnessCurve.multiplier` using catalog-wide `ItemLifecycleParams.default_freshness_alpha` and `default_freshness_decay`. Per-`Ware` overrides land in issue 04.

`Market.sample_demand` composes the freshness factor multiplicatively into the existing demand math. `Runner` plumbs `current_step` into the demand path so freshness is computed against the same step the demand draw uses.

For step-0 initial active SKUs, default to multiplier ≈ 1 (baseline-equivalent behavior — explicit per-template `init_freshness` modes land in issue 07).

**Order of factors in `Market.sample_demand` is part of the external CRN contract** (it determines floating-point identity in regression tests) and must be: `multiplier = stage_multiplier * freshness * season * promo * cross`.

**CRN demand-draw cardinality must not regress** — one `world_rng` draw per `(store, product)` per tick, even for inactive products (ADR 0003). `Runner._process_demand` continues to iterate `store.inventory.keys()` so every catalog product consumes one `world_rng` draw per tick.

## Acceptance criteria

- [ ] `FreshnessCurve.multiplier(alpha, decay, ticks_since_activation) -> float` exists as a pure function in its own module (no `Store` / `Market` / `Registry` imports)
- [ ] `Store.activation_tick: dict[str, int]` exists; populated by `activate_item(pid)` with the current step
- [ ] `Store.freshness_multiplier(pid, current_step) -> float` exists and delegates to `FreshnessCurve.multiplier` with catalog-wide defaults
- [ ] `ItemLifecycleParams.default_freshness_alpha: float | Distribution` and `default_freshness_decay: float | Distribution` added
- [ ] `Market.sample_demand` composes the freshness factor in order: `stage * freshness * season * promo * cross`
- [ ] `Runner` passes `current_step` into the demand path
- [ ] `Scenario.to_json()` / `from_json()` round-trips the new `default_freshness_*` fields
- [ ] `FreshnessCurve.multiplier` unit tests pin: `multiplier(α, β, 0) == 1 + α`; `multiplier(α, β, τ → ∞) → 1`; `multiplier(0, β, τ) == 1` for all τ; monotone-decreasing in τ for α > 0
- [ ] CRN regression test pins: a paired pair (same `(template, init_seed)`, different policies) consumes the same `world_rng` draws across an entire run; demand for an inactive product is logged with the inactive demand draw still firing one `world_rng` call per `(store, product)` per tick
- [ ] Re-activating a discontinued SKU resets τ to 0 (multiplier returns to `1 + α`) — pinned by an integration test against `Store` + `Market`
- [ ] Existing example scenarios continue to run with no demand spike at step 0 (baseline-equivalent behavior preserved)

## Testing notes

TDD `FreshnessCurve` as a pure-function unit first (one math property per test). Then a `Market` integration test that exercises the composition through `sample_demand` — don't mock the curve. The CRN cardinality test is the load-bearing regression — write it early so subsequent edits can't quietly break it. Pin behavior through public interfaces (`Store.freshness_multiplier`, `Market.sample_demand`, `Runner.run`), not internal mechanics.

## Blocked by

None - can start immediately
