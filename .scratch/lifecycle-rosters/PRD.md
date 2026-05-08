# Realistic lifecycle, freshness curve, and explicit store rosters

Status: ready-for-agent

## Problem Statement

I'm a researcher running policy-evaluation experiments on the simulator. Three concrete frictions block me today.

1. **I can't compose realistic store rosters.** The two scenario-authoring helpers — `homogeneous(template, policy, n_copies)` and `paired(template, [a, b], n_pairs)` — can't express "five stores running policies A, A, B, C, D in templates T1, T2, T3, T4, T4, T4". The Runner's data model is general enough; the helpers aren't. `paired` is hard-coded to exactly two policies.

2. **The product lifecycle is unrealistically uniform.** Every product starts at the same `init_stage` and shares one `stage_change_prob`. Lifecycle stage is a single global value across all stores, so newly-stocked SKUs at one store get no "newness hype" — and CRN-paired stores see identical lifecycle transitions even when their policies activate at different times. Worse, `Item.advance_stage` cycles `decline → introduction` after a single transition draw, so every product re-launches on its own in a few months.

3. **Initial active assortment isn't authorable.** `init_active_count` is a number; which products start active is `init_rng.sample(catalog, n_active)`. There's no way to say "this fashion specialist starts with these 30 SKUs", or "staples get more of the initial stock than fashion within the same capacity budget".

Together: I can't run "test policy B in exactly the same conditions as policy A" experiments cleanly, and I can't make the world realistic enough to trust the policy comparisons that come out.

## Solution

Three coordinated changes, all driven by the design captured in `CONTEXT.md` and ADRs 0001–0003.

1. **Replace `homogeneous` and `paired` with one declarative authoring helper, `make_stores(triples)`**, where `triples: list[tuple[StoreTemplate, int, Policy]]`. CRN comparison is "repeat the same `(template, init_seed)` with different policies"; robustness sweeps are "vary `init_seed`". No regime abstraction above the list.

2. **Two-layer lifecycle model.** Keep the global PLC (Levitt-style stages owned by `ItemRegistry`) but augment with a per-(store, product) **freshness curve** `m(τ) = 1 + α · exp(−τ/β)` owned by `Store`, where `τ` resets to `0` on every `Store.activate_item`. Add a `dead` stage to the canonical list with a 5%-of-baseline multiplier and a tunable comeback probability (default ~0.003 per tick). Replace the scalar `stage_change_prob` with a per-current-stage dict so fashion can decline fast and staples can decline slow under one schema.

3. **Per-`Ware` lifecycle/freshness/stock authoring fields**, with sensible per-category defaults; per-template explicit `init_active_products` list and `init_freshness` mode (`"baseline"` for established stores, `"fresh"` for grand-opening). The LLM `WorldBuilder` authors all of these per category and per template.

The CRN demand-draw contract — every catalog product gets one `world_rng` draw per tick, even when inactive — stays untouched (ADR 0003).

## User Stories

1. As a policy researcher, I want to declare my store roster as an explicit list of `(template, init_seed, policy)` triples, so that I can run any policy/store combination — including duplicates and singletons — without going through a regime-specific helper.

2. As a policy researcher, I want two stores that share `(template, init_seed)` to start step-0 bit-identical regardless of attached policy, so that paired comparisons measure policy difference, not noise.

3. As a policy researcher, I want to express a k-way comparison ("policies A, B, C all on the same store") by repeating one `(template, init_seed)` with three different policies, so that CRN generalises beyond two-policy paired runs without API changes.

4. As a policy researcher, I want to express a robustness sweep ("policy A across diverse stores") by listing one policy with varying `init_seed` and varying templates, so that I can read variance over store noise from a single run.

5. As a policy researcher running a mixed roster (e.g. policies A, A, B, C, D across templates T1, T2, T3, T4, T4, T4), I want this to be one literal triple list, so that I don't have to glue together helper outputs.

6. As a policy researcher, I want the old `homogeneous` and `paired` helpers gone (not deprecated wrappers), so that there's exactly one path to author a store roster.

7. As a simulator user, I want each product to advance through `[introduction, growth, maturity, decline, dead]`, so that the lifecycle has a realistic terminal state instead of looping every few months.

8. As a simulator user, I want different products to have different `init_stage` values (e.g. staples start at `maturity`, new fashion at `introduction`), so that the catalog reflects realistic age diversity at step 0.

9. As a simulator user, I want different products to have different per-stage transition probabilities (fashion decays fast, staples decay slow), so that one parameter set can express category-typical dynamics.

10. As a simulator user, I want `dead → introduction` to default to a small positive probability (~0.003 per tick, ~1-year mean comeback), so that the world has occasional re-launches while still treating most `dead` products as effectively gone.

11. As a simulator user, I want `dead → introduction = 0` to be a valid configuration, so that I can run strict-terminal-lifecycle experiments without forking the model.

12. As a policy researcher, I want demand for newly-activated SKUs to spike and decay (`m(τ) = 1 + α · exp(−τ/β)`) per (store, product), so that a "late introduction" policy is correctly penalised for missing the hype window.

13. As a policy researcher, I want the freshness curve to reset on every `activate_item` event, so that re-introducing a discontinued SKU produces fresh hype rather than resuming a stale clock.

14. As a simulator user, I want different categories to have different `(α, β)` parameters (fashion ≈ 0.4 / 30, staples = 0), so that products without a real "newness effect" don't get an unwarranted boost.

15. As a policy researcher, I want the freshness curve to be one isolated module with a single `multiplier(...)` function, so that the team can swap it for a Bass-shaped diffusion later without touching the rest of the simulator.

16. As an LLM-driven world author, I want to populate per-`Ware` `init_stage`, `stage_change_probs`, `freshness_alpha`, `freshness_decay`, and `init_stock_share` per category, so that the generated world is realistic out of the box without manual tuning.

17. As an LLM-driven world author, I want to declare each store template's `init_active_products` roster (instead of relying on random sampling), so that a "fashion specialist" template actually carries fashion items at step 0.

18. As a policy researcher, I want each store template to declare an `init_freshness` mode (`"baseline"` vs `"fresh"`), so that I can run "established store" and "grand-opening" scenarios with the same template family.

19. As a policy researcher running an established-store baseline, I want initial active SKUs to skip the hype window (multiplier = 1 at step 0), so that initial-state demand isn't artificially inflated by a hype curve that should already have decayed.

20. As a policy researcher running a grand-opening scenario, I want initial active SKUs to start at `τ = 0` (full hype), so that I can model new-store opening dynamics correctly.

21. As a policy researcher, I want per-`Ware` `init_stock_share` weights (normalised across the active set), so that staples receive more initial stock than fashion within the same `capacity * init_stock_pct` budget.

22. As a scenario author, I want any of the new per-`Ware` fields to be optional, so that I can hand-write small test catalogs without populating every parameter.

23. As a scenario author, I want sensible defaults for all per-`Ware` fields to live on `ItemLifecycleParams`, so that "no override" produces a coherent world.

24. As a policy researcher, I want demand draws for inactive products to keep happening (one `world_rng` draw per `(store, product)` per tick), so that CRN-paired stores with divergent active sets still consume identical world stochasticity. (ADR 0003 — load-bearing, must not regress.)

25. As a scenario author, I want `Scenario.to_json()` / `from_json()` to round-trip every new field on `Ware`, `StoreTemplate`, `ItemLifecycleParams`, including the per-stage `stage_change_probs` dict, so that LLM-generated worlds can be saved and replayed.

26. As a scenario author, I want the two example scenarios (`example_homogeneous.py` and `example_paired_comparison.py`) migrated to `make_stores`, so that the canonical examples teach the new pattern.

27. As a developer, I want the bit-identity contract (`(template, init_seed)` ⇒ same step-0 state) to be testable as a single explicit assertion against an extracted `init_store_state` function, so that load-bearing invariants don't sit as emergent properties of `Store.__init__`.

28. As a developer, I want the freshness multiplier to be computable in isolation (no Store/Market/Registry wiring), so that I can test the math directly.

29. As a developer reading the run log, I want the per-`Ware` resolved values for `freshness_alpha`, `freshness_decay`, `init_stage`, `stage_change_probs`, and `init_stock_share` captured in the static products parquet, so that downstream analysis can reproduce demand math without re-resolving overrides.

## Implementation Decisions

### New deep modules to extract

- **`FreshnessCurve`** — exposes `multiplier(alpha, decay, ticks_since_activation) -> float`. Single function; one place to swap the exp-decay shape for a Bass hump or polynomial later. Owns no state.

- **`LifecycleClock`** — exposes `advance_stage(rng, current_stage, stage_change_probs) -> next_stage`. Pure function. Encodes "draw once against the probability for the *current* stage; if it fires, advance to the next stage in `[introduction, growth, maturity, decline, dead]`; from `dead`, advance to `introduction` (cyclic-with-terminal-default — set `dead → introduction = 0` to disable)". Replaces the scalar `stage_change_prob` semantics.

- **`StoreInitializer`** — exposes `init_store_state(template, init_seed, catalog) -> InitialStoreState`. Pure function. Materialises the deterministic step-0 state: which SKUs are active (explicit `init_active_products` list ⇒ that list; otherwise `init_rng.sample`); per-product initial stock allocated by normalised `init_stock_share` weights, summing to ≤ `capacity * init_stock_pct`; `activation_tick` map (`baseline` mode ⇒ tick value such that the freshness multiplier is `1` at step 0; `fresh` mode ⇒ `0`); resolved scalar values for any `Distribution`-typed template fields. The bit-identity contract becomes one explicit test against this function.

### Modules modified in place

- **`scenario.py`** — delete `homogeneous` and `paired`; add `make_stores(triples) -> list[StoreInstance]`. Add `freshness_alpha`, `freshness_decay`, `init_stage`, `stage_change_probs` (dict keyed by current stage), `init_stock_share` to `Ware`. Add `init_active_products: list[str] | None` and `init_freshness: Literal["baseline", "fresh"]` to `StoreTemplate`. Add `dead` to canonical stages on `ItemLifecycleParams`; replace scalar `stage_change_prob` with `default_stage_change_probs: dict[str, float | Distribution]`; add `default_freshness_alpha`, `default_freshness_decay`, `default_init_stock_share`, `default_init_stage`. JSON round-trip covers every new field, including the per-stage dict.

- **`item_registry.py`** — read per-`Ware` overrides for `init_stage` and `stage_change_probs`, falling back to `ItemLifecycleParams.default_*`. Delegate per-tick stage advancement to `LifecycleClock.advance_stage`. Continue to advance every catalog item once per `Runner` tick, against `world_rng`.

- **`store.py`** — delegate step-0 setup to `StoreInitializer`. Add `activation_tick: dict[product_id, int]`. `activate_item(pid)` writes `activation_tick[pid] = current_step` (curve resets); `deactivate_item` may leave the entry in place — re-activation overwrites. Expose `freshness_multiplier(pid, current_step) -> float` that delegates to `FreshnessCurve.multiplier` with the per-`Ware` `(α, β)` resolved against `Ware` overrides + scenario defaults.

- **`market.py`** — `sample_demand` multiplies in `store.freshness_multiplier(pid, current_step)` alongside the existing global stage multiplier, season factor, promotion factor, cross-product factor, and price-elasticity factor. Order of factors: `multiplier = stage_multiplier * freshness * season * promo * cross`. Demand-draw cardinality preserved.

- **`runner.py`** — pass `current_step` down where needed so the freshness multiplier is computed against the same step the demand draw uses. CRN demand-draw contract preserved verbatim — `_process_demand` continues to iterate `store.inventory.keys()` so every catalog product consumes one `world_rng` draw per tick.

- **`world_builder.py` + `schemas.py`** — extend the catalog-stage Pydantic schema to capture per-`Ware` `init_stage`, `stage_change_probs`, `freshness_alpha`, `freshness_decay`, `init_stock_share`. Extend the store-templates-stage schema to capture `init_active_products` and `init_freshness`. Prompt the LLM to author per-category values from realistic ranges: α ∈ [0.1, 0.4] with α = 0 for staples; β ∈ [15, 45] ticks; transition probs ≈ `intro→growth: 0.02`, `growth→maturity: 0.005`, `maturity→decline: 0.001`, `decline→dead: 0.005`, `dead→intro: 0.003` (LLM may override per category).

- **Example scenarios** — `example_homogeneous.py` migrated to `make_stores` with three triples sharing one policy and varying `init_seed`. `example_paired_comparison.py` migrated to `make_stores` with paired triples (same `(template, init_seed)`, two policies). Both keep the same numerical scenarios so existing tests still pin behaviour.

### Migration / behaviour-preservation

- The `dead` stage and per-stage `stage_change_probs` dict change the data model of `ItemLifecycleParams`. Existing example scenarios author `stage_change_prob = 0.0`; after migration they set every per-stage entry to `0.0` (no transitions) and run identically.
- `Ware` gains optional fields. Catalogs that don't set them inherit `ItemLifecycleParams.default_*` and behave equivalently.
- The CRN demand-draw contract is **load-bearing** (ADR 0003). Tests must pin it; do not "fix" `Runner._process_demand` to skip inactive items.

### Cyclic-with-terminal-default semantics

`LifecycleClock` advances stages in a list. From `dead`, the "next" stage is `introduction` (cyclic). The transition only fires when a `world_rng` draw falls below `stage_change_probs["dead"]`. Default `stage_change_probs["dead"] = 0.003` per tick. Authors who want strict-terminal lifecycle set the entry to `0`.

## Testing Decisions

A good test pins **external behaviour** — the contract a downstream caller depends on — not internal mechanics. Don't assert the order of attribute assignment in `Store.__init__`, don't assert a specific RNG draw count from a method (unless the count is part of an external contract like CRN), don't test that a private helper exists. Do assert that two stores with the same `(template, init_seed)` produce identical step-0 state regardless of attached policy; that demand for an inactive product is logged as a `demand` value with `sales = 0`; that the freshness multiplier is `1 + α` immediately after `activate_item` and approaches `1` as `τ` grows.

### Modules that get tests

1. **`FreshnessCurve.multiplier`** — unit, no dependencies. Pin: `multiplier(α, β, 0) == 1 + α`; `multiplier(α, β, τ → ∞) → 1`; `multiplier(0, β, τ) == 1` for all `τ`; multiplier is monotone-decreasing in `τ` for `α > 0`.

2. **`LifecycleClock.advance_stage`** — unit, with synthetic `Random` objects. Pin: with `stage_change_probs[stage] = 1.0`, advance every tick; with `stage_change_probs[stage] = 0.0`, never advance; from `dead` with prob 1, return to `introduction`; with prob 0, stay in `dead` (terminal-by-default check); cyclic ordering matches `[introduction, growth, maturity, decline, dead, introduction, …]`.

3. **`StoreInitializer.init_store_state`** — unit. Pin: same `(template, init_seed, catalog)` ⇒ identical `InitialStoreState` (bit-identity); `init_active_products` set ⇒ that exact list is active; unset ⇒ random sample of size `init_active_count`; `init_stock_share` weights distribute stock proportionally and respect the `capacity * init_stock_pct` budget; `init_freshness == "baseline"` ⇒ all initial active SKUs produce freshness multiplier `1` at step 0; `init_freshness == "fresh"` ⇒ all initial `activation_tick` values are `0`.

4. **`Market.sample_demand`** (regression) — pin: an inactive product still consumes one demand draw; freshness multiplier composes multiplicatively (active product, varied `τ`, expected multiplier matches `m(τ)` × the product of the existing factors).

5. **`Runner` CRN cleanliness** (regression) — pin: a paired pair (same `(template, init_seed)`, different policies) consumes the same `world_rng` draws across the entire run; demand traces for products untouched by policy-divergent activation history are bit-identical step-by-step.

6. **`Scenario.to_json` / `from_json`** (regression) — round-trip every new field. Pin: a Scenario with per-`Ware` `stage_change_probs={"introduction": 0.02, ...}`, freshness params, `init_active_products`, and `init_freshness` survives a JSON round-trip with structural and numerical equality.

7. **Example scenarios run cleanly** — `example_homogeneous.py` and `example_paired_comparison.py` execute end-to-end and produce a non-empty run log.

### Prior art

`tests/` already contains determinism tests that read `Store` step-0 state and assert bit-identity across `(template, init_seed)` repetitions, plus CRN tests that diff demand traces between paired triples. The new tests follow the same shape: build a small `Scenario`, run, assert on the resulting log or step-0 state. No new test-harness machinery is required.

## Out of Scope

- Replacing the exp-decay freshness curve with a Bass hump. The data model accommodates the swap (`FreshnessCurve` is the swap point); this PRD ships exp-decay only.
- Per-store **global** lifecycle stages. Global PLC stays one-stage-per-product across all stores (ADR 0001 — two-layer model). Per-store lifecycle was considered and explicitly rejected.
- Per-product RNG streams. Today's `world_rng`-shared draw sequence is preserved (ADR 0003).
- Refactoring `BaselinePolicy` to read latent demand for inactive products. The data is in the run log; reading it is a future policy concern.
- A formal "policy distribution" abstraction. Policies authored as "Base with sampled hyperparameters" are constructed in the experiment script via plain Python list comprehensions; no first-class API.
- Backward-compatibility shims for `homogeneous` / `paired`. They are deleted, and the example scenarios are migrated as part of this PRD.

## Further Notes

- `CONTEXT.md` and ADRs 0001–0003 capture the design decisions in canonical form. The PRD restates the load-bearing pieces so an AFK agent has the full picture in one read; ADRs remain the single source of truth for *why*.
- The order of factors in `Market.sample_demand` is part of the external contract (it determines floating-point identity in CRN comparisons). Implementations must keep `multiplier = stage_multiplier * freshness * season * promo * cross` to maintain bit-identity in regression tests.
- The cyclic-with-terminal-default semantics of `dead → introduction` was a user-driven decision: re-launches matter for long simulations, and a `0`-probability override recovers strict terminal behaviour for users who want it.
- The deep-module split (`FreshnessCurve`, `LifecycleClock`, `StoreInitializer`) is deliberately conservative. Each one isolates a load-bearing invariant (decay shape, transition rule, bit-identity) into a tested module. We don't extract more than that — the rest of `Store` / `Market` / `ItemRegistry` remains in place to keep the diff manageable.
