# PRD: RL scale-invariance package (ADR 0007 implementation)

Status: ready-for-agent

Related ADRs:
- [ADR 0007](../../docs/adr/0007-rl-scale-invariance-package.md) — RL scale-invariance package: order-up-to action decoder, demand-units inventory feature, log-uniform domain randomisation (this PRD's spec)
- [ADR 0005](../../docs/adr/0005-demand-relative-action-decoding.md) — Action decoder expresses order quantity in lead-times-of-demand (superseded by 0007; retained as historical record of the original diagnosis)
- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family (defines `OrderUpToPolicy` / `PeriodicOrderUpToPolicy` used as the CRN comparison anchor and centring-sanity reference)
- [ADR 0004](../../docs/adr/0004-rl-training-env.md) — RL training env shape (randomised assortment, slot-shuffled observations, frozen assortment)

## Problem Statement

The RL agent's action decoder currently expresses order quantity in *capacity-units* (`qty = order_frac × free_space / K`), which is invariant in shelf-fraction but not in demand-units. The same trained policy produces wildly different cover-horizons across deployment scales: at training scale (`capacity ≈ 250`) it lays in ~5 ticks of cover; at flagship scale (`capacity = 10_000`) the same action lays in ~200 ticks of cover, far beyond the 180-tick episode horizon. The fashion-world checkpoint that beats the baseline by ~5× equity in-distribution loses ~$200k against the baseline when redeployed at flagship scale.

ADR 0005 diagnosed this correctly and proposed `qty = clamp(order_frac × demand_horizon × recent_sales_rate, 0, per_sku_free_space)`. Under design review three implementation gaps surfaced:

1. **Off-centre natural action.** The multiplicative-scalar parameterisation puts the steady-state target action at `order_frac ≈ delivery_lag / demand_horizon ≈ 0.3` rather than the PPO-init centre at `0.5`. The agent must drift downward in early updates before it can address substantive ordering choices.
2. **Cold-start deadlock.** With `init_stock_pct = 0.0` and `cold_start_qty: int = 0` (ADR 0005's default), the agent starts with zero inventory, observes zero sales, computes zero rate, emits zero qty for any action, observes zero sales again, ad infinitum. PPO sees zero reward everywhere and never learns.
3. **Encoder is not actually scale-invariant.** The observation's per-SKU features are `inventory / per_sku_capacity` (and friends), clamped to `[0, 1]`. At training scale this uses the full range; at flagship scale every feature is bunched in `[0, 0.05]`. The policy net's response surface in the low-feature regime was never explored during training. Action-side scale-invariance alone is insufficient.

A secondary problem: ADR 0005's regression-test description references `BaselinePolicy`, which ADR 0006 replaced with `OrderUpToPolicy`. The validation criterion needs re-pointing, and it needs to be stronger than "beat baseline" — under the new parameterisation, a random-init policy roughly matches the baseline by construction, so "beat baseline" no longer distinguishes a working agent from an unlearned one.

## Solution

Implement the six-part scale-invariance package decided in ADR 0007, replacing the current capacity-units decoder with an order-up-to decoder, adding a demand-units inventory feature to the encoder, and widening the training distribution to span two orders of magnitude in store size.

The user-visible effects:

- A single trained PPO checkpoint covers store sizes from `capacity = 100` (corner shop) to `capacity = 10_000` (flagship), measured against `OrderUpToPolicy` (ADR 0006 anchor) under CRN-paired evaluation. Flagship and small-scale uplift are within a factor of 2.
- The agent at `order_raw = 0` (random-init starting point) behaves identically to `PeriodicOrderUpToPolicy(R=1, S=15·rate)` from tick 1 onward — PPO learns deviations from a textbook prior rather than searching for sensible behaviour from scratch.
- The cold-start tick produces a positive order via a market-derived demand prior; no episode is structurally stuck at zero inventory.
- The encoder carries inventory in both capacity-units (existing feature, useful for pricing) *and* demand-units (new feature, useful for ordering); `N_PER_SKU` grows from 13 to 14.
- The training capacity distribution becomes log-uniform across two orders of magnitude (`LogUniform(100, 10_000)`), making both small-store and flagship regimes in-distribution for the trained policy.

Validation is gated by three tiers (Tier 1 unit tests, Tier 2 centring sanity rollout, Tier 3 paired-CRN evaluation at two scales) — all three must pass before the work merges. The ADR-0005-era `runs/fashion_run` checkpoint and any other pre-existing checkpoints will not transfer; retraining is required.

## User Stories

1. As an RL researcher deploying a single trained PPO checkpoint to a heterogeneous fleet of stores, I want the policy to produce well-scaled order quantities at every store size, so that I don't have to retrain per scale or maintain a fleet of scale-specific checkpoints.

2. As an RL researcher running paired-CRN evaluation at flagship scale (`capacity = 10_000`), I want the trained agent to beat `OrderUpToPolicy` on equity, so that the scale-transfer claim is empirically grounded rather than a hopeful extrapolation from training.

3. As an RL researcher comparing trained-agent uplift across scales, I want the small-scale uplift and flagship uplift to be within a factor of 2 of each other, so that "the agent generalises" is a falsifiable statement and not a vague gesture.

4. As an RL researcher reading the PPO learning curve, I want the random-init policy at `order_raw = 0` to roughly match a textbook periodic-order-up-to baseline, so that the agent's improvement over the first few thousand updates is interpretable as "learning deviations from a sensible prior" rather than "the random policy is so bad it cannot help but improve".

5. As an RL researcher inspecting the action decoder, I want the meaning of `order_raw = 0` to be exactly `PeriodicOrderUpToPolicy(R=1, S=15·rate)`, so that the natural-init policy's behaviour is documented and reproducible without running the env.

6. As an RL researcher inspecting the decoder at episode start, I want the first-tick order to be positive (derived from the market's `base_demand` prior) regardless of the action, so that an episode never gets structurally stuck at zero inventory and zero gradient signal.

7. As an RL researcher inspecting per-SKU ordering during steady state, I want the decoder to compute qty as `max(0, target_lt × effective_rate − inventory_position)`, so that pending in-transit orders are correctly netted against the target and the agent doesn't double-order during the lead-time gap.

8. As an RL researcher inspecting the cold-start tick at flagship scale, I want the cold-start qty to scale with `base_demand × target_centre_lead_times`, so that the cold-start order is *not* a flat constant that mis-scales across capacity ranges.

9. As an RL researcher inspecting the clamp logic, I want unused per-SKU shelf budget on quiet SKUs to be available to bulk-ordering SKUs in the same tick, so that the cold-start tick (5 SKUs each requesting ~75 units against a 200-unit total free space) allocates fairly rather than wastefully cutting each SKU to `capacity / K`.

10. As an RL researcher inspecting the observation tensor at flagship scale, I want a per-SKU feature that expresses inventory in lead-times-of-cover (saturating at 30 lead-times), so that the policy net sees a comparable feature distribution at every deployment scale and the response surface it learned at training scale remains usable at flagship.

11. As an RL researcher inspecting the observation tensor in steady state, I want the existing capacity-units inventory feature retained alongside the new demand-units feature, so that shelf-saturation signal remains available for pricing decisions even though it's no longer the primary signal for ordering.

12. As an RL researcher configuring a training run, I want `LogUniform(100, 10_000)` capacity sampling by default, so that the training distribution covers the full deployment range and I do not have to remember to widen the distribution manually for each run.

13. As an RL researcher reading the training reward curve mid-run, I want the reward to be roughly stable as a function of the sampled capacity, so that obvious scale-dependent bugs (e.g. a clamping issue that only fires at one scale) are visible early rather than only at the final paired-CRN eval.

14. As a simulator user authoring a scenario, I want `LogUniform(lo, hi)` available as a first-class `Distribution` subclass alongside `Constant` / `Uniform` / `Normal` / `Choice`, so that log-scale draws (capacity, balance, anything else multiplicatively distributed) can be expressed without ad-hoc lambdas.

15. As an engineer extending the RL stack, I want the demand-rate computation (`effective_rate = max(rolling_5_mean_sales, base_demand_prior)`) to live in one pure-function module consumed by both encoder and decoder, so that future changes to the rate primitive (e.g. EMA smoothing, stockout-bonus inheritance) land in one place and stay consistent across the two consumers.

16. As an engineer extending the RL stack, I want the two-pass fair-share allocator to live in one pure-function module, so that the same allocator can be reused if the encoder ever needs a clamp-aware feature or if the decoder grows additional constraints (per-region cash limits, etc.).

17. As an engineer reviewing the implementation, I want Tier 1 unit tests covering every branch of `decode_action` and the new encoder feature, so that regressions in the core math are caught before any expensive training cycle.

18. As an engineer reviewing the implementation, I want a Tier 2 centring-sanity test that rolls the env forward with a deterministic `action = 0` policy and asserts identical trajectories with `PeriodicOrderUpToPolicy(R=1, S=15·rate)` from tick 1 onward, so that an implementation drift that silently moves the centring point is caught without requiring a full training run.

19. As an RL researcher merging the work, I want the Tier 3 paired-CRN evaluation script reproducible from a single command, taking a checkpoint path and emitting per-scale uplift means / CIs / cold-start qty statistics, so that the validation result is auditable and re-runnable on any future checkpoint.

20. As an RL researcher post-merge, I want the `runs/fashion_run` checkpoint and any other ADR-0005-era artifacts marked as incompatible with the new obs/action layout (not silently loadable into a mismatched `Actor`), so that a stale checkpoint produces a clear error rather than a confusing silent failure.

21. As an engineer reading `CONTEXT.md`, I want the RL Env paragraph to describe the new decoder semantics in domain language (order-up-to target in lead-times, demand-units inventory feature), so that future readers reaching this code via the glossary understand the design intent without diving into the ADR.

22. As an engineer reading the existing TextbookReorderPolicy cross-reference in `CONTEXT.md`, I want it updated to point at ADR 0007 (with ADR 0005 marked as superseded in the decisions list), so that "what is the RL decoder's framing" has one authoritative answer.

## Implementation Decisions

### Modules

This is a code change across five existing modules and two new test files. No new directories.

**`Distribution` family extension (`src/sim/distributions.py`).** A new `LogUniform(lo, hi)` class added alongside the existing `Constant` / `Uniform` / `Normal` / `Choice`. Same `sample(rng) → float` contract; `sample` returns `exp(rng.uniform(log(lo), log(hi)))`. JSON-serialisable consistent with the existing pattern. Validation: `lo > 0` and `hi > lo`. This is a small, focused addition — not a deep module on its own but a load-bearing primitive for the wider work.

**RL config (`src/rl/configs/default.py`).** Add fields:
- `target_centre_lead_times: int = 15` — the order-up-to target at `order_raw = 0`, equal to `OrderUpToPolicy.S / rate` at default policy kwargs (`delivery_lag + safety_lead_ticks + cover_horizon_ticks = 3 + 2 + 10`).
- `target_half_span_lead_times: int = 15` — width of the action range around the centre.
- `target_max_lead_times: int = 30` — upper clip to prevent extreme target requests.
- `max_inventory_lt: float = 30.0` — saturation point for the new demand-units inventory feature.

Remove fields ADR 0005 mentioned but never landed in code: `demand_horizon`, `cold_start_qty`. Defaults change:
- `capacity_dist` → `LogUniform(100, 10_000)` (was `Uniform(150, 400)`).
- `balance_dist` → `LogUniform(10_000, 1_000_000)` (was `Uniform(15_000, 40_000)`).

**Effective-rate primitive (new deep module, lives in `src/rl/encoders.py`).** A pure function `compute_effective_rate(sales_history, base_demand_prior) → dict[pid, float]` that takes the env's per-pid sales-history deques and the scalar market-derived prior, returns the per-pid effective rate as `max(rolling_5_mean(sales[pid]), base_demand_prior)`. Both `encode_observation` (slot-13 feature) and `decode_action` (qty math) consume this function — single source of truth for "what rate does the RL stack think this SKU has". Trivial to unit-test in isolation; no env coupling.

**Fair-share allocator (new deep module, lives in `src/rl/encoders.py`).** A pure function `fair_share_allocate(requested, per_sku_headroom, global_free_space) → dict[pid, int]` implementing the two-pass allocator: pass 1 caps each request at the per-SKU physical headroom; pass 2 proportionally scales all requests if their sum exceeds `global_free_space`. Returns integer quantities (truncate, do not round up — under-allocation is safer than over-allocation against a hard capacity constraint). Mirrors the textbook policy family's capacity allocator (`TextbookReorderPolicy` machinery in `src/sim/policy.py`) but capacity-only — the cash-pool half is not relevant for the RL decoder, where cash is exposed to the agent's reward signal rather than enforced as a hard constraint.

**Encoder (`src/rl/encoders.py`).** `N_PER_SKU` bumps from 13 to 14. New slot 13 carries `clip(inventory[pid] / effective_rate[pid], 0, max_inventory_lt) / max_inventory_lt`. `encode_observation` gains an `effective_rate` parameter (the env computes it once per tick and passes to both encoder and decoder); when not supplied, falls back to the existing rolling-5-mean-of-sales / capacity feature for backward compatibility in tests. Docstring layout table updated for `N_PER_SKU = 14`.

**Decoder (`src/rl/encoders.py`).** `decode_action` is rewritten end-to-end:
- Gains `effective_rate: dict[str, float]` parameter (mandatory at the call site in `RLEnv.step`; default-None branch kept for unit tests).
- Order half of the action: `target_lt[slot] = clip(target_centre_lt + order_raw × target_half_span_lt, 0, target_max_lt)`, then `requested[pid] = max(0, target_lt × effective_rate[pid] − (inventory[pid] + pending[pid]))`.
- Per-SKU shelf headroom `headroom[pid] = max(0, capacity − inventory[pid] − pending[pid])` (per-SKU physical, not equal-split).
- Final qty `order[pid] = fair_share_allocate(requested, headroom, global_free_space)[pid]`.
- Price half unchanged: `price = MSRP × clip(1.0 + 0.5 × price_raw, 0.5, 1.5)`. (ADR 0007 explicitly defers pricing-side changes.)

**Env (`src/rl/env.py`).** At `reset()`, after building `Market`, extract `base_demand_prior = market.params.base_demand.sample(world_rng_clone)` if `base_demand` is a `Distribution`, else use the scalar directly. Stash on `self._base_demand_prior`. At each `step()`, compute `effective_rate = compute_effective_rate(self._sales_history, self._base_demand_prior)` once and pass to both `decode_action` and `encode_observation`. No other env logic changes.

### Interface changes summary

- `decode_action` signature gains `effective_rate: dict[str, float] | None` parameter and reads new config fields.
- `encode_observation` signature gains `effective_rate: dict[str, float] | None` parameter; output length increases by `K_active` (slot 13 per SKU).
- `observation_dim(K_active)` returns `K_active * 14 + 4` (was `K_active * 13 + 4`).
- `Actor` / `Critic` architectures: no edits beyond consuming the new `observation_dim`. Input layer rebuilds automatically from the env's observation space.
- `RLConfig` field deltas: see above.
- New `LogUniform(lo, hi)` class on `Distribution`.

### Architectural decisions

- **Effective rate and fair-share allocation are pure-function deep modules.** They're testable in isolation, encapsulate the decision-rich logic, and have a single well-defined interface. The decoder and encoder become thin glue that compose these primitives.
- **No reuse of the textbook policy's allocator implementation in the RL decoder.** The textbook version is bundled with cash-budget logic that doesn't apply here. Duplicating the capacity-only pass-1 / pass-2 logic is ~20 lines; the shared "behavioural" guarantee is captured in the test suite, not in shared code.
- **Cold-start uses the market's `base_demand` directly, not a flat config constant.** The prior is derived per-episode from the market state the env already has access to; no new RNG, no new config indirection. If `base_demand` is a `Distribution` (the common case), the env samples it once at reset against the world RNG; if it's a scalar, the env uses it directly.
- **`runs/fashion_run` is not migrated.** New obs shape (`N_PER_SKU` 13 → 14) makes the checkpoint unloadable. The `Actor`'s `load_state_dict` will raise — this is the desired failure mode (loud and early, not silent and wrong).

## Testing Decisions

### Test philosophy

Test external behaviour at module boundaries. The "module boundary" for the two new deep modules (`compute_effective_rate`, `fair_share_allocate`) is their function signature — assert input-output contracts, not internal state. The boundary for the decoder and encoder is the public function (`decode_action`, `encode_observation`) — assert the produced numpy array / dict, not which internal helper produced which slot. The boundary for the env is `(obs, reward, done, info)` — Tier 2 asserts trajectory equivalence with `PeriodicOrderUpToPolicy`, not which encoder slot held which scalar at which tick.

Avoid testing implementation details: e.g. don't assert that `fair_share_allocate` uses two passes — assert that the returned quantities (a) respect per-SKU headroom, (b) sum to ≤ `global_free_space`, and (c) preserve the relative proportions of the requested amounts when binding.

### Modules under test

**Tier 1 — pure-function unit tests (no env, no training, fast).**

- `compute_effective_rate`: empty sales history returns `{pid: prior}` for every pid in the input; populated history of less than 5 entries returns the partial-window mean clamped to the prior; populated history of 5+ entries returns the rolling-5 mean clamped to the prior; SKUs not in the history dict are not in the output.
- `fair_share_allocate`: sum-below-free-space returns requested unchanged; sum-above-free-space proportionally scales while no SKU exceeds its headroom; per-SKU headroom binds tighter than global scaling when needed; empty input returns empty output; zero global free space returns all zeros.
- `decode_action`: at `effective_rate = 0` everywhere and zero inventory, returns positive qty equal to `target_lt × base_demand_prior` (via cold-start branch through `compute_effective_rate`); at `position = target_lt × effective_rate` returns zero qty regardless of action sign; total returned qty across all SKUs is always ≤ `capacity − total_inventory − total_pending`; price half is in `[0.5 × MSRP, 1.5 × MSRP]` exactly as before.
- `encode_observation`: slot 13 is finite, in `[0, 1]`, and equals `min(1.0, inventory / (effective_rate × max_inventory_lt))` for each active SKU; existing slot 0 (capacity-units inventory) is unchanged from the prior implementation.
- `LogUniform.sample`: returns values in `[lo, hi]`; mean of many samples is between geometric-mean and arithmetic-mean of `[lo, hi]`; same RNG seed produces same sample (determinism); rejects `lo <= 0` and `hi <= lo`.

**Tier 2 — centring-sanity integration test (env, no training).**

- Roll the RL env forward for the full 180-tick episode with a constant `action = zeros(2*K)` policy on a fixed `(world_seed, init_seed)`.
- Roll the simulator forward with `PeriodicOrderUpToPolicy(R=1, cover_horizon_ticks=10, safety_lead_ticks=2)` on the same seeds — match by constructing the env's underlying `Scenario` directly, then running the simulator step loop with the textbook policy attached.
- Assert per-tick orders, prices, inventory, and final balance match within a tolerance of `1` unit / `0.01` price-unit from tick 1 onward.
- Tick 0 is *not* asserted to match — the RL decoder uses the market-derived prior; the textbook policy uses its cash-budget pilot. Both are valid cold-starts; they diverge by design. Tick-0 behaviour for both is covered by their respective unit tests.

**Tier 3 — paired-CRN evaluation (post-training, slow, run once per merge candidate).**

- Train one policy on the new defaults (`LogUniform` capacity + balance) to the full env-step budget.
- Build two held-out 32-seed eval sets disjoint from the training seed range: `small` uses `capacity_dist = Uniform(150, 400)`; `flagship` uses `capacity_dist = Constant(10_000)`. Balance widened symmetrically.
- For each seed in each set, run paired CRN: same `(world_seed, init_seed)` for the RL policy and `OrderUpToPolicy`; record return for each.
- Pass criteria (all three must hold to merge):
  - **Learning:** mean paired-uplift > 0 at both scales, with 95% bootstrap CI excluding zero.
  - **Scale-invariance:** `|log(flagship_uplift / small_uplift)| ≤ ln 2` (flagship within 2× of small).
  - **Cold-start non-pathology:** average tick-0 qty per active SKU across the 64 seeds is in `[0.5, 2.0] × base_demand_prior × target_centre_lead_times`. This catches both "cold-start fires at zero" (the ADR 0005 deadlock) and "cold-start over-orders by 10×" (a coding bug in the prior wiring).

### Prior art

- `tests/rl/test_encoders.py` already covers the existing `encode_observation` / `decode_action` shape, dtype, and bound assertions — extend in place rather than starting a new file. Tier 1 unit tests for the new functions live alongside the existing ones.
- `tests/sim/test_distributions.py` (if present — check on implementation) covers `Constant` / `Uniform` / `Normal` / `Choice`. The `LogUniform` test follows the same shape.
- `src/rl/eval.py` already implements paired-CRN evaluation via `evaluate(rl_policy_fn, baseline_policy_factory, eval_specs, config)`. Tier 3 extends this with a two-scale runner (one call per scale, results combined) rather than rewriting it.

## Out of Scope

- **Retraining the RL agent and running Tier 3 to green.** The training run takes hours and produces a checkpoint that must be reviewed separately. This PRD covers code + Tier 1 + Tier 2; Tier 3 is gated on the trained checkpoint produced after this PRD's work merges.
- **Recurrent policy with action history.** ADR 0007 alternative (h). Deferred until Tier 3 produces a measurable gap that would justify the architectural cost.
- **Pricing decoder changes.** ADR 0007 explicitly retains the existing `[0.5, 1.5] × MSRP` price decoder. The price half of the action vector is scale-invariant by construction (price is per-SKU and per-MSRP). Re-examining the price decoder is out of scope.
- **Renormalising the global observation block (cash / total_inventory features).** `cash / initial_cash` is already scale-invariant; `total_inventory / capacity` saturates the same way per-SKU features do, but at one feature it's a smaller surface than the per-SKU saturation. If Tier 3 reveals a problem driven by the global block, that's a follow-up.
- **Replacing the capacity-units inventory feature outright (option Z in the design tree).** Both features coexist by design (ADR 0007 alternative (f)). Removing the capacity-units feature is a separate decision.
- **Adapting non-RL callers of `Distribution`.** `LogUniform` is added as a new option; existing callers that don't use it are unaffected. No migration of `MarketParams.base_demand` or other `Uniform` users is required.
- **Updating notebooks.** Notebooks that load `runs/fashion_run` will break (intentionally — the obs shape changes). Re-running them with a freshly trained checkpoint is a follow-up task; this PRD does not include notebook fixes.

## Further Notes

- ADR 0007 is the spec; this PRD is its implementation contract. If a design question surfaces during implementation that the ADR doesn't answer, prefer "minimal compatible change" + a follow-up ADR over inline divergence.
- The decoder rewrite changes the action-to-qty mapping. The `Actor` / `Critic` architectures don't need code edits, but the saved `state_dict`s from any prior training run are incompatible with the new observation shape. The `load_state_dict` failure on a stale checkpoint is desired behaviour — silent partial loading would mask the obs-shape change.
- ADR 0007 lists `LogUniform(10_000, 1_000_000)` for `balance_dist`. This is wider than the historical `Uniform(15_000, 40_000)` and means episodes will see opening balances that may be inconsistent with the LLM-built world's hand-authored balance fields. The intended behaviour is that the episode sampler's draw fully overrides the template's `init_balance` (as it already does — see `episode_sampler.sample_episode`); confirm during implementation that no path leaks the template's balance into the runtime store at flagship-scale runs.
- The Tier 2 centring-sanity test is the single highest-leverage test in the suite. If forced to skip one tier, do not skip this one — Tier 1 catches off-by-one errors but cannot catch "the centring point is off by 50% because of a subtle index error in slot_perm"; Tier 3 catches that eventually but only after a full training run. Tier 2 catches it in seconds.
- The work is intentionally batched (decoder + encoder + DR + tests) rather than split into separate landings. A partial landing (e.g. decoder without encoder) would still require a retraining cycle and would leave a known scale-transfer gap uncorrected. One retraining cycle for the full package is strictly cheaper than two retraining cycles for two partial landings.
