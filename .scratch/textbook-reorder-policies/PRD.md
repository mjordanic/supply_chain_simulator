# PRD: Textbook reorder-policy family + heuristic-baseline rename

Status: ready-for-agent

Related ADRs:
- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family replaces the heuristic baseline as the RL comparison anchor
- [ADR 0005](../../docs/adr/0005-demand-relative-action-decoding.md) — Action decoder expresses order quantity in lead-times-of-demand (compounding sibling change on the RL side)

## Problem Statement

The current `BaselinePolicy` is a 660-line heuristic in `src/sim/policy.py` exposing 20+ entangled hyperparameters that govern reorder quantity, dynamic pricing, promotions, catalog rotation, and cooldowns. As a CRN-paired comparison anchor for the RL policy it has two failure modes:

1. **Behaviour is kwarg-sensitive.** Whether RL "beats baseline" depends materially on which kwargs the baseline was given. Comparison claims are therefore non-portable across worlds — every new world archetype requires another tuning pass on the baseline.
2. **The baseline loses money outright on flagship-scale stores in the LLM fashion world.** On `capacity=10_000` / `balance=$1M` with the kwargs shipped in `notebooks/07-rl_vs_baseline_per_product.ipynb`, the heuristic finishes a 180-tick episode with negative net P&L. A CRN test where both sides lose carries no information about whether the RL agent learned anything useful.

The reward function (per-tick balance delta = revenue − total_cost, with holding cost charged on residual inventory) and the simulation mechanics are correct. The comparison anchor is what fails.

A second problem follows from the first: the kitchen-sink heuristic remains genuinely useful as a *demonstration* of every knob the `Policy` framework exposes (dynamic pricing, promotions, deactivation, periodic review). Deleting it would discard a working example; keeping it as the comparison anchor would preserve the un-fixable problems above. The two roles need to be separated.

## Solution

Replace `BaselinePolicy` with a family of four textbook reorder policies sharing one abstract base, and rename the current `BaselinePolicy` to `HeuristicPolicy` for example-only use. The new family lives in the same module (`src/sim/policy.py`), shares a common `TextbookReorderPolicy` abstract base, and consists of:

- `OrderUpToPolicy` — (s,S) continuous review. Becomes the canonical CRN comparison anchor for RL.
- `ReorderPointPolicy` — (s,Q) continuous review.
- `PeriodicOrderUpToPolicy` — (R,S) periodic review.
- `PeriodicReorderPolicy` — (R,s,S) periodic review with reorder-point min.

All four are pure textbook inventory rules: they emit a flat `price[pid] = base_price[pid]` (no cost floor, no dynamic pricing, no promotions), an empty `activate=[] / deactivate=[]` (frozen assortment as in the RL env per ADR 0004), and an `order[pid]` computed from a recent-sales-rate estimator. Levels `s` and `S` are expressed in lead-times-of-demand so the policies are scale-invariant in capacity by construction — the same defaults work for a $20k corner shop and a $1M flagship, mirroring the demand-units framing of ADR 0005.

The previous `BaselinePolicy` is renamed `HeuristicPolicy` (no code-body change) and retained only in `scenarios/example_*.py` for parameter-demo purposes. No class named `BaselinePolicy` survives the change — every existing `from src.sim.policy import BaselinePolicy` raises `ImportError`, forcing the importer to consciously pick its intent (kitchen-sink demo vs (s,S) comparison anchor). No compatibility alias.

## User Stories

1. As an RL researcher comparing the trained PPO actor against a competitor policy, I want a single, well-defined CRN comparison anchor (`OrderUpToPolicy`) whose behaviour does not depend on per-world tuning, so that "RL beats the baseline" is a portable claim across world archetypes.
2. As an RL researcher running `notebooks/06-compare_rl_vs_baseline.ipynb`, I want the paired-uplift metric to compare RL against a policy that does not itself lose money in-distribution, so that a positive uplift is informative rather than "RL lost less than the other thing that was also losing."
3. As an RL researcher running `notebooks/07-rl_vs_baseline_per_product.ipynb` on per-product equity / cumulative-P&L curves, I want the baseline to be the textbook (s,S) policy at the *training distribution* scale (`Uniform(150, 400)` capacity, `Uniform(15_000, 40_000)` balance), so that the per-product breakdown reflects an apples-to-apples competitor.
4. As an RL researcher writing an ablation report, I want a *ladder* of textbook competitors — (s,S), (s,Q), (R,S), (R,s,S) — so that a comparative claim ("RL beats (s,S) by X but only ties (R,s,S)") becomes possible without writing four bespoke policies.
5. As a simulator user authoring a new scenario, I want to import `OrderUpToPolicy()` with no kwargs and have a working policy out of the box, so that running a comparison scenario doesn't require copying a 20-kwarg constructor from another scenario file.
6. As a simulator user examining the existing `scenarios/example_homogeneous.py` to learn how the policy framework works, I want a `HeuristicPolicy` that still exercises every knob (dynamic pricing, promo cooldowns, deactivation, periodic review), so that the example continues to demonstrate the full surface.
7. As a CI maintainer, I want existing `from src.sim.policy import BaselinePolicy` imports to fail loudly at import time rather than silently change behaviour, so that the migration is auditable from grep-output.
8. As a CONTEXT.md reader unfamiliar with the codebase, I want the `Policy` glossary entry and a new `TextbookReorderPolicy family` entry to make the two-policy-families structure obvious, so that I do not have to read 1200 lines of `policy.py` to understand what the comparison anchor is.
9. As a Python developer extending the policy framework, I want `TextbookReorderPolicy` to be a public abstract base with two well-named abstract methods (`_trigger`, `_quantity`), so that adding a fifth textbook variant later (e.g. an EOQ-based policy) is a small subclass file rather than a refactor.
10. As a developer changing the rate-estimator or allocator logic in a follow-up, I want the rate-estimator and the two-pass allocator extracted as pure functions, so that a unit test in `tests/sim/test_textbook_policy.py` can verify the maths against synthetic inputs without standing up a Store/Market.
11. As a store at flagship scale (`capacity=10_000`, `balance=$1M`) being benchmarked, I want `OrderUpToPolicy` defaults to *also* produce a profitable trajectory, not just on `capacity=200` stores, so that the comparison is not silently broken at scale.
12. As a store starting an RL episode with zero inventory (`init_stock_pct=0.0`), I want the textbook policy to place a cash-budget pilot order on tick 0 to bootstrap demand observation, so that the policy doesn't die in a no-sales / no-orders death spiral.
13. As a store with active SKUs that have hit a stockout, I want the option (off by default) to bump safety stock so that the next replenishment cycle absorbs the censored-demand signal, so that a *single* stockout doesn't pin the policy at a permanently-under-ordering equilibrium.
14. As a CRN-paired-eval consumer (`src/rl/eval.py::evaluate`), I want the `baseline_policy_factory` injection contract to remain unchanged, so that switching from `BaselinePolicy()` to `OrderUpToPolicy()` is a one-line change in `src/rl/train.py` and does not require any eval-machinery edits.
15. As the author of `tests/sim/test_regression_snapshot.py`, I want a fresh trajectory snapshot pinned on `OrderUpToPolicy` rather than on the old `BaselinePolicy`, so that future tuning of (s,S) parameters is a regression-test-failure rather than silent drift.
16. As the author of `tests/sim/test_baseline_policy.py`, I want the file renamed to `test_heuristic_policy.py` with all `BaselinePolicy` references updated to `HeuristicPolicy`, so that the test name reflects what it actually tests after the rename.
17. As a scenario author for `scenarios/llm_world_20.py` (the canonical RL training world), I want a clean `OrderUpToPolicy(policy_seed=seed)` constructor with no further kwargs, so that the scenario file documents the comparison anchor rather than encoding 20 lines of heuristic kwargs.
18. As a scenario author for `scenarios/example_paired_comparison.py`, I want to retain the two-`HeuristicPolicy`-variant pattern (aggressive vs conservative) so that the paired-comparison demo continues to show the framework's CRN-comparison capability.
19. As a developer reading `notebooks/05-monitor_rl_training.ipynb` or `notebooks/06-compare_rl_vs_baseline.ipynb` markdown, I want the prose to refer to `OrderUpToPolicy` explicitly (not the generic "baseline"), so that the comparison protocol is self-explanatory.
20. As an observer of paired-uplift training metrics over the course of training, I want the RL policy to be evaluated against the same `OrderUpToPolicy` factory throughout, so that a single number ("eval/paired_uplift") is interpretable over time.
21. As a CI runner executing the test suite, I want all existing test files to pass after the migration (with the snapshot regenerated), so that the rename does not introduce a multi-PR migration window where main is red.
22. As a developer who runs `uv run python scenarios/llm_world_20.py` directly, I want the scenario to keep working end-to-end with the new `OrderUpToPolicy`, including the `DataExporter` artefact paths, so that the scenario runner remains the canonical "smoke test" for production scenarios.
23. As an RL researcher iterating on `cover_horizon_ticks`, `safety_lead_ticks`, or `opening_budget_pct` defaults, I want those defaults exposed as named kwargs on `OrderUpToPolicy.__init__`, so that I can override them per-scenario without subclassing.
24. As a maintainer worried about implicit behaviour, I want `min_qty` on the textbook policies to default to `0` (no floor), so that the textbook decision rule is published exactly as the textbook describes it. Scenario authors who care about order-fee amortisation can set `min_qty` explicitly.
25. As a CRN-paired-eval consumer who relies on `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` determinism, I want `OrderUpToPolicy` to consume *only* `policy_rng` (not `world_rng` or `init_rng`), so that the CRN-disjoint property of paired evaluation is preserved.
26. As an integrator who upgrades the version of this repo across a project boundary, I want `MIGRATION` notes inside `docs/adr/0006-*.md` to list every renamed import path, so that I can produce a one-line `sed` over my downstream scenarios.

## Implementation Decisions

### Module structure

All work lands in a single file, `src/sim/policy.py`. Within that file:

- `Policy` ABC — unchanged.
- `NoopPolicy` — unchanged.
- `RLPolicy` — unchanged.
- `HeuristicPolicy` — renamed from `BaselinePolicy`, otherwise byte-identical. All 20+ kwargs preserved. `__all__` updated.
- `TextbookReorderPolicy` — new abstract base class subclassing `Policy`. Public (no underscore prefix), abstract by virtue of two `@abstractmethod` hooks. Holds shared state and machinery (rate estimator, pilot allocator, two-pass fair-share allocator).
- `OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy` — four concrete subclasses, each ~15–30 LOC. Implement `_trigger(pid, step, position, rate) → bool` and `_quantity(pid, position, rate) → int`. Periodic variants also accept a `review_interval: int` kwarg.

Two deep helpers extracted as private module-level functions so they can be unit-tested in isolation against synthetic inputs without the Store/Market machinery:

- `_estimate_rate(sales_log_view, demand_window, inv, pending, cold_start_handled_by_caller=False, stockout_safety_bonus_ticks=0) → float`
- `_allocate_two_pass_fair_share(desired: dict[pid, int], unit_costs: dict[pid, float], free_space: int, cash: float, min_qty: int, pilot_pids: set[str]) → dict[pid, int]`

Layout choice: one file, not a `policies/` package. Rationale: ~1200 LOC stays grep-friendly; splitting later is a pure refactor with no information loss.

### Naming

| Class | Variant | Public? |
|---|---|---|
| `HeuristicPolicy` | (renamed kitchen-sink) | yes |
| `TextbookReorderPolicy` | abstract base | yes |
| `OrderUpToPolicy` | (s,S) — RL CRN anchor | yes |
| `ReorderPointPolicy` | (s,Q) | yes |
| `PeriodicOrderUpToPolicy` | (R,S) | yes |
| `PeriodicReorderPolicy` | (R,s,S) | yes |

No `BaselinePolicy` alias. Hard break.

### Rate estimator

Statistic: **censored sales** (`store.sales[pid]`, what a real store-manager actually observes — RL encoder uses the same convention so the two policies share assumptions).

Window: `demand_window` defaults to `delivery_lag` read from the observation (no extra kwarg unless overriding).

Aggregator: simple mean of the last `demand_window` observations.

Cold-start escape: when a SKU has never been observed (`pid not in sales_log`) **and** has zero on-hand + in-transit stock (`inventory[pid] + outstanding_orders[pid] == 0`), a cash-budget pilot order fires *before* the trigger loop. Pilot qty per SKU = `int(opening_budget_pct × balance / K_active / unit_cost[pid])`, clamped to free space. Defaults: `opening_budget_pct=0.50`. Pilots bypass `min_qty` (they exist *to* probe demand even at small qty). The inventory gate keeps the probe from firing when the store opens with stock (`init_stock_pct > 0`) — demand surfaces through natural sales in that case, so no probe is needed.

Stockout-adaptive safety stock: opt-in. When `stockout_safety_bonus_ticks > 0`, the policy bumps `safety_lead_ticks` by that bonus for any pid whose recent window contains stockout ticks (sales[pid] == inventory_before_settle[pid]). Default `stockout_safety_bonus_ticks = 0` (off — textbook-pure).

### Inventory position

`position = inventory[pid] + pending[pid]` (on-hand + in-transit). Textbook-canonical. Avoids phantom-reorder during lead-time and prevents fixed-order-fee waste.

### Reorder levels (demand-units framing)

```
s = (delivery_lag + safety_lead_ticks) × rate
S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate
```

Defaults: `safety_lead_ticks=2`, `cover_horizon_ticks=10`. Both are kwargs on `TextbookReorderPolicy.__init__` (inherited by all four subclasses). `delivery_lag` is read from the per-pid observation, so future per-SKU lead times work without policy changes.

### Per-variant decision rules (subclass hooks)

| Variant | `_trigger(pid, step, position, rate)` | `_quantity(pid, position, rate)` |
|---|---|---|
| `OrderUpToPolicy` (s,S) | `position < s` | `S − position` |
| `ReorderPointPolicy` (s,Q) | `position < s` | `Q` (kwarg, default `(cover_horizon_ticks) × rate`) |
| `PeriodicOrderUpToPolicy` (R,S) | `step % review_interval == 0` | `max(0, S − position)` |
| `PeriodicReorderPolicy` (R,s,S) | `step % review_interval == 0 and position < s` | `S − position` |

`review_interval` defaults to `delivery_lag` (review at lead-time cadence). Configurable.

### Two-pass fair-share allocator

Pass 1: each active SKU is allocated `min(desired, space_block, cash_block_in_qty)` where the blocks are fair-shared per SKU from pool / K. Iteration-order-independent within pass 1.

Pass 2: water-filling. In each round, fair-share the *remaining* pool among SKUs whose shortfall > 0, cap by shortfall. Iterate until either no shortfall remains or no progress in a round. Bounded to ≤ K rounds.

Final mop-up: greedy pass over remaining shortfalls collects the integer-rounding tail (≤ K-1 units).

`min_qty` floor: applied only to non-pilot allocations after passes complete. Default `min_qty=0` (textbook-pure).

### Action shape

Every variant emits:
- `order` — non-zero entries from the allocator
- `price` — `{pid: obs["base_prices"][pid]}` for every pid in `obs["active_products"]`
- `activate` — `[]`
- `deactivate` — `[]`
- `promotions` — `{}`

No cost floor on price. No `unit_cost`-based markup. Catalog authoring bugs (where `base_price < unit_cost`) will surface as negative P&L rather than be masked.

### Active assortment handling

Policy reads `obs["active_products"]` each tick and treats it as the universe. Never emits `activate` / `deactivate`. Mirrors the RL env's frozen-assortment-per-episode property (ADR 0004). The set is whatever the Store was constructed with — scenarios choose their own `init_active_count` / `init_active_products` and the policy respects it.

### Migration surface (20 files)

**Bucket A — adopt `HeuristicPolicy` (rename only):**
- `scenarios/example_homogeneous.py`
- `scenarios/example_paired_comparison.py`
- `scenarios/example_llm_world.py`
- `scenarios/example_llm_world_offline.py`
- `tests/sim/test_baseline_policy.py` → rename file to `test_heuristic_policy.py`
- `tests/test_examples_and_cli.py`

**Bucket B — adopt `OrderUpToPolicy` with default kwargs (old heuristic kwargs are *dropped* — they refer to behaviour the textbook policy does not have):**
- `scenarios/llm_world_20.py`
- `scenarios/llm_world_100.py`
- `scenarios/llm_world_1000.py`
- `tests/sim/test_regression_snapshot.py` (regenerate snapshot)
- `tests/sim/test_full_run.py`
- `tests/sim/test_scenario.py`
- `tests/rl/test_episode_sampler.py`
- `tests/rl/test_eval_crn.py`
- `notebooks/06-compare_rl_vs_baseline.ipynb` (imports + markdown narrative + `build_baseline_policy()` body)
- `notebooks/07-rl_vs_baseline_per_product.ipynb` (imports + markdown narrative + `build_baseline_policy()` body)
- `src/rl/eval.py` (type-hint import only)
- `src/rl/train.py` (factory body `_baseline_factory() → OrderUpToPolicy()`)
- `CONTEXT.md` (already updated as part of design session)
- `README.md` (baseline references)

### `src/rl/eval.py` invariants

`evaluate(rl_policy_fn, baseline_policy_factory, …)` keeps its current injection signature. The eval module itself is policy-agnostic; only the type-hint import (`from src.sim.policy import Policy`) and the type-doc references need updating.

### Determinism / CRN invariants

`TextbookReorderPolicy` consumes only `self.policy_rng` (currently used: none — the policies are deterministic given the obs; the seed is preserved as a stylistic convention and for symmetry with `HeuristicPolicy`). No `world_rng`, no `init_rng`. Two paired CRN rollouts with the same `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` and the same `OrderUpToPolicy` defaults produce bit-identical trajectories.

## Testing Decisions

### What makes a good test here

Test the **textbook semantics**, not the Python wiring. A test that asserts "after `delivery_lag` ticks of constant demand `d`, `OrderUpToPolicy.decide` returns an `order[pid]` such that resulting position equals `S`" is testing the contract; a test that asserts "`_allocate_two_pass_fair_share` was called once" is testing the implementation. We want the former.

Tests should drive the policy through a real `Store` / `Market` / `ItemRegistry` stack where appropriate (integration), so that contract violations against the Store's observation surface or the action-dict shape are caught at the right boundary. The two extracted pure helpers (`_estimate_rate`, `_allocate_two_pass_fair_share`) are the exception — they get unit tests against synthetic inputs.

### Test modules

**New unit tests — pure helpers (`tests/sim/test_textbook_helpers.py`):**

1. `_allocate_two_pass_fair_share` invariants:
   - Pass-1 fair-share: every active SKU receives at least `min(desired, fair_share)` regardless of iteration order.
   - Capacity-binding: when `sum(desired) > free_space`, total allocated equals `free_space` (modulo integer rounding ≤ K).
   - Cash-binding: same but for cash pool.
   - Water-filling convergence: when one SKU asks for far more than its share and others ask for less, the leftover redistributes to the over-share SKU up to its desired qty.
   - Integer mop-up: a synthetic case where pass-2 rounding leaves slack confirms the greedy mop-up cleans it.
   - Edge cases: K=1, all-zero desired, `min_qty` clamp, pilot bypass of `min_qty`.

2. `_estimate_rate` invariants:
   - Empty sales log returns `0.0` (caller handles pilot upstream).
   - Constant-demand history returns the constant.
   - Censored history (all-zero sales with zero inv & pending) — caller-side concern; the estimator returns the censored mean, the policy's pilot path handles the bootstrap.
   - Stockout-adaptive bonus fires when `stockout_safety_bonus_ticks > 0` and the recent window contains stockout ticks.

**New integration tests — per-variant (`tests/sim/test_textbook_policy.py`):**

3. `OrderUpToPolicy`: drive a Store with constant-demand market for `4 × delivery_lag` ticks; assert (a) tick-0 pilot order fires and lands at lead-time; (b) once steady-state reached, position oscillates between approximately `s` and `S`; (c) no order fires while position > s.
4. `ReorderPointPolicy`: same setup; assert qty when triggered equals the configured `Q`, regardless of how far below `s` position fell.
5. `PeriodicOrderUpToPolicy`: assert orders only fire at `step % review_interval == 0`; assert qty at fire time brings position to `S`.
6. `PeriodicReorderPolicy`: assert no order fires when `step % R == 0` but `position ≥ s`; assert order fires when both conditions met.

**Smoke / migration tests:**

7. Import-failure test (`tests/sim/test_textbook_policy.py::test_no_baseline_policy_alias`): `from src.sim.policy import BaselinePolicy` raises `ImportError` (hard break confirmation).
8. CRN regression snapshot (`tests/sim/test_regression_snapshot.py`): regenerated on `OrderUpToPolicy` with default kwargs. Pins step-by-step trajectory; future tuning of `cover_horizon_ticks` or `safety_lead_ticks` will surface as snapshot-diff.

**Rebadged existing test:**

9. `tests/sim/test_baseline_policy.py` → renamed to `tests/sim/test_heuristic_policy.py`, imports switched to `HeuristicPolicy`. No assertion changes — proves the rename is byte-identical behaviour.

### Prior art

- `tests/sim/test_baseline_policy.py` — observation-shape contract tests, kwargs round-trip, RNG-isolation checks. Same shape works for `test_heuristic_policy.py` after rename and for the per-variant integration tests (drive Store, inspect decisions).
- `tests/sim/test_full_run.py::test_baseline_policy_places_orders` — "policy emits at least one positive order across the run" style smoke test. Mirror for each new variant.
- `tests/rl/test_eval_crn.py` — CRN self-comparison pattern ("RL vs RL on the same seed gives zero uplift, baseline vs baseline gives non-zero finite uplift"). Repurpose for `OrderUpToPolicy` self-comparison.
- `tests/sim/test_regression_snapshot.py` — pinned-trajectory regression style. Same harness, new snapshot.

### Tests not written

- No direct test of `TextbookReorderPolicy.decide` outside a subclass (the class is abstract).
- No regression snapshot for the three non-anchor variants (`ReorderPointPolicy`, periodic variants); their contract is captured by the per-variant integration test instead.
- No backward-compat alias test (there is no alias by design).

## Out of Scope

- **Tuning `OrderUpToPolicy` defaults for the LLM fashion world specifically.** The PRD ships with the documented defaults (`cover_horizon_ticks=10`, `safety_lead_ticks=2`, `opening_budget_pct=0.50`, `min_qty=0`). A drive-by tuning pass to extract additional performance is deferred to a follow-up PRD.
- **Retraining the RL agent against the new `OrderUpToPolicy` baseline.** The existing checkpoint at `runs/fashion_run/checkpoints/actor_step0001000000.pt` continues to work — only the comparison curve changes. ADR 0005's decoder change *does* require retraining (separate work).
- **A fifth textbook variant** (EOQ-only, base-stock with backorders, newsvendor). The abstract base is structured to allow it cleanly, but the four-variant set is what ships.
- **Pricing as a separable RL ablation.** With flat-price baseline, the future "RL with pricing-only vs RL with ordering-only" comparison becomes possible — but is not built here.
- **`scenarios/llm_world_20.py` docstring fix.** The file's docstring incorrectly mentions "1000-item" (likely copy-paste from `llm_world_1000.py`). Flag in implementation but do not auto-correct unless the user OKs it during PR review.
- **`HeuristicPolicy` ranking against `OrderUpToPolicy`.** Comparing the two on the LLM fashion world is interesting but is a separate analysis notebook, not part of this PRD's scope.
- **Public API on `_estimate_rate` and `_allocate_two_pass_fair_share`.** They are *private* module helpers extracted for testability. If a future need arises to call them from outside `policy.py`, that public-API decision is deferred.
- **`MIGRATION.md` document.** Per project convention, the ADR's Migration section is the canonical migration note. Standalone migration docs are out of scope.

## Further Notes

- The new family operates entirely through `obs["active_products"]` and never iterates over the full Store inventory dict. Scenarios that ship a Store with many catalog products but a small active assortment are correctly handled (orders fire only for active SKUs; non-active SKU prices come back as base_price unchanged but the policy will *not* place orders for them).
- The `HeuristicPolicy` rename is byte-identical behaviour. The existing snapshot in `tests/sim/test_regression_snapshot.py` is regenerated against `OrderUpToPolicy` because the *anchor identity* changes; if we wanted both pinned, that would be an additional snapshot file (rejected per Q9b/Q9e).
- `src/rl/eval.py::evaluate` continues to take a `baseline_policy_factory: Callable[[], Policy]` — *not* `Callable[[], OrderUpToPolicy]`. The factory is policy-agnostic by design; only the call sites care about which concrete class is returned.
- ADR 0005 (demand-relative action decoding for RL) and ADR 0006 (this work) are independent commits but compounding effects. After ADR 0005 lands and a fresh RL checkpoint exists, the natural follow-up notebook is "RL (post-ADR-0005) vs `OrderUpToPolicy` across capacity scales" — the existence of which is enabled by both ADRs.
- Cold-start `opening_budget_pct=0.50` was deliberately set high (the user's call). Reducing it later is a one-line change; the high default is conservative for the worst case (zero-stock RL episodes) and harmless when the store opens with positive `init_stock_pct` (no pilot fires).
- `stockout_safety_bonus_ticks` is the only opt-in heuristic in the family. It exists because pure textbook (s,S) with censored sales has a known failure mode (stockout → under-order → stockout). The default `0` keeps the policy textbook-pure; scenario authors who observe perpetual-stockout pathology can flip it on.
