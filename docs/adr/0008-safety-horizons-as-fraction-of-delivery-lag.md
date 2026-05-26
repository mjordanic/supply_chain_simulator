# Reparameterise textbook safety horizons as fractions of delivery lag

ADR 0006 introduced the `TextbookReorderPolicy` family and named (s,S) `OrderUpToPolicy` as the canonical RL CRN comparison anchor. Levels are computed per tick from a censored-sales rate estimate:

```
s = (delivery_lag + safety_lead_ticks) × rate
S = s + cover_horizon_ticks × rate
```

`cover_horizon_ticks` (default 10) and `safety_lead_ticks` (default 2) are absolute integers in tick units, with the opt-in `stockout_safety_bonus_ticks` (default 0) adding a temporary additive bump to `safety_lead_ticks` during stockout-containing windows. The framing "scale-invariant in capacity by construction" (ADR 0006) holds: `rate` carries the scale, the horizons are time, the product is in demand units, so the same kwargs work across two orders of magnitude of capacity.

Two problems surface as soon as the policy is asked to *vary* parameters or run on a catalog with heterogeneous lead times.

(1) **Absolute safety horizons mis-scale across SKUs.** `Store.observe()` already exposes per-pid `delivery_lags`, and `TextbookReorderPolicy._get_delivery_lag(pid, observation)` already reads it (see `src/sim/policy.py:_get_delivery_lag`). But `safety_lead_ticks` is a single absolute integer applied uniformly across SKUs. On a catalog where SKU A has `lag=1` and SKU B has `lag=10`, `safety_lead_ticks=2` translates to safety equal to **200 %** of lead time on A and **20 %** of lead time on B — exactly backwards from the textbook newsvendor result, which says safety scales *with* lead time (`SS ∝ σ · √L`). The bug is latent on hand-authored worlds where every SKU shares a lag (the CLI template ships `delivery_lag=3` uniformly) and surfaces on LLM-built worlds where per-category lead times are authored independently.

(2) **The current parameterisation is hostile to hyperparameter tuning.** Once an Optuna-style search is run over `safety_lead_ticks`, the search-space integer "2" means something different in a `lag=3` world (safety = 67 % of lag) than in a `lag=8` world (safety = 25 % of lag). A tuner that optimises against a fixed log-uniform capacity domain will produce kwargs that do not transfer across worlds — the value is a number of ticks, not a piece of textbook logic. ADR 0006 alternative (a) rejected per-world baseline tuning as a methodology precisely because the heuristic kwargs were not transferable; the textbook family inherits the *same* failure mode as long as its safety horizon stays absolute.

**Decision — Rename `safety_lead_ticks` to `safety_lead_pct_of_lag` (float, ratio of delivery lag) and `stockout_safety_bonus_ticks` to `stockout_safety_bonus_pct_of_lag` (float, ratio of delivery lag). Keep `cover_horizon_ticks` absolute.**

The two safety horizons are computed per pid inside `TextbookReorderPolicy.decide()` by multiplying the ratio by that pid's `delivery_lag` and rounding to int. Concretely:

```
effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
effective_bonus_ticks  = round(stockout_safety_bonus_pct_of_lag × delivery_lag[pid])  # when stockout-detected
s = (delivery_lag[pid] + effective_safety_ticks + effective_bonus_ticks) × rate
S = s + cover_horizon_ticks × rate
```

New defaults are `safety_lead_pct_of_lag = 2/3 ≈ 0.667` and `stockout_safety_bonus_pct_of_lag = 0.0`, chosen so that on the canonical CLI template (`delivery_lag=3`) the resulting `effective_safety_ticks = round(0.667 × 3) = 2` and `effective_bonus_ticks = 0` — bit-identical to ADR 0006 defaults at the test scale. The CRN regression snapshot in `tests/sim/test_regression_snapshot.py` is regenerated and confirmed to produce the same trajectory at `lag=3` (any deviation surfaces as a snapshot diff and blocks the migration).

**Why `cover_horizon_ticks` stays absolute.** Cover horizon drives `S − s` — the cycle length per replenishment. The textbook EOQ result for cycle length is

```
T* = √(2K / (D · h))
```

where K is the fixed ordering cost, h is the holding-cost rate, and D is the demand rate. **Lead time does not appear.** Cycle length is a function of the cost-tradeoff between ordering and holding; lead time is a function of supplier physics. Tying `cover_horizon_ticks` to `delivery_lag` would conflate two economically independent decisions and bake in a wrong invariant (if a supplier becomes slower, you do not want larger orders — you want to reorder earlier, which is what the `s` shift already does). The absolute integer in ticks of demand cover is the natural unit.

**Why the safety horizons become ratios.** The textbook safety-stock formula

```
SS = z · σ_demand · √L
```

is explicit that safety scales with lead time. A linear approximation `SS ≈ k · L · rate` (which is what `safety_lead_pct_of_lag × delivery_lag × rate` is) is one term off from the textbook square-root form, but it preserves the qualitative invariant ("more lead time → more safety") and is what most practical inventory systems use as an approximation. It also matches the existing additive structure of the rest of the policy (everything is `ticks × rate`) without introducing a new mechanism. The float type and ratio range (0.0–3.0 for safety, 0.0–2.0 for stockout bonus) are also strictly more expressive than the old integer absolute (which was clamped at small values for most lead times).

**Why `opening_budget_pct` and `min_qty` are not affected.** `opening_budget_pct` is already a ratio (fraction of opening cash); it is independent of lead time. `min_qty` is an allocator-floor in unit-of-stock, not a horizon; it can stay absolute and ADR 0006's textbook-pure default `min_qty = 0` is preserved.

Considered alternatives:

(a) **Keep both kwargs (old absolute + new ratio) with mutual exclusion.** Soft fallback so existing callsites do not need to change. Rejected for the same ADR 0006 migration philosophy: a hard `ImportError` / `TypeError` at the call site forces every importer to consciously pick its intent. A soft fallback obscures the per-SKU correctness fix and creates a class with two parameter-mode branches that future maintainers must keep aligned.

(b) **Scale safety as a fraction of `cover_horizon_ticks` instead of `delivery_lag`.** The original user proposal during design grilling. Rejected on textbook grounds: cycle length and lead-time-risk are independent concepts and using `cover_horizon_ticks` as the denominator would cross-couple them. A scenario author who chooses to order weekly (`cover_horizon_ticks = 7`) would automatically dial *down* safety relative to lead-time, which is the opposite of what is wanted under variable supplier reliability.

(c) **Drop `safety_lead_ticks` entirely and derive safety from a service-level kwarg (e.g. `target_service_level = 0.95`).** Closer to textbook-pure: safety derives from the z-score against the demand standard deviation in the rolling window. Rejected because the rolling-window demand variance is itself unstable on the censored series the policy estimates from (one stockout corrupts the variance estimate for the entire window); the ratio form is robust and explains 80 % of the practical effect with one float. A service-level kwarg is a sensible follow-up if a future need arises.

(d) **Implement the square-root form `SS = k · √L · rate`.** Strictly closer to textbook. Rejected because the gain over the linear form is modest in the lead-time range this codebase operates in (1 – 10 ticks) and the kwarg becomes harder to reason about (`k = 1.5` corresponds to different safety percentages at different lead times). The linear form is what most practical (s,S) implementations use; the textbook square-root term is dominated by the demand-variability factor `σ` which we are already approximating implicitly through the censored mean.

**Migration.** Hard rename, no compatibility alias. Five callsite buckets:

1. `src/sim/policy.py` — `TextbookReorderPolicy.__init__` signature changes; `decide()` body changes to multiply per pid; class docstring gains an "EOQ vs safety stock" explanation block; ADR 0006 cross-reference updated.
2. `src/rl/eval.py`, `src/rl/train.py` — `OrderUpToPolicy()` call sites use new defaults implicitly (no explicit kwargs in current code).
3. `notebooks/06-compare_rl_vs_baseline.ipynb`, `notebooks/07-rl_vs_baseline_per_product.ipynb` — `baseline_factory` bodies use the renamed kwargs explicitly if they currently override; markdown text referencing "safety lead ticks" updated.
4. `tests/sim/*` — any test instantiating `OrderUpToPolicy(safety_lead_ticks=…)` flips to `safety_lead_pct_of_lag=…`.
5. `tests/sim/test_regression_snapshot.py` — regenerate the pinned trajectory after migration; assert bit-identical at the standard `lag=3` scale, surface any rounding-flip as a deliberate review item.

`CONTEXT.md` updates the `TextbookReorderPolicy family` and `Policy` glossary entries to mention the new kwarg names and the per-SKU correctness property. `min_qty` and `opening_budget_pct` are unchanged.

The CRN-paired eval contract (`src/rl/eval.py::evaluate(rl_policy_fn, baseline_policy_factory, …)`) is unchanged. The canonical RL baseline trajectory at the standard `lag=3` test scale is bit-identical. Trajectories on worlds with heterogeneous `delivery_lags` will differ from pre-migration runs — this is the bug being fixed; any downstream consumer comparing pre/post-change numbers on such worlds is told via the migration note.
