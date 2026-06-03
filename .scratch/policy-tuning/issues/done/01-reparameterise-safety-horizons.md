# 01 — Reparameterise textbook safety horizons as fractions of delivery lag

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0008-safety-horizons-as-fraction-of-delivery-lag.md`

## What to build

Rename two `TextbookReorderPolicy` kwargs from absolute integer tick counts to ratios of delivery lag, and compute the per-pid safety horizon by multiplying that ratio by the pid's `delivery_lag` inside `decide()`. Hard rename — no compatibility alias, old kwarg names raise `TypeError`.

The two affected kwargs:

- `safety_lead_ticks: int = 2` → `safety_lead_pct_of_lag: float = 2 / 3`
- `stockout_safety_bonus_ticks: int = 0` → `stockout_safety_bonus_pct_of_lag: float = 0.0`

`cover_horizon_ticks` stays absolute (EOQ cycle length is independent of lead time). `opening_budget_pct` and `min_qty` are unchanged.

Per-pid computation inside `decide()`:

```
effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
effective_bonus_ticks  = round(stockout_safety_bonus_pct_of_lag × delivery_lag[pid])  # when stockout-detected
s = (delivery_lag[pid] + effective_safety_ticks + effective_bonus_ticks) × rate
S = s + cover_horizon_ticks × rate
```

The class docstring gains a top-level "Why these denominators" block with three short paragraphs covering: (i) the EOQ cycle-length result `T* = √(2K/(D·h))` and why lead time does not appear there, (ii) the textbook newsvendor `SS = z·σ·√L` and why safety scales with lead time, (iii) the linear approximation `SS ≈ k·L·rate` and why it matches the rest of the policy's `ticks × rate` structure.

Migrate every call site that names the old kwargs. The defaults at `lag=3` are bit-identical to ADR 0006 (`round(2/3 × 3) = 2`), so the canonical CRN regression snapshot regenerates identically at the standard scale; trajectories on heterogeneous-lag worlds will differ (this is the bug being fixed).

The two shared helpers in `src/sim/policy.py` (`_estimate_rate`, `_allocate_two_pass_fair_share`) are unchanged. `evaluate(rl_policy_fn, baseline_policy_factory, …)` in `src/rl/eval.py` is unchanged.

PRD user stories covered: 4, 5, 6, 7, 8, 9, 29, 30.

### Call-site migration buckets

- `src/sim/policy.py` — `TextbookReorderPolicy.__init__` signature, `decide()` body, class docstring, ADR cross-reference.
- `src/rl/eval.py` — type-hint imports only (no explicit kwarg passing in current code).
- `src/rl/train.py` — type-hint imports only.
- `notebooks/06-compare_rl_vs_baseline.ipynb`, `notebooks/07-rl_vs_baseline_per_product.ipynb` — `baseline_factory` bodies if they override; markdown referencing "safety lead ticks".
- `tests/sim/test_regression_snapshot.py` — regenerate the pinned trajectory; assert bit-identical at the standard `lag=3` scale.
- Any other test that names `safety_lead_ticks=` or `stockout_safety_bonus_ticks=` flips to the new names.

### Tests added

In `tests/sim/test_textbook_policy.py` (extend if present, follow existing test style):

- `test_safety_pct_of_lag_per_sku` — Two-SKU synthetic catalog with `lag=1` and `lag=10`; drive a Store for 20 ticks of constant demand; assert `s` for the `lag=10` SKU is roughly 5× the `s` for the `lag=1` SKU (10·rate + round(0.667×10)·rate ≈ 17·rate vs 1·rate + round(0.667×1)·rate ≈ 2·rate at default kwargs).
- `test_safety_pct_of_lag_default_equivalence` — Default `OrderUpToPolicy()` on the canonical `lag=3` template; drive a Store through 50 ticks; assert the resulting `decide()` outputs are bit-identical to a pinned pre-migration baseline. Self-snapshotting test: any drift is a deliberate review item.
- `test_no_safety_lead_ticks_kwarg` — `OrderUpToPolicy(safety_lead_ticks=2)` raises `TypeError`; same for `stockout_safety_bonus_ticks=`.
- `test_safety_pct_of_lag_monotonic_in_safety` — At `lag=3`, increasing `safety_lead_pct_of_lag` from `0.0` → `1.0` → `3.0` produces monotonically-increasing `s` levels across a 50-tick rollout on a single SKU.

In `tests/sim/test_regression_snapshot.py`: regenerate the pinned trajectory (the harness is unchanged; only the snapshot values move).

In `tests/rl/test_eval_crn.py`: no assertion changes; re-run to confirm the CRN-paired contract is preserved.

### CONTEXT.md update (in this slice)

Update the `TextbookReorderPolicy family` glossary entry to mention the renamed kwargs, the per-pid computation, the default-equivalence at `lag=3`, and the ADR 0008 cross-reference. Add an entry for ADR 0008 in the Decisions list. The new `Policy tuning study` glossary entry and the `CRN-paired eval` addition are out of scope for this slice (covered by issue 08).

## Acceptance criteria

- [ ] `TextbookReorderPolicy.__init__` accepts `safety_lead_pct_of_lag: float = 2/3` and `stockout_safety_bonus_pct_of_lag: float = 0.0`; rejects `safety_lead_ticks=` and `stockout_safety_bonus_ticks=` with `TypeError`.
- [ ] `decide()` computes the per-pid safety horizon by `round(safety_lead_pct_of_lag × delivery_lag[pid])` and applies it to `s`.
- [ ] `cover_horizon_ticks`, `opening_budget_pct`, and `min_qty` are unchanged in name, type, and default.
- [ ] Class docstring contains the "Why these denominators" block covering EOQ + newsvendor + linear-approximation rationale.
- [ ] `tests/sim/test_regression_snapshot.py` passes against a regenerated snapshot at the canonical `lag=3` scale; trajectory values are bit-identical to the pre-migration baseline.
- [ ] All four new tests in `tests/sim/test_textbook_policy.py` pass.
- [ ] All notebooks and call sites referencing the old kwarg names are migrated; `grep -r safety_lead_ticks` and `grep -r stockout_safety_bonus_ticks` return no matches outside the ADR text.
- [ ] `CONTEXT.md`'s `TextbookReorderPolicy family` glossary entry mentions the renamed kwargs, the per-pid computation, and ADR 0008.
- [ ] `uv run pytest tests/sim/ tests/rl/` is green.

## Blocked by

None — can start immediately.
