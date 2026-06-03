# 06 — `PeriodicReorderPolicy` (R,s,S)

Status: ready-for-agent

## Parent

PRD: `.scratch/textbook-reorder-policies/PRD.md`
ADR: `docs/adr/0006-textbook-reorder-policy-family.md`

## What to build

The (R,s,S) periodic-review variant of the textbook reorder-policy family. Subclasses `TextbookReorderPolicy` (added in issue 01) and ships as `PeriodicReorderPolicy` in `src/sim/policy.py`. Combines the (R,S) periodic-review cadence with an (s,S) reorder-point gate: orders fire only on review ticks AND only when position is below `s`. When both conditions hold, the order brings position up to `S`.

This is the most conservative of the four textbook variants — periodic review reduces ordering overhead, and the reorder-point gate avoids small "top-off" orders on review ticks where position is already healthy. PRD user story 4 motivates including it in the ladder: an RL agent that beats (s,S) but loses to (R,s,S) on one world has learned something specific about continuous review rather than general inventory policy.

### Subclass shape

`PeriodicReorderPolicy(TextbookReorderPolicy)`:

- One additional constructor kwarg:
  - `review_interval: int | None = None` — review cadence in ticks. When `None` (default), `review_interval` equals `delivery_lag` read from the observation (review at lead-time cadence). Explicit positive integers are honoured verbatim. Same semantics as `PeriodicOrderUpToPolicy` (issue 05).
- `_trigger(pid, step, position, rate)` returns `step % review_interval == 0 and position < s` where `s = (delivery_lag + safety_lead_ticks) × rate`. Both conditions must hold.
- `_quantity(pid, position, rate)` returns `int(round(S − position))` where `S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate`. No `max(0, …)` clamp needed because the trigger requires `position < s < S`, so `S − position > 0` is guaranteed.

All other behaviour — pilot order, two-pass fair-share allocator, flat pricing, frozen assortment, RNG isolation — is inherited unchanged from `TextbookReorderPolicy`.

### Tests added

Append to `tests/sim/test_textbook_policy.py`:

- `test_periodic_reorder_pilot_fires_on_tick_zero` — fresh Store with `init_stock_pct=0.0` places a pilot order on tick 0 (pilots are handled before the trigger loop, independent of both gates).
- `test_periodic_reorder_no_order_on_non_review_tick` — even when `position < s`, no order fires unless `step % review_interval == 0`.
- `test_periodic_reorder_no_order_on_review_tick_when_above_s` — on a review tick with `position >= s`, no order fires.
- `test_periodic_reorder_fires_when_both_conditions_met` — drive a Store under constant demand long enough that `position < s` aligns with a review tick; assert exactly one order fires that tick with qty `S − position` (within integer rounding).
- `test_periodic_reorder_crn_self_consistency` — two `PeriodicReorderPolicy()` instances on the same CRN seed produce bit-identical trajectories.

## Acceptance criteria

- [ ] `src/sim/policy.py` exports `PeriodicReorderPolicy` (added to `__all__`).
- [ ] `PeriodicReorderPolicy()` instantiates with no arguments; the implicit `review_interval=None` default falls back to per-pid `delivery_lag`.
- [ ] `PeriodicReorderPolicy(review_interval=7)` instantiates and uses the fixed integer.
- [ ] All five base-class invariants from issue 01 hold (CRN isolation, pilot, flat pricing, frozen assortment, allocator).
- [ ] The five new tests listed above are added to `tests/sim/test_textbook_policy.py` and pass under `uv run pytest tests/sim/test_textbook_policy.py`.
- [ ] No regression in `uv run pytest tests/sim/`.

## Blocked by

- `01-textbook-base-and-order-up-to.md` (subclasses `TextbookReorderPolicy`).
