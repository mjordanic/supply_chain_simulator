# 05 — `PeriodicOrderUpToPolicy` (R,S)

Status: ready-for-agent

## Parent

PRD: `.scratch/textbook-reorder-policies/PRD.md`
ADR: `docs/adr/0006-textbook-reorder-policy-family.md`

## What to build

The (R,S) periodic-review variant of the textbook reorder-policy family. Subclasses `TextbookReorderPolicy` (added in issue 01) and ships as `PeriodicOrderUpToPolicy` in `src/sim/policy.py`. Reviews inventory only every `review_interval` ticks; on review ticks it orders enough to bring position to `S`. On non-review ticks no order fires regardless of position.

This is the classical "periodic-review, order-up-to" policy from inventory theory. Together with `PeriodicReorderPolicy` (issue 06) it forms the periodic pair of the four-variant ladder. PRD user story 4 motivates the ladder.

### Subclass shape

`PeriodicOrderUpToPolicy(TextbookReorderPolicy)`:

- One additional constructor kwarg:
  - `review_interval: int | None = None` — review cadence in ticks. When `None` (default), `review_interval` equals `delivery_lag` read from the observation (review at lead-time cadence). Explicit positive integers are honoured verbatim.
- `_trigger(pid, step, position, rate)` returns `step % review_interval == 0`. Position is ignored — the periodic schedule alone gates ordering.
- `_quantity(pid, position, rate)` returns `max(0, int(round(S − position)))` where `S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate`. The `max(0, …)` clamp matters here in a way it doesn't for `OrderUpToPolicy`: because the trigger fires unconditionally on review ticks, position may already exceed `S` (e.g. after a pilot order) and we must not emit a negative order.

All other behaviour — pilot order, two-pass fair-share allocator, flat pricing, frozen assortment, RNG isolation — is inherited unchanged from `TextbookReorderPolicy`.

### Tests added

Append to `tests/sim/test_textbook_policy.py`:

- `test_periodic_order_up_to_pilot_fires_on_tick_zero` — fresh Store with `init_stock_pct=0.0` places a pilot order on tick 0 (independent of `review_interval`, since pilots are handled before the trigger loop).
- `test_periodic_order_up_to_orders_only_on_review_ticks` — drive a Store for `4 × review_interval` ticks under constant demand; assert orders fire only when `step % review_interval == 0` and no orders fire on intermediate ticks (even when position drops below `s`).
- `test_periodic_order_up_to_brings_position_to_S` — on a review tick the requested qty equals `S − position` (within integer rounding), bringing position to `S` after the order lands.
- `test_periodic_order_up_to_no_negative_order_when_above_S` — synthetic setup with `position > S` on a review tick: requested qty is `0`, not negative.
- `test_periodic_order_up_to_crn_self_consistency` — two `PeriodicOrderUpToPolicy()` instances on the same CRN seed produce bit-identical trajectories.

## Acceptance criteria

- [ ] `src/sim/policy.py` exports `PeriodicOrderUpToPolicy` (added to `__all__`).
- [ ] `PeriodicOrderUpToPolicy()` instantiates with no arguments; the implicit `review_interval=None` default falls back to per-pid `delivery_lag`.
- [ ] `PeriodicOrderUpToPolicy(review_interval=7)` instantiates and uses the fixed integer.
- [ ] All five base-class invariants from issue 01 hold (CRN isolation, pilot, flat pricing, frozen assortment, allocator).
- [ ] The five new tests listed above are added to `tests/sim/test_textbook_policy.py` and pass under `uv run pytest tests/sim/test_textbook_policy.py`.
- [ ] No regression in `uv run pytest tests/sim/`.

## Blocked by

- `01-textbook-base-and-order-up-to.md` (subclasses `TextbookReorderPolicy`).
