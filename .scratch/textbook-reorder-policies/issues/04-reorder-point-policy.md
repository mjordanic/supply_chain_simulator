# 04 — `ReorderPointPolicy` (s,Q)

Status: ready-for-agent

## Parent

PRD: `.scratch/textbook-reorder-policies/PRD.md`
ADR: `docs/adr/0006-textbook-reorder-policy-family.md`

## What to build

The (s,Q) continuous-review variant of the textbook reorder-policy family. Subclasses `TextbookReorderPolicy` (added in issue 01) and ships as `ReorderPointPolicy` in `src/sim/policy.py`. Behaves identically to `OrderUpToPolicy` except in the quantity hook: rather than ordering up to a level `S`, it orders a fixed quantity `Q` each time the reorder point is crossed.

This is the classical "reorder-point, fixed-order-quantity" policy from inventory theory (Silver/Pyke/Peterson). Together with `OrderUpToPolicy` it forms the continuous-review pair of the four-variant ladder. PRD user story 4 motivates the ladder: "I want a *ladder* of textbook competitors so that a comparative claim ('RL beats (s,S) by X but only ties (s,Q)') becomes possible without writing four bespoke policies."

### Subclass shape

`ReorderPointPolicy(TextbookReorderPolicy)`:

- One additional constructor kwarg:
  - `Q: int | None = None` — fixed reorder quantity. When `None` (default), `Q` is computed per-pid each tick as `int(round(cover_horizon_ticks × rate))` so the per-cycle order is one cover-horizon's worth of demand. Explicit integer values are honoured verbatim.
- `_trigger(pid, step, position, rate)` returns `position < s` where `s = (delivery_lag + safety_lead_ticks) × rate` (identical to `OrderUpToPolicy`'s trigger).
- `_quantity(pid, position, rate)` returns `Q` (the configured kwarg, or the rate-derived default). Note: unlike `OrderUpToPolicy`, the order qty does NOT depend on how far below `s` position fell — that's the defining (s,Q) property.

All other behaviour — pilot order, two-pass fair-share allocator, flat pricing, frozen assortment, RNG isolation — is inherited unchanged from `TextbookReorderPolicy`.

### Tests added

Append to `tests/sim/test_textbook_policy.py`:

- `test_reorder_point_pilot_fires_on_tick_zero` — same property as the (s,S) version: fresh Store with `init_stock_pct=0.0` places a pilot order on tick 0 sized from `opening_budget_pct`.
- `test_reorder_point_quantity_is_fixed_Q` — drive a Store with constant-demand market; whenever `_trigger` returns true, the requested qty equals `Q` (the configured value or the rate-derived default), independent of how far below `s` the position fell. Test with both `Q=50` explicit and `Q=None` (rate-derived).
- `test_reorder_point_no_order_when_position_above_s` — no order fires while `position > s` (inherited contract; the test confirms the subclass doesn't accidentally re-order on every tick).
- `test_reorder_point_crn_self_consistency` — two `ReorderPointPolicy()` instances on the same CRN seed produce bit-identical trajectories.

## Acceptance criteria

- [ ] `src/sim/policy.py` exports `ReorderPointPolicy` (added to `__all__`).
- [ ] `ReorderPointPolicy()` instantiates with no arguments; the implicit `Q=None` default triggers per-tick rate-derived quantity.
- [ ] `ReorderPointPolicy(Q=50)` instantiates and uses the fixed integer.
- [ ] All five base-class invariants from issue 01 hold for `ReorderPointPolicy` (CRN isolation, pilot, flat pricing, frozen assortment, allocator).
- [ ] The four new tests listed above are added to `tests/sim/test_textbook_policy.py` and pass under `uv run pytest tests/sim/test_textbook_policy.py`.
- [ ] No regression in `uv run pytest tests/sim/`.

## Blocked by

- `01-textbook-base-and-order-up-to.md` (subclasses `TextbookReorderPolicy`).
