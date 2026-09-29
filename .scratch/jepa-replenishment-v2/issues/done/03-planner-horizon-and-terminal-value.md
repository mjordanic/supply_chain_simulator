# 03: Decouple the planner's scoring horizon and add a terminal value

**What to build:** A planner that orders. Today the scoring horizon equals the
delivery lead time, so an order is charged to imagined cash inside the window
while the units it buys arrive at or after the window's edge. The
profit-maximising proposal is therefore always the one that orders least, no
matter how good the world model is. Substituting a perfect world model — exact
inventory recursion with correct arrival timing — into the unmodified planner
scores 0.336 against the learned model's 0.333 on the same seeds. The learned
buyer placed orders on 7 of 365 ticks, 94 units for a year, while the random
behaviour policy that generated its training data ordered roughly 8,900.

Measured with an oracle model, moving the horizon from 3 to 10 moves the score
from 0.336 to 3.444; 30 gives 3.488. The published anchor reference is 3.903.

After this ticket the scoring horizon is a named multiple of the delivery lead
time and the end of the horizon carries a continuation value, so the boundary
stops being a cliff and the multiple stops needing delicate tuning.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

**Prior art:** existing planner tests; the `Latent replenishment planner`,
`Realized profit`, `Holding cost`, `Order fee` and `Order cost` entries in
`CONTEXT.md`; ADR 0024 (ordering-only learned policy) for what the action may
contain.

- [ ] The scoring horizon is `horizon_multiple * delivery_lag` with
      `horizon_multiple` a named planner knob, defaulting to roughly 4.
- [ ] A terminal value on ending inventory position is applied at the end of the
      horizon.
- [ ] A behavioural test asserts the planner orders a sensible quantity under
      known-good conditions — this is the test whose absence let the refusal-to-
      order bug ship, so it must fail against the pre-fix behaviour.
- [ ] The planner's score is insensitive to `horizon_multiple` across a
      reasonable range once the terminal value is present; that insensitivity is
      asserted, not assumed.
- [ ] The learned buyer stays ordering-only per ADR 0024: `list_price` from
      `base_price`, no promotions, no learned minimum-order floor.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      knob is updated in this ticket.
