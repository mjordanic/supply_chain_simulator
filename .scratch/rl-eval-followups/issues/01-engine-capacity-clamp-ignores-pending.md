# 01 · Engine capacity clamp ignores pending stock and same-tick siblings

Status: needs-triage

## Problem

`execute_buy` (`src/sim/allocation.py:169-180`) clamps each fill to
`capacity − sum(inventory.values())`. Two gaps:

1. **In-transit `pending` stock is not counted.** Purchases land in `pending`, not
   on-hand, so stock already ordered does not reduce the headroom seen by the next
   order.
2. **Headroom is not decremented across multiple order lines within the same tick.**
   All K same-tick lines for one buyer clamp against the same on-hand snapshot.

Any policy that bypasses the RL Arbiter (textbook policies don't over-order in
practice, but nothing stops one) can stack pending deliveries to many times the node
capacity. Measured during the 2026-06-11 eval-harness investigation: the pre-ADR-0022
arbiter-less RL eval path drove node `S` to a max on-hand of 1,335–1,696 units at
capacity 101–136 (10–16× over) on the notebook-06a §5b starved specs. This was the
entire source of the previously reported "RL beats baseline in the starved regime"
numbers.

## Proposed fix

Clamp against `capacity − on_hand − pending_total`, and decrement a per-tick
remaining-capacity accumulator per buyer so multiple lines in one tick share the
headroom.

## Notes

- **Behaviour-changing**: engine dynamics shift, so recorded runs / golden tests /
  CRN-paired results move. Needs a re-baseline decision before merging.
- Needs a test: one buyer placing multiple same-tick order lines whose joint quantity
  exceeds headroom must be clipped jointly, and pending stock must count against
  capacity.
- Prior art: the RL Arbiter (`src/rl/arbiter.py`) already implements the correct
  headroom formula (`capacity − inv − pending`, plus cash budget) — the engine clamp
  should agree with it.
- Cross-reference: ADR 0022 (eval-harness rebuild); notebook 06a intro + §5b describe
  the symptom.
