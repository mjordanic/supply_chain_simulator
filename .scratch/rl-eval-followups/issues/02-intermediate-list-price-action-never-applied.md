# 02 · Intermediate `list_price` / `min_order_imposed` actions are never applied

Status: needs-triage

## Problem

The demand-pull walk (`src/sim/runner.py:434-497`) reads only `action["order"]` for
`IntermediateNode`s; the `list_price` and `min_order_imposed` entries a policy returns
from `decide()` are never written to the node. Factories *do* get their prices applied
(`src/sim/runner.py:523-525`). No commit in git history ever applied them for
intermediates — this has been inert since the demand-pull schedule landed (ADR 0018).

Consequences:

- The RL price head (action column 0) has no effect on dynamics — in training and eval
  alike (so it is *not* a train/eval parity gap, just dead weight in the action space).
  Confirmed by notebook 06's `mean_price_pct_of_msrp = 1.000` for both arms.
- Any pricing-based textbook policy attached to an intermediate node is equally inert.
- The notebook-06a "implicit assortment" story ("stop carrying via prices") overstated
  what the engine supports; a caveat now marks it (06a §6).

## Decision needed

Either:

1. **Apply the actions** — write `list_price` / `min_order_imposed` to the node at
   decide time, before the next publish, mirroring the factory path. Touches engine
   dynamics (behaviour-changing for any policy that sets prices), RL action-space
   semantics (the price head becomes live → retraining to exploit it), and notebooks.
2. **Document as out of scope** — declare intermediate pricing unsupported, remove the
   price head from the RL action space (action goes `(K_MAX, 3)` → `(K_MAX, 2)` with
   priority kept), and update the decoders/notebooks. Checkpoint-breaking.

## Notes

- Prior art: factory price application in the demand-pull walk; `decode_set_action`
  price decoding in `src/rl/set_encoder.py`.
- Cross-reference: notebook 06a §6 caveat points at this issue by filename.
