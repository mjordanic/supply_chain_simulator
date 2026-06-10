# Set encoder/decoder: (K_max, F) layout with mask, slimmed row, layout version

Status: done

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decisions 1–2)

## What to build

A rewritten observation/action codec for the variable-K layout, alongside the existing flat
codec (the old one is retired in issue 05; both exist until then so the suite stays green).

Observation: a `(K_max, F)` tensor flattened for the Box space, with `K_max = 32`. Per-product
row contains:

- The ~10 live per-SKU features — the survivors of the current 18 after dropping the dead
  constants (lifecycle one-hot, in-season flag, ticks-since-activation, mean-lead-time, all
  degenerate while `item_registry` is None). Head semantics for the surviving features are
  unchanged.
- The 4 global features broadcast onto each row.
- 2 contention aggregates: total proposed quantity over free space, and total estimated order
  cost over cash budget — so products sense competition for shared resources without seeing
  each other's identities.
- 1 mask channel (1.0 for active rows, 0.0 for padding). Mask recoverable from the layout.

Padded rows are all-zero. Action layout: `(K_max, 3)` — price multiplier and order-up-to target
with semantics unchanged from the scale-invariance package (ADR 0007: price `[-1,1] → [0.5,1.5]`
× MSRP; order-up-to in lead-time units), plus the priority scalar (always present; ignored by
the proportional Arbiter so both arbiter arms share one checkpoint format).

Decoding honours the mask: padded slots produce no orders. Decode feeds the Arbiter (issue 01)
and emits the per-pid dict format the engine's `decide` contract expects, for active products
only.

The encoder module owns a single observation-layout version constant — the value checkpoints
validate against (issue 04). Any future layout change has exactly one place to bump.

## Acceptance criteria

- [ ] Row layout matches a documented feature list (docstring/constants), with named index constants — no magic offsets.
- [ ] Mask correct for every K < K_max; padded rows are exactly zero.
- [ ] Decode honours the mask: no orders or prices emitted for padded slots.
- [ ] Round-trip through the Arbiter produces per-pid dicts only for active products.
- [ ] Layout version constant exists in the encoder module and is the single source of truth.
- [ ] Dedicated test module (one of the three pure-core modules with dedicated tests per the PRD).
- [ ] Existing flat codec and its tests untouched (retirement in issue 05).

Prior art: `test_encoders` for layout/decoding test style; ADR 0007 for head semantics.

## Blocked by

- `01-arbiter-module.md` (decode round-trip tests exercise the Arbiter)
