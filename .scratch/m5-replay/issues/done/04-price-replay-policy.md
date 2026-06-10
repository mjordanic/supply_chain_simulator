# 04: PriceReplayPolicy wrapper

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also ADR 0020 and `.scratch/m5-replay/STUDY.md`)

## What to build

An opt-in wrapper intermediate policy that replays observed selling prices: it delegates
`decide()` to any inner `IntermediatePolicy` (textbook, tuned, anything), then overwrites
the `list_price` portion of the decision with values from per-product, per-tick daily
price arrays. Ordering decisions pass through untouched. Unwrapped policies price freely —
opt-in by construction.

The wrapper consumes plain per-tick arrays and knows nothing about Walmart week numbering
(weekly→daily expansion is the M5 adapter's job, issue 05). Prices it posts flow through
the normal offer-publication path so buyers see them in the central table.

Independent of the replay-sink slices — works on any scenario today.

## Acceptance criteria

- [ ] Wrapper delegates to the inner policy: order decisions are bit-identical with and without the wrapper (same seeds)
- [ ] Offers published by a wrapped node carry the observed price for that tick, for every carried product with a price array
- [ ] Products without a price array keep the inner policy's price (partial coverage works)
- [ ] Price array shorter than `n_steps` fails fast at construction/build time
- [ ] Composes with the multi-supplier textbook policy family (the policies under evaluation)

## Blocked by

None - can start immediately
