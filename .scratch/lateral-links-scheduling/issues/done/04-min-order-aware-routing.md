# 04 — Min-order-aware routing

Status: done

## Parent

`.scratch/lateral-links-scheduling/PRD.md` (ADR 0018)

## What to build

Fix the silent lost sale that the relaxed graph makes common: when a buyer's routed quantity is
below a supplier's `min_order`, the policy-less default action currently rejects it silently instead
of trying the next feasible supplier.

- `_default_sink_action` skips a supplier whose `min_order` the quantity it would route there cannot
  meet, and falls through to the next-cheapest feasible supplier.
- No change to `_split_across_suppliers`: it delegates to `_default_cheapest_first_strategy`, which
  already computes `effective_min = max(offer.min_order, buyer_min_order_floor)` and skips a supplier
  both pre-allocation (`remaining < effective_min`) and post-clamp (`qty < effective_min`). Both
  min-order layers are already honoured — the only confirmed gap is the `_default_sink_action`
  default path.

This makes the intended tier economics work: a warehouse-type intermediate (low price, high
`min_order`) and a shop-type (higher price, low `min_order`) differ purely through these fields with
no engine branching — a small sink falls through the warehouse to a feasible shop rather than losing
the sale.

## Acceptance criteria

- [ ] `_default_sink_action` skips a too-small (below `min_order`) cheapest supplier and routes to the next feasible one.
- [ ] A low-demand sink reaching a high-`min_order` warehouse with a feasible shop also available fills from the shop — no silent lost sale.
- [ ] When *no* supplier is feasible, the demand is genuinely unmet (no spurious fill) — leaves the lost-sale record to issue 05.
- [ ] Unit/integration test asserts the fall-through behavior against an engineered min-order mismatch.

## Blocked by

- Issue 01 (type-based validation — needed so warehouse→sink / relaxed edges that make this matter are authorable). Can be implemented in parallel with issues 02/03.
