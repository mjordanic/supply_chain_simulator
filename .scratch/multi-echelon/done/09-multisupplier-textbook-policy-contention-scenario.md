# MultiSupplierTextbookPolicy + textbook subclass migration + routing strategy + contention scenario + tests

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Lift the textbook policy family onto the multi-supplier `IntermediatePolicy` contract and demonstrate supply contention end-to-end on a 2F-2S scenario.

`MultiSupplierTextbookPolicy(IntermediatePolicy)` becomes the new base. All four concrete subclasses (`OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy`) re-route through it. The per-pid trigger / quantity hooks (rate estimator, safety horizon, two-pass fair-share allocator) are preserved unchanged — only the order-emission step is upgraded to multi-supplier splits.

The new method `_split_across_suppliers(pid, qty_total, suppliers, central_table) -> list[(supplier_id, qty)]` is the pure routing function — encapsulates min-order discipline + price ordering + partial-fill arithmetic. Default strategy: cheapest-first, both min-order layers honoured (buyer-side and supplier-imposed), partial fills surfaced through `fill_rate_recent`. Pluggable via `routing_strategy` ctor kwarg.

`OrderUpToPolicy` remains the canonical CRN comparison anchor — its behaviour on the degenerate 3-node chain must match issue 6's `SingleSupplierAdapter` exactly (single-supplier case is a degenerate multi-supplier).

Ship `scenarios/example_two_factories_two_shops.py`: two `FactoryNode` (`F_lo` cheap-slow, `F_hi` premium-fast), two `IntermediateNode` (`S1`, `S2`), one `DemandSinkNode` per shop, same product. Should produce a visible `fill_rate_recent` drop during contention bursts in the run log.

Ship tests:
- `tests/sim/test_multisupplier_routing.py` — pure unit tests on `_split_across_suppliers`: cheapest-first ordering, both min-order layers reject sub-min lines, partial-fill arithmetic, pluggable `routing_strategy` invoked when supplied
- `tests/sim/test_determinism.py` generalised — the four legacy `Store` invariants restated as graph invariants on the two-factories-two-shops scenario

## Acceptance criteria

- [ ] `MultiSupplierTextbookPolicy(IntermediatePolicy)` base class lands
- [ ] `OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy` all re-rooted on `MultiSupplierTextbookPolicy`
- [ ] Per-pid trigger / quantity hooks preserved unchanged (rate estimate, safety horizon math)
- [ ] `_split_across_suppliers(pid, qty_total, suppliers, central_table)` pure function: cheapest-first default, both min-order layers honoured, partial-fill arithmetic correct
- [ ] `routing_strategy` ctor kwarg pluggable; the default and at least one alternative (e.g. fill-rate-weighted) shipped
- [ ] `OrderUpToPolicy` behaves identically to issue 6's `SingleSupplierAdapter` on the degenerate single-supplier chain (CRN anchor preserved)
- [ ] `scenarios/example_two_factories_two_shops.py` exists; `uv run python scenarios/example_two_factories_two_shops.py` writes a run log; `fill_rate_recent` series shows contention drops
- [ ] `tests/sim/test_multisupplier_routing.py` covers the four routing cases above
- [ ] `tests/sim/test_determinism.py` restates the four legacy invariants as graph invariants
- [ ] `uv run pytest tests/sim -x` is green

## Blocked by

- `.scratch/multi-echelon/issues/07-allocation-execute-buy-full.md`
- `.scratch/multi-echelon/issues/08-allocation-rng-shuffle-ema.md`
