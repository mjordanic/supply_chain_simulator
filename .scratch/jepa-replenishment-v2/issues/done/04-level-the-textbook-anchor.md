# 04: Level the textbook anchor

**What to build:** A textbook anchor that faces the same constraints as the
learned buyer, so a profit gap between them means something.

Today `MultiSupplierTextbookPolicy` builds its internal observation with a
`unit_cost` that defaults to 1.0 while the real supplier price is near 11, and
the tuning search space never overrides it. Its internal budget check is
therefore roughly an order of magnitude too loose: it over-requests and relies on
the Allocation clamp to cut it back, while the learned planner self-clamps
honestly against the true price. It also fires a cold-start pilot worth half its
cash, which the planner has no equivalent of.

A second asymmetry arrives with multiple suppliers: the anchor's search space
tunes `routing_strategy` over cheapest-first and fill-rate-weighted, and it tunes
`per_supplier_min_order_floor`, while the ordering-only planner is pinned to one
routing strategy and a zero floor. Once fill rates vary the anchor could win on
routing alone.

**Blocked by:** 01.

**Status:** ready-for-agent

**Prior art:** existing textbook-policy tests; the `TextbookReorderPolicy family`
and `Allocation` entries in `CONTEXT.md`; ADR 0006 (textbook reorder policy
family), ADR 0008 (safety horizons as fraction of delivery lag), ADR 0024
(ordering-only learned policy).

- [ ] The textbook policy family reasons with the true supplier unit cost rather
      than a default of 1.0.
- [ ] A test asserts the anchor's internal budget check actually binds when cash
      is short, rather than deferring to the Allocation clamp.
- [ ] Routing strategy and minimum-order floor are shared settings applied
      identically to the anchor and the planner; neither is left free for one arm
      to tune into while the other is pinned.
- [ ] Cold-start behaviour for anchor and planner is stated explicitly and
      matched, so an opening-order advantage is a deliberate choice.
- [ ] Baselines elsewhere in the suite that move because the textbook policy now
      costs orders correctly are re-established.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by this
      change is updated in this ticket.
