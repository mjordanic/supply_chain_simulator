# 02 — Per-tick flow log + flow DataFrame builder

Status: done

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019, amends ADR 0013 Rules 4–5)

## What to build

Make a plain `Runner.run()` emit the minimal per-tick **flows** that state snapshots cannot recover
(inventory on tick *t* reflects both sales and arrivals, so units sold are not derivable after the
fact), and provide one tested, pure function that turns the run log into a tidy DataFrame every
downstream consumer shares.

End-to-end: extend the per-tick snapshot with two flow record types, then build the frame.

- **Per `(node, pid)`:** `sales` (units the node sold as a supplier this tick), `demand` (units
  requested of it; for a `DemandSinkNode`, the exogenous `demand_target`), `price` (the node's
  `list_price` at decision time), `stockout` (boolean: decision-time on-hand == 0 — captured during
  the walk, not inferred from the closing snapshot a same-tick replenishment would mask). The
  current-tick sales already computed transiently for the un-lagged demand signal (ADR 0018) are
  persisted rather than discarded.
- **Per `(buyer, supplier, pid)`:** `qty_filled` and `cash_paid` (= `qty_filled × supplier
  list_price`, from `execute_buy`). This supersedes today's `node_orders: {pid: qty}` aggregate —
  derive the old aggregate from these rows so existing readers stay green.
- The existing end-of-tick `node_inventory` / `node_cash` / `node_pending` snapshots are unchanged
  (inspect / equity / parquet readers depend on them).
- **Flow DataFrame builder (deep module, inspect layer):** a pure function, run log → tidy long-form
  DataFrame, one row per `(node, pid, tick)` for the per-product flows, plus access to the
  per-`(buyer, supplier, pid, tick)` purchase rows. It owns *all* run-log-schema knowledge so
  `metrics.py` (issue 04) stays schema-free.

Only raw flows are logged — holding-cost and order-fee totals stay derived in analysis (issues 04/05).

## Acceptance criteria

- [ ] Each tick snapshot carries the per-`(node, pid)` and per-`(buyer, supplier, pid)` records described above.
- [ ] The old `node_orders` aggregate is derived from the purchase rows and existing readers are unaffected.
- [ ] The builder returns a frame with the documented column contract and `(node, pid, tick)` row grain.
- [ ] On a small hand-checkable scenario, `sales` / `demand` / `price` / `stockout` are correct; `stockout` reflects decision-time on-hand, not the closing snapshot.
- [ ] Per-`(buyer, supplier, pid)` `cash_paid` equals `qty_filled × supplier list_price`.
- [ ] A saved run reloads to the same flow frame as the fresh run (extend `test_saved_run_roundtrip.py`).

## Blocked by

- None — can start immediately. Independent of charging (issue 03); both read the same run.
