# 03 — Engine charges holding cost + order fee

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019, implements ADR 0013 Rules 4–5)

## What to build

Make the simulation's economics real: charge the holding cost and order fee that ADR 0013 mandates
but the engine never implemented, so realized profit (change in equity) already reflects them. Uses
the per-node parameters from issue 01.

- **Holding cost:** each tick, `IntermediateNode.cash -= Σ_pid closing_on_hand[pid] × holding_rate ×
  unit_cost[pid]` — on the end-of-tick inventory snapshot, valued at catalog `unit_cost`, charged in
  the final tick phase (with `consume_demand_sinks`, ADR 0014). Intermediates only (factories are
  zero-margin bookkeeping per ADR 0013 Rule 3; sinks hold no inventory).
- **Order fee:** `order_fee` charged once per `(node, supplier)` purchase order per tick — a
  multi-SKU PO to one supplier is one fee; ordering from two suppliers in a tick is two fees.
  Charged to the ordering `IntermediateNode` only.
- Both charges "go to the void" (credited to no other node) — the conservation identity in issue 06
  accounts for them.
- Re-baseline the cash/equity regression and parquet fixtures intentionally: every run's cash/equity
  trajectory changes. Regenerate `test_regression_snapshot.py` and the `products`/`stores`/
  `timeseries` fixtures and treat the new values as the intended behavior.

## Acceptance criteria

- [ ] Holding cost charged equals `closing_on_hand × holding_rate × unit_cost` summed over products, for a known inventory trajectory.
- [ ] Holding cost is charged in the final tick phase and only against `IntermediateNode`s.
- [ ] A two-SKU order to one supplier incurs exactly one `order_fee`; ordering from two suppliers in a tick incurs two fees.
- [ ] The order fee is charged only to the ordering intermediate.
- [ ] `test_regression_snapshot.py` and the parquet fixtures are regenerated and pass with the new, reviewed-as-intended values.
- [ ] Operational KPIs are unaffected by charging (no change to service/stockout/turnover/price behavior).

## Blocked by

- Issue 01 (economic parameters on `IntermediateNode`).
