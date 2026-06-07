# 05 — Rejection & lost-sale log

Status: done

## Parent

`.scratch/lateral-links-scheduling/PRD.md` (ADR 0018)

## What to build

Give the modeller a per-tick record of why orders went unfilled, so debugging unmet demand is no
longer guesswork. Always-on (no flag).

- `AllocationResult` gains `reason: str | None`, set at the **binding constraint** in `execute_buy`:
  `no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`,
  `insufficient_capacity` — including the case where a clamp drives the fill to zero.
- `Simulation` accumulates a per-tick `_tick_rejections: list[dict]` (reset each tick), surfaced in
  the run log as `ticks[i]["rejections"]` alongside `node_orders` / `node_pending`. Each entry:
  `{tick, buyer_id, supplier_id, pid, qty_requested, qty_filled, qty_rejected, reason}`.
- Sink-level `unmet_demand` entries (`supplier_id=None`,
  `qty_rejected = demand_target − total_filled`) record genuine lost sales — demand no feasible
  supplier could fill, which (after issue 04's fall-through) is the true lost-sale measure.

Out of scope: tracing routing-layer *skips* (proactive supplier skips that successfully route
elsewhere) — only actual rejections and genuine lost sales are logged. No toggle.

## Acceptance criteria

- [ ] `AllocationResult.reason` is correct for each clamp path (no offer, below min-order, stock, cash, capacity), including when a clamp drives the fill to zero.
- [ ] An engineered shortage produces a per-tick `rejections` stream with correct quantities and reasons, inspectable via the existing run-log tooling.
- [ ] `unmet_demand` entries (`supplier_id=None`) appear with the correct residual when no feasible supplier can fill a sink's demand.
- [ ] The log is always-on and the run remains bit-deterministic from the seed.

## Blocked by

- Issue 02 (rejections accumulate during the unified demand-pull walk)
- Issue 04 (min-order fall-through must run before a residual counts as genuine `unmet_demand`)
