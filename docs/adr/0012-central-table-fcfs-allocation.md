# Central table + sequential FCFS allocation contract

Status: Accepted

Within one tick, multiple buyers at the same echelon level compete for limited upstream inventory. The allocation mechanism determines how that inventory is divided when demand exceeds supply.

Three mechanisms were evaluated: (a) **snapshot fair-share** — each buyer sees a tick-start snapshot and receives a proportional fill based on declared demand; (b) **sequential FCFS with buyer shuffle** — buyers are ordered randomly and execute one at a time, each seeing the live table updated by prior buyers; (c) **two-tick request/response** — buyers send an order in tick N, the supplier responds in tick N+1. The PRD grilling session (see "Further Notes") chose sequential FCFS.

**Decision — Use a globally visible `CentralTable` of live offers and a sequential FCFS allocation protocol, with buyers shuffled once per phase by `allocation_rng` (ADR 0016).**

**`CentralTable` semantics.** `CentralTable.rows` is a dict keyed by `(seller_id, pid)` mapping to `Offer(available_qty, list_price, min_order, fill_rate_recent)`. All sellers publish offers at the start of each tick via `publish(seller_id, pid, offer)`. During a phase, buyers call `commit(seller_id, pid, qty)` which decrements `available_qty` live and updates the rolling qty-weighted EMA on `fill_rate_recent` (EMA window = 10 ticks). `snapshot_for_buyer(pid)` returns the post-commit state — a buyer that calls this after prior allocations have landed sees an accurate picture of residual supply, not a stale tick-start snapshot. Global visibility: every node sees every offer; topology-restricted views (e.g. "you only see your named suppliers") are explicitly out of scope.

**`execute_buy` single call.** `allocation.execute_buy(buyer, supplier, pid, qty_requested, table, cash_ledger, *, current_tick, lead_time) -> AllocationResult` is the atomic allocation primitive. It:

1. Rejects when `qty_requested < supplier-imposed min_order` (server-side threshold on the `Offer`) or `< buyer-side policy min_order` — the **two-layer min-order rule**. Buyer-side minimums are enforced by the buyer's own decision logic; supplier-side minimums are enforced here.
2. Clamps by live `available_qty` (never over-allocates what the seller has).
3. Clamps by `buyer.cash / list_price` (buyer pays at allocation time per ADR 0013; can't spend what it doesn't have).
4. Clamps by remaining buyer capacity (prevents overfilling).
5. Calls `table.commit(seller_id, pid, qty_filled)`, debits buyer cash, credits seller cash, and schedules a delivery callback at `current_tick + lead_time`.
6. Returns `AllocationResult(qty_filled, qty_rejected, cash_paid)`.

**Why not snapshot fair-share?** Fair-share requires a global solver pass after all buyers have declared demand, which couples decisions that should be made greedily. It also hides supply depletion from later buyers within a phase, making it impossible for a buyer to observe "this supplier is nearly out" and redirect to a backup. Sequential FCFS with a shuffled order is strictly more informative for buyers that care about supplier reliability signal.

**Why not two-tick request/response?** The added tick delay makes the feedback loop slower for all policies and makes the implementation significantly more complex (mailboxes, response buffers, partial-fill callbacks across tick boundaries). The PRD explicitly out-of-scopes this as "Model 3."

**Fill-rate EMA.** `fill_rate_recent` on each `Offer` is an exponential moving average (alpha = 2/(10+1) ≈ 0.18) of the per-commit fill fraction (qty_filled / available_qty at commit time). It serves as the "supplier reliability" signal that `IntermediatePolicy.decide(obs, central_table)` can read when choosing among suppliers.

## Cross-references

- ADR 0011 — Multi-echelon graph (the nodes whose inventory this table tracks)
- ADR 0013 — Cash flow conservation (payment at `execute_buy` time is the mechanism that enforces buyer-side cash clamping)
- ADR 0014 — Tick phasing (the phase cascade determines when `publish_offers` and each buyer shuffle happen relative to `produce` and `deliver`)
- ADR 0016 — RNG/CRN extension: allocation sub-seed (the `allocation_rng` that drives buyer shuffle order each phase)
