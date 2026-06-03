# allocation.execute_buy full implementation + clamp paths + unit tests

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Flesh out `allocation.execute_buy` to the full FCFS allocation primitive per ADR 0012. This slice owns the clamp/payment/delivery contract end-to-end; it does not yet wire the buyer shuffle (issue 8) or the multi-supplier routing in policies (issue 9).

`execute_buy(buyer, supplier, pid, qty_requested, table, cash_ledger) -> AllocationResult`:
1. Reject when `qty_requested < min(supplier-imposed min_order, buyer-side policy min_order)` per ADR 0012's two-layer rule
2. Clamp by live `available_qty`, by `buyer.cash / list_price` (allocation-time payment per ADR 0013), by remaining buyer capacity
3. `table.commit(...)`, debit buyer cash, credit supplier cash, schedule delivery callback at `current_tick + lead_time(supplier, buyer, pid)`
4. Return `AllocationResult(qty_filled, qty_rejected, cash_paid)`

This is a deep-module slice — tested in isolation with mocked buyer/supplier/table fixtures. The clamp logic and the cash-conservation side effect are where the regressions hide.

After this slice, the engine still runs only on single-supplier topologies (no shuffle), but the allocation primitive can be exercised across all clamp paths through unit tests, ready to feed the contention scenarios in issue 9.

## Acceptance criteria

- [ ] `src/sim/allocation.py` exports `execute_buy`, `AllocationResult`
- [ ] `AllocationResult` carries `qty_filled`, `qty_rejected`, `cash_paid`
- [ ] Inventory-limited clamp path correct
- [ ] Cash-limited clamp path correct (`buyer.cash / list_price` at allocation time)
- [ ] Capacity-limited clamp path correct (remaining buyer capacity)
- [ ] Two-layer min-order rejection: supplier-imposed AND buyer-side policy min — when `qty_requested < min(supplier_min, buyer_min)`
- [ ] Payment debits buyer and credits supplier by `qty_filled × list_price` exactly
- [ ] Delivery callback scheduled at `current_tick + lead_time(supplier, buyer, pid)`
- [ ] `tests/sim/test_allocation_fcfs.py` covers every clamp path (inventory-limited, cash-limited, capacity-limited, min-order-rejected-by-either-layer) on a pure unit fixture
- [ ] `uv run pytest tests/sim -x` is green

## Blocked by

- `.scratch/multi-echelon/issues/05-graph-simulation-runner-chain-engine.md`
