# Cash flow conservation across nodes

Status: Proposed

The single-store simulator had implicit cash flow: `Store.balance` tracked a retailer's profit-and-loss, but there was no upstream actor receiving payment. Orders were "anonymous lambdas" — inventory materialised at `unit_cost` without a supplier balance ever changing. This made it impossible to model supplier-side economics or reason about system-wide cash conservation.

**Decision — Model explicit cash transfers between all nodes, with demand-sinks as the only source of new cash and factories as zero-margin cash sinks.**

**Rules:**

1. **Demand-sinks create cash.** `DemandSinkNode.cash += income_rate` each tick. `income_rate` is the single knob for consumer purchasing power. Unspent cash accumulates across ticks.

2. **Payment at allocation time.** When `execute_buy` is called (ADR 0012), cash transfers immediately: `buyer.cash -= qty_filled × list_price` and `seller.cash += qty_filled × list_price`. Payment happens at allocation time, not at delivery time. This ensures a buyer cannot claim more inventory than it can pay for (the cash-clamp in `execute_buy`).

3. **Factories are zero-margin.** `FactoryNode.list_price == FactoryNode.unit_cost` by construction. The factory's cash balance is therefore a no-op bookkeeping field: it receives `qty_filled × unit_cost` from the buyer and "spends" it on manufacturing at the same rate. Factory margin is architecturally zero; if production economics become important in a future scenario, a new ADR is needed.

4. **Holding cost and order fee go to the void.** `IntermediateNode` holding cost and fixed order fee are charged against the node's balance but credited to no other node. They are not redistributed. This preserves the accounting convention from the single-store era and keeps cash conservation a per-trade property rather than a strict system-level invariant.

5. **Cash conservation identity.** Within the trade graph: Σ demand-sink cash created == Σ balances at factories + intermediates + sinks + cumulative (holding cost + order fee) to void + inventory value in transit. This identity holds tick-by-tick as a system invariant and is asserted by the graph-runner integration tests.

**Why payment at allocation vs. at delivery?** Payment at delivery would allow a buyer to accumulate more obligations than its cash balance can support (by placing orders on credit). Payment at allocation ties purchasing power to cash on hand, giving supplier-side policies a meaningful signal about buyer creditworthiness. It also aligns with the `execute_buy` atomic semantics: a single call handles both the inventory commit and the cash transfer, eliminating any window where inventory is reserved but not yet paid for.

**Why `income_rate` on the sink rather than exogenous revenue from demand?** The revenue-from-sales model (a sink earns money by selling to an external market) requires a downstream price and a demand model at the sink level. `income_rate` is simpler and sufficient for the current research questions: consumer purchasing power is a first-class scenario lever without requiring the sink to participate in two-sided markets. Revenue-from-sales can be added as a `DemandSinkNode` subclass if needed.

## Cross-references

- ADR 0011 — Multi-echelon graph (the node types whose cash fields are governed by these rules)
- ADR 0012 — Central table + FCFS allocation (the `execute_buy` call where buyer-/seller+ cash transfer is executed)
- ADR 0014 — Tick phasing (income_rate is credited to sinks during `consume_demand_sinks`, the final phase of each tick)
