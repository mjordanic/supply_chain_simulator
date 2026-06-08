# Charge holding cost and order fee; flow-logged business metrics

Status: Accepted — amends ADR 0013 (implements Rules 4–5 and fixes their mechanics). Implementation pending (until merged, the engine charges neither cost and the run log carries no flow data).

Two gaps share one root cause.

1. **ADR 0013 Rules 4–5 are unimplemented.** Rule 4 says `IntermediateNode` holding cost and a fixed order fee are charged against the node's balance and "go to the void" (credited to no node); Rule 5 states a system conservation identity that includes those void terms and claims it "is asserted by the graph-runner integration tests." In fact `src/sim/node.py` charges neither, no test asserts the identity, and it holds only trivially because both terms are 0.

2. **Business KPIs can only be produced by the tuning/RL stack.** `src/sim/metrics.py` is pure math, but the only things that build its `RunSlice` input are `src/tuning/rollout.py` (a `_TrackingDemandSinkNode` monkey-patch + `_record_active_subset`) and `src/rl/eval.py` (two inline collectors). A plain `Runner.run()` — and therefore notebooks 02 / 02a — cannot score a run on service level, turnover, stockout, or profit.

Both stem from the same omission: the `Runner` emits no per-tick **flow** data (sales, demand, price, realized purchase cost) and does not charge the costs the economics assume. State snapshots alone cannot recover flows — inventory on tick *t* reflects both sales and replenishment arrivals, so units sold are not derivable after the fact.

**Decision — Charge holding cost and order fee in the engine with concrete mechanics, log per-tick flows in the run log, and make `metrics.py` derive KPIs from the run log (retiring `RunSlice`).**

**Rules:**

1. **Holding cost.** Each tick, `IntermediateNode.cash -= Σ_pid closing_on_hand[pid] × holding_rate × unit_cost[pid]`, on the end-of-tick inventory snapshot, valued at catalog `unit_cost`, charged in the final tick phase (with `consume_demand_sinks`, ADR 0014). Intermediates only — factories are zero-margin bookkeeping (ADR 0013 Rule 3) and sinks hold no inventory.

2. **Order fee.** A fixed `order_fee` is charged once per `(node, supplier)` purchase order per tick: a multi-SKU PO to one supplier is one fee; ordering from two suppliers in a tick is two fees. Charged to the ordering `IntermediateNode` only.

3. **Order cost is the real cash paid.** The purchase-cost component of the profit decomposition is the actual cash transferred — `execute_buy.cash_paid` = `qty_filled × supplier list_price` — not catalog `unit_cost`. This reconciles the decomposition to the cent with the buyer's real cash outflow, including across intermediate→intermediate lateral links (ADR 0018), where `list_price > unit_cost`.

4. **Economic parameters live on the node.** `holding_rate` and `order_fee` are per-`IntermediateNode` fields sourced from `setup.yaml` (carried into the run's `config/` snapshot, so 02a reads them). Placing them on the node — not in `src/tuning/config.py` — makes them observable to policies in future and removes the tuning layer's ownership of them.

5. **Per-tick flow log.** Each tick snapshot gains two record types: (a) per `(node, pid)` — `sales` (units sold as supplier), `demand` (units requested of it; for a sink, the exogenous `demand_target`), `price` (`list_price` at decision time), and a `stockout` boolean (decision-time on-hand == 0); (b) per `(buyer, supplier, pid)` — `qty_filled` and `cash_paid`. Record (b) supersedes today's `node_orders: {pid: qty}` (which dropped the supplier and the price); the old aggregate is derived from it. Only raw flows are logged — holding cost and order-fee totals are *derived* in analysis from the logged flows + the node's `holding_rate`/`order_fee`, keeping the log minimal and the rates swappable.

6. **`metrics.py` becomes DataFrame-native; `RunSlice` is retired.** KPI functions consume a tidy per-`(node, pid, tick)` flow DataFrame, so a `node_id` column yields per-node *and* system-wide KPIs via `groupby` (`RunSlice` was structurally single-node). `src/tuning/rollout.py` and `src/rl/eval.py` delete their bespoke collectors and build the frame from the run log. The rewrite is **value-preserving for the operational KPIs** (service_level, stockout_rate, inventory_turnover, mean_price_pct_of_msrp — these are the tuner/RL objective and must not shift); the **profit family is re-baselined on purpose** (charging holding/fee and using real order cost changes the numbers).

7. **Conservation identity becomes a real test.** The Rule 5 identity — Σ demand-sink cash created == Σ node balances + cumulative (holding cost + order fee) to void + inventory value in transit — is asserted by a graph-runner integration test, now with non-zero void terms.

**Why a fee per `(node, supplier)` PO rather than per tick or per line?** A fixed ordering cost models the overhead of raising one purchase order. In a multi-echelon graph a node may source from several suppliers in a tick, and each PO is a distinct fixed cost — so per-supplier is the faithful unit. Bundling multiple SKUs to the same supplier into one PO (one fee) matches the real-world notion and avoids penalising assortment breadth.

**Why real cash paid for order cost rather than catalog `unit_cost`?** Catalog cost only equals the price paid when the supplier is zero-margin (factories, Rule 3). Lateral links (ADR 0018) sell at a margin, so a `unit_cost`-based decomposition would understate purchase spend and fail to reconcile with realized cash. Using `cash_paid` makes the decomposition an exact partition of the node's real cash flow.

**Why closing inventory for holding, but decision-time for stockout?** Holding cost is the carrying cost of what you *end the period* holding — the standard accounting basis, and it reuses the snapshot already logged. Stockout is a *decision-time* condition (could the node fill demand when asked?), so it is captured as a boolean during the walk rather than inferred from the closing snapshot, which a same-tick replenishment would mask.

**Why retire `RunSlice` rather than keep it and add a builder?** The list-of-lists existed to "avoid a pandas dependency," but `aggregate_episode` runs once per episode (not in any hot loop) and pandas is already a core dependency. Its single-node `tick × sku` shape cannot express per-node KPIs across a graph, which is the direction of the project. A tidy DataFrame subsumes it and unifies two duplicated collectors into one.

**Why not log the derived costs per tick too?** Holding cost, order-fee totals, revenue, and net profit are all functions of the raw flows plus the node's rates. Logging only the irreducible flows keeps the run log small, lets analysis re-run the economics under different rate assumptions, and keeps a single source of truth for each number.

## Consequences

- Every run's cash/equity trajectory changes. `tests/sim/test_regression_snapshot.py` and the `products`/`stores`/`timeseries` parquet fixtures are re-baselined intentionally.
- Realized profit (Δequity) and the `metrics.py` decomposition reconcile by construction; the earlier "two profits" distinction collapses to one number with a faithful breakdown.
- The previously-null `sales`/`demand`/`price`/`revenue`/… columns in graph-mode `timeseries.parquet` (`data_exporter.py`) are populated from the flow log.

## Cross-references

- ADR 0011 — Multi-echelon graph (the node types charged here)
- ADR 0012 — Central table + FCFS allocation (`execute_buy`, source of `cash_paid`)
- ADR 0013 — Cash flow conservation (this ADR implements its Rules 4–5 and fixes their mechanics)
- ADR 0014 — Tick phasing (holding cost is charged in the final phase)
- ADR 0018 — Lateral links (why order cost must use real `cash_paid`, not `unit_cost`)
