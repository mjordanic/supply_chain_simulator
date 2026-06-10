# PRD: Flow-logged business metrics & engine cost charging

Status: ready-for-agent

Governing ADR: [0019](../../docs/adr/0019-charge-holding-order-fee-and-flow-logged-metrics.md) (amends ADR 0013 Rules 4–5). Respects ADR 0012 (`execute_buy`/`cash_paid`), ADR 0014 (tick phasing), ADR 0016 (allocation RNG), ADR 0018 (lateral links).

## Problem Statement

After running a simulation, I can inspect *state* (cash, inventory, pending, orders, rejections) in notebooks 02 and 02a, but I cannot see the **business metrics** — service level, stockout rate, inventory turnover, price realization, and a profit decomposition. Those KPIs exist (`src/sim/metrics.py`) but can only be produced by the tuning/RL stack, which builds the metric input (`RunSlice`) with bespoke, monkey-patching instrumentation. A plain `Runner.run()` produces no flow data, so a saved run cannot be scored.

Worse, the economics the KPIs assume are not real: the base engine never charges holding cost or order fee, even though ADR 0013 (Rules 4–5) says an `IntermediateNode` should. So "profit" today is a tuning-only proxy that does not match the simulation's actual cash flow, and there is no single trustworthy profit number.

## Solution

Make business metrics a first-class property of *any* run, derivable from its saved artifacts, and make the simulation's economics real so there is one honest profit number.

1. The engine charges holding cost and order fee against each `IntermediateNode` (implementing ADR 0013 Rules 4–5), so realized profit (change in equity) already reflects them.
2. The `Runner` logs the minimal per-tick **flows** (sales, demand, price, stockout, realized purchase cost) that state snapshots cannot recover.
3. `metrics.py` derives KPIs from a tidy per-`(node, pid, tick)` DataFrame built from the run log — so notebooks 02 and 02a (and tuning, and RL) all consume the same path. The single-node `RunSlice` is retired.
4. The `metrics.py` profit decomposition becomes a faithful breakdown of realized cash flow (revenue − real order cost − holding − order fees), reconciling to the cent with Δequity, including across lateral links.

## User Stories

1. As an analyst, I want to see service level, stockout rate, inventory turnover, and mean price-vs-MSRP for a run, so that I can judge operational performance, not just inventory levels.
2. As an analyst, I want a profit decomposition (revenue, order cost, holding cost, order fees, net profit) for a run, so that I understand *why* a policy made or lost money.
3. As an analyst, I want the profit decomposition to reconcile with the change in equity, so that I trust the breakdown is the real cash flow and not an unrelated proxy.
4. As an analyst, I want these business metrics in notebook 02 right after running a simulation, so that I can score a policy I just ran.
5. As an analyst, I want the same business metrics in notebook 02a for a saved run loaded from disk, so that I can score a run produced earlier from the terminal without re-running it.
6. As an analyst, I want per-node KPIs across a multi-echelon graph, so that I can compare the performance of each selling node, not just one.
7. As an analyst, I want a system-wide KPI roll-up, so that I have a single scorecard for the whole supply chain.
8. As an analyst, I want the economic assumptions (holding rate, order fee) recorded with the run, so that a saved run is self-describing and the numbers are reproducible.
9. As a modeller, I want holding cost charged on each intermediate's closing inventory every tick, so that carrying stock has a real cost in the simulation.
10. As a modeller, I want a fixed order fee charged once per purchase order to each supplier, so that placing frequent or fragmented orders has a real cost.
11. As a modeller, I want ordering from two suppliers in one tick to incur two fees, and a multi-SKU order to one supplier to incur one fee, so that the fee reflects purchase orders, not order lines.
12. As a modeller, I want order cost measured as the actual cash paid to each specific supplier, so that lateral-link margins (supplier `list_price` > `unit_cost`) are reflected accurately.
13. As a modeller, I want `holding_rate` and `order_fee` to be per-`IntermediateNode` fields in `setup.yaml`, so that different nodes can carry different economics and the values travel with the scenario.
14. As a policy author, I want the economic parameters to live on the node where the policy can observe them in future, so that policies can eventually reason about their own holding and ordering costs.
15. As a maintainer, I want the cash-conservation identity (ADR 0013 Rule 5) asserted by an integration test with non-zero void terms, so that the new charges provably leak nowhere except the void.
16. As a maintainer, I want the operational KPIs (service level, stockout, turnover, price%) to be value-preserving through the refactor, so that the tuner/RL objective does not silently shift.
17. As a maintainer, I want the profit/cash/equity regression fixtures re-baselined intentionally, so that the deliberate behavior change is captured, not hidden.
18. As a tuning user, I want the tuner to consume the shared run-log-derived metrics, so that the bespoke `_TrackingDemandSinkNode`/`_record_active_subset` collector is deleted.
19. As an RL user, I want evaluation to consume the same shared metrics path, so that the two inline `RunSlice` collectors in `rl/eval.py` are deleted.
20. As an analyst, I want the previously-null `sales`/`demand`/`price`/`revenue` columns in graph-mode `timeseries.parquet` populated from the flow log, so that the exported table is complete.
21. As an analyst, I want stockout measured at decision time (could the node fill demand when asked), so that a same-tick replenishment does not mask a stockout.
22. As an analyst, I want holding cost measured on closing inventory, so that it matches the standard carrying-cost accounting basis.
23. As an analyst, I want only raw flows logged (not derived KPIs), so that I can re-run the economics under different rate assumptions in analysis.
24. As a notebook user, I want a per-node business-metrics table and a profit-decomposition chart in 02 and 02a, so that the scorecard is visual and immediately legible.
25. As a maintainer, I want a single deep `build_*` function that turns a run log into the flow DataFrame, so that every consumer shares one tested code path.
26. As a maintainer, I want `metrics.py` to be DataFrame-native and free of run-log-schema knowledge, so that the metric math stays pure and the schema-awareness lives in the inspect layer.

## Implementation Decisions

### Module breakdown (confirmed with developer)

- **Engine — cost charging.** In the per-tick walk and final tick phase: charge `IntermediateNode` holding cost on closing inventory, and the order fee per `(node, supplier)` purchase order. Intermediates only (factories are zero-margin per ADR 0013 Rule 3; sinks hold no inventory).
- **Engine — flow logging.** Extend the per-tick snapshot with the two flow record types (below). The current-tick sales already computed transiently for the un-lagged demand signal (ADR 0018) are persisted rather than discarded.
- **`IntermediateNode` economic params.** Add `holding_rate` and `order_fee` fields; parse/serialize them through the setup IO layer (`setup.yaml`), so they appear in the run's `config/` snapshot.
- **Flow DataFrame builder (deep module, in the inspect layer).** A pure function: run log → tidy long-form DataFrame, one row per `(node, pid, tick)` for the per-product flows, plus access to the per-`(buyer, supplier, pid, tick)` purchase rows. Owns all run-log-schema knowledge.
- **DataFrame-native metrics (deep module, `metrics.py`).** KPI functions consume the flow DataFrame; add a `business_metrics(...)`-style entry point that returns per-node and system-wide KPI rows (`groupby` on `node_id`). `RunSlice` and its list-of-lists representation are removed.
- **Conservation identity.** A pure helper computing the ADR 0013 Rule 5 terms from a run log + scenario, plus a graph-runner integration test asserting the identity with non-zero void terms.
- **Consumer refactors.** `src/tuning/rollout.py` and `src/rl/eval.py` delete their bespoke collectors and call the builder + metrics.
- **DataExporter.** Populate the previously-null graph-mode flow columns from the flow log.
- **Notebooks 02 and 02a.** Add a per-node business-metrics table, a profit-decomposition view, and the reconciliation against Δequity.

### Per-tick flow log schema (the contract everything reads)

- Per `(node, pid)`: `sales` (units the node sold as a supplier this tick), `demand` (units requested of it; for a `DemandSinkNode`, the exogenous `demand_target`), `price` (the node's `list_price` at decision time), `stockout` (boolean: decision-time on-hand == 0).
- Per `(buyer, supplier, pid)`: `qty_filled`, `cash_paid` (= `qty_filled × supplier list_price`, from `execute_buy`). This supersedes the current `node_orders: {pid: qty}` aggregate, which is derived from these rows.
- The existing end-of-tick `node_inventory`/`node_cash`/`node_pending` snapshots are unchanged (the inspect/equity/parquet readers depend on them).

### Charging mechanics (ADR 0019)

- **Holding cost:** each tick, `cash -= Σ_pid closing_on_hand[pid] × holding_rate × unit_cost[pid]` on the end-of-tick inventory, valued at catalog `unit_cost`, charged in the final tick phase (with `consume_demand_sinks`).
- **Order fee:** `order_fee` once per `(node, supplier)` PO per tick — a multi-SKU PO to one supplier is one fee; two suppliers is two fees.
- **Order cost (decomposition):** real `cash_paid` summed across purchases, so the decomposition reconciles with realized cash exactly, including lateral links.
- Holding cost and order-fee *totals* are **derived in analysis** from the logged flows + the node's rates — they are not logged per tick.

### Metrics semantics

- Operational KPIs (`service_level`, `stockout_rate`, `inventory_turnover`, `mean_price_pct_of_msrp`) keep their current definitions and **values**; only their input representation changes (DataFrame, not `RunSlice`).
- `stockout_rate` uses the decision-time `stockout` boolean; `holding_cost` in the decomposition uses closing inventory (aligned so the decomposition is an exact partition of realized cash).
- KPIs are computable per selling node and as a system-wide roll-up via `groupby` on `node_id`.

### Parameter ownership

- `holding_rate` / `order_fee` move out of `src/tuning/config.py` ownership onto the `IntermediateNode` (sourced from `setup.yaml`). Tuning/RL read them from the scenario. A single canonical default is used where a scenario does not specify them.

## Testing Decisions

Good tests assert **external behavior** — the numbers and the column contract a consumer relies on — not internal structure. Reuse existing prior art: `tests/sim/test_inspect.py` (pure run-log → DataFrame assertions), `tests/sim/test_metrics.py` and `tests/rl/test_metrics.py` (KPI value assertions), `tests/sim/test_regression_snapshot.py` and `tests/sim/test_data_exporter_parquet_parity.py` (fixture re-baselining), and `tests/sim/test_saved_run_roundtrip.py` (the 02a round-trip).

Modules to test (developer-selected — all four):

1. **Flow DataFrame builder.** Assert the column contract and row grain; that `sales`/`demand`/`price`/`stockout` are correct against a small hand-checkable scenario; that per-`(buyer, supplier, pid)` `cash_paid` rows are present and equal `qty_filled × supplier list_price`. Round-trip: a saved run reloads to the same flow frame as the fresh run (extend `test_saved_run_roundtrip.py`).
2. **DataFrame-native metrics.** Value-preserving regression: the operational KPI numbers equal the current `RunSlice`-era values for the same scenario (port the existing metric tests to the new input). The profit decomposition reconciles to Δequity (net profit == change in equity over the episode, within tolerance), including a lateral-link scenario where `cash_paid`-based order cost differs from a `unit_cost`-based one. Per-node and system roll-ups aggregate correctly.
3. **Conservation identity (Rule 5).** Integration test over a graph run: Σ demand-sink cash created == Σ node balances + cumulative (holding cost + order fee) to void + in-transit inventory value, asserted tick-by-tick or at end-of-run, with the void terms non-zero.
4. **Engine charging correctness.** Holding cost equals `closing_on_hand × holding_rate × unit_cost` for a known inventory trajectory; the order fee is charged once per `(node, supplier)` PO (one fee for a two-SKU single-supplier order; two fees for two suppliers); order cost equals real cash paid across a lateral link.

Re-baselining: `test_regression_snapshot.py` and the parquet fixtures are regenerated as part of the change, with the new values reviewed as the intended behavior.

## Out of Scope

- **Factory finished-goods holding cost.** ADR 0013 Rule 4 scopes holding to `IntermediateNode`; factories hold `inventory._total` but are zero-margin bookkeeping. A future ADR amendment if ever wanted.
- **Policies actually reading the economic parameters.** Putting `holding_rate`/`order_fee` on the node *enables* this; wiring them into observations/policy logic is a separate effort.
- **Changing the operational KPI definitions** or adding new KPIs beyond the existing five-plus-decomposition surface.
- **Revenue-from-sales at the sink** (ADR 0013 keeps `income_rate` as the sink cash source).
- **Redistributing holding cost / order fee** to another node (ADR 0013 fixes them as "to the void").

## Further Notes

- This is fundamentally *closing a conformance gap*: ADR 0013 already mandates the charges and the conservation identity; the engine just never implemented them. The mechanics specifics (per-PO fee, real order cost, closing-inventory holding) are in ADR 0019.
- Because charging holding/fee changes every run's cash/equity, expect the profit/cash family of fixtures to change; the operational KPIs must not.
- The flow log is intentionally minimal (flows only). Everything economic is derived in analysis so the rates stay swappable and there is one source of truth per number.
- One implementation dependency to verify early: whether `rl/eval.py` runs through `Runner.run()` (and thus gets a standard run log to build the frame from) or uses a bespoke stepping loop that must be routed through the same flow logging.
