# `src/sim/inspect.py` run-log deep module + tests

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

A small deep module, `src/sim/inspect.py` (aligns with ADR 0010 — `src/sim/` is the home for
rollout primitives), that converts an in-memory run-log into tidy per-tier DataFrames so the
inspection/showcase notebooks stay thin and the parsing is unit-tested once instead of duplicated
across three notebooks. Pure functions over the run-log — no plotting, no file I/O.

Authoritative run-log shape the parser consumes:

```
run_log = {
  n_steps: int,
  ticks: [{tick, node_cash: {nid: float},
                 node_inventory: {nid: {_total: int, <pid>: int}},
                 node_pending:   {nid: {...}},
                 node_orders:    {nid: {...}}}],
  global: {time, market_supply, market_demand, products, events},
}
```

Functions:
- `node_timeseries_df(run_log, scenario) -> DataFrame` — long form
  `(tick, node_id, node_type, region, level, cash, inventory_total, pending_total, orders_total)`,
  joining each tick's `node_*` dicts to node metadata from `scenario.nodes`.
- `global_timeseries_df(run_log) -> DataFrame` — `(tick, market_supply, market_demand, …)` from
  `run_log['global']`.
- `per_product_df(run_log, scenario, node_id) -> DataFrame` — per-`(tick, pid)` inventory/orders for
  one node, expanding the per-pid keys in `node_inventory` and **excluding** the synthetic `_total`
  key (which is reused as the tier total in `node_timeseries_df`).
- An equity helper (cash + inventory-at-cost + outstanding-at-cost), usable per node, valuing
  inventory against catalog `unit_cost`.

This slice is additive: no engine behaviour, policy, RL, tuner, or `DataExporter` code changes.

## Acceptance criteria

- [ ] `src/sim/inspect.py` exports `node_timeseries_df`, `global_timeseries_df`, `per_product_df`, and the equity helper
- [ ] `node_timeseries_df` returns one row per `(tick, node)` with correct `node_type`/`region`/`level` joins from the scenario, across **all** tiers (factory, warehouse/shop, sink)
- [ ] `per_product_df` returns one row per `(tick, pid)` for the given node and never emits a `_total` row
- [ ] The equity helper computes `cash + inventory-at-cost + outstanding-at-cost` valued against catalog `unit_cost`
- [ ] `tests/sim/test_inspect.py` asserts on external behaviour only (column sets, row cardinality, `node_type`/`level` joins, `_total`-key handling, equity arithmetic) using a tiny 3-node-chain run (few ticks) or a hand-written run-log dict — not private internals or exact float trajectories of a full run
- [ ] `uv run pytest tests/sim/test_inspect.py -x` is green; no existing test modified

## Blocked by

None — can start immediately
