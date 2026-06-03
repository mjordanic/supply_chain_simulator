# 04 — `metrics` pure-function module

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

A pure-function module `src/rl/metrics.py` that computes business KPIs from a `run_slice` populated by the env's per-tick `info` dicts. No parquet I/O, no DataFrame coupling — just in-memory dataclass aggregation.

Public surface:

- `RunSlice` dataclass (or equivalent) — fields covering per-active-SKU-per-tick traces for sales, demand, inventory at decision time, price, MSRP, revenue, holding cost, order cost, order fee.
- `service_level(run_slice) → float` — `sum(sales) / max(1, sum(demand))` across active SKUs and ticks.
- `stockout_rate(run_slice) → float` — fraction of `(active_SKU, tick)` pairs where inventory was 0 at decision time.
- `mean_price_pct_of_msrp(run_slice) → float` — mean of `price / MSRP` across active-SKU ticks.
- `inventory_turnover(run_slice) → float` — `sum(sales) / max(1, mean(inventory))`.
- `profit_decomposition(run_slice) → dict[str, float]` — totals for revenue, holding cost, order cost, order fees, net profit. Net profit = revenue − (holding + order + fees).
- `aggregate_episode(run_slice) → dict[str, float]` — bundles all of the above for one episode.

All functions are pure and handle empty/zero-demand slices without divide-by-zero.

## Acceptance criteria

- [ ] `src/rl/metrics.py` exports `RunSlice` and the six aggregator functions
- [ ] All functions deterministic and pure — no I/O, no global state
- [ ] `tests/rl/test_metrics.py` covers:
  - [ ] Hand-crafted slice with known sales/demand → expected `service_level`
  - [ ] Hand-crafted slice with inventory hitting zero on K of T ticks → expected `stockout_rate`
  - [ ] `profit_decomposition` net total adds back to the balance delta computed from the same slice
  - [ ] Empty / zero-demand slices return finite values (no `nan` or division errors)
  - [ ] `aggregate_episode` keys match the documented schema and values agree with individual function calls
- [ ] Tests run under `uv run pytest tests/rl/test_metrics.py` and pass

## Blocked by

None — can start immediately.
