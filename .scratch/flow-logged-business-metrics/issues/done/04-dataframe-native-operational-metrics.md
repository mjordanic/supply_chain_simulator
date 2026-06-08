# 04 — DataFrame-native operational metrics (RunSlice retired) + notebook scorecard

Status: done

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019)

## What to build

Make business metrics a first-class property of any run by deriving the operational KPIs from the
flow frame (issue 02), and surface them in the notebooks. This is the end-to-end tracer bullet:
run → flow frame → KPIs → scorecard.

- `metrics.py` becomes DataFrame-native: KPI functions consume the tidy per-`(node, pid, tick)` flow
  frame, so a `node_id` column yields per-node *and* system-wide KPIs via `groupby` (`RunSlice` was
  structurally single-node).
- Add a `business_metrics(...)`-style entry point returning per-node and system-wide KPI rows for the
  operational KPIs: `service_level`, `stockout_rate` (using the decision-time `stockout` boolean),
  `inventory_turnover`, `mean_price_pct_of_msrp`.
- Remove `RunSlice` and its list-of-lists representation. `metrics.py` carries no run-log-schema
  knowledge — that lives in the builder (issue 02).
- The rewrite is **value-preserving for the operational KPIs** — these are the tuner/RL objective and
  must not shift; only their input representation changes.
- Notebooks 02 and 02a gain a per-node business-metrics table (operational KPIs), so an analyst can
  score a run they just ran (02) or one loaded from disk (02a) without re-running.

Profit decomposition is deliberately deferred to issue 05 (it needs real charging, issue 03).

## Acceptance criteria

- [ ] Operational KPI values equal the current `RunSlice`-era values for the same scenario (port the existing metric tests to the new input).
- [ ] KPIs compute per selling node and as a system-wide roll-up; the roll-up aggregates correctly.
- [ ] `stockout_rate` uses the decision-time `stockout` boolean.
- [ ] `RunSlice` is removed and `metrics.py` consumes only the DataFrame.
- [ ] Notebooks 02 and 02a render a per-node operational business-metrics table.

## Blocked by

- Issue 02 (flow log + DataFrame builder).
