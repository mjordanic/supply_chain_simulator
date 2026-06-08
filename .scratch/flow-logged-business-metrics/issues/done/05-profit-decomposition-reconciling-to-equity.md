# 05 — Profit decomposition reconciling to Δequity + notebook profit view

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019)

## What to build

Give the analyst one honest profit number with a faithful breakdown that ties out to the change in
equity — so the earlier "two profits" distinction (a tuning-only proxy vs. real cash flow) collapses
to a single trustworthy figure.

- Profit decomposition = `revenue − order cost − holding cost − order fees`, where:
  - **order cost** is the real cash paid summed across purchases (`cash_paid` from the flow frame,
    issue 02) — *not* catalog `unit_cost` — so it reconciles to the cent including across
    intermediate→intermediate lateral links (ADR 0018) where `list_price > unit_cost`;
  - **holding cost** and **order-fee totals** are *derived* in analysis from the logged flows + the
    node's `holding_rate` / `order_fee` (issue 01), aligned with what the engine charged (issue 03).
- Net profit reconciles with Δequity (change in equity over the episode) within tolerance, by
  construction.
- Notebooks 02 and 02a gain the profit-decomposition view (revenue, order cost, holding, fees, net)
  and the reconciliation check against Δequity.

## Acceptance criteria

- [ ] Decomposition net profit equals Δequity over the episode within tolerance.
- [ ] Order cost uses real `cash_paid`; a lateral-link scenario shows it differs from a `unit_cost`-based figure and still reconciles.
- [ ] Holding and order-fee totals are derived (not logged per tick) and match the amounts the engine charged in issue 03.
- [ ] Notebooks 02 and 02a render the profit decomposition and the Δequity reconciliation.

## Blocked by

- Issue 02 (flow log + builder), issue 03 (real charging), issue 04 (DataFrame-native metrics path).
