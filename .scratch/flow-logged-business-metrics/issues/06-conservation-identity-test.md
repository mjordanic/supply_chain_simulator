# 06 — Conservation identity test (ADR 0013 Rule 5)

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019, implements ADR 0013 Rule 5)

## What to build

Prove the new charges leak nowhere except the void. ADR 0013 Rule 5 states a system cash-conservation
identity and claims it "is asserted by the graph-runner integration tests" — but no test asserts it,
and it held only trivially while the void terms were zero (issue 03 makes them non-zero).

- A pure helper computing the Rule 5 terms from a run log + scenario: Σ demand-sink cash created;
  Σ node balances; cumulative (holding cost + order fee) to void; in-transit (outstanding) inventory
  value. The holding/fee void terms are derived from the logged flows + the node rates (consistent
  with issues 03/05).
- A graph-runner integration test asserting the identity — Σ sink cash created == Σ node balances +
  cumulative (holding + fee) to void + in-transit inventory value — at end-of-run (and/or tick by
  tick), with the void terms non-zero.

## Acceptance criteria

- [ ] A pure helper returns each Rule 5 term from `(run log, scenario)`.
- [ ] An integration test over a graph run asserts the conservation identity holds.
- [ ] The void terms (cumulative holding + order fee) are non-zero in the asserted scenario.
- [ ] The derived void totals match the amounts the engine charged in issue 03.

## Blocked by

- Issue 02 (flow log — source of derived fee/holding terms), issue 03 (engine charging — non-zero void terms).
