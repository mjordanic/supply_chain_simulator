# 01 — Economic parameters on IntermediateNode

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019, amends ADR 0013 Rules 4–5)

## What to build

Give each `IntermediateNode` its own economic parameters so a scenario is self-describing and the
values travel with it through save/load. This is the foundation the engine charging (issue 03) and
the self-describing saved run (notebooks 02a) build on.

- Add `holding_rate` and `order_fee` fields to `IntermediateNode`.
- Parse and serialize them through the setup IO layer so they live in `setup.yaml` and survive a
  `Scenario` → disk → `Scenario` round-trip.
- They are carried verbatim into the run's `config/` snapshot, so a saved run reloads with the same
  economics (and notebook 02a can read them).
- A single canonical default is applied where a setup does not specify them. Place the parameters on
  the node — not in `src/tuning/config.py` — so policies can observe them in future and the tuning
  layer no longer owns them.
- Factories and demand sinks are unaffected (factories are zero-margin bookkeeping per ADR 0013
  Rule 3; sinks hold no inventory).

No charging behavior yet — this slice only adds and persists the fields. The engine does not read
them until issue 03, so existing run output is unchanged.

## Acceptance criteria

- [ ] `IntermediateNode` exposes `holding_rate` and `order_fee`.
- [ ] A `setup.yaml` carrying per-node `holding_rate`/`order_fee` round-trips through load → save → load with the values preserved per node.
- [ ] A setup that omits the fields loads with the canonical default applied.
- [ ] The values appear in a saved run's `config/` snapshot and reload identically (extends/keeps `test_saved_run_roundtrip.py` green).
- [ ] No change to any run's cash/equity output (charging is not wired yet).

## Blocked by

- None — can start immediately.
