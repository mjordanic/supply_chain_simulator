# GraphSimulation + GraphRunner + build_graph_world + Market.demand_multiplier + minimum single-supplier execute_buy

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Stand up a runnable graph engine that processes the tick cascade end-to-end on single-supplier-per-buyer topologies. No contention, no buyer shuffle, no allocation_rng yet — those land in issue 8. This slice's job is the phase cascade, the cash flow, the lead-time delivery scheduling, and the Market multiplier wiring.

`GraphSimulation.tick()` runs: `tick_world` (market multiplier + event_engine + item_registry) → `publish_offers` → for `p in 1..max_level`: per buyer at level `p` observe → decide → execute_buy per order line → `produce` (factories) → `deliver` (scheduled event_engine callbacks fire) → `consume_demand_sinks`.

`allocation.execute_buy` lands in minimal form: single-supplier path with payment debit/credit, central-table commit, delivery scheduling at `current_tick + lead_time(supplier, buyer, pid)`. Multi-supplier clamping logic and min-order rules are stubbed (single-supplier-no-contention won't exercise them) — full impl in issue 7.

`Market` gains `demand_multiplier(pid, region, tick) -> float` that bundles seasonality + regional shock + active disruption. `sample_demand` is kept for now (legacy `Store` engine still uses it) — deletion is in issue 11.

`build_graph_world(scenario, *, policy_overrides=None)` is the parallel entry point to today's `build_world`. Both coexist; selection driven by `Scenario.is_graph`.

The legacy `Store` engine continues to run unchanged — `main.py scenarios/example_homogeneous.py` is still green.

## Acceptance criteria

- [ ] `src/sim/runner.py` adds `GraphSimulation`, `GraphRunner`, `build_graph_world(scenario, *, policy_overrides=None)`
- [ ] `GraphSimulation.tick()` runs the cascade in the order: `tick_world` → `publish_offers` → phase cascade (1..max_level) → `produce` → `deliver` → `consume_demand_sinks`
- [ ] Orders placed at phase N arrive at `current_tick + lead_time`, not within the same tick (lead time still delays physical delivery)
- [ ] Cash transfers (buyer−, seller+) executed at allocation time
- [ ] `src/sim/allocation.py::execute_buy` implements the single-supplier path: payment, central-table commit, delivery scheduling
- [ ] `src/sim/market.py` adds `demand_multiplier(pid, region, tick) -> float`; `sample_demand` retained (deletion in issue 11)
- [ ] Existing tests stay green; `uv run python main.py scenarios/example_homogeneous.py` continues to work
- [ ] `uv run pytest tests/sim -x` is green

## Blocked by

- `.scratch/multi-echelon/issues/03-node-hierarchy-scenario-extension.md`
