# 03 — Introduce `Simulation` / `build_world` / `TickResult` in `src/sim/runner.py` + CRN determinism gate

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADRs: `docs/adr/0003-crn-demand-for-all-products.md` (CRN bit-identity contract), `docs/adr/0004-rl-training-env.md` (slot-permutation invariant)

## What to build

Split `src/sim/runner.py` so the per-tick state machine is callable independently of the verbose `run_log` collection. All four current rollout copies will eventually compose these primitives instead of re-implementing them.

This slice introduces the new sim surface and the CRN determinism gate, but does not yet migrate the consumer modules — the four rollout copies in tuning and RL keep their current bodies. The new public symbols become available; downstream migrations land in issues 06–08.

### New sim surface

- `build_world(scenario, *, policy_overrides: list[Policy] | None = None) -> Simulation` — module-level free function. Returns a `Simulation` (mutable bundle: `scenario`, `world_rng`, `item_registry`, `market`, `event_engine`, `stores`). When `policy_overrides` is supplied, `Store`s are constructed with the override policies in place of `StoreInstance.policy`. **Override wins when both are set.** This is the canonical attach pattern for spec-based callers (tuning, RL eval, RL env) whose specs carry `policy=None`. Scenario-authoring callers (main.py, hand-authored scenarios) leave `policy_overrides=None` and let `StoreInstance.policy` flow through.

- `Simulation.tick_world() -> list[WorldEvent]` — phase 1: world advances (`market.tick → event_engine.tick → item_registry.tick`).

- `Simulation.tick_decide_and_settle() -> TickResult` — phase 2: per-store `observe → decide → _dispatch_orders → _process_demand`. RL action-injection happens in the seam between phases: encode obs from post-world state, run the actor, decode, `RLPolicy.set_pending_action(...)`, then call this method so `store.decide` returns the pending action.

- `Simulation.tick() -> TickResult` — convenience composing both phases. Tuning, the rewritten `Runner.run()`, and `_run_baseline` use this. `_run_rl` and `RLEnv.step` use the two halves.

- `TickResult` — frozen dataclass `(actions: dict[int, dict[str, Any]], demand_traces: dict[int, dict[str, int]], active_events: list[WorldEvent])`.

### `Runner.run()` refactor

`Runner` keeps its public API (`Runner(scenario).run() -> dict`). Body shrinks to a composition of `build_world` + `Simulation.tick()` + the existing `_init_run_log` / `_log_state` log-collection helpers. DataExporter, `main.py`, notebooks, and existing tests see no API change.

The per-tick demand-sampling order from ADR 0003 (every catalog product, every tick, in `store.inventory` iteration order) must be preserved exactly in `Simulation.tick_decide_and_settle()`.

### CRN determinism gate (`tests/sim/test_runner_refactor.py`)

Two tests pin the CRN bit-identity contract that issues 06–08 must keep green.

1. **Snapshot test.** Before any code change in this slice, capture `Runner(canonical_small_scenario).run()` on the current `main` SHA and serialise the `run_log` to a golden parquet at `tests/sim/fixtures/runner_snapshot_pre.parquet`. Commit the fixture. After the refactor, regenerate and assert `pd.DataFrame.equals` on numeric columns and exact equality on integer / categorical columns. Catches Runner-output regressions caused by the factor.

2. **Cross-path equivalence.** After the refactor, drive the same scenario via `Runner(scenario).run()` and via `sim = build_world(scenario); for _ in range(n_steps): sim.tick()`. Assert end-of-run state equality on `(world_rng.getstate(), stores[0].balance, stores[0].inventory, stores[0].demand, stores[0].sales, market.market_state, item_registry lifecycle stage map)`. Confirms the two public sim entry points share the same trajectory bit-for-bit.

### CONTEXT.md update (inline)

- Add a new "Simulation" architecture entry describing the bundle returned by `build_world`, the two-phase tick API, and the override-wins rule for `policy_overrides`.
- Update "Runner" entry: note it now composes `build_world` + `Simulation.tick()` and the public API is unchanged.
- Update "Scenario" entry: document the `build_world(scenario, *, policy_overrides=...)` attach pattern.

## Acceptance criteria

- [ ] `from src.sim.runner import Runner, Simulation, TickResult, build_world` works.
- [ ] `build_world(scenario)` returns a `Simulation`; `build_world(scenario, policy_overrides=[my_policy])` constructs stores with the override policy; override wins when `StoreInstance.policy` is also set.
- [ ] `Simulation.tick_world()`, `Simulation.tick_decide_and_settle()`, and `Simulation.tick()` are exposed; `tick()` is functionally equivalent to `tick_world(); tick_decide_and_settle()`.
- [ ] `Runner(scenario).run() -> dict` public API is unchanged; existing callers in `main.py`, DataExporter, and any notebooks continue to work.
- [ ] `tests/sim/fixtures/runner_snapshot_pre.parquet` is committed (captured from pre-refactor `main` SHA).
- [ ] `tests/sim/test_runner_refactor.py::test_runner_snapshot_equals_golden` passes.
- [ ] `tests/sim/test_runner_refactor.py::test_runner_run_equals_simulation_tick_loop` passes.
- [ ] CONTEXT.md has new "Simulation" entry; "Runner" and "Scenario" entries updated.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

None — can start immediately. (The four rollout consumers are migrated in issues 06–08; they are not touched in this slice.)
