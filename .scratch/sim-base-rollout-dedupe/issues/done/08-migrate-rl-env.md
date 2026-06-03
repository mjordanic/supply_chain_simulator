# 08 — Migrate `src/rl/env.py::RLEnv.step` to the two-phase sim API

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADR: `docs/adr/0004-rl-training-env.md` (RL training env)

## What to build

Rewrite `RLEnv.step` so the per-tick state machine delegates to `Simulation` instead of carrying a fourth copy. The body uses the same two-phase shape as `_run_rl` from issue 07: `tick_world()` → encode obs → run actor → decode → `RLPolicy.set_pending_action(...)` → `tick_decide_and_settle()`. Gymnasium concerns (`observation_space`, `action_space`, `terminated`, `truncated`, the `info` dict, episode length tracking, reward shaping) stay in `RLEnv` — they are framework-contract, not per-tick state machine.

Private helpers in `src/rl/env.py` that duplicate `_dispatch_orders` / `_process_demand` / `_make_delivery_callback` are deleted.

`RLEnv.reset()` constructs the `Simulation` via `build_world(spec.spec.scenario, policy_overrides=[RLPolicy()])`. The `RLPolicy` instance is reachable via `sim.stores[0].policy` for `set_pending_action` calls inside `step`.

### CRN bit-identity gate

The CRN tests from issue 03 must stay green. Additionally, this slice runs the end-to-end RL training smoke as manual verification: `uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2` before and after; the reward curve at the first evaluation point must be identical to within `1e-9` on the same `--seed`. This guarantees the migration is invisible to the RL training contract.

### Notes from the PRD's risks section

- `RLEnv.step` is the fourth and final rollout copy. After this slice, the per-tick state machine has exactly one implementation: `Simulation.tick_*` in `src/sim/runner.py`.
- The gymnasium contract is independent of the sim refactor; `observation_space` / `action_space` / `terminated` / `truncated` / `info` definitions in `RLEnv` are unchanged.

## Acceptance criteria

- [ ] `src/rl/env.py::RLEnv.step` body uses `Simulation.tick_world()` and `Simulation.tick_decide_and_settle()` with the actor injection in the seam.
- [ ] `src/rl/env.py::RLEnv.reset()` constructs the `Simulation` via `build_world(scenario, policy_overrides=[RLPolicy()])`.
- [ ] Any helpers in `src/rl/env.py` duplicating `_dispatch_orders` / `_process_demand` / `_make_delivery_callback` are deleted.
- [ ] Gymnasium-contract symbols (`observation_space`, `action_space`, `terminated`, `truncated`, `info`) are unchanged in shape and content.
- [ ] CRN determinism tests from issue 03 remain green.
- [ ] RL training smoke: `uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2` reward curve at the first eval point identical to within `1e-9` before and after.
- [ ] Grepping `src/rl/env.py` for `_dispatch_orders`, `_process_demand`, `_make_delivery_callback` returns no definitions (call sites either delegate or are gone).
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

- `03-simulation-build-world-tick-result.md` — uses `build_world`, `Simulation.tick_world`, `Simulation.tick_decide_and_settle`.
