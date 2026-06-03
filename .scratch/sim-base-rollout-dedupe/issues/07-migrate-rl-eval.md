# 07 — Migrate `src/rl/eval.py::_run_baseline` and `::_run_rl` to sim primitives (two-phase API for `_run_rl`)

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADRs: `docs/adr/0003-crn-demand-for-all-products.md` (CRN bit-identity for paired eval), `docs/adr/0004-rl-training-env.md` (slot-permutation invariant stays RL-specific)

## What to build

Rewrite the two rollout halves of `src/rl/eval.py` to compose sim primitives. `_run_baseline` mirrors the tuning rollout shape (single-phase `sim.tick()`); `_run_rl` uses the two-phase API (`tick_world()` → encode obs → run actor → decode → `RLPolicy.set_pending_action(...)` → `tick_decide_and_settle()`). Helper functions `_build_world`, `_dispatch_orders`, `_process_demand`, `_make_delivery_callback` in `src/rl/eval.py` are deleted — the per-tick state machine no longer lives in this file.

RL-specific state stays in `src/rl/eval.py`: `sales_history`, `effective_rate`, `base_demand_prior`, observation encoder, action decoder, slot permutation. None of this leaks into `src/sim/`. The public `evaluate(...)` function in `src/rl/eval.py` keeps its CRN-paired contract.

Target shape for `_run_rl` (illustrative):

```python
sim = build_world(spec.spec.scenario, policy_overrides=[RLPolicy()])
rl_policy = sim.stores[0].policy  # the RLPolicy we just attached
for tick in range(episode_length):
    sim.tick_world()                                    # phase 1
    obs = encode_observation(sim.stores[0], sim.market, sim.item_registry, ...)
    action_vec = rl_policy_fn(obs)
    action_dict = decode_action(action_vec, spec.slot_permutation, ...)
    rl_policy.set_pending_action(action_dict)
    result = sim.tick_decide_and_settle()              # phase 2
    _record_active_subset(...)
return aggregate_episode(run_slice)
```

`_run_baseline` uses `build_world(spec.spec.scenario, policy_overrides=[policy])` then the same `sim.tick()` loop with active-subset trace recording.

Note the `spec.spec.scenario` access path: RL specs are `RLEpisodeSpec(spec: EpisodeSpec, slot_permutation: tuple[int, ...])` after issue 04, so the scenario lives at `spec.spec.scenario`.

### CRN bit-identity gate

The CRN tests from issue 03 must stay green. Additionally, this slice runs the end-to-end RL eval smoke as manual verification: a short RL eval run before this slice and after; per-eval mean reward identical to within `1e-9` on the same seed. This guarantees the paired uplift number in `notebooks/06-compare_rl_vs_baseline.ipynb` does not silently shift.

### Grep guard

After this slice, `grep -r "src.tuning" src/rl/` must return nothing. The full grep audit lives in issue 09 but the RL eval portion of it is locked in here.

## Acceptance criteria

- [ ] `src/rl/eval.py::_run_baseline` uses `build_world` + `Simulation.tick()`; no in-place mutation of spec state.
- [ ] `src/rl/eval.py::_run_rl` uses the two-phase API (`tick_world` → action injection → `tick_decide_and_settle`).
- [ ] The helpers `_build_world`, `_dispatch_orders`, `_process_demand`, `_make_delivery_callback` are deleted from `src/rl/eval.py`.
- [ ] RL-specific state (`sales_history`, `effective_rate`, `base_demand_prior`, observation encoder, action decoder, slot permutation) remains in `src/rl/eval.py`; none of it leaks into `src/sim/`.
- [ ] `src/rl/eval.py::evaluate(...)` public signature and CRN-paired contract are unchanged.
- [ ] CRN determinism tests from issue 03 remain green.
- [ ] RL eval smoke: per-eval mean reward identical to within `1e-9` on the same seed before and after this slice.
- [ ] `grep -r "src.tuning" src/rl/eval.py` returns nothing.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

- `01-lift-metrics-to-sim.md` — imports `RunSlice` and `aggregate_episode` from `src.sim.metrics`.
- `03-simulation-build-world-tick-result.md` — uses `build_world`, `Simulation.tick`, `Simulation.tick_world`, `Simulation.tick_decide_and_settle`.
- `04-lift-episode-sampler.md` — uses `RLEpisodeSpec` (composing sim's `EpisodeSpec`).
