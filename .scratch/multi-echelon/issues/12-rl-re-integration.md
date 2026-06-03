# RL re-integration: episode sampler, env, encoders, RLIntermediatePolicy, eval CRN

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Train and evaluate RL on the graph engine. Single-store RL is preserved as a degenerate 3-node episode (1 factory → 1 trainable intermediate → 1 sink), so the simplest entry point onto the new engine continues to work.

**Changes:**
- `src/rl/episode_sampler.py::sample_episode` builds the 3-node graph; intermediate is the trainable node
- `src/rl/env.py::RLEnv.reset` calls `build_world(spec.scenario, policy_overrides_by_id={"S": RLIntermediatePolicy()})` (override pattern keyed by node id, not list index)
- `RLEnv.step(action)` flow: `tick_world` → `publish_offers` → run phases except S's → encode obs from post-tick S state → actor → decode → `RLIntermediatePolicy.set_pending_action` → run S's phase → run higher phases → `consume_demand_sinks` → reward = `S.cash_delta`
- `src/rl/encoders.py`: observation extended with `central_table_snapshot[product_slot]` block (`supplier_count`, `min_price`, `mean_lead_time`, `mean_fill_rate`); bump `observation_dim`
- Action decoder emits per-supplier splits so the encoder/decoder pair generalises when an episode later runs against a multi-supplier graph
- `src/rl/eval.py`: CRN tuple expanded to include the `allocation` sub-seed; anchor remains `OrderUpToPolicy` (now `MultiSupplierTextbookPolicy`)

**Tests rewritten:** `tests/rl/test_eval_crn.py`, `test_env_smoke.py`, `test_episode_sampler.py`, `test_encoders.py`, `test_two_scale_eval_smoke.py`.

**Verification:** `uv run pytest tests/rl`. `uv run python -m src.rl.train --config src/rl/configs/default.py --steps 5000` smoke; reward learns above `OrderUpToPolicy` baseline on the 32-seed held-out set.

## Acceptance criteria

- [ ] `src/rl/episode_sampler.py::sample_episode` builds the 3-node graph; trainable intermediate identified by node id
- [ ] `src/rl/env.py::RLEnv.reset` uses `policy_overrides_by_id`
- [ ] `RLEnv.step(action)` runs the phase-aware flow (S's phase deferred to post-encode/decode)
- [ ] `src/rl/encoders.py` adds `central_table_snapshot[product_slot]` block with the four sub-features; `observation_dim` bumped accordingly
- [ ] Action decoder emits per-supplier split actions (degenerate single-supplier case = list of length 1)
- [ ] `RLIntermediatePolicy(IntermediatePolicy)` with `set_pending_action`
- [ ] `src/rl/eval.py` CRN tuple expanded with `allocation` sub-seed
- [ ] All `tests/rl/*` rewritten and green: `test_eval_crn.py`, `test_env_smoke.py`, `test_episode_sampler.py`, `test_encoders.py`, `test_two_scale_eval_smoke.py`
- [ ] `uv run pytest tests/rl` is green
- [ ] `uv run python -m src.rl.train --config src/rl/configs/default.py --steps 5000` completes; smoke reward learns above `OrderUpToPolicy` baseline on 32-seed held-out set

## Blocked by

- `.scratch/multi-echelon/issues/11-retire-legacy-store-engine.md`
