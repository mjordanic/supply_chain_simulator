# 05 — `RLPolicy` shim + `RLEnv` Gymnasium env

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

The central integration slice. Adds a minimal `RLPolicy(Policy)` shim and a Gymnasium-compatible `RLEnv` that exposes the simulator's `Market` / `EventEngine` / `ItemRegistry` / `Store` subsystems as `reset()` / `step()` semantics.

### `RLPolicy` (extension to `src/sim/policy.py`)

Minimal `Policy` subclass. Holds a "pending action" attribute set by the env immediately before `Store.decide()` is invoked. `decide(observation)` returns the pending action dict and clears it. Does not consume `policy_rng`. The env constructs the action dict via `encoders.decode_action` before storing it.

### `RLEnv` (`src/rl/env.py`)

Public surface:

- `RLEnv(catalog, base_template, market_params, lifecycle_params, disruption_params, config)` — accepts a pre-built catalog and an `RLConfig`.
- `reset(seed=None) → (obs, info)` — calls `episode_sampler.sample_episode`, builds a fresh Scenario, instantiates the simulator subsystems plus one `Store` with an attached `RLPolicy`, returns the initial observation.
- `step(action) → (obs, reward, terminated, truncated, info)` — performs one tick in the same order as `Runner.run()`:
  1. `market.tick()`
  2. `event_engine.tick(market)`
  3. `item_registry.tick()`
  4. Set pending action on `RLPolicy` via `encoders.decode_action`
  5. `store.decide(store.observe(...))`
  6. Dispatch orders (same as `Runner._dispatch_orders`)
  7. Settle demand (same as `Runner._process_demand`), accumulating reward
  8. Build next observation via `encoders.encode_observation`
  9. Increment step counter; terminate when `step == config.episode_length`.
- `info` dict carries unmasked per-SKU traces needed for business-metric computation (sales, demand, inventory, revenue, costs).
- `observation_space` / `action_space` are `gymnasium.spaces.Box`, shapes derived from `(K_active, n_per_sku_features, n_global_features)`.

The env owns its own RNG used only for slot permutation; world stochasticity flows through `world_seed` exactly as in `Runner`. Promotions disabled, assortment frozen, lifecycle and disruptions left enabled at simulator defaults.

The existing `Runner` is not modified — the env must reuse the same subsystems but never edit `Runner` itself.

## Acceptance criteria

- [ ] `src/sim/policy.py` exports a new `RLPolicy` class (subclass of `Policy`) with a setter for the pending action
- [ ] `src/rl/env.py` exports `RLEnv` that satisfies the Gymnasium `Env` interface (`reset`, `step`, `observation_space`, `action_space`)
- [ ] `RLEnv.step()` advances the simulator in the exact ordering of `Runner.run()` (no divergent simulation loop)
- [ ] Promotions are disabled and the assortment is frozen for the duration of an episode
- [ ] `tests/rl/test_env_smoke.py` covers:
  - [ ] `reset()` returns an observation matching `observation_space.shape`
  - [ ] One full episode (180 `step()` calls) runs end-to-end without raising
  - [ ] `terminated` becomes `True` exactly at step `episode_length`
  - [ ] Reward across one full episode sums to the balance delta minus any non-active-SKU contributions
  - [ ] Two `RLEnv` instances called with `reset(seed=s)` produce identical first-observation tensors
- [ ] `uv run pytest tests/sim/` continues to pass (no regression in the existing ~50 simulator tests)
- [ ] `uv run pytest tests/rl/test_env_smoke.py` passes

## Blocked by

- `.scratch/rl-policy-framework/issues/01-rl-deps-and-config.md`
- `.scratch/rl-policy-framework/issues/02-encoders-module.md`
- `.scratch/rl-policy-framework/issues/03-episode-sampler-module.md`
