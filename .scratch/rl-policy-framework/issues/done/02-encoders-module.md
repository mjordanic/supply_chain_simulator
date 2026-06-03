# 02 — `encoders` pure-function module

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

A pure-function module `src/rl/encoders.py` that converts simulator state into observation tensors and decodes RL action vectors into the action dict the simulator's `Policy.decide()` expects. No simulator state held; no I/O; deterministic.

Public functions:

- `encode_observation(store, market, registry, step, slot_perm, K_active) → ndarray` — returns a flat `float32` 1-D tensor.
  - Per-SKU features (placed in `slot_perm` order): inventory normalised by per-slice capacity, recent sales (rolling 5-tick mean), pending orders, current price / MSRP, MSRP / mean MSRP, unit cost / MSRP, lifecycle stage (one-hot of 5), ticks-since-activation (clipped + log-scaled), in-season flag (0/1).
  - Global features: cash / initial cash, total inventory / capacity, sin(2π · step / 360), cos(2π · step / 360).
  - Regional `market_demand` / `market_supply` MUST NOT appear in the observation.

- `decode_action(action_vec, slot_perm, store, K_active, base_prices) → action_dict` — input is `[-1, 1]^(2K)`. First K → price multipliers in `[0.5, 1.5]` (linear). Second K → order quantities as fractions of free space in `[0, 1]` (linear from `[-1, 1]`). The inverse `slot_perm` maps the K slots back to the active SKU ids. Output is `{"order": dict[pid, int], "price": dict[pid, float], "activate": [], "deactivate": [], "promotions": {}}`.

- `observation_dim(K_active) → int` and `action_dim(K_active) → int` — used by `RLEnv` to declare its `gymnasium.spaces.Box`.

## Acceptance criteria

- [ ] `src/rl/encoders.py` exports `encode_observation`, `decode_action`, `observation_dim`, `action_dim`
- [ ] No simulator state mutated; the module's functions are pure and deterministic
- [ ] `tests/rl/test_encoders.py` covers:
  - [ ] Slot-shuffle round-trip: applying decode-then-encode (or encode-then-decode for actions) for an arbitrary permutation returns the original action dict
  - [ ] Action decoding bounds: random action vectors in `[-1, 1]^(2K)` produce prices in `[0.5·MSRP, 1.5·MSRP]` and order quantities in `[0, free_space]`
  - [ ] Observation shape matches `observation_dim(K)`
  - [ ] Observation feature ranges: every feature falls in its documented bound (e.g. `[-1, 1]` or `[0, 1]` after normalisation)
  - [ ] Determinism: same `(store, market, slot_perm, step)` → identical tensor
- [ ] Tests run under `uv run pytest tests/rl/test_encoders.py` and pass

## Blocked by

None — can start immediately.
