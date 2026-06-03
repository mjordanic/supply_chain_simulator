# 05 — Encoder demand-units inventory feature (slot 13)

Status: in-progress

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

Grow the per-SKU observation block from 13 to 14 features by adding a demand-units inventory feature in slot 13. The existing capacity-units inventory feature in slot 0 is **retained** — it carries shelf-saturation signal that is useful for pricing decisions even though it is the wrong frame for ordering. Both features coexist; the policy net learns which to attend to when.

PRD user stories: 10 (demand-units inventory feature at flagship scale), 11 (capacity-units feature retained alongside), 20 (stale checkpoint produces clear shape-mismatch error).

### RLConfig addition

In `src/rl/configs/default.py`, add one frozen field:

- `max_inventory_lt: float = 30.0` — saturation point for the demand-units inventory feature, in lead-times of cover.

### Encoder change

In `src/rl/encoders.py`:

- Bump `N_PER_SKU` from `13` to `14`.
- Update the module-level docstring layout table to include slot 13: `clip(inventory[pid] / effective_rate[pid], 0, max_inventory_lt) / max_inventory_lt`, range `[0, 1]`.
- `encode_observation` gains two parameters:
  - `effective_rate: dict[str, float] | None = None` — when supplied, slot 13 uses `inventory / effective_rate[pid]` for each active SKU. When `None`, slot 13 falls back to `0.0` (test-helper backward compatibility).
  - `max_inventory_lt: float = 30.0` — saturation point, defaulted so existing test callers don't need to pass it explicitly.
- Slot 13 formula per active SKU: `slot_13 = clip(inventory[pid] / max(effective_rate[pid], epsilon), 0.0, max_inventory_lt) / max_inventory_lt`, with `epsilon = 1e-9` to avoid division-by-zero (slice 2's `compute_effective_rate` already floors at the prior, so this guard is belt-and-braces).
- Capacity-units slot 0 is unchanged — same formula as before.
- The global block (4 features) is unchanged; total obs length becomes `K_active * 14 + 4`.

### Env change

In `src/rl/env.py`:

- `RLEnv.step()` already computes `effective_rate` for the decoder (slice 4). Pass the same dict to `encode_observation` via `_build_observation`.
- `RLEnv._build_observation()` reads `self.config.max_inventory_lt` and the cached effective_rate; passes both into `encode_observation`. The first observation built in `reset()` is before any tick has fired, so `effective_rate` there is the all-prior dict — compute it once at the end of `reset()` for symmetry with `step()`.

### Stale-checkpoint failure-mode test

`Actor` and `Critic` in `src/rl/agents/ppo.py` build their input layer from the env's observation space, so they automatically widen to `K_active * 14 + 4`. A pre-existing `state_dict` saved against `K_active * 13 + 4` will fail to load with a clear shape-mismatch from `torch.nn.Module.load_state_dict`. This is the desired failure mode — loud and early, not silent.

Add a regression test asserting this:

- `test_loading_stale_obs_shape_checkpoint_raises_shape_mismatch` — build an `Actor` against an observation space of width `K_active * 13 + 4` (the pre-ADR-0007 shape), save its `state_dict` to a temp path, then attempt to load it into an `Actor` built against the new `K_active * 14 + 4` observation space. Assert that `load_state_dict` raises a `RuntimeError` (PyTorch's default) and that the error message mentions a size mismatch on the first linear layer.

Where to put this test depends on existing layout — if `tests/rl/test_ppo_smoke.py` already covers Actor instantiation, append there; otherwise add it to `tests/rl/test_train_driver.py`.

### Tests added

Append to `tests/rl/test_encoders.py`:

- `test_observation_dim_increased_to_14_per_sku` — `observation_dim(K=5) == 5 * 14 + 4 == 74` (was `5 * 13 + 4 == 69`).
- `test_encoder_slot_13_matches_demand_units_formula` — for each active SKU, slot 13 of the obs equals `min(1.0, inventory / (effective_rate × max_inventory_lt))`. Test with concrete numbers: `inventory = 100`, `effective_rate = 10`, `max_inventory_lt = 30` → slot 13 = `min(1.0, 100 / 300) ≈ 0.333`.
- `test_encoder_slot_13_saturates_at_max_inventory_lt` — `inventory = 10_000`, `effective_rate = 10`, `max_inventory_lt = 30`: slot 13 = `1.0` (saturated).
- `test_encoder_slot_13_zero_when_effective_rate_is_none` — when `effective_rate=None`, slot 13 = `0.0` for every active SKU (backward-compat branch).
- `test_encoder_slot_0_capacity_units_inventory_unchanged` — slot 0 still equals `clip(inventory / per_sku_capacity, 0, 1)` for every active SKU (regression guard on the retained feature).
- `test_encoder_scale_invariance_of_slot_13` — same per-SKU `inventory / effective_rate` ratio with two different absolute capacities produces identical slot-13 values (the scale-invariance property the feature is designed to provide).
- `test_loading_stale_obs_shape_checkpoint_raises_shape_mismatch` — described above.

## Acceptance criteria

- [ ] `src/rl/configs/default.py` exports `RLConfig` with `max_inventory_lt: float = 30.0`.
- [ ] `src/rl/encoders.py` `N_PER_SKU == 14`.
- [ ] `observation_dim(K)` returns `K * 14 + 4` for any `K`.
- [ ] `encode_observation` signature accepts `effective_rate` and `max_inventory_lt` parameters with the documented defaults.
- [ ] Slot 13 formula matches the spec for every active SKU; saturates at `max_inventory_lt`.
- [ ] Capacity-units slot 0 is unchanged (regression test passes).
- [ ] `RLEnv` passes `effective_rate` and `max_inventory_lt` into both encoder calls (the one in `reset()` and the one in `_build_observation()`).
- [ ] An `Actor` checkpoint saved against the prior obs shape fails to load with a clear `RuntimeError` size mismatch.
- [ ] All new tests pass under `uv run pytest tests/rl/test_encoders.py` and the chosen Actor-test file.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`.

## Blocked by

- `04-order-up-to-decoder.md` — slice 4 introduces the env's `effective_rate` plumbing this slice consumes.
