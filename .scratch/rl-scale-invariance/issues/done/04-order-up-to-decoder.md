# 04 — Order-up-to decoder + `base_demand_prior` env plumbing

Status: done

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

Replace the capacity-units order-fraction decoder with an order-up-to decoder expressed in lead-times of expected demand. After this slice, `order_raw = 0` (the natural-init action under PPO's symmetric Gaussian) produces order quantities identical to `PeriodicOrderUpToPolicy(R=1, S=15·rate)` from tick 1 onward — PPO learns deviations from a textbook periodic order-up-to baseline rather than searching for a sensible operating point from scratch.

PRD user stories: 5 (order_raw=0 ≡ textbook periodic order-up-to), 6 (positive cold-start qty from market prior), 7 (qty = max(0, target_lt·rate − position)), 8 (cold-start scales with base_demand × target_centre), 9 (fair-share allocator across SKUs, not equal-split), 17 (Tier 1 unit-test coverage).

### RLConfig additions

In `src/rl/configs/default.py`, add three frozen fields:

- `target_centre_lead_times: int = 15` — order-up-to target at `order_raw = 0`. Matches `OrderUpToPolicy.S / rate` at default policy kwargs (`delivery_lag + safety_lead_ticks + cover_horizon_ticks = 3 + 2 + 10`).
- `target_half_span_lead_times: int = 15` — width of the action range around the centre.
- `target_max_lead_times: int = 30` — upper clip preventing extreme target requests.

No existing fields are removed (`demand_horizon` / `cold_start_qty` were ADR-0005 proposals that were never landed; nothing to delete).

### Decoder rewrite

Rewrite `decode_action` in `src/rl/encoders.py` end-to-end:

- Add a mandatory call-site parameter `effective_rate: dict[str, float]` (the env will compute this once per tick and pass to both decoder and encoder). For backward compatibility with existing unit tests that construct decoder calls directly, allow `effective_rate: dict[str, float] | None = None` and treat `None` as "no rate signal available" (every active SKU contributes zero requested qty).
- Add three call-site parameters surfacing the new config fields: `target_centre_lead_times`, `target_half_span_lead_times`, `target_max_lead_times`. Default values mirror the `RLConfig` defaults so test callers can opt out of passing them.
- Order half of the action vector:
  - `target_lt[slot] = clip(target_centre_lt + order_raw × target_half_span_lt, 0, target_max_lt)`
  - `requested[pid] = max(0.0, target_lt[slot] × effective_rate[pid] − (inventory[pid] + pending[pid]))`
- Per-SKU shelf headroom: `headroom[pid] = max(0, capacity − inventory[pid] − pending[pid])` (per-SKU physical, **not** equal-split).
- Final qty: compose through `fair_share_allocate(requested, headroom, global_free_space)` (from slice 3) where `global_free_space = max(0, capacity − total_inventory − total_pending)`.
- Price half **unchanged**: `price[pid] = MSRP × clip(0.5 + (price_raw + 1) × 0.5, 0.5, 1.5)`.

### Env plumbing

In `src/rl/env.py`:

- At `reset()`, after building `Market` and before returning the first obs:
  - Read `market.params.base_demand`. If it is a `Distribution`, sample it once against a fresh `Random(world_seed + 1)` (a CRN-disjoint stream — the world RNG must not advance, otherwise market draws become non-bit-identical to a Runner-based run). If it is a scalar, use it directly.
  - Stash on `self._base_demand_prior: float`.
- At each `step()`, before calling `decode_action`:
  - Compute `effective_rate = compute_effective_rate(self._sales_history, self._base_demand_prior)` (from slice 2) once per tick.
  - Pass `effective_rate` to `decode_action` and to the corresponding slice-5 encoder call. Until slice 5 lands, only the decoder consumes it; the encoder call site is unaffected.
- Read the three new `target_*_lead_times` fields from `self.config` and pass them through to `decode_action`.

### Tests added

Append to `tests/rl/test_encoders.py`:

- `test_decode_action_zero_action_matches_periodic_order_up_to_formula` — at `action_vec = zeros(2*K)`, `target_lt = 15`. With `effective_rate = {pid: 10}` per SKU, `inventory = {pid: 30}`, `pending = {pid: 0}`, large capacity, decoded `qty[pid]` equals `max(0, 15 × 10 − 30) = 120`.
- `test_decode_action_position_at_target_returns_zero_qty` — with `inventory + pending = target_lt × rate` for every SKU, decoded qty is zero regardless of `order_raw` sign (the `max(0, ...)` floor catches negative requests).
- `test_decode_action_total_qty_respects_global_free_space` — across randomised inputs, `sum(qty) <= capacity − total_inventory − total_pending`.
- `test_decode_action_per_sku_qty_respects_per_sku_headroom` — across randomised inputs, `qty[pid] <= capacity − inventory[pid] − pending[pid]`.
- `test_decode_action_price_half_unchanged` — at `price_raw = 0`, `price[pid] == MSRP[pid] × 1.0`; at `price_raw = +1`, `price[pid] == MSRP[pid] × 1.5`; at `price_raw = -1`, `price[pid] == MSRP[pid] × 0.5`.
- `test_decode_action_uses_inventory_position_not_just_inventory` — with `inventory = 0`, `pending = 50`, `effective_rate = 10`, `target_centre_lt = 15`: requested qty equals `max(0, 15 × 10 − (0 + 50)) = 100` (pending in-transit nets against the target).
- `test_decode_action_cold_start_qty_scales_with_prior` — with empty sales history (so `effective_rate = base_demand_prior`), `inventory = 0`, `pending = 0`: cold-start qty per SKU is `target_centre_lt × base_demand_prior` (modulo the fair-share clamp). Run this at both `base_demand_prior = 2` and `base_demand_prior = 20` and confirm the ratio of cold-start qty is 10× (scale-invariance property).
- `test_decode_action_effective_rate_none_returns_zero_qty` — `effective_rate=None` produces zero qty for every SKU (covers the test-helper backward-compat branch).

Append to `tests/rl/test_env_smoke.py` (or a new env smoke test if cleaner):

- `test_env_reset_stashes_base_demand_prior` — after `env.reset(seed=42)`, `env._base_demand_prior > 0` (the default market `base_demand = Uniform(2, 8)` always samples positive).
- `test_env_step_at_zero_action_produces_positive_cold_start_order` — `env.reset()`, `env.step(zeros(2*K))`: the action dict that was decoded (inspect via `env._rl_policy.pending_action` or the info dict) has at least one positive order across active SKUs.

## Acceptance criteria

- [ ] `src/rl/configs/default.py` exports `RLConfig` with the three new fields at the documented defaults.
- [ ] `decode_action` in `src/rl/encoders.py` matches the contract above: `effective_rate` parameter accepted, qty math is `max(0, target_lt × rate − position)` composed through `fair_share_allocate`, price half unchanged.
- [ ] `RLEnv.reset()` stashes `self._base_demand_prior` (positive float).
- [ ] `RLEnv.step()` computes `effective_rate` once per tick and passes it to `decode_action`.
- [ ] CRN bit-identity invariant preserved: existing `tests/rl/test_eval_crn.py` continues to pass (sampling `base_demand` against a disjoint `Random(world_seed + 1)` must not perturb the existing world-RNG advance pattern).
- [ ] All new tests pass under `uv run pytest tests/rl/test_encoders.py tests/rl/test_env_smoke.py`.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`.

## Blocked by

- `02-compute-effective-rate.md` — decoder composes through `compute_effective_rate` at the env layer.
- `03-fair-share-allocate.md` — decoder composes through `fair_share_allocate`.
