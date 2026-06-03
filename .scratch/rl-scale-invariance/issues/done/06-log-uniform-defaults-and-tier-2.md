# 06 — Log-uniform DR defaults + Tier 2 centring-sanity test

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

Widen the per-episode capacity and balance distributions to span two orders of magnitude, and land the highest-leverage validation gate in the package: the Tier 2 centring-sanity test that asserts the order-up-to decoder at `action = 0` produces trajectories identical to `PeriodicOrderUpToPolicy(R=1, S=15·rate)` from tick 1 onward.

This slice is the de-facto merge gate. Once it lands green, the order-up-to-decoder + demand-units-feature + log-uniform-DR package is behaviourally complete; slices 7 and 8 are non-blocking follow-ups.

PRD user stories: 12 (`LogUniform(100, 10_000)` capacity default), 13 (training reward roughly stable across capacity range — a property the Tier 2 test partially anchors), 18 (Tier 2 centring-sanity integration test).

### Config default changes

In `src/rl/configs/default.py`, rewrite the two default-factory helpers:

- `_default_capacity_dist()` returns `LogUniform(100, 10_000)` (was `Uniform(150, 400)`).
- `_default_balance_dist()` returns `LogUniform(10_000, 1_000_000)` (was `Uniform(15_000, 40_000)`).

Imports change from `Uniform` to `LogUniform`. No field declarations or `RLConfig` shape change beyond the default factories.

### Existing tests that may need adjustment

Several tests construct an `RLConfig()` and depend on specific capacity / balance ranges:

- `tests/rl/test_episode_sampler.py` — likely asserts capacity / balance values land in the old `Uniform(150, 400)` / `Uniform(15_000, 40_000)` ranges. Update the expected ranges, or refactor to pin `capacity_dist` / `balance_dist` to the old values when range-checking is the test's actual concern.
- `tests/rl/test_env_smoke.py` — may construct a default-config env and indirectly depend on small numbers. Inspect; adjust only if a real assertion breaks.
- `tests/rl/test_train_driver.py` — same review pass.

Default principle: tests that assert behavioural shape (e.g. "capacity comes from the configured distribution") stay agnostic to the default; tests that depend on specific numbers either pin the config explicitly or update their expectations.

### Tier 2 centring-sanity test

New file `tests/rl/test_centring_sanity.py`. One test:

`test_zero_action_matches_periodic_order_up_to_from_tick_one`:

1. Pick a fixed `episode_seed = 7` (or any small deterministic value).
2. Build an `RLEnv(catalog, base_template, config=RLConfig(...))` where `config` pins `capacity_dist = Constant(500)` and `balance_dist = Constant(50_000)` so the test is independent of the log-uniform defaults. Use the same `_default_market_params` the episode sampler already uses, so `base_demand = Uniform(2, 8)` and the env's `_base_demand_prior` matches the comparison run's.
3. Roll the env for the full `config.episode_length` ticks with `action = np.zeros(2 * K, dtype=np.float32)` each step. Record per-tick `action_dict` (orders, prices) from `env._rl_policy.pending_action` (or wire an info channel), plus per-tick `store.inventory` and final balance.
4. Build a second `Scenario` directly from the same `EpisodeSpec` (`sample_episode(catalog, base_template, config, episode_seed=7)`) and replace the single `StoreInstance.policy` with `PeriodicOrderUpToPolicy(R=1, cover_horizon_ticks=10, safety_lead_ticks=2)`. Run it through `Runner` to completion.
5. Assert from tick 1 onward (tick 0 diverges by construction — RL uses the market prior, the textbook policy uses its cash-budget pilot):
   - Per-tick orders match per pid within `±1` unit.
   - Per-tick prices match per pid within `±0.01` price-unit (both should be flat at `1.0 × MSRP` since neither policy adjusts pricing here).
   - Per-tick inventory matches per pid within `±1` unit (the order tolerance propagates).
   - Final `store.balance` matches within `±5.0` currency-units (sub-percent of expected balance for a `Constant(50_000)` opening).
6. Tick 0 is explicitly *not* asserted to match; the test must skip the tick-0 comparison or compare with a much looser tolerance, with a comment pointing at the cold-start divergence (RL uses `compute_effective_rate(empty_history, base_demand_prior)` = prior; textbook uses `opening_budget_pct × cash / K`).

The test should run in under 10 seconds — a single 180-tick episode at K=5, twice. If it doesn't, something is wrong with the bit-identity of the two world rollouts (likely a stray RNG advance in the env not present in the Runner, or vice versa).

### CRN bit-identity prerequisite

The test only works if `RLEnv.step()` advances `world_rng` identically to `Runner.run()` on the same scenario. The existing `tests/rl/test_eval_crn.py` covers this property and must continue to pass — if it doesn't after slices 4 and 5, this slice is blocked on fixing the env's RNG-advance pattern.

## Acceptance criteria

- [ ] `_default_capacity_dist()` returns `LogUniform(100, 10_000)`; `_default_balance_dist()` returns `LogUniform(10_000, 1_000_000)`.
- [ ] `tests/rl/test_centring_sanity.py` exists and contains `test_zero_action_matches_periodic_order_up_to_from_tick_one`.
- [ ] The Tier 2 test passes under `uv run pytest tests/rl/test_centring_sanity.py`.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`. Any test that broke due to the default-distribution widening is either correctly updated or refactored to pin its own distribution.
- [ ] `tests/rl/test_eval_crn.py` continues to pass (CRN bit-identity preserved).

## Blocked by

- `01-log-uniform-distribution.md` — new defaults import `LogUniform`.
- `04-order-up-to-decoder.md` — Tier 2 asserts the order-up-to decoder produces textbook-policy-equivalent trajectories.
- `05-encoder-demand-units-feature.md` — the obs shape change must be in place; the test's env construction uses the full slice-4 + slice-5 stack.
