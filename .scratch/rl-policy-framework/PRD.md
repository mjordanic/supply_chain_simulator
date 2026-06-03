# PRD: Reinforcement-Learning Policy Framework

Status: ready-for-agent

## Problem Statement

The simulator already supports pluggable policies via the `Policy` ABC, and a heuristic `BaselinePolicy` is in place. We want to demonstrate that a Reinforcement-Learning agent can learn something sensible — i.e. converge to a policy that beats random and is competitive with `BaselinePolicy` — using the same observe→decide→advance→log loop. The current `Runner` exposes a one-shot `run()` API, which is not directly usable as a stepwise RL environment, and there is no observation/action encoding, training loop, evaluation harness, or business-metric tracking. We need an end-to-end RL framework: env wrapper, encoders, sampler, agent, trainer, evaluator, and metrics — built into this repo, tightly coupled to the existing `Scenario` / `Store` / `Market` / `ItemRegistry` machinery, and designed so a future swap to a heavier model is just a network class change.

## Solution

Add a new `src/rl/` package that exposes the simulator as a Gymnasium-compatible environment, ships a CleanRL-style single-file PPO agent, and provides a CRN-paired evaluation harness against `BaselinePolicy`. The env wraps the existing subsystems (`Market`, `EventEngine`, `ItemRegistry`, `Store`) and ticks them in step-by-step semantics via `reset()` / `step()`, with each episode drawing a fresh assortment of active SKUs from a fixed 100-product LLM-built catalog, randomising capacity and opening balance, and shuffling the slot order in the observation tensor so the agent cannot memorise SKU identity. Training writes TensorBoard scalars; evaluation runs every ~50k env steps over a frozen set of 32 held-out CRN seeds, comparing RL against `BaselinePolicy` on bit-identical worlds.

A new `RLPolicy(Policy)` shim lets the agent participate through the existing `Store.decide()` pathway without bypassing the simulator's mutation contracts. The action space stays simple: 10-dim continuous in `[-1, 1]`, decoded to 5 price multipliers and 5 order-fraction values for the active subset. Episode length is 180 ticks (half-year). Reward is per-tick profit summed across the active subset.

## User Stories

1. As a researcher, I want a Gymnasium-compatible env wrapping the simulator, so that I can plug in any RL library that speaks the standard env interface.
2. As a researcher, I want each `reset()` to produce a freshly randomised episode (assortment, capacity, balance, world seed, slot order), so that the agent learns to generalise rather than memorise.
3. As a researcher, I want the active subset sampled uniformly per episode from a 100-product LLM-built catalog, so that ~75M unique assortments make memorisation impossible.
4. As a researcher, I want the per-SKU observation features placed in slots that are reshuffled every episode, so that the agent must learn from features rather than positional identity.
5. As a researcher, I want regional `market_demand` and `market_supply` to remain hidden from the agent, so that the partial-observability assumption from the baseline policy is preserved.
6. As a researcher, I want `lifecycle_stage`, `ticks_since_activation`, and a cyclic encoding of the calendar date exposed to the agent, so that the agent has access to the information a real store-manager would have.
7. As a researcher, I want the agent's action represented as 10 continuous floats in `[-1, 1]`, so that PPO and SAC both work without custom action wrappers.
8. As a researcher, I want the action mapped to 5 price multipliers in `[0.5×, 1.5×]` of MSRP and 5 order quantities as fractions of free space, so that the action lives on a bounded, well-conditioned manifold.
9. As a researcher, I want promotions disabled entirely during RL runs, so that price is the only price-lever and apples-to-apples comparison against the baseline is clean.
10. As a researcher, I want the assortment frozen for the duration of an episode (no activation/deactivation during the run), so that the catalog-rotation head can be deferred to a follow-up.
11. As a researcher, I want the per-tick reward to equal the sum of `revenue − total_cost` over the active subset, so that episode return equals total profit and matches the balance evolution exactly.
12. As a researcher, I want a `RLPolicy(Policy)` shim that receives the decoded action via a setter and returns it from `decide()`, so that the env uses the existing `Store.decide()` pathway without bypassing simulator invariants.
13. As a researcher, I want a single-file CleanRL-style PPO with an MLP actor-critic, so that I can read every line of the algorithm and modify it freely.
14. As a researcher, I want 8 vectorised sync envs by default, so that PPO rollouts collect enough trajectories per update without GPU contention on a single laptop.
15. As a researcher, I want TensorBoard scalars logged at every PPO update, so that I can watch learning curves in real time.
16. As a researcher, I want a frozen set of 32 held-out CRN evaluation seeds, so that mid-training evaluations are paired against the baseline on bit-identical worlds.
17. As a researcher, I want evaluation to run every ~50k env steps, so that I see ~20–40 eval points across a typical 1M-step training run.
18. As a researcher, I want the eval to compute mean RL return, mean baseline return, paired uplift, and win-rate, so that I can tell when RL has started to beat the baseline and by how much.
19. As a researcher, I want business KPIs logged alongside reward — `service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, plus the profit decomposition (revenue / holding / order / fee) — so that I can detect degenerate solutions like "max profit by stripping the shelves and dodging holding cost".
20. As a researcher, I want capacity sampled from `Uniform(150, 400)` and opening balance from `Uniform(15000, 40000)` per episode, so that the env sits in the "tight but solvable" regime where stockouts are avoidable with good reorder timing.
21. As a researcher, I want `init_freshness="fresh"` and zero starting inventory each episode, so that the agent always faces a grand-opening scenario and must build inventory itself.
22. As a researcher, I want delivery lead time fixed at 3 ticks during training, so that there is a meaningful planning horizon for the order action.
23. As a researcher, I want order fee non-zero (default 50), so that micro-orders are punished and the agent learns to batch.
24. As a researcher, I want hyperparameters captured in a dataclass config, so that I can sweep them without editing the agent file.
25. As a researcher, I want a clear PPO-first, SAC-later progression, so that the first cut is minimally complex.
26. As a researcher, I want SAC available as a second single-file module when I'm ready, so that I can compare on-policy vs off-policy on the same env without rewriting plumbing.
27. As a researcher, I want a smoke test that one episode reset/step round-trips end-to-end without crashing, so that env integration regressions are caught immediately.
28. As a researcher, I want a determinism test for `episode_sampler` (same seed → identical scenario), so that experiment reproducibility is guaranteed.
29. As a researcher, I want a round-trip test for `encoders` (slot-shuffle inverse correctness, action decode bounds), so that silent encoding bugs cannot tank training.
30. As a researcher, I want a hand-crafted-input test for `metrics`, so that service-level and turnover formulas don't drift.
31. As a researcher, I want a CRN self-eval test (`BaselinePolicy` against itself on identical seeds yields zero uplift), so that the eval plumbing can't silently mis-pair seeds.
32. As a researcher, I want a short PPO smoke training run (~1000 steps) that verifies loss decreases and KL stays bounded, so that integration regressions in the training loop are caught.
33. As a developer maintaining the simulator, I want all RL dependencies (torch, gymnasium, tensorboard) under `[project.optional-dependencies] rl`, so that non-RL workflows are not affected.
34. As a developer, I want the existing `Runner` left untouched, so that the ~50 existing simulator tests continue to pass.
35. As a developer, I want the new env to use the same `Market` / `EventEngine` / `ItemRegistry` / `Store` subsystems as `Runner`, so that there is no divergent fork of the simulation loop.
36. As a developer, I want pure-function deep modules (`encoders`, `episode_sampler`, `metrics`), so that the testable surface is wide and the implementation surface is narrow.
37. As a future developer, I want swapping the PPO actor-critic for a heavier model (transformer, attention over SKUs) to be a single-class change, so that the framework scales beyond the demo without re-architecture.
38. As a future developer, I want catalog-rotation, larger N, and stratified assortment sampling all expressible as additions on top of this framework, so that the demo is a foundation rather than a dead end.

## Implementation Decisions

### New domain concepts (to be added to `CONTEXT.md`)

- **RL Env** — a Gymnasium-compatible environment wrapping the simulator's `Market` / `EventEngine` / `ItemRegistry` / `Store` subsystems in step-by-step semantics. Each `reset()` produces a fresh `Scenario` via `episode_sampler`; each `step(action)` advances one tick using the same sequence as `Runner.run()` but with the action injected through an `RLPolicy` shim. Distinct from `Runner`, which remains for batch runs and tests.
- **RL Episode** — one `reset()`-to-terminated pass through the env. Fixed at 180 ticks. Distinct from the "Scenario" notion which describes a full experiment.
- **Active subset** — the K (default 5) product ids sampled uniformly per episode from the 100-product catalog universe; the only SKUs the RL agent decides for.
- **Slot-shuffled observation** — observation tensor where the K active SKUs occupy K slots, with the slot-to-SKU mapping permuted per episode. The permutation is a per-episode RNG draw; the encoder applies it on observation and the decoder applies the inverse on action.
- **CRN-paired eval** — evaluation protocol where the RL policy and `BaselinePolicy` are run on bit-identical `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` tuples; uplift is computed paired per seed.

### `RLPolicy` (extension to `src/sim/policy.py`)

A new minimal `Policy` subclass. Holds a "pending action" attribute set by the env immediately before `Store.decide()` is invoked. `decide(observation)` returns the pending action dict and clears it. No `policy_rng` consumption. Action dict format: `{"order": dict[pid, int], "price": dict[pid, float], "activate": [], "deactivate": [], "promotions": {}}`. The env constructs the action dict via `encoders.decode_action` before storing it.

### `RLEnv` (Gymnasium env, `src/rl/env.py`)

Public surface:

- `RLEnv(catalog, base_template, market_params, lifecycle_params, disruption_params, config)` — constructor accepts a pre-built catalog and a config dataclass.
- `reset(seed=None) → (obs, info)` — calls `episode_sampler` with the given seed, builds a fresh Scenario, instantiates `Market` / `EventEngine` / `ItemRegistry` / one `Store`, attaches an `RLPolicy`, returns initial observation.
- `step(action) → (obs, reward, terminated, truncated, info)` — performs one tick using the same ordering as `Runner.run()`:
  1. `market.tick()`
  2. `event_engine.tick(market)`
  3. `item_registry.tick()`
  4. Set the pending action on `RLPolicy` via `encoders.decode_action`
  5. `store.decide(store.observe(...))`
  6. Dispatch orders (same as `Runner._dispatch_orders`)
  7. Settle demand (same as `Runner._process_demand`), accumulating reward
  8. Build next observation via `encoders.encode_observation`
  9. Increment step counter; terminate when step == episode_length.
- `info` dict carries the unmasked per-SKU traces needed for business-metric computation (sales, demand, inventory, revenue, costs).
- `observation_space` and `action_space` are `gymnasium.spaces.Box`; concrete shapes derived from `(K_active, n_per_sku_features, n_global_features)`.

The env constructs and owns its own RNG used only for slot permutation; world stochasticity flows through `world_seed` exactly as in `Runner`.

### `encoders` (`src/rl/encoders.py`, deep pure-function module)

Public functions:

- `encode_observation(store, market, registry, step, slot_perm, K_active) → ndarray` — returns a flat 1-D float32 tensor. Per-SKU features (placed in `slot_perm` order): inventory (normalised by per-slice capacity), recent sales (rolling 5-tick mean), pending orders, current price / MSRP, MSRP / mean MSRP, unit cost / MSRP, lifecycle stage (one-hot of 5), ticks since activation (clipped + log-scaled), in-season flag (0/1). Global features: cash / initial cash, total inventory / capacity, sin(2π · step / 360), cos(2π · step / 360).
- `decode_action(action_vec, slot_perm, store, K_active, base_prices) → action_dict` — input is the env action in `[-1, 1]^(2K)`. First K entries decode to price multipliers in `[0.5, 1.5]` (linear), second K decode to order quantities as fractions of free space in `[0, 1]` (linear from `[-1, 1]`). The inverse `slot_perm` maps the K slots back to the active SKU ids. Output is the action dict in the format `Policy.decide` expects.
- `observation_dim(K_active) → int` and `action_dim(K_active) → int` — used by `RLEnv` to declare spaces.

No simulator state held; no I/O; deterministic.

### `episode_sampler` (`src/rl/episode_sampler.py`, deep pure-function module)

Public function:

- `sample_episode(catalog, base_template, config, episode_seed) → EpisodeSpec` where `EpisodeSpec` is a small dataclass holding:
  - `scenario: Scenario` — full Scenario object ready to feed into the env (n_steps = episode_length, world_seed derived from episode_seed).
  - `slot_permutation: tuple[int, ...]` — slot index → position in active subset.
  - `active_subset: tuple[str, ...]` — the K product ids selected this episode.

The function takes one `episode_seed`, splits it into independent sub-seeds (assortment, capacity, balance, world, slot permutation), and produces a fully deterministic `EpisodeSpec`. Capacity and balance sampled from `config.capacity_dist` / `config.balance_dist` (default `Uniform(150, 400)` / `Uniform(15000, 40000)`). Active subset sampled uniformly without replacement from the catalog. `init_active_products` on the synthesised StoreTemplate is set to the active subset; `init_stock_pct = 0.0`; `init_freshness = "fresh"`.

### `metrics` (`src/rl/metrics.py`, deep pure-function module)

Public functions:

- `service_level(run_slice) → float` — `sum(sales) / max(1, sum(demand))` across active SKUs and ticks.
- `stockout_rate(run_slice) → float` — fraction of `(active_SKU, tick)` pairs where inventory was 0 at decision time.
- `mean_price_pct_of_msrp(run_slice) → float` — mean of `price / MSRP` across active-SKU ticks.
- `inventory_turnover(run_slice) → float` — `sum(sales) / max(1, mean(inventory))`.
- `profit_decomposition(run_slice) → dict[str, float]` — totals for revenue, holding cost, order cost, order fees, net profit.
- `aggregate_episode(run_slice) → dict[str, float]` — bundles all of the above for one episode.

`run_slice` is a `dataclass` populated from the env's per-tick `info` dicts; trivial to construct from in-memory state, no parquet I/O.

### `agents/ppo.py` (CleanRL-style single-file PPO)

Adapted from CleanRL's `ppo_continuous_action.py`:

- `Actor` and `Critic` as separate small MLPs (e.g. 64-64 hidden, tanh activation). Actor outputs a `Normal` distribution over actions; action is squashed to `[-1, 1]` via tanh at sample time.
- Standard PPO loop: rollout buffer of `n_steps × n_envs` transitions, GAE-λ advantages, K epochs of minibatch updates with PPO clip + value clip + entropy bonus + KL early stop.
- Default hyperparameters: `lr=3e-4`, `n_steps=128`, `n_epochs=10`, `n_minibatches=4`, `clip_coef=0.2`, `ent_coef=0.0`, `vf_coef=0.5`, `gae_lambda=0.95`, `gamma=0.99`, `max_grad_norm=0.5`, `target_kl=None`.
- TensorBoard scalars logged every update: `train/episodic_return`, `train/episodic_length`, `losses/value_loss`, `losses/policy_loss`, `losses/entropy`, `losses/approx_kl`, `losses/clipfrac`, `charts/learning_rate`, `charts/SPS`.

The architecture is intentionally swappable: a future model change touches only the `Actor` / `Critic` class definitions.

### `eval.py` (CRN-paired evaluation)

- `build_eval_seeds(catalog, config, n_seeds=32) → list[EpisodeSpec]` — generates a fixed list of `EpisodeSpec` objects from a deterministic eval-seed sequence (disjoint from training seeds via a config flag).
- `evaluate(rl_policy_fn, baseline_policy_factory, eval_specs) → dict[str, float]` — for each spec, runs the simulator twice (RL and baseline) on the same Scenario, aggregates business metrics per run, computes paired uplift and win-rate. Returns a flat dict ready to log to TensorBoard under `eval/*`.

Eval uses the same simulator subsystems as the env; the RL policy is invoked via a callable that takes an observation tensor and returns an action vector (so the same eval can serve PPO, SAC, or any future agent).

### `train.py`

`argparse` driver. Constructs:

1. The catalog (load via `load_or_build_world("rl_train", ...)` from `src/llm/world_builder`, or load from `data/worlds/rl_train/world.json` if cached).
2. A `gymnasium.vector.SyncVectorEnv` of 8 `RLEnv` instances (configurable).
3. A PPO agent (`agents/ppo.py`).
4. A `SummaryWriter` pointing at `runs/<experiment_name>/`.
5. The training loop, with periodic eval and checkpoint saves (`runs/<experiment_name>/checkpoints/`).

### `configs/default.py`

A dataclass `RLConfig` capturing every knob: `episode_length`, `K_active`, `K_catalog`, `capacity_dist`, `balance_dist`, `delivery_lag`, `holding_rate`, `order_fee`, PPO hyperparams, eval cadence, eval seed count, vector-env count, total env steps, TB log dir, checkpoint dir, world archetype/path.

### `pyproject.toml`

Add `[project.optional-dependencies] rl = ["torch>=2.0", "gymnasium>=0.29", "tensorboard"]`. Install via `uv sync --extra rl`. Non-RL workflows unaffected.

### ADR

`docs/adr/0004-rl-training-env.md` documenting: (a) randomised assortment per episode, (b) slot-shuffled observation encoding, (c) hidden regional market state, (d) frozen-assortment-within-episode scope. Cites the alternatives considered (set-based attention, per-product factored policy, exposing market state) and the rationale.

## Testing Decisions

### What makes a good test here

Test external behaviour, not implementation details. For pure deep modules, tests should pin the *contract* (shapes, ranges, invariants, determinism, formula correctness) without locking the internal structure. For the env, tests should round-trip through `reset()` / `step()` against shape and reward invariants — never against specific scalar values that could shift if a default changes. For the agent and CRN eval, tests should verify integration health (loss decreases, KL stays bounded, baseline-against-itself uplift is zero) rather than performance thresholds (which are noisy).

### Modules under test

- **`encoders`** — `tests/rl/test_encoders.py`:
  - Slot-shuffle round-trip: applying `decode_action(encode_action_back)` for an arbitrary permutation returns the original action dict.
  - Action decoding bounds: random action vectors in `[-1, 1]^(2K)` produce prices in `[0.5×MSRP, 1.5×MSRP]` and order quantities in `[0, free_space]`.
  - Observation shape matches `observation_dim(K)`.
  - Observation feature ranges: every feature is in `[-1, 1]` or `[0, 1]` as documented (normalised).
  - Determinism: same `(store, market, slot_perm, step)` → identical tensor.

- **`episode_sampler`** — `tests/rl/test_episode_sampler.py`:
  - Determinism: same `episode_seed` → identical `EpisodeSpec` (including capacity, balance, assortment, slot perm, world_seed).
  - Assortment coverage: across many seeds, every product in the catalog appears in some assortment.
  - Distribution sanity: sampled capacities and balances fall inside the configured ranges; mean within a tolerance.
  - Scenario validity: the returned Scenario round-trips through `Runner` for one tick without raising.

- **`metrics`** — `tests/rl/test_metrics.py`:
  - Hand-crafted slice with known sales/demand → expected `service_level`.
  - Hand-crafted slice with inventory hitting zero on K of T ticks → expected `stockout_rate`.
  - `profit_decomposition` totals add back to the balance delta computed from the same slice.
  - Empty/zero-demand slices handled without divide-by-zero.

- **`RLEnv`** — `tests/rl/test_env_smoke.py`:
  - `reset()` returns an observation matching `observation_space.shape`.
  - One episode (180 `step()` calls) runs end-to-end without raising.
  - `terminated` becomes True at step 180.
  - Reward equals per-tick balance delta (`balance_after − balance_before − any non-active-SKU contributions`), validated across one full episode.
  - Same `reset(seed=s)` from two `RLEnv` instances produces identical first-observation tensors.

- **CRN eval** — `tests/rl/test_eval_crn.py`:
  - Running `BaselinePolicy` against itself on the same `EpisodeSpec` yields zero uplift (paired difference exactly 0 across all seeds).
  - Running `RLPolicy` returning random actions versus `BaselinePolicy` on the same seeds produces a non-zero uplift (negative, almost certainly) — proves the comparison is wired.

- **PPO smoke training** — `tests/rl/test_ppo_smoke.py`:
  - Train PPO for ~1000 env steps (small `n_steps`, single env) on a fixed seed.
  - Assertions: loss is finite throughout; KL stays below a generous threshold (e.g. 0.5); training completes without raising; TensorBoard event file is written.
  - Marked `@pytest.mark.slow` so it can be excluded from the fast CI loop.

### Prior art

The new tests should mirror the structure of `tests/sim/` — small fixtures, focus on deep modules, no mocks of simulator internals, deterministic by construction. The existing `test_init_state_independent_of_policy` pattern (seeded runs compared bit-for-bit) is the model for the determinism tests. The existing `test_examples_and_cli.py` end-to-end smoke is the model for the env smoke test.

## Out of Scope

- **Catalog rotation by the RL agent** — `activate` / `deactivate` decisions remain frozen for the duration of an episode. A follow-up can layer this on as either a discrete top-level head (every K ticks) or a separate hierarchical agent.
- **Promotions head** — disabled entirely for this PRD. Re-enabling promotions either as RL-controlled actions or as baseline-side-car shocks is a follow-up.
- **Multi-store training** — one store per env instance. Multi-store fleets are expressible by composing multiple `RLEnv`s but not part of this PRD.
- **Larger K (more active SKUs) or larger catalog universe** — fixed at K=5 active from a 100-product universe. Scaling up should work mechanically; demonstrating scale is not the goal of the first cut.
- **SAC and other algorithms** — only PPO is required for this PRD. SAC is staged as a parallel file later (`agents/sac.py`), reusing the same env and eval harness.
- **Heavyweight network architectures** — MLP only. Attention / DeepSets / transformers are deferred; the actor/critic class boundary is designed so they swap in cleanly.
- **Curriculum learning** — no progressive difficulty schedule (e.g. start with K=2, expand to K=5). Single-stage training only.
- **Hyperparameter sweeps** — no automated sweep infrastructure. Hand-tuning via config edits + multiple TB runs.
- **W&B integration** — TensorBoard only.
- **Tuning of capacity/balance/lead-time defaults beyond a sanity-check smoke run** — pinned values are the starting point; revisit only if the smoke run shows the env is in a pathological regime.

## Further Notes

- **Stochastic disruption events** are kept enabled at the default rate (e.g. `event_prob=0.05`). They manifest to the agent only through their effect on realised sales (since `market_demand` / `market_supply` are hidden). This is intentional: the agent must learn to recover from supply/demand shocks.
- **Lifecycle transitions** are kept enabled; with default-stage-change probabilities tuned slowly, most episodes will see at most one stage change among the active subset. The agent has `lifecycle_stage` in its observation and can react.
- **Reward shape sanity**: per-tick profit is approximately on the order of `25 × $18 − 25 × $6 = ~$300` in steady state at base prices, so episode returns will be on the order of `$50,000` ± noise. We may want to scale reward by `1/init_balance` later for cross-store normalisation, but not for the first cut.
- **Seed discipline**: training and evaluation use disjoint `episode_seed` spaces (e.g. evaluate uses `episode_seed` in `[10_000_000, 10_000_031]`, training samples below `10_000_000`). The `episode_sampler` splits each seed into independent sub-seeds for assortment, capacity, balance, world, and slot permutation so any subset of randomisations can be frozen independently for ablations.
- **Migration to heavy models**: only the `Actor` / `Critic` class bodies in `agents/ppo.py` change. The env contract, encoder layout, and eval harness are all unchanged. Permutation invariance via the slot-shuffle means a later attention-based set encoder can be dropped in without re-training data pipelines.
- **Future addition of catalog-rotation head**: extend `encoders.decode_action` with a third head producing a small discrete activate/deactivate decision every K ticks; the simulator already supports `activate_item` / `deactivate_item` via `Store.decide()`, so no new sim machinery is needed.
- **CONTEXT.md update**: add the five new domain concepts inline when implementation lands (RL Env, RL Episode, Active subset, Slot-shuffled observation, CRN-paired eval).
- **ADR 0004** to be written alongside implementation, covering randomised-assortment / slot-shuffled-observation / hidden-market-state / frozen-within-episode decisions.
