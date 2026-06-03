# 07 — CleanRL-style PPO agent (`agents/ppo.py`)

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

A single-file CleanRL-style PPO implementation at `src/rl/agents/ppo.py`. Adapted from CleanRL's `ppo_continuous_action.py`, but split so the algorithm/update logic lives here while the argparse driver lives in a separate `train.py` slice.

Contents:

- `Actor` and `Critic` as separate small MLPs (64-64 hidden, tanh activation). Actor outputs a `Normal` distribution; actions squashed to `[-1, 1]` via tanh at sample time.
- Rollout buffer of `n_steps × n_envs` transitions, GAE-λ advantages, K epochs of minibatch updates with PPO clip + value clip + entropy bonus + optional KL early stop.
- A training entry function (e.g. `train_ppo(envs, config, writer, eval_fn=None) -> None`) that owns the rollout loop and update loop. The `train.py` driver (slice 08) constructs `envs`, `writer`, and `eval_fn` and calls into this function.
- TensorBoard scalars logged every update: `train/episodic_return`, `train/episodic_length`, `losses/value_loss`, `losses/policy_loss`, `losses/entropy`, `losses/approx_kl`, `losses/clipfrac`, `charts/learning_rate`, `charts/SPS`.
- The `Actor` / `Critic` boundary is the only place a future heavier model (transformer, attention-over-SKUs) needs to change.

Default hyperparameters come from `RLConfig` (slice 01): `lr=3e-4`, `n_steps=128`, `n_epochs=10`, `n_minibatches=4`, `clip_coef=0.2`, `ent_coef=0.0`, `vf_coef=0.5`, `gae_lambda=0.95`, `gamma=0.99`, `max_grad_norm=0.5`, `target_kl=None`.

## Acceptance criteria

- [ ] `src/rl/agents/__init__.py` exists
- [ ] `src/rl/agents/ppo.py` exports `Actor`, `Critic`, and `train_ppo` (or equivalent named entry function)
- [ ] Reads all hyperparameters from `RLConfig` — no magic numbers buried inline
- [ ] Logs the documented TensorBoard scalars to the provided `SummaryWriter`
- [ ] `tests/rl/test_ppo_smoke.py` covers:
  - [ ] Trains PPO for ~1000 env steps (small `n_steps`, single env) on a fixed seed
  - [ ] Loss values remain finite throughout
  - [ ] KL divergence stays below a generous threshold (e.g. 0.5)
  - [ ] Training completes without raising
  - [ ] A TensorBoard event file is written to the configured log dir
- [ ] The smoke test is marked `@pytest.mark.slow` and is excluded from the default fast test run
- [ ] `uv run pytest -m slow tests/rl/test_ppo_smoke.py` passes

## Blocked by

- `.scratch/rl-policy-framework/issues/05-rl-policy-shim-and-env.md`
- `.scratch/rl-policy-framework/issues/06-crn-eval-harness.md`
