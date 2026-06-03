# 08 — `train.py` driver (vec envs, TB writer, eval cadence, checkpoints)

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

An `argparse` driver at `src/rl/train.py` that wires the pieces together for an end-to-end PPO run.

Responsibilities:

1. Load (or build via `load_or_build_world("rl_train", ...)`) the 100-product catalog from `src/llm/world_builder`.
2. Construct a `gymnasium.vector.SyncVectorEnv` of `config.n_envs` `RLEnv` instances (default 8).
3. Construct a PPO agent from `src/rl/agents/ppo.py`.
4. Create a `SummaryWriter` pointing at `runs/<experiment_name>/`.
5. Wire periodic evaluation: every `config.eval_cadence_env_steps` (~50k), call `evaluate(...)` from `src/rl/eval.py` with the frozen 32 CRN-paired held-out seeds and log results under `eval/*`.
6. Save checkpoints at `runs/<experiment_name>/checkpoints/` (e.g. every eval).
7. Accept config overrides via `argparse` (or read a config name) so sweeps are possible without editing source.

The driver should not contain training algorithm logic — it delegates to `train_ppo` from slice 07.

## Acceptance criteria

- [ ] `src/rl/train.py` runs via `uv run python -m src.rl.train --total-env-steps 5000` (or similar) without raising
- [ ] Constructs a `SyncVectorEnv` of `config.n_envs` `RLEnv` instances
- [ ] Periodically invokes `evaluate(...)` at `config.eval_cadence_env_steps` and logs results under `eval/*`
- [ ] Writes checkpoints to `runs/<experiment_name>/checkpoints/`
- [ ] Writes TensorBoard event files to `runs/<experiment_name>/`
- [ ] The argparse interface exposes at minimum: `--total-env-steps`, `--n-envs`, `--experiment-name`, `--seed`
- [ ] A short end-to-end run (a few thousand env steps) completes without error in CI and produces both a TB event file and at least one checkpoint
- [ ] `uv run pytest tests/sim/` continues to pass

## Blocked by

- `.scratch/rl-policy-framework/issues/07-ppo-agent.md`
