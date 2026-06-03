# 01 — RL dependencies and `RLConfig` dataclass

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

Set up the dependency and configuration plumbing the rest of the RL package will sit on. Add an `[rl]` optional-dependency group to `pyproject.toml` covering `torch`, `gymnasium`, and `tensorboard`, and create a `src/rl/configs/default.py` module that exports an `RLConfig` dataclass capturing every knob the agent, env, sampler, and eval will read.

`RLConfig` should hold:

- Episode shape — `episode_length`, `K_active`, `K_catalog`
- Episode randomisation — `capacity_dist` (default `Uniform(150, 400)`), `balance_dist` (default `Uniform(15000, 40000)`)
- Simulator knobs — `delivery_lag` (3), `holding_rate` (0.01), `order_fee` (50)
- PPO hyperparameters — `lr=3e-4`, `n_steps=128`, `n_epochs=10`, `n_minibatches=4`, `clip_coef=0.2`, `ent_coef=0.0`, `vf_coef=0.5`, `gae_lambda=0.95`, `gamma=0.99`, `max_grad_norm=0.5`, `target_kl=None`
- Training driver — `n_envs=8`, `total_env_steps=1_000_000`
- Eval — `eval_cadence_env_steps=50_000`, `n_eval_seeds=32`, `eval_seed_offset=10_000_000`
- Output — `tb_log_dir`, `checkpoint_dir`, `experiment_name`
- World — `world_archetype`, `world_cache_path`

Defaults match the PRD values exactly. The dataclass should be importable and instantiable with no arguments.

`uv sync --extra rl` must install the new deps. Non-`rl` workflows (existing simulator tests) must not be affected.

## Acceptance criteria

- [ ] `pyproject.toml` has `[project.optional-dependencies]` with an `rl` group containing `torch>=2.0`, `gymnasium>=0.29`, `tensorboard`
- [ ] `uv sync --extra rl` completes without error
- [ ] `uv run pytest tests/sim/` continues to pass (no regressions for non-RL workflows)
- [ ] `src/rl/__init__.py` exists (package marker)
- [ ] `src/rl/configs/__init__.py` exists
- [ ] `src/rl/configs/default.py` exports `RLConfig` with all fields and defaults listed above
- [ ] `RLConfig()` instantiates with no arguments and yields PRD-spec defaults
- [ ] `RLConfig` is a frozen dataclass (or otherwise immutable) so configs can be safely shared across vector envs

## Blocked by

None — can start immediately.
