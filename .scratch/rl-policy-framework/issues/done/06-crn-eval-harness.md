# 06 — CRN-paired evaluation harness

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

An evaluation module `src/rl/eval.py` that runs the RL policy and `BaselinePolicy` on bit-identical `EpisodeSpec` tuples (Common Random Numbers) and reports paired uplift, win-rate, mean returns, and business KPIs.

Public surface:

- `build_eval_seeds(catalog, base_template, config, n_seeds=32) → list[EpisodeSpec]` — generates a fixed list of `EpisodeSpec` objects from a deterministic eval-seed sequence (disjoint from training seeds via `config.eval_seed_offset`).
- `evaluate(rl_policy_fn, baseline_policy_factory, eval_specs, *, config) → dict[str, float]` — for each spec, runs the simulator twice (RL and baseline) on the same Scenario, aggregates business metrics per run via `src/rl/metrics.py`, computes paired uplift (RL − baseline per seed) and win-rate (fraction of seeds where RL ≥ baseline). Returns a flat dict ready to log to TensorBoard under `eval/*` keys (e.g. `eval/rl_return`, `eval/baseline_return`, `eval/paired_uplift`, `eval/win_rate`, `eval/rl_service_level`, …).

`rl_policy_fn` is a callable `(obs_tensor) → action_vec` so the same eval can serve PPO, SAC, or any future agent without rewriting plumbing. `baseline_policy_factory` produces a fresh `BaselinePolicy` per seed (so per-policy RNG state is reset between paired runs).

CRN guarantees: both runs of a given `EpisodeSpec` must hit identical `world_seed`, capacity, balance, active subset, and slot permutation.

## Acceptance criteria

- [ ] `src/rl/eval.py` exports `build_eval_seeds` and `evaluate`
- [ ] Returned dict has flat string keys prefixed with `eval/` and `float` values
- [ ] `tests/rl/test_eval_crn.py` covers:
  - [ ] CRN self-eval — running `BaselinePolicy` against itself on the same `EpisodeSpec` yields paired difference exactly 0 across all seeds
  - [ ] Random vs baseline — running a random-action `RLPolicy` against `BaselinePolicy` produces a non-zero paired uplift (almost certainly negative), proving the comparison is wired up
  - [ ] Eval seed disjointness — `build_eval_seeds` returns seeds in the configured eval range and they do not overlap the training range
- [ ] `uv run pytest tests/rl/test_eval_crn.py` passes

## Blocked by

- `.scratch/rl-policy-framework/issues/04-metrics-module.md`
- `.scratch/rl-policy-framework/issues/05-rl-policy-shim-and-env.md`
