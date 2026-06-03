# 05 — `confirm_top_k()` + `holdout.parquet` / `holdout_summary.json`

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add the top-K re-evaluation flow on top of `run_study()`. Re-evaluates the top-K trials of a completed study on a held-out CRN seed set (disjoint from the 16 search seeds) plus the published default (`OrderUpToPolicy()` with no kwargs) on the same seeds, and writes `holdout.parquet` + `holdout_summary.json`.

This is the defensive measure against TPE overfitting to the 16 search seeds: if the top-K re-evaluation on 32 disjoint seeds shows the search winner does not generalise, the headline number is the *re-evaluation* mean, not the search mean. Both are recorded for transparency.

### `confirm_top_k()` signature

In `src/tuning/study.py` alongside `run_study()`:

```python
def confirm_top_k(
    study: optuna.Study,
    policy_space: Callable[[optuna.Trial], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_dir: str,  # absolute path under runs/tuning/<study_name>/
) -> dict[str, Any]:
```

Returns the same summary dict that gets written to `holdout_summary.json`.

### Behaviour

- Select the top `tuning_config.top_k_for_holdout` completed trials by `mean_normalised_return` (descending).
- For each top-K trial, re-instantiate its policy via `optuna.trial.FixedTrial(trial.params)` passed to `policy_space`. This guarantees the same kwargs from the original trial.
- Plus the published default: `OrderUpToPolicy()` with no kwargs.
- Build `tuning_config.n_holdout_seeds` `EpisodeSpec` objects seeded from `tuning_config.holdout_seed_offset` (disjoint from `seed_offset` used by `run_study`).
- Call `evaluate_policy_normalised` on each of the K+1 policies against the same held-out spec list (CRN paired comparison).
- Write `study_dir/holdout.parquet`: one row per `(label, seed)`. `(top_k_for_holdout + 1) × n_holdout_seeds` rows total. Columns: same as `per_seed.parquet` plus `label` (one of `"tuned_rank_1"`, …, `"tuned_rank_K"`, `"default"`).
- Compute bootstrap confidence intervals (e.g. 1000 resamples, 95% CI) on the per-seed `net_profit` arrays for each label and on the paired difference `tuned_best − default`.
- Write `study_dir/holdout_summary.json` with fields: `tuned_best_label`, `tuned_best_mean_net_profit`, `tuned_best_ci_low`, `tuned_best_ci_high`, `default_mean_net_profit`, `default_ci_low`, `default_ci_high`, `headline_uplift`, `headline_uplift_ci_low`, `headline_uplift_ci_high`, `n_holdout_seeds`.

### Public API additions

Extend `src/tuning/__init__.py`:

```python
from src.tuning.study import confirm_top_k

__all__ += ["confirm_top_k"]
```

### Tests added

In `tests/tuning/test_study.py`:

- `test_confirm_top_k_smoke` — Run a 5-trial study with `n_holdout_seeds=2` and `top_k_for_holdout=2` on a tmp dir; call `confirm_top_k(study, order_up_to_space, ...)`. Assert: (a) `holdout.parquet` exists with 6 rows (`(2 + 1) × 2`); (b) the `label` column contains `{"tuned_rank_1", "tuned_rank_2", "default"}`; (c) `holdout_summary.json` exists with all documented fields; (d) `holdout_summary.json["tuned_best_label"]` is one of `"tuned_rank_1"` … `"tuned_rank_K"`.
- `test_confirm_top_k_seeds_disjoint_from_search` — After running a smoke study + holdout, assert that the `seed` values in `holdout.parquet` are disjoint from the `seed` values in `per_seed.parquet` (no overlap).
- `test_confirm_top_k_default_is_published` — Assert the `"default"`-labelled rows in `holdout.parquet` are reproducible by directly evaluating `OrderUpToPolicy()` (no kwargs) on the holdout specs.
- `test_confirm_top_k_paired_crn` — Assert that for each seed, the `tuned_rank_1` row and the `default` row use the same `capacity` and `initial_cash` (paired evaluation on identical worlds).

PRD user stories covered: 19.

## Acceptance criteria

- [ ] `confirm_top_k()` is exported from `src.tuning`.
- [ ] On a 5-trial smoke study with `top_k_for_holdout=2`, `n_holdout_seeds=2`, the call writes `holdout.parquet` (6 rows) and `holdout_summary.json` (all documented fields).
- [ ] The holdout seed set is disjoint from the search seed set (verified by `test_confirm_top_k_seeds_disjoint_from_search`).
- [ ] The `"default"` rows are produced by `OrderUpToPolicy()` with no kwargs.
- [ ] The paired CRN property holds: same seed → same `capacity` and `initial_cash` across all labels.
- [ ] All four tests in `tests/tuning/test_study.py` (new + existing from issue 04) pass.
- [ ] `uv run pytest tests/tuning/` is green.

## Blocked by

- `04-run-study-and-artifacts.md` — `confirm_top_k` reads the live `optuna.Study` returned by `run_study` and writes into the same `study_dir`.
