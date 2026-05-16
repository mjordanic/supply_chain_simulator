# 04 — `run_study()` orchestration + `trials.parquet` / `per_seed.parquet` / `study.json`

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add `src/tuning/study.py` with the `run_study()` entry point that runs an Optuna study against `evaluate_policy_normalised` and persists the trial-level artifacts to disk. This is the orchestration layer on top of the deep `evaluator.py`.

### `run_study()` signature

```python
def run_study(
    policy_space: Callable[[optuna.Trial], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_name: str,
    output_dir: str = "runs/tuning",
) -> optuna.Study:
```

Returns the live in-memory `optuna.Study` so a caller (notebook demo cell) can inspect it without re-reading from disk.

### Behaviour

- Build `eval_specs: list[EpisodeSpec]` once at study start: `tuning_config.n_search_seeds` specs sampled via the existing episode-sampler against `tuning_config.seed_offset`. The same spec list is reused for every trial (CRN at the trial level).
- `optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tuning_config.sampler_seed), storage=None)`. No pruner. `catch=()` so trial exceptions surface loudly.
- Trial objective: instantiate a policy via `policy_space(trial)`, call `evaluate_policy_normalised`, stash the full KPI dict in `trial.user_attrs`, return `mean_normalised_return`.
- Sequential trials (`n_jobs=1`). No trial timeout.
- After `study.optimize(n_trials=tuning_config.n_trials)` completes, write three artifact files under `{output_dir}/{study_name}/` (created if absent):

#### `trials.parquet` — one row per trial

Columns: `trial_id`, `state`, `datetime_start`, `datetime_complete`, `duration_s`, `mean_normalised_return`, `mean_net_profit`, `mean_service_level`, `mean_stockout_rate`, `mean_inventory_turnover`, `mean_revenue`, `mean_mean_price_pct_of_msrp`, plus one column per search-space parameter (column names match the `trial.suggest_*` names).

#### `per_seed.parquet` — one row per (trial, seed)

Long format. Columns: `trial_id`, `seed`, `capacity`, `initial_cash`, `net_profit`, `normalised_return`, `service_level`, `stockout_rate`, `inventory_turnover`, `revenue`, `mean_price_pct_of_msrp`. Row count: `n_trials × n_search_seeds`.

#### `study.json` — self-describing study metadata

Fields: `study_name`, `n_trials`, `n_search_seeds`, `n_holdout_seeds`, `episode_length`, `seed_offset`, `holdout_seed_offset`, `sampler_seed`, `search_space` (dict of param name → `{type, low, high}`), `best_trial_id`, `best_params`, `best_value`, `started_at`, `finished_at`, `git_sha` (subprocess `git rev-parse HEAD`; empty string if unavailable), `policy_class_name` (string introspected from the factory).

### Public API additions

Extend `src/tuning/__init__.py`:

```python
from src.tuning.study import run_study

__all__ += ["run_study"]
```

`confirm_top_k` and the CLI are added in issues 05 and 06.

### Tests added

In `tests/tuning/test_study.py`:

- `test_run_study_smoke_3_trials` — Run `run_study(order_up_to_space, …, tuning_config=TuningConfig(n_trials=3, n_search_seeds=2, episode_length=10), study_name="smoke", output_dir=tmp_path)`. Assert: (a) returns an `optuna.Study` with `len(study.trials) == 3`; (b) `tmp_path/smoke/trials.parquet` exists with 3 rows and all documented columns; (c) `tmp_path/smoke/per_seed.parquet` exists with 6 rows (3 trials × 2 seeds) and all documented columns; (d) `tmp_path/smoke/study.json` exists with all documented fields.
- `test_run_study_artifacts_load_with_pandas_only` — After running the smoke study, re-open the three files using only `pandas` and `json` (no `optuna` import in the test body). Assert schemas are usable without Optuna at read time.
- `test_run_study_crn_at_trial_level` — Run two 2-trial studies with `policy_space` returning a fixed `OrderUpToPolicy()` regardless of trial params. Assert the `per_seed.parquet` rows for the two trials are bit-identical (same eval_specs reused, same policy → same trajectory).
- `test_run_study_writes_git_sha` — Assert `study.json["git_sha"]` is either a 40-char hex string or the empty string.

PRD user stories covered: 2, 10 (partial — CLI is in 06), 11, 18.

## Acceptance criteria

- [ ] `run_study()` is exported from `src.tuning`.
- [ ] A 3-trial smoke study runs to completion in under 30 seconds and writes all three artifact files.
- [ ] `trials.parquet`, `per_seed.parquet`, `study.json` schemas match the documented columns/fields exactly.
- [ ] `pd.read_parquet(.../trials.parquet)` and `pd.read_parquet(.../per_seed.parquet)` load cleanly without `optuna` in the import path.
- [ ] The study uses `storage=None` (in-memory; no SQLite file is created in the output dir).
- [ ] `optuna.samplers.TPESampler` is constructed with `seed=tuning_config.sampler_seed`.
- [ ] Trial exceptions surface (no silent `None` outcomes; `catch=()` on `study.optimize`).
- [ ] All four tests in `tests/tuning/test_study.py` pass.
- [ ] `uv run pytest tests/tuning/` is green.

## Blocked by

- `02-tuning-module-skeleton-and-evaluator.md` — uses `TuningConfig` and `evaluate_policy_normalised`.
- `03-search-space-factories.md` — `order_up_to_space` is the canonical test factory.
