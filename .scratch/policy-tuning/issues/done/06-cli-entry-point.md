# 06 — CLI entry point `python -m src.tuning.study`

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add a `__main__` block to `src/tuning/study.py` that exposes the tuning tool as a CLI. The canonical invocation is:

```
uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1
```

A single CLI command runs `run_study()` followed by `confirm_top_k()` and produces all four artifact files under `runs/tuning/<study_name>/`.

### Flags

- `--policy {order_up_to|reorder_point|periodic_order_up_to|periodic_reorder}` (required) — selects which bundled trial-callback factory.
- `--trials <int>` (default `TuningConfig().n_trials`) — overrides `n_trials`.
- `--study-name <str>` (required) — used in the output path `runs/tuning/<study_name>/`.
- `--world <archetype>` (default `rl_train`) — selects the catalog + base template used to build `EpisodeSpec` objects. Use the same archetype/loader as the RL training pipeline so the tuner sees the same distribution.
- `--n-search-seeds <int>` (default `TuningConfig().n_search_seeds`).
- `--n-holdout-seeds <int>` (default `TuningConfig().n_holdout_seeds`).
- `--seed-offset <int>` (default `TuningConfig().seed_offset`).
- `--sampler-seed <int>` (default `TuningConfig().sampler_seed`).
- `--top-k <int>` (default `TuningConfig().top_k_for_holdout`).
- `--skip-holdout` (default `False`) — skip the `confirm_top_k` re-evaluation for fast smoke runs.
- `--output-dir <path>` (default `runs/tuning`) — root output directory.

The CLI uses `argparse`. The `--policy` flag maps to the four bundled factories via a small dispatch dict (e.g. `{"order_up_to": order_up_to_space, ...}`).

### Behaviour

1. Parse flags, construct `TuningConfig` with the overridden fields.
2. Load the world archetype (catalog + base template).
3. Call `run_study(...)`.
4. If `not --skip-holdout`: call `confirm_top_k(...)`.
5. Print a one-line summary to stdout: study name, n_trials, best mean normalised return, best params, output path.

Exit code 0 on success. Non-zero on argparse failure or any unhandled exception (the underlying `study.optimize` runs with `catch=()`, so trial failures bubble up).

### Tests added

In `tests/tuning/test_study.py`:

- `test_cli_smoke_full_pipeline` — Invoke `python -m src.tuning.study --policy order_up_to --trials 3 --study-name cli_smoke --n-search-seeds 2 --n-holdout-seeds 2 --top-k 2 --output-dir <tmp>` via `subprocess.run`. Assert exit code 0 and all four artifact files exist under `<tmp>/cli_smoke/`.
- `test_cli_skip_holdout` — Same invocation with `--skip-holdout`. Assert `trials.parquet`, `per_seed.parquet`, `study.json` exist but `holdout.parquet` and `holdout_summary.json` do not.
- `test_cli_policy_dispatch` — For each of the four `--policy` values, invoke with `--trials 1 --n-search-seeds 1 --episode-length 5 --skip-holdout` (smaller smoke) and assert exit code 0 and `study.json["policy_class_name"]` is the expected class name.

Note: CLI smoke tests may be slow (each runs the full eval loop). Mark them `slow` if the project has a slow-test marker convention; otherwise they run by default.

PRD user stories covered: 10.

## Acceptance criteria

- [ ] `uv run python -m src.tuning.study --policy order_up_to --trials 3 --study-name smoke --n-search-seeds 2 --n-holdout-seeds 2 --top-k 2` runs to completion with exit code 0.
- [ ] All documented flags work and override the corresponding `TuningConfig` field.
- [ ] `--policy` accepts the four bundled factory names; an unknown value exits non-zero with a clear argparse error.
- [ ] `--skip-holdout` cleanly skips `confirm_top_k` and writes only the three search-phase artifacts.
- [ ] Stdout summary line includes study name, n_trials, best mean normalised return, best params, and output directory.
- [ ] All three new tests in `tests/tuning/test_study.py` pass.
- [ ] `uv run pytest tests/tuning/` is green.

## Blocked by

- `05-confirm-top-k-and-holdout.md` — the CLI's default flow calls both `run_study` and `confirm_top_k`.
