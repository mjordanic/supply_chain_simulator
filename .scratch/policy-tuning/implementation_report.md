# Implementation Report — `policy-tuning`

- **Feature**: policy-tuning
- **PRD**: [PRD.md](./PRD.md)
- **Integration branch**: `hyperparameter-optimization`
- **Started**: 2026-05-16T02:06:50+02:00
- **Last updated**: 2026-05-16T06:30:00+02:00
- **Parallelism cap**: 1 (in-place sequential)
- **Preflight assumptions**:
  - Untracked `.claude/settings.local.json` left as-is (harness-owned settings file, not feature work).
  - All 8 issues carry `Status: ready-for-agent`.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-reparameterise-safety-horizons | Reparameterise textbook safety horizons as fractions of delivery lag | 1 | committed | 937bec6c90145284acdcb63099fd61d9b9f696d1 | 937bec6c90145284acdcb63099fd61d9b9f696d1 | 2026-05-16T02:35:00+02:00 | 2026-05-16T03:10:00+02:00 | In-place (cap=1). 408 tests pass + 4 new ADR-0008 tests. |
| 02-tuning-module-skeleton-and-evaluator | `src/tuning/` skeleton + `TuningConfig` + `evaluate_policy_normalised` | 2 | committed | 8a5eb2fe0cff84c54c479c441ca1b9af275688d0 | 8a5eb2fe0cff84c54c479c441ca1b9af275688d0 | 2026-05-16T03:15:00+02:00 | 2026-05-16T03:45:00+02:00 | In-place (cap=1). optuna added. 16 new tuning tests pass; 246 sim + 177 rl tests pass. |
| 03-search-space-factories | Bundled search-space factories for the four textbook variants | 3 | committed | f626173 | f626173 | 2026-05-16T03:55:00+02:00 | 2026-05-16T04:15:00+02:00 | In-place (cap=1). 24 new tests pass; 40 total tuning tests green. All four factories export from src.tuning public API. |
| 04-run-study-and-artifacts | `run_study()` orchestration + `trials.parquet` / `per_seed.parquet` / `study.json` | 4 | committed | 840f09e | 840f09e | 2026-05-16T04:20:00+02:00 | 2026-05-16T04:50:00+02:00 | In-place (cap=1). 19 new study tests pass; 59 total tuning tests green. |
| 05-confirm-top-k-and-holdout | `confirm_top_k()` + `holdout.parquet` / `holdout_summary.json` | 5 | committed | 3a99d17362a21fc16909d9604f9064e17b51c512 | 3a99d17362a21fc16909d9604f9064e17b51c512 | 2026-05-16T05:10:00+02:00 | 2026-05-16T05:30:00+02:00 | In-place (cap=1). 10 new tests pass; 69 total tuning tests green. confirm_top_k exported from src.tuning. Issue file moved to done/. |
| 06-cli-entry-point | CLI entry point `python -m src.tuning.study` | 6 | committed | — | — | 2026-05-16T06:00:00+02:00 | 2026-05-16T06:30:00+02:00 | In-place (cap=1). __main__ block added to study.py; 7 new CLI tests pass (76 total tuning tests green). Issue file moved to done/. |
| 07-tune-textbook-policy-notebook | Notebook `notebooks/08-tune_textbook_policy.ipynb` | 7 | pending | — | — | — | — | — |
| 08-context-md-update-for-tuning | `CONTEXT.md` updates for the tuning tool | 8 | pending | — | — | — | — | — |

## Dependency graph

```
01 ──► 02 ──► 03 ──► 04 ──► 05 ──► 06 ──► 07 ──► 08
       ▲      ▲      ▲
       │      │      │
       └──────┴──────┘   (03 also blocked-by 01; 04 also blocked-by 02)
```

Reduced (no redundancy): linear chain 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

## Wave plan

1. Wave 1 — `01-reparameterise-safety-horizons`
2. Wave 2 — `02-tuning-module-skeleton-and-evaluator`
3. Wave 3 — `03-search-space-factories`
4. Wave 4 — `04-run-study-and-artifacts`
5. Wave 5 — `05-confirm-top-k-and-holdout`
6. Wave 6 — `06-cli-entry-point`
7. Wave 7 — `07-tune-textbook-policy-notebook`
8. Wave 8 — `08-context-md-update-for-tuning`

## Activity log

- 2026-05-16T02:06:50+02:00 — Orchestrator initialised report. HEAD = 7666ebc. Cap=1 (in-place). 8 waves of 1 issue each.
- 2026-05-16T02:34:45+02:00 — Resume reconcile (orchestrator). HEAD = a9574a0. `done/` empty, no `<id>:`-prefixed commits, no worktrees. All 8 issues remain `pending`. Dispatching wave 1.
- 2026-05-16T03:10:00+02:00 — Wave 1 complete. 01-reparameterise-safety-horizons committed at 937bec6 (in-place). 408 sim+rl tests pass; 4 new ADR-0008 tests added. Issue file moved to done/.
- 2026-05-16T03:15:00+02:00 — Wave 2 started. Dispatching 02-tuning-module-skeleton-and-evaluator in-place.
- 2026-05-16T03:45:00+02:00 — Wave 2 complete. 02-tuning-module-skeleton-and-evaluator committed at 8a5eb2f (in-place). optuna added; src/tuning/{__init__,config,evaluator}.py + tests/tuning/{test_config,test_evaluator}.py created. 16 new tuning tests pass.
- 2026-05-16T03:55:00+02:00 — Wave 3 started. Dispatching 03-search-space-factories in-place.
- 2026-05-16T04:15:00+02:00 — Wave 3 complete. 03-search-space-factories committed at f626173 (in-place). src/tuning/search_spaces.py + tests/tuning/test_search_spaces.py created; src/tuning/__init__.py updated. 24 new tests pass (40 total tuning tests green). Issue file moved to done/.
- 2026-05-16T04:20:00+02:00 — Wave 4 started. Dispatching 04-run-study-and-artifacts in-place.
- 2026-05-16T04:50:00+02:00 — Wave 4 complete. 04-run-study-and-artifacts committed (in-place). src/tuning/study.py + tests/tuning/test_study.py created; src/tuning/__init__.py updated with run_study export. 19 new tests pass (59 total tuning tests green). Issue file moved to done/.
- 2026-05-16T05:10:00+02:00 — Wave 5 started. Dispatching 05-confirm-top-k-and-holdout in-place.
- 2026-05-16T05:30:00+02:00 — Wave 5 complete. 05-confirm-top-k-and-holdout committed at 3a99d17 (in-place). confirm_top_k() + _bootstrap_ci() added to src/tuning/study.py; src/tuning/__init__.py updated. 10 new tests pass (69 total tuning tests green). Issue file moved to done/.
- 2026-05-16T06:00:00+02:00 — Wave 6 started. Dispatching 06-cli-entry-point in-place.
- 2026-05-16T06:30:00+02:00 — Wave 6 complete. 06-cli-entry-point committed (in-place). __main__ block + _cli_main() added to src/tuning/study.py; all flags work including --skip-holdout and --episode-length. 7 new CLI tests pass (76 total tuning tests green). Issue file moved to done/.

## Outstanding follow-ups

_None yet._

## Resume instructions

Re-run `/implement-issues .scratch/policy-tuning` from the `hyperparameter-optimization` branch. Phase 2 (resume reconcile) will:

1. Re-read this status table.
2. Mark any issue whose file is in `.scratch/policy-tuning/issues/done/` as `committed`.
3. Mark any commit with subject prefix `<id>:` on `hyperparameter-optimization` as `committed`.
4. Re-queue any issue stuck in `in-progress` (prior subagent died) as `pending`.
5. Sweep `.claude/worktrees/issue-*` for orphans / salvage candidates.
6. Dispatch the next wave with at least one pending issue.
