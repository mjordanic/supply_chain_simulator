# Implementation report — `sim-base-rollout-dedupe`

- **Feature**: sim-base-rollout-dedupe
- **PRD**: [PRD.md](PRD.md)
- **Started (UTC)**: 2026-05-16T18:26:49Z
- **Last updated (UTC)**: 2026-05-16T21:50:00Z
- **Integration branch**: `hyperparameter-optimization`
- **Parallelism cap**: 3
- **Preflight assumptions**: All 9 issues had explicit `Status: ready-for-agent`. Untracked `.claude/settings.local.json` in repo root is local-only settings (not gitignored, not part of feature work) — left in place; wave-runners operate in isolated worktrees so it does not affect commits.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-lift-metrics-to-sim | Lift `RunSlice` + `aggregate_episode` to `src/sim/metrics.py` | 1 | committed | 8ec90ad | 25088fb | 2026-05-16T18:30:00Z | 2026-05-16T20:35:32Z | Salvaged from existing worktree |
| 02-lift-world-dataclass-to-sim | Lift `World` dataclass to `src/sim/world.py` | 1 | committed | 48dc571 | 09f08a3 | 2026-05-16T20:40:00Z | 2026-05-16T20:43:44Z | — |
| 03-simulation-build-world-tick-result | Introduce `Simulation` / `build_world` / `TickResult` in `src/sim/runner.py` + CRN determinism gate | 1 | committed | 5c69de3 | 1994925 | 2026-05-16T20:44:00Z | 2026-05-16T20:49:49Z | — |
| 04-lift-episode-sampler | Lift episode sampling to `src/sim/episode_sampler.py`; introduce `RLEpisodeSpec`; retire `TuningEpisodeSpec` | 2 | committed | bd47d6b | 5fe044c | 2026-05-16T21:00:00Z | 2026-05-16T21:01:17Z | Salvaged from existing worktree; cherry-pick was already integrated (empty) |
| 05-lift-world-loader | Lift world loading to `src/sim/world_loader.py`; tuning + RL train consume via thin Config-adapters | 2 | committed | f0fd6a3 | 11fd6b8 | 2026-05-16T21:05:00Z | 2026-05-16T21:15:34Z | — |
| 08-migrate-rl-env | Migrate `src/rl/env.py::RLEnv.step` to the two-phase sim API | 2 | committed | aada570 | 99ec747 | 2026-05-16T21:05:00Z | 2026-05-16T21:15:44Z | — |
| 06-migrate-tuning-rollout | Migrate `src/tuning/rollout.py::run_policy_episode` to sim primitives; amend ADR 0009 section (a) | 3 | committed | 6367035 | fc29c51 | 2026-05-16T21:20:00Z | 2026-05-16T21:30:00Z | — |
| 07-migrate-rl-eval | Migrate `src/rl/eval.py::_run_baseline` and `::_run_rl` to sim primitives (two-phase API for `_run_rl`) | 3 | committed | 9157623 | acd16a9 | 2026-05-16T21:20:00Z | 2026-05-16T21:32:00Z | — |
| 09-final-gate-context-grep-smoke | CONTEXT.md consolidation + grep audit + end-to-end smoke | 4 | committed | dacdf7e | e6ea46a | 2026-05-16T21:40:00Z | 2026-05-16T21:50:00Z | Grep audit clean; tuning smoke pass; RL train smoke pass; notebook 08 clean; 22 pre-existing failures unchanged |

## Dependency graph

```
01-lift-metrics-to-sim ─────┐
                            ├─→ 04-lift-episode-sampler ─┐
03-simulation-build-world ──┤                            ├─→ 06-migrate-tuning-rollout ─┐
                            └────────────────────────────┼─→ 07-migrate-rl-eval ────────┤
                            │                            │                              │
                            └─→ 08-migrate-rl-env ───────┼──────────────────────────────┤
                                                         │                              │
02-lift-world-dataclass-to-sim ─→ 05-lift-world-loader ──┴──────────────────────────────┤
                                                                                        ↓
                                                                  09-final-gate-context-grep-smoke
```

Edges (blocker → dependent):
- `01 → 04, 06, 07, 09`
- `02 → 05, 09`
- `03 → 04, 06, 07, 08, 09`
- `04 → 06, 07, 09`
- `05 → 09`
- `06 → 09`
- `07 → 09`
- `08 → 09`

## Wave plan

- **Wave 1** (size 3, parallel): `01-lift-metrics-to-sim`, `02-lift-world-dataclass-to-sim`, `03-simulation-build-world-tick-result`
- **Wave 2** (size 3, parallel): `04-lift-episode-sampler`, `05-lift-world-loader`, `08-migrate-rl-env`
- **Wave 3** (size 2, parallel): `06-migrate-tuning-rollout`, `07-migrate-rl-eval`
- **Wave 4** (size 1): `09-final-gate-context-grep-smoke`

## Activity log

- 2026-05-16T18:26:49Z — orchestrator: wrote initial report; integration branch `hyperparameter-optimization` at `79bc1ba`; no prior report, reconcile clean (no stray worktrees, no salvage commits).
- 2026-05-16T20:55:00Z — wave-1 runner: salvaged issue-01 commit 8ec90ad via cherry-pick → integrated SHA 25088fb.
- 2026-05-16T20:55:00Z — wave-1 runner: implemented issue-02 in worktree, cherry-pick 48dc571 → integrated SHA 09f08a3.
- 2026-05-16T20:55:00Z — wave-1 runner: implemented issue-03 in worktree, cherry-pick 5c69de3 → integrated SHA 1994925.
- 2026-05-16T20:55:00Z — wave-1 runner: removed all 3 worktrees and branches; integration branch at 1994925.
- 2026-05-16T21:01:00Z — wave-2 runner: salvaged issue-04 commit bd47d6b; cherry-pick attempt was empty (already integrated at 5fe044c); skipped via --skip; recorded integrated SHA 5fe044c.
- 2026-05-16T21:15:34Z — wave-2 runner: implemented issue-05 in worktree (issue-04 commit cherry-picked into worktree to get episode_sampler dep); cherry-pick f0fd6a3 → integrated SHA 11fd6b8.
- 2026-05-16T21:15:44Z — wave-2 runner: implemented issue-08 in worktree; cherry-pick aada570 → integrated SHA 99ec747.
- 2026-05-16T21:20:00Z — wave-2 runner: removed all 3 worktrees and branches; integration branch at 99ec747.
- 2026-05-16T21:24:00Z — wave-3 runner: created worktrees for issue-06 and issue-07 from hyperparameter-optimization.
- 2026-05-16T21:30:00Z — wave-3 runner: implemented issue-06 in worktree; cherry-pick 6367035 → integrated SHA fc29c51. rollout.py shrunk from 150→72 lines; ADR 0009(a) amended; all tuning tests pass; grep guard clean.
- 2026-05-16T21:32:00Z — wave-3 runner: implemented issue-07 in worktree; cherry-pick 9157623 → integrated SHA acd16a9. _run_rl uses two-phase API; _run_baseline uses build_world+tick(); all RL tests pass; src.tuning grep returns nothing.
- 2026-05-16T21:35:00Z — wave-3 runner: removed both worktrees and branches; integration branch at acd16a9.
- 2026-05-16T21:40:00Z — wave-4 runner: created worktree for issue-09 from hyperparameter-optimization at acd16a9.
- 2026-05-16T21:45:00Z — wave-4 runner: implemented issue-09 in worktree; CONTEXT.md consolidated (CRN-paired eval + Policy tuning study entries updated to name sim modules explicitly); grep audit clean; tuning smoke passed; RL train smoke passed; notebook 08 executed cleanly; commit dacdf7e.
- 2026-05-16T21:48:00Z — wave-4 runner: cherry-pick dacdf7e → integrated SHA e6ea46a on hyperparameter-optimization.
- 2026-05-16T21:50:00Z — wave-4 runner: removed worktree and branch; all 9 issues committed; integration branch at e6ea46a.

## Outstanding follow-ups

- Pre-existing test failures (22 total, unrelated to this feature):
  - `tests/tuning/test_search_spaces.py` — 19 failures related to search space boundary checks (pre-existing before wave 1).
  - `tests/llm/test_world_builder.py::test_build_market_domain_params_merges_handset_defaults` — 1 failure (asserts cycle_amp == 0.0 but constant is 0.0065; pre-existing before wave 1).
  - `tests/sim/test_scenario_dataframe_views.py::test_catalog_df_columns_match_world_catalog_df` — 1 failure (column mismatch: init_stage/stage_change_probs present in scenario.catalog_df but not in world.catalog_df; pre-existing).
  - `tests/llm/test_world_dataframe_inspection.py::test_catalog_df_column_set` — 1 failure (same column mismatch; pre-existing).
- RL eval smoke: `src/rl/eval.py` imports `from src.rl.agents.actor_critic import Actor` but the module is `src.rl.agents.ppo`. This is pre-existing (exists on hyperparameter-optimization before wave 1) and unrelated to the sim-base-rollout-dedupe refactor. Requires a separate fix.

## Resume instructions

If this session is killed, re-run `/implement-issues .scratch/sim-base-rollout-dedupe/` on the same `hyperparameter-optimization` branch. Phase 2 will:

1. Mark any issue whose file has moved to `issues/done/` as `committed`.
2. Mark any issue with a `<id>:`-prefixed commit on `hyperparameter-optimization` as `committed`.
3. Salvage commits from any surviving `.claude/worktrees/issue-*` worktrees that already have a `<id>:`-prefixed commit (cherry-pick into `hyperparameter-optimization`; conflicts → mark `failed`, do not auto-resolve).
4. Sweep stale worktrees with no unique commits.
5. Re-dispatch only still-`pending` / `failed` issues in the next waves.
