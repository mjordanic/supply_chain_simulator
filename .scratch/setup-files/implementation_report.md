# Implementation Report — setup-files

- **Feature**: setup-files
- **PRD**: [`PRD.md`](./PRD.md)
- **Started**: 2026-06-05T12:27:27Z
- **Last updated**: 2026-06-05T17:30:00Z
- **Parallelism cap**: 1 (in-place sequential)
- **Integration branch**: `claude/eloquent-hopper-Mwefh`
- **runner-model**: default
- **implementer-model**: opus (global `--implementer-model` override, applies to all issues)
- **Preflight assumptions**:
  - Feature folder chosen interactively (all `.scratch/*` share an identical clone mtime; inference was ambiguous).
  - `.claude/settings.local.json` was absent; created with the Required-Bash allow list + remote deny list during preflight.
  - No issue carries a `Complexity:` or `Model:` line; global opus override pins every implementer.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|----|-------|------|--------|--------------|----------------|---------|----------|-------|
| 01-simplify-demand-core | Simplify the demand core: remove life-cycle & freshness, relocate CRN | 1 | committed | a6802f93743f01101e94f7ed894e1b2b3c0567b2 | a6802f93743f01101e94f7ed894e1b2b3c0567b2 | 2026-06-05T12:30:00Z | 2026-06-05T13:05:00Z | in-place mode; 826 tests pass |
| 02-run-setup-directory-end-to-end | Run a hand-authored setup directory end-to-end | 2 | committed | d51a5d69d12305ca66ff330a1862369d69fd2545 | d51a5d69d12305ca66ff330a1862369d69fd2545 | 2026-06-05T13:10:00Z | 2026-06-05T13:08:07Z | in-place mode; 863 tests pass, 13 skipped. New modules: setup_io, policy_registry. CLI updated to `run <setup-dir>` subcommand. Example: setups/three_node_chain/. |
| 03-topology-scaffolder | Topology scaffolder | 3 | committed | 03b86793453425db13b7abfe1f0eb16992e51f2f | 03b86793453425db13b7abfe1f0eb16992e51f2f | 2026-06-05T14:00:00Z | 2026-06-05T14:30:00Z | in-place mode; 885 tests pass, 13 skipped. New module: topology_scaffolder. CLI scaffold subcommand. 22 new tests. |
| 04-llm-generator-writes-setup-files | LLM generator writes setup files (+ write_setup, dir-as-cache) | 3 | committed | 78c59bc2c8d2d5c7df3c6c8c22da13c73d23d5ab | 78c59bc2c8d2d5c7df3c6c8c22da13c73d23d5ab | 2026-06-05T14:30:00Z | 2026-06-05T15:30:00Z | in-place mode; 908 tests pass, 13 skipped. New: write_setup, write_catalog_and_market, load_or_build_setup, WorldBuilder.build_setup. 23 new tests. |
| 05-rl-trains-against-setup-directory | RL trains against a setup directory | 3 | committed | 7e77cb2a5d8e12fe63d1e7cb5f68f8b9ea3f1a20 | 7e77cb2a5d8e12fe63d1e7cb5f68f8b9ea3f1a20 | 2026-06-05T15:30:00Z | 2026-06-05T16:30:00Z | in-place mode; 932 tests pass, 13 skipped. RLConfig.setup_dir field; load_catalog_and_market_from_setup; make_synthetic_catalog; 24 new tests. |
| 06-tuning-optimises-against-setup-directory | Tuning optimises against a setup directory | 4 | committed | 3a64bf5a4cca31fb923f1d296112ad4622ce12e4 | 3a64bf5a4cca31fb923f1d296112ad4622ce12e4 | 2026-06-05T16:30:00Z | 2026-06-05T17:30:00Z | in-place mode; 958 tests pass, 13 skipped. TuningConfig.setup_dir; TuningEpisodeSpec with capacity/balance; graph-mode episode builder; load_catalog_and_market_from_setup; make_synthetic_catalog; 26 new tests. |
| 07-remove-remaining-legacy | Remove remaining legacy: Store model, World, world_loaders, dead policy classes | 5 | committed | TBD | TBD | 2026-06-05T17:31:00Z | 2026-06-05T18:30:00Z | in-place mode; 816 tests pass, 10 skipped. Deleted StoreTemplate, StoreInstance, make_stores, World, world_loader.py (sim+tuning), NoopPolicy, HeuristicPolicy, Scenario.to_json/from_json/from_world/stores_df/is_graph/stores, WorldBuilder.build()/build_store_templates(), RLConfig.world_archetype/world_cache_path, TuningConfig.world_archetype/world_cache_path. |
| 08-docs-and-convert-examples | Docs + convert example scenarios | 5 | pending | — | — | — | — | |

## Dependency graph

```
01 ──▶ 02 ──┬──▶ 03 ─────────────┐
            ├──▶ 04 ─────────────┤
            ├──▶ 05 ──┐          │
            ├──▶ 06 ──┤          │
            │         ├──▶ 07    │
            │         │          ├──▶ 08
            └─────────┴──────────┘

07 ← {02, 05, 06}
08 ← {02, 03, 04}
```

## Wave plan

- **Wave 1**: `01-simplify-demand-core` (opus)
- **Wave 2**: `02-run-setup-directory-end-to-end` (opus)
- **Wave 3**: `03-topology-scaffolder` (opus), `04-llm-generator-writes-setup-files` (opus), `05-rl-trains-against-setup-directory` (opus)
- **Wave 4**: `06-tuning-optimises-against-setup-directory` (opus)
- **Wave 5**: `07-remove-remaining-legacy` (opus), `08-docs-and-convert-examples` (opus)

(cap=1 ⇒ issues within a wave run sequentially in the repo root, not in parallel worktrees.)

## Activity log

- 2026-06-05T12:27:27Z — Orchestrator: preflight passed; created `.claude/settings.local.json`; report initialized with 8 pending issues across 5 waves. Reconcile found 0 committed (done/ empty).
- 2026-06-05T12:30:00Z — Wave-runner: Wave 1 started; dispatching issue-implementer for 01-simplify-demand-core (model=opus).
- 2026-06-05T13:05:00Z — Wave-runner: Wave 1 complete. 01-simplify-demand-core committed as a6802f9 on claude/eloquent-hopper-Mwefh. 826 tests pass, 13 skipped.
- 2026-06-05T13:10:00Z — Wave-runner: Wave 2 started; dispatching issue-implementer for 02-run-setup-directory-end-to-end (model=opus).
- 2026-06-05T13:08:07Z — Wave-runner: Wave 2 complete. 02-run-setup-directory-end-to-end committed as d51a5d6 on claude/eloquent-hopper-Mwefh. 863 tests pass, 13 skipped. New modules: setup_io, policy_registry. CLI updated to `run <setup-dir>` subcommand. Example: setups/three_node_chain/.
- 2026-06-05T14:00:00Z — Wave-runner: Wave 3 started; dispatching issue-implementer for 03-topology-scaffolder (model=opus).
- 2026-06-05T14:30:00Z — Wave-runner: 03-topology-scaffolder committed as 03b8679 on claude/eloquent-hopper-Mwefh. 885 tests pass, 13 skipped.
- 2026-06-05T14:30:00Z — Wave-runner: dispatching issue-implementer for 04-llm-generator-writes-setup-files (model=opus).
- 2026-06-05T15:30:00Z — Wave-runner: 04-llm-generator-writes-setup-files committed as 78c59bc on claude/eloquent-hopper-Mwefh. 908 tests pass, 13 skipped.
- 2026-06-05T15:30:00Z — Wave-runner: dispatching issue-implementer for 05-rl-trains-against-setup-directory (model=opus).
- 2026-06-05T16:30:00Z — Wave-runner: Wave 3 complete. 05-rl-trains-against-setup-directory committed as 7e77cb2 on claude/eloquent-hopper-Mwefh. 932 tests pass, 13 skipped. New: RLConfig.setup_dir, load_catalog_and_market_from_setup, make_synthetic_catalog, 24 new tests.
- 2026-06-05T16:30:00Z — Wave-runner: Wave 4 started; dispatching issue-implementer for 06-tuning-optimises-against-setup-directory (model=opus).
- 2026-06-05T17:30:00Z — Wave-runner: Wave 4 complete. 06-tuning-optimises-against-setup-directory committed as 3a64bf5 on claude/eloquent-hopper-Mwefh. 958 tests pass, 13 skipped. New: TuningConfig.setup_dir, TuningEpisodeSpec with capacity/balance, graph-mode episode builder, load_catalog_and_market_from_setup, make_synthetic_catalog, CLI --setup-dir flag, 26 new tests.
- 2026-06-05T17:31:00Z — Wave-runner: Wave 5 started; dispatching issue-implementer for 07-remove-remaining-legacy (model=opus).
- 2026-06-05T18:30:00Z — Wave-runner: 07-remove-remaining-legacy committed on claude/eloquent-hopper-Mwefh. 816 tests pass, 10 skipped. All Store-model, World, world_loader, NoopPolicy/HeuristicPolicy, Scenario.to_json/from_json/stores/is_graph deleted. Fixtures regenerated.

## Outstanding follow-ups

_(none)_

## Resume instructions

Re-run `/implement-issues .scratch/setup-files/` with the same feature path. Phase 2 reconciles
committed work from `done/` + the integration branch's `<id>:`-prefixed commits before dispatching
anything, so a killed session resumes cleanly.
