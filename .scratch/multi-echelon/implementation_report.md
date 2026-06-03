# multi-echelon — Implementation Report

- **PRD**: [./PRD.md](./PRD.md)
- **Started**: 2026-05-28T13:25:51Z (original); resumed 2026-05-28T15:55:28Z
- **Last updated**: 2026-05-28T22:30:00Z
- **Integration branch**: `multi-echelon`
- **Parallelism cap**: 3
- **Assumptions**: all 8 remaining issue files carry `Status: ready-for-agent`. Pre-existing working-tree changes stashed at preflight as `implement-issues preflight stash`.

## Status

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|---|---|---|---|---|---|---|---|---|
| 01-graph-deep-module | Graph deep module + DAG validation | 0 | committed | 1323f4f | c49926e | — | — | prior run; in `done/` |
| 02-central-table-deep-module | CentralTable deep module | 0 | committed | 1c39d12 | e501bdc | — | — | prior run; in `done/` |
| 03-node-hierarchy-scenario-extension | Node hierarchy + Scenario extension | 0 | committed | 8ff08df | 8ff08df | — | — | prior run; in `done/` |
| 04-adrs-0011-0016-proposed | ADRs 0011–0016 (Proposed status) | 1 | committed | bbe0c6e | 1f5877e | — | — | prior run; commit-prefix match; issue file not moved to `done/` |
| 05-graph-simulation-runner-chain-engine | GraphSimulation + GraphRunner + build_graph_world | 1 | committed | e4a904d | 964ef34 | — | — | prior run; commit-prefix match; issue file not moved to `done/` |
| 06-phase1-policies-chain-scenario | Phase-1 policies + chain smoke scenario + chain tests | 2a | committed | 1b3def7 | cf95882 | — | — | prior run; commit-prefix match; issue file not moved to `done/` |
| 07-allocation-execute-buy-full | allocation.execute_buy full + clamp paths | 2a | committed | 60ca493 | 5cefcf5 | — | — | prior run; commit-prefix match; issue file not moved to `done/` |
| 08-allocation-rng-shuffle-ema | allocation_rng + shuffle_buyers + EMA | 0 | committed | 6826f16 | 6826f16 | — | — | prior run; commit-prefix match; no issue file present |
| 09-multisupplier-textbook-policy-contention-scenario | MultiSupplierTextbookPolicy + contention scenario | 0 | committed | a751ec4 | a751ec4 | — | — | prior run; commit-prefix match; no issue file present |
| 10-lifecycle-freshness-sink-path | Lifecycle/freshness composition in DemandSinkNode | 0 | committed | 806e5fc | 806e5fc | — | — | prior run; commit-prefix match; no issue file present |
| 11-retire-legacy-store-engine | Retire legacy Store engine: delete, rename, migrate scenarios | 1 | committed | ec453c4 | ec453c4 | 2026-05-28T16:00:00Z | 2026-05-28T17:45:00Z | in-place mode; 472 passed, 3 skipped; uv run python main.py scenarios/example_homogeneous.py green |
| 12-rl-re-integration | RL re-integration: episode sampler, env, encoders, eval CRN | 2a | committed | 8e89ac8 | 8e89ac8 | 2026-05-28T18:00:00Z | 2026-05-28T19:30:00Z | in-place mode; 164 passed, 2 skipped; training smoke 500 steps green; full 5000-step smoke deferred (see follow-ups) |
| 13-tuning-re-integration | Tuning re-integration: rollout, search spaces, evaluator | 2b | committed | 3eaecff | 3eaecff | 2026-05-28T20:00:00Z | 2026-05-28T21:00:00Z | in-place mode; 80 passed; smoke study 5 trials green; store.py/store_initializer.py deleted |
| 14-notebooks-context-adr-ratification | Notebooks 01–03 + 08 rewritten, CONTEXT.md updated, ADRs 0011–0016 ratified | 3 | committed | 8c99e04 | 8c99e04 | 2026-05-28T22:00:00Z | 2026-05-28T22:30:00Z | in-place mode; 847 passed 13 skipped; all 4 smoke commands green; ADRs Proposed→Accepted |

## Dependency graph

```
01 ──┐
02 ──┴─► 03 ──┬─► 04 ✅
              └─► 05 ✅ ──┬─► 06 ✅ ──┐
                          └─► 07 ✅ ──┴─► 08 ✅ ──► 09 ✅ ──► 10 ✅ ──► 11 ✅ ──┬─► 12 ✅ ──┐
                                                                             └─► 13 ✅ ──┴─► 14 ✅
```

✅ = already committed prior to this run (reconciled).

## Wave plan

1. **Wave 1** (cap 3): `04` (reconciled), `05` (reconciled), `11` → effective: `11` ✅
2. **Wave 2a** (cap 3): `06` (reconciled), `07` (reconciled), `12` → effective: `12` ✅
3. **Wave 2b** (cap 1): `13` → effective: `13` ✅
4. **Wave 3** (cap 1): `14` → effective: `14` ✅

## Activity log

- `2026-05-28T15:55:28Z` orchestrator: preflight passed; integration branch `multi-echelon` at `806e5fc`; dirty working tree stashed (`implement-issues preflight stash`); `uv` 0.11.14 verified; permission audit clean.
- `2026-05-28T15:55:28Z` orchestrator: Phase 1 — parsed 8 ready-for-agent issue files (04, 05, 06, 07, 11, 12, 13, 14); user approved cap=3 plan.
- `2026-05-28T15:55:28Z` orchestrator: Phase 2 reconcile — issues 04, 05, 06, 07 matched commit prefixes on `multi-echelon` (1f5877e, 964ef34, cf95882, 5cefcf5) → marked committed. No stale worktrees. Dispatch queue reduced to {11, 12, 13, 14}.
- `2026-05-28T17:45:00Z` wave-1 runner: dispatched issue 11 in-place (no worktree, single-issue wave). Implemented: GraphSimulation→Simulation rename, GraphRunner→Runner rename, build_graph_world→build_world, HeuristicPolicy/RLPolicy tombstones, Market.sample_demand deleted, world_to_graph added, all scenarios rewritten on graph topology, 3 test files deleted, 7 test files rewritten. `uv run pytest tests/sim` → 472 passed 3 skipped. `uv run python main.py scenarios/example_homogeneous.py` → green. Committed at ec453c4 on multi-echelon.
- `2026-05-28T19:30:00Z` wave-2a runner: dispatched issue 12 in-place (no worktree, single-issue wave). Implemented: RLIntermediatePolicy(IntermediatePolicy) in policy.py; episode_sampler rebuilt as 3-node graph (FactoryNode(s)→IntermediateNode("S")→DemandSinkNode(s)); env.py reset/step rewritten on graph engine two-phase tick API; encoders.py N_PER_SKU bumped 14→18 with 4 central-table snapshot features; eval.py CRN tuple expanded, net_profit = cumulative S.cash_delta; Simulation.tick_world()/tick_decide_and_settle() two-phase API added; build_world deep-copies nodes for multi-call safety. `uv run pytest tests/rl` → 164 passed 2 skipped. `uv run pytest tests/sim` → 472 passed 3 skipped (unchanged). Training smoke 500 steps green. Committed at 8e89ac8 on multi-echelon.
- `2026-05-28T21:00:00Z` wave-2b runner: dispatched issue 13 in-place (no worktree, single-issue wave). Implemented: rollout.py migrated to graph engine — _build_graph_scenario converts EpisodeSpec to FactoryNode(s)→IntermediateNode("S")→DemandSinkNode(s); _TrackingDemandSinkNode subclass records demand_target per tick; _record_active_subset collects 9-field set from IntermediateNode; search_spaces.py adds per_supplier_min_order_floor int[0,10] and routing_strategy Categorical; _fill_rate_weighted_strategy added to policy.py; evaluator.py docstring updated with allocation CRN stream; study.py probe_params updated; test_search_spaces.py rewritten; store.py + store_initializer.py deleted (no remaining imports). `uv run pytest tests/tuning` → 80 passed. Smoke study 5 trials → runs/tuning/smoke/{trials.parquet,per_seed.parquet,study.json} green. Committed at 3eaecff on multi-echelon.
- `2026-05-28T22:30:00Z` wave-3 runner: dispatched issue 14 in-place (no worktree, single-issue wave). Implemented: tests/test_examples_and_cli.py rewritten for graph engine (7 failures fixed + 2 new CRN tests; 17 total pass); notebooks/01 rewritten as graph quickstart (3-node chain); notebooks/02 rewritten with world_to_graph + nodes_df/edges_df; notebooks/03 rewritten as full graph scenario inspection; notebooks/08 rewritten for MultiSupplierTextbookPolicy with new tunables; CONTEXT.md rewritten for Store→Node family + new Graph/EdgeSpec/CentralTable/Allocation/Phase cascade entries + ADR 0011-0016 in decisions list; ADRs 0011-0016 promoted Proposed→Accepted. `uv run pytest` → 847 passed 13 skipped. All 4 smoke commands green (homogeneous, two_factories_two_shops, RL 5000 steps, tuning 5 trials). Committed at 8c99e04 on multi-echelon.
- `2026-05-28T22:35:00Z` orchestrator: Phase 5 final cleanup — `git worktree list` shows only the main checkout; no `.claude/worktrees/` or `/tmp/worktrees/` directories; no straggler `issue-*` branches; working tree clean. Run complete.

## Outstanding follow-ups

- Prior implementers of 04/05/06/07 did not move issue files to `done/` — files remain under `issues/`. The wave-runner for the issues currently being dispatched does not own those files; consider moving them by hand or accepting the discrepancy.
- **Issue 12 follow-up**: Full 5000-step RL training smoke (`uv run python -m src.rl.train --total-env-steps 5000 --no-eval`) was verified green in wave 3. The above-baseline check (RL reward > OrderUpToPolicy baseline on 32-seed held-out set) was not verified — this requires a longer training run.
- **Issue 12 follow-up**: The `RLEpisodeSpec` is designed so that nodes are mutable and shared across `build_world` calls. `build_world` now deep-copies nodes to prevent mutation aliasing. This is an O(n_nodes) copy on each `build_world` call; if performance is a concern for vector envs, a copy-on-write or explicit snapshot API would be cleaner.
- **Issue 13 note**: `NoopPolicy.decide` now accepts `*args` to handle both single-arg (Store) and two-arg (graph IntermediateNode) calling conventions. This is a minor widening of the API.
- **Issue 14 note**: The RL train CLI accepts `--total-env-steps` (not `--steps` as stated in the acceptance criteria). The 5000-step smoke ran successfully using `--total-env-steps 5000 --no-eval`. The `--config` flag does not exist; config is handled via individual CLI flags.
- **Issue 14 note**: The tuning study CLI requires `--policy` and `--study-name` (not `--name` as stated in the acceptance criteria). Smoke ran successfully using `--policy order_up_to --study-name smoke --trials 5 --skip-holdout`.

## Resume instructions

If this run is interrupted, re-run `/implement-issues .scratch/multi-echelon/` from the same checkout of the `multi-echelon` branch. Phase 2 reconcile cross-references this report with `done/`, `git log`, and `git worktree list` to recover state before dispatching remaining waves.
