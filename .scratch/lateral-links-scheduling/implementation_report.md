# Implementation Report — lateral-links-scheduling

- **PRD**: [PRD.md](./PRD.md) (ADR 0018)
- **Started**: 2026-06-07T18:35:40Z
- **Last updated**: 2026-06-07T20:05:00Z
- **Parallelism cap**: 3
- **Integration branch**: `claude/eloquent-davinci-pTc8q`
- **Runner model**: default
- **Implementer model**: opus (global `--implementer-model opus`; no per-issue `Model:` overrides)
- **Preflight assumptions**: feature folder `lateral-links-scheduling` selected by mtime (most recently modified; next candidate `setup-files/` ~3 min earlier, no tie). All 5 issues carry `Status: ready-for-agent`. Prior run (18:31:24Z) died after writing the initial report — no `<id>:` commits landed, no `done/`, no worktrees; reconcile resets all to `pending`.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|----|-------|------|--------|--------------|----------------|---------|----------|-------|
| 01-type-based-graph-validation | Type-based graph validation (lateral edges legal) | 1 | committed | a2a1ee4809eba7b95083eae361d05898b2d3420a | 1bfba1d | 2026-06-07T18:40:00Z | 2026-06-07T18:47:23Z | in-place mode; integrated SHA after author-reset rebase |
| 04-min-order-aware-routing | Min-order-aware routing | 2 | committed | 2f4bcaf35262d7e2c93933843414078e7db8c62b | cad5ca8 | 2026-06-07T19:00:00Z | 2026-06-07T19:15:00Z | worktree mode; integrated SHA after author-reset rebase |
| 02-demand-pull-scheduler-unified-walk | Demand-pull topological scheduler + unified walk | 2 | committed | c1b40fa2ee4904eaac293654479937da7dbeebea | fddc18a | 2026-06-07T19:00:00Z | 2026-06-07T19:30:00Z | worktree mode; integrated SHA after author-reset rebase |
| 03-unlagged-demand-signal | Un-lagged demand signal (`observed_sales`) | 3 | committed | 5d005db23839dd6895d18eb44eb7ca9df148321a | 74a45e9 | 2026-06-07T19:32:00Z | 2026-06-07T19:43:00Z | worktree mode; integrated SHA after author-reset rebase |
| 05-rejection-and-lost-sale-log | Rejection & lost-sale log | 3 | committed | b7d74c42ffedd7bf360b2ae6e723a69a89d9cace | 1fac114 | 2026-06-07T19:32:00Z | 2026-06-07T19:48:00Z | worktree mode; integrated SHA after author-reset rebase |

## Dependency graph

```
01 ─┬─► 02 ──► 03
    │    │
    └─► 04 ─┘
         └──► 05 ◄── 02
```

- 01 — blocked by: none
- 02 — blocked by: 01
- 03 — blocked by: 02
- 04 — blocked by: 01
- 05 — blocked by: 02, 04

## Wave plan

- **Wave 1**: `01-type-based-graph-validation` (opus)
- **Wave 2**: `02-demand-pull-scheduler-unified-walk` (opus), `04-min-order-aware-routing` (opus)
- **Wave 3**: `03-unlagged-demand-signal` (opus), `05-rejection-and-lost-sale-log` (opus)

## Activity log

- 2026-06-07T18:35:40Z — orchestrator: preflight passed (branch `claude/eloquent-davinci-pTc8q`, clean tree, uv 0.8.17, permissions in place — all required allow patterns present).
- 2026-06-07T18:35:40Z — orchestrator: Phase 2 reconcile — prior run never landed commits; all 5 issues reset to `pending`.
- 2026-06-07T18:35:40Z — orchestrator: initial report written; implementer model pinned to `opus`; dispatching Wave 1.
- 2026-06-07T18:40:00Z — wave-runner: Wave 1 started; dispatching issue-implementer for 01-type-based-graph-validation (in-place mode, model=opus).
- 2026-06-07T18:47:23Z — wave-runner: 01-type-based-graph-validation committed a2a1ee4809eba7b95083eae361d05898b2d3420a on claude/eloquent-davinci-pTc8q (in-place, no cherry-pick needed). 821 tests pass, 10 skipped.
- 2026-06-07T19:00:00Z — wave-runner: Wave 2 started; dispatching issue-implementers for 02-demand-pull-scheduler-unified-walk and 04-min-order-aware-routing (worktree mode, model=opus).
- 2026-06-07T19:15:00Z — wave-runner: 04-min-order-aware-routing committed in worktree (2f4bcaf), cherry-picked clean as 77d738e. 829 tests pass, 10 skipped.
- 2026-06-07T19:30:00Z — wave-runner: 02-demand-pull-scheduler-unified-walk committed in worktree (c1b40fa), cherry-picked clean as 3a000e2. 852 tests pass, 10 skipped. Worktrees removed.
- 2026-06-07T19:32:00Z — wave-runner: Wave 3 started; dispatching issue-implementers for 03-unlagged-demand-signal and 05-rejection-and-lost-sale-log (worktree mode, model=opus).
- 2026-06-07T19:43:00Z — wave-runner: 03-unlagged-demand-signal committed in worktree (5d005db), cherry-picked clean as 22765ec. 855 tests pass, 10 skipped.
- 2026-06-07T19:48:00Z — wave-runner: 05-rejection-and-lost-sale-log committed in worktree (b7d74c4), cherry-picked clean as 0855845. 869 tests pass, 10 skipped. Worktrees removed.
- 2026-06-07T20:05:00Z — orchestrator: Phase 5 — all 3 waves dispatched, all 5 issues committed. No `.claude/worktrees/issue-*` dirs or `lateral-links-scheduling/issue-*` branches remain. Full suite on integration branch: 872 passed, 10 skipped. Report committed.
- 2026-06-07T20:10:00Z — orchestrator: author-reset rebase (committer Claude <noreply@anthropic.com>) over the 6-commit range to satisfy the verified-commit check. Integrated SHAs rewritten: 01→1bfba1d, 04→cad5ca8, 02→fddc18a, 03→74a45e9, 05→1fac114 (tree contents unchanged).

## Outstanding follow-ups

- 01-type-based-graph-validation: `_compute_bfs_levels` in `src/sim/graph.py` is now unused (BFS same-level check removed). Safe to delete in a cleanup pass; left in place per AGENTS.md §3.
- 02-demand-pull-scheduler-unified-walk: Censored-demand problem (empty inventory → observed_sales={} → rate=0) is an inherent tradeoff of demand-pull ordering. Textbook policy tests that exercise the (R,s,S) trigger now use init_stock_pct>0 to avoid cold-start censoring. Issue 03 (unlagged demand signal) addresses this by injecting the correct sales signal.

## Resume instructions

Re-run `/implement-issues .scratch/lateral-links-scheduling/` with the same branch checked out. Phase 2 reconcile rebuilds state from `git log` + `done/` + this report; already-`committed` issues are dropped from the dispatch queue.
