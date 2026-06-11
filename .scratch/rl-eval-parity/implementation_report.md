# Implementation Report — RL eval parity & attach-anywhere RL policy

- **Feature**: `.scratch/rl-eval-parity/`
- **PRD**: [PRD.md](./PRD.md)
- **Started**: 2026-06-11T11:59:15Z
- **Last updated**: 2026-06-11T14:45:00Z
- **Parallelism cap**: 3
- **Integration branch**: `RL-rewrite` (base SHA `7c5b41f`)
- **runner-model**: default (frontmatter)
- **implementer-model**: fable (global `--implementer-model`; all issues, no per-issue `Complexity:`/`Model:` overrides)
- **Preflight assumptions**: none — all 6 issues carry `Status: ready-for-agent`; permission audit passed with no changes; ADR sequence ends at 0021 so `0022` is the next number; both target notebooks tracked (not git-ignored).

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|----|-------|------|--------|--------------|----------------|---------|----------|-------|
| 01-extract-shared-arbitration-helper | Extract shared arbitration helper | 1 | committed | 4ea813b | 4ea813b | 2026-06-11T12:00:00Z | 2026-06-11T12:05:00Z | in-place mode; 289 tests pass |
| 02-rlnodepolicy | RLNodePolicy — Runner-compatible RL policy | 2 | committed | 86bda7d | 86bda7d | 2026-06-11T12:10:00Z | 2026-06-11T12:20:00Z | in-place mode |
| 03-eval-harness-rebuild | Rebuild offline eval harness on RLNodePolicy + Runner | 3 | committed | d361ad7 | d361ad7 | 2026-06-11T12:25:00Z | 2026-06-11T12:45:00Z | in-place mode; 1122 tests pass |
| 04-notebook-06a-rewrite | Notebook 06a — fixed-path re-run + attach demo | 4 | committed | 06fb31a | dd7e1b3 | 2026-06-11T14:00:00Z | 2026-06-11T14:38:00Z | worktree mode; removed stale Arbiter caveats; added section 8 attach demo; re-executed |
| 05-notebook-06-update | Notebook 06 — prose update + re-execution | 4 | committed | a41b52e | f53a59d | 2026-06-11T14:00:00Z | 2026-06-11T14:32:00Z | worktree mode; updated eval prose + "Where to next"; re-executed (tiny demo agent retrained) |
| 06-docs-adr-0022 | Docs — ADR 0022, RL README, CONTEXT.md | 4 | committed | 0aa7d67 | 8b887dc | 2026-06-11T14:00:00Z | 2026-06-11T14:29:00Z | worktree mode; ADR 0022 created; RL README attach recipe; CONTEXT.md Arbiter two-homes |

## Dependency graph

```
01 ──> 02 ──> 03 ──┬──> 04
                   ├──> 05
                   └──> 06
```

## Wave plan

- **Wave 1**: `01-extract-shared-arbitration-helper` (fable)
- **Wave 2**: `02-rlnodepolicy` (fable)
- **Wave 3**: `03-eval-harness-rebuild` (fable)
- **Wave 4** (parallel, cap 3): `04-notebook-06a-rewrite` (fable), `05-notebook-06-update` (fable), `06-docs-adr-0022` (fable)

## Activity log

- 2026-06-11T11:59:15Z — orchestrator: report initialised, 6 issues pending across 4 waves; base `RL-rewrite`@`7c5b41f`.
- 2026-06-11T12:00:00Z — wave-runner wave 1: dispatching issue-implementer for 01-extract-shared-arbitration-helper (in-place mode, model: fable).
- 2026-06-11T12:05:00Z — wave-runner wave 1: 01-extract-shared-arbitration-helper committed 4ea813b on RL-rewrite; 289 RL tests pass; issue moved to done/.
- 2026-06-11T12:10:00Z — wave-runner wave 2: dispatching issue-implementer for 02-rlnodepolicy (in-place mode, model: fable).
- 2026-06-11T12:20:00Z — wave-runner wave 2: 02-rlnodepolicy committed 86bda7d on RL-rewrite; issue moved to done/.
- 2026-06-11T12:25:00Z — wave-runner wave 3: dispatching issue-implementer for 03-eval-harness-rebuild (in-place mode, model: fable).
- 2026-06-11T12:45:00Z — wave-runner wave 3: 03-eval-harness-rebuild committed d361ad7 on RL-rewrite; 1122 tests pass (313 RL, 1 new regression); issue moved to done/.
- 2026-06-11T14:00:00Z — wave-runner wave 4: creating 3 worktrees (04, 05, 06) from cf223ba on RL-rewrite; implementing all issues directly.
- 2026-06-11T14:29:00Z — wave-runner wave 4: 06-docs-adr-0022 committed 0aa7d67 in worktree; cherry-picked as 8b887dc on RL-rewrite.
- 2026-06-11T14:32:00Z — wave-runner wave 4: 05-notebook-06-update committed a41b52e in worktree; cherry-picked as f53a59d on RL-rewrite.
- 2026-06-11T14:38:00Z — wave-runner wave 4: 04-notebook-06a-rewrite committed 06fb31a in worktree; cherry-picked as dd7e1b3 on RL-rewrite.
- 2026-06-11T14:45:00Z — wave-runner wave 4: all 3 worktrees cleaned up; all 6 issues committed; feature complete.
- 2026-06-11T14:46:00Z — orchestrator housekeeping: waves 1–2 left their issue files duplicated at the old `issues/` path (copied, not moved); removed the duplicates in commit `461fc39`. Wave 3 committed the move with a stale `Status: ready-for-agent`; corrected to `done` in commit `cf223ba`. Wave 4 moves were clean (housekeeping note applied).
- 2026-06-11T14:46:30Z — orchestrator Phase 5: final cleanup verified — no `.claude/worktrees/issue-*` dirs, no `rl-eval-parity`/`issue-*` branches; working tree clean (only this report untracked). All 6 issues `committed`, all issue files in `done/`.

## Outstanding follow-ups

_(none)_

## Resume instructions

Re-run `/implement-issues .scratch/rl-eval-parity/`. Phase 2 reconciles from disk + git (issue files in `done/`, `<id>:`-prefixed commits on `RL-rewrite`) before dispatching any remaining waves.
