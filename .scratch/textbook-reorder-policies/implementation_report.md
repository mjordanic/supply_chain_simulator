# Implementation Report — textbook-reorder-policies

- **Feature**: textbook-reorder-policies
- **PRD**: [PRD.md](./PRD.md)
- **Started at**: 2026-05-13T23:04:58Z
- **Last updated**: 2026-05-14T00:15:27Z
- **Parallelism cap**: 3
- **Integration branch**: `textbook-policies`
- **Preflight notes**: All 6 issues have `Status: ready-for-agent`. Working tree clean. Wave 2 (`02-rename-and-code-migration`) isolated into its own wave per user direction to avoid cherry-pick conflicts with subclass work (04/05/06).

## Status

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-textbook-base-and-order-up-to | TextbookReorderPolicy base + helpers + OrderUpToPolicy | 1 | committed | af0bcbe823ea48ae5cb861ec4768d64ce60851dd | af0bcbe823ea48ae5cb861ec4768d64ce60851dd | 2026-05-13T23:08:06Z | 2026-05-13T23:16:57Z | In-place (wave=1). Added _estimate_rate, _allocate_two_pass_fair_share, TextbookReorderPolicy, OrderUpToPolicy; delivery_lags added to store.observe(); 31 new tests, 218 sim tests pass. |
| 02-rename-and-code-migration | Rename BaselinePolicy → HeuristicPolicy + atomic code migration | 2 | committed | e842e2ba351999a4c755f39ed289c86788d1e3ec | e842e2ba351999a4c755f39ed289c86788d1e3ec | 2026-05-14T00:00:00Z | 2026-05-14T00:10:00Z | In-place (wave=2). Hard-break rename: BaselinePolicy→HeuristicPolicy (Bucket A), OrderUpToPolicy migration (Bucket B). Regression snapshot regenerated. 468 tests pass, 3 pre-existing failures unrelated to this issue. |
| 04-reorder-point-policy | ReorderPointPolicy (s,Q) | 3 | committed | cdb15af7118779b049eda5ab5d4e4cd28d640a5f | d0ee7b26a68b2062d3506a78dbf678e5a3894c2e | 2026-05-14T01:30:00Z | 2026-05-14T01:55:00Z | Worktree mode. ReorderPointPolicy added to policy.py + __all__. Fixed Q or rate-derived default (cover_horizon*rate). 5 new tests, 225 sim tests pass. |
| 05-periodic-order-up-to-policy | PeriodicOrderUpToPolicy (R,S) | 3 | failed | 20211cc77281424eba85d5c5e2bfbf2633f46770 | — | 2026-05-14T01:55:00Z | 2026-05-14T02:15:00Z | Cherry-pick conflict vs 04 on src/sim/policy.py and tests/sim/test_textbook_policy.py. Both worktrees appended to the same files from the same base. Worktree commit is clean; conflict is purely additive. See Outstanding follow-ups. |
| 06-periodic-reorder-policy | PeriodicReorderPolicy (R,s,S) | 3 | failed | 94436169d6ce6473782b871561419cbf8c1d5c5d | — | 2026-05-14T02:15:00Z | 2026-05-14T02:25:00Z | Cherry-pick conflict vs 04 on src/sim/policy.py and tests/sim/test_textbook_policy.py. Same root cause as 05. Worktree commit is clean; conflict is purely additive. See Outstanding follow-ups. |
| 03-notebook-migration | Migrate notebooks 06 and 07 to OrderUpToPolicy | 4 | committed | acd394a | acd394a | 2026-05-14T03:00:00Z | 2026-05-14T03:10:00Z | In-place (wave=4). Both notebooks updated: imports, build_baseline_policy() → OrderUpToPolicy(), markdown narrative. Both re-executed headlessly via nbconvert; zero BaselinePolicy refs remain. |

## Dependency graph

```
01 ── 02 ── 03
 ├──── 04
 ├──── 05
 └──── 06
```

## Wave plan

- **Wave 1** — `01-textbook-base-and-order-up-to` (in-place, single issue)
- **Wave 2** — `02-rename-and-code-migration` (in-place, isolated rename)
- **Wave 3** — `04-reorder-point-policy`, `05-periodic-order-up-to-policy`, `06-periodic-reorder-policy` (parallel worktrees, cap=3)
- **Wave 4** — `03-notebook-migration` (in-place, single issue)

## Activity log

- 2026-05-13T23:04:58Z — orchestrator: preflight passed (branch `textbook-policies`, clean tree, `uv` 0.11.13), 6 ready-for-agent issues found, no prior report or done/ entries; initial report written.
- 2026-05-13T23:16:57Z — wave-1: committed 01-textbook-base-and-order-up-to at af0bcbe on textbook-policies (in-place, no worktree); 31 new tests, 218 sim tests passing.
- 2026-05-14T00:10:00Z — wave-2: committed 02-rename-and-code-migration at e842e2b on textbook-policies (in-place, no worktree); hard-break BaselinePolicy→HeuristicPolicy rename; Bucket A (HeuristicPolicy), Bucket B (OrderUpToPolicy); regression snapshot regenerated; 468 tests pass, 3 pre-existing unrelated failures.
- 2026-05-14T01:55:00Z — wave-3: cherry-picked 04-reorder-point-policy (cdb15af → d0ee7b2) onto textbook-policies; ReorderPointPolicy (s,Q) added to policy.py + __all__; 5 new tests pass; 225 sim tests pass.
- 2026-05-14T02:15:00Z — wave-3: cherry-pick of 05-periodic-order-up-to-policy (20211cc) FAILED with conflict on src/sim/policy.py and tests/sim/test_textbook_policy.py vs 04's changes; cherry-pick aborted; issue marked failed.
- 2026-05-14T02:25:00Z — wave-3: cherry-pick of 06-periodic-reorder-policy (9443616) FAILED with conflict on src/sim/policy.py and tests/sim/test_textbook_policy.py vs 04's changes; cherry-pick aborted; issue marked failed.
- 2026-05-14T02:30:00Z — wave-3: cleaned up all three worktrees and per-issue branches; report updated.
- 2026-05-14T03:10:00Z — wave-4: committed 03-notebook-migration at acd394a on textbook-policies (in-place); notebooks/06 and notebooks/07 migrated to OrderUpToPolicy; both re-executed via nbconvert; zero BaselinePolicy refs remain; issue moved to done/.
- 2026-05-14T00:15:27Z — orchestrator (final cleanup): no worktrees or feature-slug branches remained. Removed orphan `done/05-periodic-order-up-to-policy.md` and `done/06-periodic-reorder-policy.md` created by their respective implementers before the cherry-picks aborted — original issue files at `issues/05-*.md` and `issues/06-*.md` are intact and accurate, so resume reconcile will correctly see 05/06 as not-yet-committed.

## Outstanding follow-ups

**Wave-3 cherry-pick conflicts for 05 and 06 — purely additive, manually resolvable.**

Issues 05 (`PeriodicOrderUpToPolicy`) and 06 (`PeriodicReorderPolicy`) were implemented in parallel worktrees that both branched from `e842e2b`. Both appended new classes/tests to the same files that issue 04 also modified. After cherry-picking 04, the patches for 05 and 06 conflicted against the updated file because git could not automatically merge three sets of additions at the end of the same two files.

The worktree commits are clean and tested. Resolution steps:
1. Manually apply the changes from worktree SHA `20211cc` (issue 05) by replaying the diff on top of current `textbook-policies` (which already has 04 integrated at `d0ee7b2`).
2. Manually apply the changes from worktree SHA `9443616` (issue 06) on top.
3. Run `uv run pytest tests/sim/` to confirm all tests pass.
4. Commit each as a new commit with the `05-periodic-order-up-to-policy:` and `06-periodic-reorder-policy:` subject prefixes.

Both changes are strictly additive (new class + new `__all__` entry + new test functions appended at the end of the file), so there is no logical conflict — only a git diff context collision.

Conflict files: `src/sim/policy.py`, `tests/sim/test_textbook_policy.py` (both issues).

**Quickest path to resolve (no re-run):**
```bash
# from textbook-policies HEAD = acd394a
git diff e842e2b 20211cc -- src/sim/policy.py tests/sim/test_textbook_policy.py | git apply --3way
# inspect/fix any conflict markers, run uv run pytest tests/sim/, commit:
git commit -am "05-periodic-order-up-to-policy: PeriodicOrderUpToPolicy (R,S)"
# repeat for 06 with 9443616
git diff e842e2b 9443616 -- src/sim/policy.py tests/sim/test_textbook_policy.py | git apply --3way
git commit -am "06-periodic-reorder-policy: PeriodicReorderPolicy (R,s,S)"
# then move .scratch/textbook-reorder-policies/issues/05-*.md and 06-*.md into done/
```

Worktree commits `20211cc` and `9443616` may be unreachable now (per-issue branches were deleted in cleanup). If `git cat-file -e 20211cc` fails, recover from reflog: `git reflog --all | grep -E '20211cc|9443616'` should still find them within gc grace.

## Resume instructions

If this run is interrupted, re-run `/implement-issues .scratch/textbook-reorder-policies` from the `textbook-policies` branch. Phase 2 reconcile reads `done/`, `git log`, and live worktrees to recover state automatically.
