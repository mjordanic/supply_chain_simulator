# Implement Issues

Orchestrate parallel/sequential implementation of `ready-for-agent` issues in a `.scratch/<feature>/` folder using the `issue-implementer` subagent. Maintains a resumable `implementation_report.md` in the feature folder so a credit-out / killed session can pick up cleanly on next invocation.

This is an **alternative** to `bash ralph/afk.sh`. The `afk.sh` flow is untouched — both stay available. Differences from `afk.sh`:
1. **Issue selection happens in this skill, not inside the implementer subagent.** Per-issue work (analyze → /tdd → commit) is identical.
2. **Multi-issue waves run in parallel via per-issue git worktrees** branched from the current integration branch. Commits are cherry-picked back. Single-issue waves run in-place on the integration branch with no worktree overhead.

## Inputs

- **Argument (optional)**: a feature folder path (e.g., `.scratch/lifecycle-rosters/`). If omitted, infer from the most recently modified `.scratch/<feature>/` directory. If multiple are recent, ask the user.
- **Parallelism cap (optional)**: default `3`. Configurable per invocation. `cap == 1` ⇒ in-place sequential; `cap > 1` ⇒ worktree-per-issue parallel within each wave.

## Phase 0 — Capture integration context (run FIRST, every invocation)

Before touching anything, capture and lock the orchestration anchors:

1. **Integration branch** — `BASE_BRANCH = $(git symbolic-ref --short HEAD)` from the main project root. Refuse to proceed if HEAD is detached or `BASE_BRANCH` is `main`/`master` (we never want feature work landing directly on those).
2. **Working tree must be clean** — `git status --porcelain` is empty *except* for `.scratch/<feature>/implementation_report.md` (which the orchestrator owns) and any pre-existing items the user explicitly opts into. If dirty, stop and ask the user to stash/commit first; do not run a partial dispatch over uncommitted work.
3. **`uv` is available** — `uv --version` succeeds. If not, stop and ask.

Record `BASE_BRANCH` in the report header so resume runs can verify they're back on the same branch.

## Phase 1 — Plan

1. List every `*.md` under `<feature>/issues/` (do **not** descend into `done/`).
2. For each file, extract:
   - **ID**: filename without `.md`.
   - **Title**: first H1.
   - **Status**: the `Status:` line.
   - **Blocked by**: parse the `## Blocked by` section. Treat issue IDs of files already in `<feature>/issues/done/` as satisfied.
3. Keep only issues with `Status: ready-for-agent`. Skip `needs-triage`, `needs-info`, `ready-for-human`, `wontfix`, and anything missing the line. **Exception**: if every file in the feature lacks a `Status:` line and the user invoked the skill anyway, treat them all as `ready-for-agent` and record this in the report header. Surface the assumption to the user before dispatching.
4. Build a directed dependency graph. **Halt on cycles** — surface them to the user; cycles indicate `/to-issues` produced non-vertical slices.
5. Compute **waves**: wave N is every issue whose blockers are all in `done/` or in waves `< N`. Issues within a wave can run in parallel; waves run sequentially. Cap each wave by the parallelism cap (split oversized waves).

Print the wave plan to the user before dispatching. Let them abort, reorder, or adjust the cap.

## Phase 2 — Resume reconcile (run EVERY invocation, even fresh)

Before dispatching anything:

1. If `<feature>/implementation_report.md` exists, read its status table.
2. Run `git log --oneline -n 100` and `git status --porcelain`.
3. Cross-reference — three sources of truth, in priority order:
   - **Issue file moved to `done/`** → `committed` (this is the strongest signal, since the implementer moves the file as its last step before reporting).
   - **A commit on `BASE_BRANCH` with subject prefix `<issue-id>:`** → `committed`. Capture the SHA.
   - **Report says `in-progress` but neither of the above** → the prior subagent died. Mark `pending` again.
4. **Stale worktree sweep**: list `git worktree list` and prune any `.claude/worktrees/issue-*` whose branch has no unique commits ahead of `BASE_BRANCH` (i.e., the prior agent died before committing). Remove with `git worktree remove --force` and delete the orphan branch.
5. **Salvage sweep**: for any `.claude/worktrees/issue-<id>/` whose branch HAS a `<id>:`-prefixed commit not yet on `BASE_BRANCH`, cherry-pick it now (Phase 3 cherry-pick logic) and clean up. Then mark the issue `committed`.
6. Rewrite the report's status table with this reconciled view **before** spawning anything.
7. Drop any `committed` issue from the dispatch queue.

This phase is cheap and idempotent. Do not skip it on a fresh run.

## Phase 3 — Dispatch

For each wave, dispatch by branch arithmetic — never by hope.

### Wave with one issue (or `cap == 1`)

Dispatch a single `issue-implementer` subagent **in-place** on `BASE_BRANCH`. Pass:

- Workspace path: the main project root (absolute).
- Issue file path: relative to the workspace.
- PRD path: relative to the workspace.
- An explicit instruction: **"this issue has already been selected for you — do not re-prioritize, do not skip."**

The subagent commits directly onto `BASE_BRANCH`. No cherry-pick needed.

### Wave with N>1 issues (parallelism `cap >= 2`)

For each issue in the wave (up to the cap), the orchestrator:

1. **Creates a worktree.** `git worktree add <repo>/.claude/worktrees/issue-<id> -b <feature-slug>/issue-<id> <BASE_BRANCH>`. The worktree path lives under `.claude/worktrees/` (gitignored). The branch must be unique per issue and prefixed by the feature slug so it's easy to garbage-collect later.
2. **Dispatches an `issue-implementer` subagent** (in parallel, up to the cap). Pass:
   - **Workspace path**: the absolute worktree path. Tell the agent to `cd` there and stay inside.
   - **Issue file path**: relative to the workspace, e.g., `.scratch/<feature>/issues/<id>.md`.
   - **PRD path**: relative to the workspace.
   - The pre-selection notice.
3. **Never** uses the `Agent` tool's built-in `isolation: "worktree"` flag. That flag branches from `origin/main`, which silently desyncs from `BASE_BRANCH` and leaves the agent without the feature folder. Worktrees in this skill are always created explicitly via `git worktree add` from `BASE_BRANCH`.

Wait for **all** subagents in the wave to return before integrating. Then, for each return (one at a time, serialized — git operations on the integration branch must not race):

1. Parse the `report` block from the subagent's summary.
2. Verify ground truth: `git -C <worktree> rev-parse HEAD`, `git -C <worktree> log -1 --format=%s` (subject must start with `<id>:`).
3. If `status: committed` and the subject matches:
   - `git cherry-pick <sha>` on `BASE_BRANCH` in the main repo. If clean: report row → `committed` with the new SHA on `BASE_BRANCH` (cherry-pick produces a new SHA — record both: the worktree SHA and the integrated SHA).
   - If conflict: `git cherry-pick --abort`, mark issue `failed`, log conflict files in Outstanding follow-ups, and surface to the user. Do NOT auto-resolve.
4. If `status: blocked`:
   - Cherry-pick the WIP commit anyway (so the partial scaffolding is durable on `BASE_BRANCH`), but mark report row `blocked`. Log the blocker.
   - If conflict on a blocked WIP cherry-pick: abort, mark `failed`, log the conflict.
5. If `status: failed`: nothing to integrate; mark `failed`, log diagnostics.
6. **Always clean up**: `git worktree remove --force <worktree>` and `git branch -D <feature-slug>/issue-<id>` after integration. Leftover worktrees confuse the next reconcile.
7. Update the report (Phase 4) **immediately**, before integrating the next return.

Wait for all wave returns + integrations to complete before starting the next wave.

## Phase 4 — Incremental reporting (CRITICAL)

**Update `<feature>/implementation_report.md` after every subagent return + integration step — not at end of wave, not at end of run.** This is the user's defense against credit exhaustion. If the session dies between waves, the report plus git log plus issue-file moves are enough to resume.

The report must contain, in this order:

1. **Header** — feature name, link to `PRD.md`, started-at, last-updated, parallelism cap, integration branch (`BASE_BRANCH`).
2. **Status table** — one row per issue scheduled in this run: `ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes`. Report statuses: `pending`, `in-progress`, `committed`, `failed`, `blocked`. The label `done` on an issue file maps to report status `committed`. For in-place dispatches, Worktree SHA == Integrated SHA.
3. **Dependency graph** — fenced ASCII or mermaid block.
4. **Wave plan** — ordered list, member issues per wave.
5. **Activity log** — append-only, timestamped one-liners for every state transition. Include commit SHAs and worktree paths when available.
6. **Outstanding follow-ups** — aggregated from subagent reports plus any cherry-pick conflicts.
7. **Resume instructions** — a short paragraph telling future-you (or the user) exactly how to continue: re-run `/implement-issues <feature-path>` and reconcile will catch up (including the worktree salvage sweep).

Build full content in memory, write atomically (single `Write` call).

## Phase 5 — Final summary

When all waves are complete or no further progress is possible:

1. Final consistency pass on the report.
2. **Final cleanup**: confirm no `.claude/worktrees/issue-*` directories or `<feature-slug>/issue-*` branches remain. Force-remove any stragglers and log the cleanup.
3. Summarize in chat: shipped count, failed count, blocked count, total commits, key follow-ups for human review.
4. Point the user at the report file and any issues now needing human attention.

## Hard rules

- Do **NOT** modify `ralph/afk.sh`, `ralph/once.sh`, or `ralph/prompt.md`. This skill is purely additive to that flow.
- Do **NOT** trust the `Agent` tool's `isolation: "worktree"` flag for this skill — it branches from `origin/main`, not the integration branch. Always use `git worktree add` explicitly from `BASE_BRANCH`.
- Do **NOT** dispatch parallel subagents into the main project root. Concurrent edits to the same working tree race on the index and produce corrupt commits.
- Do **NOT** batch report updates. Every subagent return + every integration step → immediate file write.
- Do **NOT** write feature code yourself. You orchestrate; the subagent implements.
- Do **NOT** exceed the parallelism cap.
- Do **NOT** auto-resolve cherry-pick conflicts. Abort, mark `failed`, surface to the user.
- A failed subagent (or a failed cherry-pick) does **not** cascade-fail the wave. Mark `failed`, log it, continue integrating the rest of the wave's returns. Surface failures prominently in the final summary.
- If the user's environment lacks something the subagent will need (e.g., `uv` missing), stop and ask — don't improvise around tooling rules.
