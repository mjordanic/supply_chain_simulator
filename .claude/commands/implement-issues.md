# Implement Issues

Orchestrate dependency-ordered implementation of `ready-for-agent` issues in a `.scratch/<feature>/` folder. Builds a wave plan, dispatches one fresh `wave-runner` subagent per wave (each runs the wave end-to-end and returns a small summary), and maintains a resumable `implementation_report.md` so a credit-out / killed session can pick up cleanly on the next invocation.

**Question window**: the user is available only during Phases 0–1 (preflight + planning). The moment Phase 2 starts (resume reconcile + wave dispatch), the run is fully unattended — anomalies are recorded in the report and the run continues; nothing blocks for human input. Use the planning phase to surface anything ambiguous up front so dispatch can run cleanly.

The wave-runner indirection is a context firewall: per-wave git activity, worktree management, and per-issue subagent transcripts stay inside the wave-runner's context. The orchestrator only sees a small return summary per wave, so its context stays roughly constant regardless of how many issues or waves are involved.

## Inputs

- **Argument (optional)**: a feature folder path (e.g., `.scratch/<feature>/`). If omitted, infer the most recently modified `.scratch/<feature>/` directory. If two or more candidates were modified within 60s of each other, record an "ambiguous feature folder" preflight failure and exit.
- **Parallelism cap (optional)**: default `3`. `cap == 1` ⇒ in-place sequential; `cap > 1` ⇒ worktree-per-issue parallel within each wave. Wave size is computed from the dependency graph, not assumed.

## Phase 0 — Preflight

Run FIRST, every invocation. **This is one of the two phases where you may ask the user.** Surface anything ambiguous; do not paper over it just to keep moving.

1. **Integration branch** — `BASE_BRANCH = $(git symbolic-ref --short HEAD)` from the repo root. Refuse if HEAD is detached or `BASE_BRANCH` ∈ {`main`, `master`} (ask the user to switch to a feature branch).
2. **Working tree clean** — `git status --porcelain` empty *except* for `<feature>/implementation_report.md` (skill-owned). If dirty, ask the user to stash/commit; do not run a partial dispatch over uncommitted work.
3. **`uv` available** — `uv --version` succeeds. If not, ask.
4. **Feature folder** — if no argument was given and inference returns multiple candidates modified within 60s of each other, ask the user which one. If the user-supplied argument doesn't exist, ask.
5. **PRD exists** — `<feature>/PRD.md` is present. If not, ask.

Record `BASE_BRANCH`, started-at, and parallelism cap in the report header so resume runs can verify the same branch.

## Phase 1 — Plan

1. List every `*.md` under `<feature>/issues/` (skip `done/`).
2. For each file, extract: ID (filename minus `.md`), Title (first H1), Status (`Status:` line), Blocked-by (`## Blocked by` section). Treat IDs already in `done/` as satisfied.
3. Keep only issues with `Status: ready-for-agent`. Skip the rest. **Exception**: if every file in the feature lacks a `Status:` line, surface this to the user and ask whether to treat them all as `ready-for-agent`. Record the answer in the report header.
4. Build the dependency graph. **Cycles → ask the user.** Cycles indicate `/to-issues` produced non-vertical slices and need human review; do not invent a tie-breaker.
5. Compute waves: wave N is every issue whose blockers are all done or in waves `< N`. If a wave is larger than the parallelism cap, split it into consecutive sub-waves of cap-sized chunks.
6. **Present the wave plan to the user** and let them abort, reorder waves, drop issues, or adjust the parallelism cap. This is the **last point at which questions are allowed** — once you start Phase 2, the run is unattended.
7. Write the initial report (schema in Phase 4) with every issue as `pending`.

## Phase 2 — Resume reconcile

Run EVERY invocation, even fresh. Cheap, idempotent, no external state.

1. If a prior `implementation_report.md` exists, read its status table.
2. Run `git log --oneline -n 100` on `BASE_BRANCH` and `git status --porcelain`.
3. Cross-reference, in priority order:
   - **Issue file in `done/`** → `committed` (strongest signal — implementer moves the file as its last step).
   - **Commit on `BASE_BRANCH` with subject prefix `<id>:`** → `committed`. Capture the SHA.
   - **Report says `in-progress` but neither of the above** → prior subagent died. Mark `pending` again.
4. **Stale worktree sweep**: `git worktree list`; prune any `.claude/worktrees/issue-*` whose branch has no unique commits ahead of `BASE_BRANCH`. `git worktree remove --force` and delete the orphan branch.
5. **Salvage sweep**: for any `.claude/worktrees/issue-<id>/` whose branch HAS a `<id>:`-prefixed commit not yet on `BASE_BRANCH`, cherry-pick it now (apply Phase 3's cherry-pick rules: clean → `committed`; conflict → abort + mark `failed` + log files). Clean up the worktree afterward.
6. Rewrite the report's status table with this reconciled view **before** spawning anything.
7. Drop any `committed` issue from the dispatch queue.

## Phase 3 — Dispatch waves

For each wave with at least one pending issue, sequentially:

1. Dispatch ONE fresh `wave-runner` subagent. Pass:
   - Repo root (absolute).
   - Feature path (absolute).
   - Report path (absolute).
   - `BASE_BRANCH`.
   - Wave number.
   - Parallelism cap.
   - List of pending issue IDs in this wave.
   - Feature slug (for branch naming).
2. Wait for the subagent's return summary (small `summary` block listing committed / failed / blocked / conflicts).
3. Merge its counts into your running totals.
4. **Continue to the next wave regardless of failures inside this one.** A failed wave does not cascade.

The wave-runner owns: worktree creation, dispatching `issue-implementer` subagents, cherry-pick integration, per-wave report updates, worktree cleanup. You only do dispatch + summary aggregation.

## Phase 4 — Report schema

The report file is the contract between this skill, its wave-runners, and the human reviewer. It must contain, in this order:

1. **Header** — feature name, link to `PRD.md`, started-at, last-updated, parallelism cap, integration branch (`BASE_BRANCH`), and any preflight assumptions (e.g., `Status:` lines missing across the board).
2. **Status table** — one row per scheduled issue: `ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes`. Statuses: `pending`, `in-progress`, `committed`, `failed`, `blocked`. Issue file at `done/` ⇔ report status `committed`. For in-place dispatches, Worktree SHA == Integrated SHA.
3. **Dependency graph** — fenced ASCII or mermaid block.
4. **Wave plan** — ordered list, member issues per wave.
5. **Activity log** — append-only, timestamped one-liners for every state transition (commit SHAs, worktree paths, cherry-pick outcomes).
6. **Outstanding follow-ups** — aggregated from subagent reports plus any cherry-pick conflicts.
7. **Resume instructions** — re-run the skill with the same feature path; reconcile catches up.

The orchestrator owns sections 1, 3, 4, 7 plus the *initial pending* rows of section 2. Each wave-runner owns the rows in section 2 for its wave and appends to sections 5 and 6. Atomic writes only (build full content in memory, single `Write` call).

## Phase 5 — Final summary

When all waves are dispatched (or an early-exit preflight happened):

1. Read the final report; recompute counts from the status table.
2. **Final cleanup**: confirm no `.claude/worktrees/issue-*` directories or `<feature-slug>/issue-*` branches remain. Force-remove stragglers; append the cleanup to the activity log.
3. Print to chat: shipped, failed, blocked, total commits, top follow-ups, path to the report file.

## Hard rules

- **No user prompts after Phase 1 ends.** Phases 0 and 1 are the only window for questions; everything from Phase 2 onward must be unattended. Mid-run anomalies are recorded in the report and the run continues with whatever is still actionable.
- Do **NOT** trust the `Agent` tool's `isolation: "worktree"` flag. It branches from `origin/main`, not the integration branch. Always use `git worktree add` explicitly from `BASE_BRANCH`. (Wave-runner enforces the same rule.)
- Do **NOT** dispatch parallel `issue-implementer` subagents into the repo root. Concurrent edits to the same working tree race on the index and corrupt commits. (Wave-runner enforces this — the orchestrator never dispatches `issue-implementer` directly.)
- Do **NOT** write feature code yourself. You orchestrate; subagents implement.
- Do **NOT** auto-resolve cherry-pick conflicts (in the salvage sweep). Abort, mark `failed`, log files.
- Do **NOT** cascade-fail. A failed issue does not stop the wave; a failed wave does not stop the run.
- A killed session is recoverable: re-run the skill with the same feature path. Phase 2 reconciles from disk + git.
