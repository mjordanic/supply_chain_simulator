# Implement Issues

Orchestrate parallel/sequential implementation of `ready-for-agent` issues in a `.scratch/<feature>/` folder using the `issue-implementer` subagent. Maintains a resumable `implementation_report.md` in the feature folder so a credit-out / killed session can pick up cleanly on next invocation.

This is an **alternative** to `bash ralph/afk.sh`. The `afk.sh` flow is untouched — both stay available. The only behavioral difference here is **issue selection happens in this skill, not inside the implementer subagent**. Per-issue work (analyze → /tdd → commit) is identical.

## Inputs

- **Argument (optional)**: a feature folder path (e.g., `.scratch/lifecycle-rosters/`). If omitted, infer from the most recently modified `.scratch/<feature>/` directory. If multiple are recent, ask the user.
- **Parallelism cap (optional)**: default `3`. Configurable per invocation.

## Phase 1 — Plan

1. List every `*.md` under `<feature>/issues/` (do **not** descend into `done/`).
2. For each file, extract:
   - **ID**: filename without `.md`.
   - **Title**: first H1.
   - **Status**: the `Status:` line.
   - **Blocked by**: parse the `## Blocked by` section. Treat issue IDs of files already in `<feature>/issues/done/` as satisfied.
3. Keep only issues with `Status: ready-for-agent`. Skip `needs-triage`, `needs-info`, `ready-for-human`, `wontfix`, and anything missing the line.
4. Build a directed dependency graph. **Halt on cycles** — surface them to the user; cycles indicate `/to-issues` produced non-vertical slices.
5. Compute **waves**: wave N is every issue whose blockers are all in `done/` or in waves `< N`. Issues within a wave can run in parallel; waves run sequentially. Cap each wave by the parallelism cap (split oversized waves).

Print the wave plan to the user before dispatching. Let them abort, reorder, or adjust the cap.

## Phase 2 — Resume reconcile (run EVERY invocation, even fresh)

Before dispatching anything:

1. If `<feature>/implementation_report.md` exists, read its status table.
2. Run `git log --oneline -n 100` and `git status --porcelain`.
3. Cross-reference — three sources of truth, in priority order:
   - **Issue file moved to `done/`** → `committed` (this is the strongest signal, since the implementer moves the file as its last step before reporting).
   - **A commit with subject prefix `<issue-id>:`** → `committed`. Capture the SHA.
   - **Report says `in-progress` but neither of the above** → the prior subagent died. Mark `pending` again.
4. Rewrite the report's status table with this reconciled view **before** spawning anything.
5. Drop any `committed` issue from the dispatch queue.

This phase is cheap and idempotent. Do not skip it on a fresh run.

## Phase 3 — Dispatch

For each issue in the current wave, spawn the `issue-implementer` subagent (in parallel, up to the cap). Pass it:

- Absolute path to the issue file.
- Absolute path to `<feature>/PRD.md` for context.
- An explicit instruction: **"this issue has already been selected for you — do not re-prioritize, do not skip."**

Wait for all subagents in a wave to return before starting the next wave.

## Phase 4 — Incremental reporting (CRITICAL)

**Update `<feature>/implementation_report.md` after every subagent return — not at end of wave, not at end of run.** This is the user's defense against credit exhaustion. If the session dies between waves, the report plus git log plus issue-file moves are enough to resume.

The report must contain, in this order:

1. **Header** — feature name, link to `PRD.md`, started-at, last-updated, parallelism cap.
2. **Status table** — one row per issue scheduled in this run: `ID | Title | Wave | Status | Commit SHA | Started | Finished | Notes`. Report statuses: `pending`, `in-progress`, `committed`, `failed`, `blocked`. The label `done` on an issue file maps to report status `committed`.
3. **Dependency graph** — fenced ASCII or mermaid block.
4. **Wave plan** — ordered list, member issues per wave.
5. **Activity log** — append-only, timestamped one-liners for every state transition. Include commit SHA when available.
6. **Outstanding follow-ups** — aggregated from subagent reports.
7. **Resume instructions** — a short paragraph telling future-you (or the user) exactly how to continue: re-run `/implement-issues <feature-path>` and reconcile will catch up.

Build full content in memory, write atomically (single `Write` call).

## Phase 5 — Final summary

When all waves are complete or no further progress is possible:

1. Final consistency pass on the report.
2. Summarize in chat: shipped count, failed count, blocked count, total commits, key follow-ups for human review.
3. Point the user at the report file and any issues now needing human attention.

## Hard rules

- Do **NOT** modify `ralph/afk.sh`, `ralph/once.sh`, `ralph/prompt.md`, or any existing skill. This skill is purely additive.
- Do **NOT** batch report updates. Every subagent return → immediate file write.
- Do **NOT** write feature code yourself. You orchestrate; the subagent implements.
- Do **NOT** exceed the parallelism cap.
- A failed subagent does **not** cascade-fail the wave. Mark `failed`, log it, continue dispatching downstream-independent work. Surface failures prominently in the final summary.
- If the user's environment lacks something the subagent will need (e.g., `uv` missing), stop and ask — don't improvise around tooling rules.
