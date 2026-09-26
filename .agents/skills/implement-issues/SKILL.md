---
name: implement-issues
description: "Orchestrate dependency-ordered, unattended implementation of ready-for-agent issues in a .scratch feature folder."
disable-model-invocation: true
---

# Implement Issues

Orchestrate dependency-ordered implementation of `ready-for-agent` issues in a `.scratch/<feature>/` folder. Builds a wave plan, dispatches each issue to `issue-implementer` with the resolved model slug, and maintains a resumable `implementation_report.md` so a killed session, dropped connection, or expired subagent can pick up on the next invocation. On Cursor the orchestrator makes that Task call itself so any model reaches `/implement`. On Claude Code a `wave-runner` makes it.

Each issue is implemented by an `issue-implementer` that **reads and follows** `.agents/skills/implement/SKILL.md` (the `/implement` skill). That skill is the implementation loop. This skill only plans, isolates, hands off, integrates, and resumes.

**Question window**: the user is available only during Phases 0–1 (preflight + planning). The moment Phase 2 starts, the run is fully unattended — anomalies go into the report and the run continues. Surface ambiguity in planning so dispatch can run cleanly.

**Context firewall**: on Claude Code, per-wave git activity, isolation, and implementer transcripts stay inside the wave-runner. On Cursor the orchestrator runs that workflow in-process, because a nested Task call cannot pass an arbitrary model.

**Harness**: Cursor is first-class (Task tool, optional cloud VMs). Claude Code still works. Detect the harness in Phase 0 and pass harness-native model slugs and isolation flags. Do not send Claude Code `opus`/`sonnet`/`haiku` into Cursor's Task `model` parameter, and do not send Cursor slugs into Claude Code's Agent `model` parameter.

## Inputs

- **Argument (optional)**: a feature folder path (e.g. `.scratch/<feature>/`). If omitted, infer the most recently modified `.scratch/<feature>/` directory. If two or more candidates were modified within 60s of each other, record an "ambiguous feature folder" preflight failure and exit.
- **Parallelism cap (optional)**: default `3`. `cap == 1` ⇒ in-place sequential; `cap > 1` ⇒ isolated parallel within each wave. Wave size comes from the dependency graph, not from the cap.
- **Isolation (optional)**: `--isolation worktree|cloud|inplace`. Defaults: `inplace` when `cap == 1`; `cloud` when `cap > 1` and the harness is Cursor and `origin` has an upstream for `BASE_BRANCH`; `worktree` otherwise. `cloud` is Cursor-only (each implementer on its own VM). Read [references/isolation.md](references/isolation.md) when the value is `cloud` or `worktree`, or when a resume report header already recorded an isolation mode.
- **Model selection (optional)**. Two independent knobs plus a shorthand:
  - `--runner-model <model>` — model for `wave-runner` (git plumbing).
  - `--implementer-model <model>` — model for `issue-implementer` (the `/implement` loop).
  - `--model <model>` — sets both. An explicit `--runner-model` / `--implementer-model` wins over `--model`.
  - Read [references/models.md](references/models.md) whenever a flag is supplied, a `Model:` / `Complexity:` line is present, or you are about to pass a `model` parameter to Task/Agent. **Default implementer: `grok`. Default runner: `inherit`.** On a resume run, an omitted flag inherits the report header; a supplied flag overrides it. The resolved selection is always rewritten to the header.
  - **Per-issue implementer only**, first match wins: `Model:` line → global `--implementer-model`/`--model` → `Complexity:` mapping in [references/models.md](references/models.md) → `grok` default.

## Phase 0 — Preflight

Run FIRST, every invocation. **This is one of the two phases where you may ask the user.** Done when every check below has a recorded pass, a user answer, or a preflight failure in the report.

1. **Harness** — Cursor if the Task tool accepts `environment: "cloud"` / `subagent_type`. Claude Code otherwise. Record it in the report header.
2. **Integration branch** — `BASE_BRANCH = $(git symbolic-ref --short HEAD)` from the repo root. Refuse if HEAD is detached or `BASE_BRANCH` ∈ {`main`, `master`} (ask the user to switch to a feature branch).
3. **Working tree clean** — `git status --porcelain` empty *except* for `<feature>/implementation_report.md` (skill-owned). If dirty, ask the user to stash/commit; do not dispatch over uncommitted work.
4. **`uv` available** — `uv --version` succeeds. If not, ask.
5. **`/implement` present** — `.agents/skills/implement/SKILL.md` exists (committed with this repo). `.agents/skills/tdd/SKILL.md` and `.agents/skills/code-review/SKILL.md` should exist on this machine (gitignored mattpocock install). If those two are missing, ask the user to run `npx skills@latest add mattpocock/skills`; still proceed if they confirm — `/implement` has a fallback when they are absent. If `/implement` itself is missing, abort.
6. **Feature folder** — if no argument was given and inference returns multiple candidates modified within 60s of each other, ask. If the user-supplied argument doesn't exist, ask.
7. **PRD exists** — `<feature>/PRD.md` is present. If not, ask.
8. **Isolation** — resolve `--isolation` vs the cap/harness default vs a resume header. `cloud` on Claude Code is a preflight failure (ask to drop to `worktree`). For `cloud` on Cursor, follow the cloud preflight in [references/isolation.md](references/isolation.md) until BASE_BRANCH is on the remote at HEAD or the user aborts.
9. **Permission audit** — Claude Code only. Follow [references/permissions.md](references/permissions.md). Cursor: skip the Claude settings file; local worktrees use the Shell sandbox plus explicit `git worktree` checkouts; `cloud` uses Cursor's VM. If Cursor will need `git_write` / worktree commands, say so once during the Phase 1 plan prompt so the user expects the approval card — then never block the unattended run on a prompt you can front-load.
10. **Model resolution (global tier).** Resolve `runner-model` (final here) and the global `implementer-model` override using [references/models.md](references/models.md). Unknown alias → preflight failure; ask. A raw slug is passed through. Never silently downgrade because a nested tool's model list is shorter than the orchestrator's.

Record `BASE_BRANCH`, harness, isolation, started-at, parallelism cap, and resolved `runner-model` / `implementer-model` in the report header.

## Phase 1 — Plan

Done when the wave plan is written into the report with every scheduled issue `pending`, or the user aborted.

1. List every `*.md` under `<feature>/issues/` (skip `done/`).
2. For each file, extract: ID (filename minus `.md`), Title (first H1), Status (`Status:` line), Blocked-by (`## Blocked by`), optional `Complexity:` (`high` / `standard` / `low`), optional `Model:` (alias or harness slug). Absent ⇒ fall through. Treat IDs already in `done/` as satisfied.
3. Keep only `Status: ready-for-agent`. Skip the rest. **Exception**: if every file in the feature lacks a `Status:` line, ask whether to treat them all as `ready-for-agent`. Record the answer in the report header.
4. Build the dependency graph. **Validate blocker IDs first**: every `Blocked by` reference must resolve to an issue file under `issues/` or `done/`. Unknown IDs → ask. **Cycles → ask.** Do not invent a tie-breaker.
5. Compute waves: wave N is every issue whose blockers are all done or in waves `< N`. If a wave is larger than the cap, split it into consecutive sub-waves of cap-sized chunks.
6. **Resolve each issue's implementer model** via the Inputs precedence and [references/models.md](references/models.md). Record the resolved *alias* and the *harness slug you will pass* against each issue.
7. **Pre-dispatch consistency scan**: for every file path an issue mandates committing, `git check-ignore -q <path>` — a hit means the artifact would be silently untracked; skim each issue against the PRD for direct contradictions. Fold findings into the wave-plan question.
8. **Present the wave plan** — isolation mode, per-issue implementer model (alias → slug), wave membership — and let the user abort, reorder, drop issues, change the cap, change isolation, or override any per-issue model. **Last point at which questions are allowed.**
9. Write the initial report (schema in Phase 4) with every issue as `pending`.

## Phase 2 — Resume reconcile

Run EVERY invocation, even fresh. Cheap, idempotent, no external state. Follow [references/handover.md](references/handover.md) for envelopes, agent-id resume, and worktree salvage. Done when the status table matches disk + git and no dispatchable issue is still marked `in-progress`.

1. If a prior `implementation_report.md` exists, read its status table and header (isolation, harness, models).
2. Run `git log --oneline -n 100` on `BASE_BRANCH` and `git status --porcelain`.
3. Cross-reference, in priority order:
   - **Issue file in `done/`** → `committed` (strongest signal — implementer moves the file as its last step).
   - **Commit on `BASE_BRANCH` with subject prefix `<id>:`** → `committed`. Capture the SHA.
   - **`in-progress` with an Agent ID** → attempt Task/Agent **resume** with that id (handover.md). Success → treat the return as a normal wave result. Failure → git salvage.
   - **`in-progress` with neither done/ nor a matching commit** → prior subagent died. Salvage then mark `pending` unless salvage produced a `<id>:` commit.
4. **Stale worktree sweep** + **salvage sweep** as specified in handover.md (owned paths: `.agents/worktrees/issue-*`, legacy `.claude/worktrees/issue-*`, `/tmp/worktrees/issue-*`, and any `.cursor/worktrees/*issue-*`).
5. **Cloud salvage** when header isolation is `cloud`: handover.md.
6. Rewrite the status table with this reconciled view **before** spawning anything.
7. Drop any `committed` issue from the dispatch queue.

## Phase 3 — Dispatch waves

Done when every wave with pending issues has a `summary` fence merged into running totals (or the session stopped after persisting agent ids so Phase 2 can resume).

For each wave with at least one pending issue, sequentially:

1. Persist `in-progress` rows for this wave's issues (timestamp, isolation, worktree path or `cloud` placeholder) **before** launching. That write is the resume handle if the connection drops mid-dispatch.
2. Dispatch the implementer with the resolved model slug. On Cursor the orchestrator makes that Task call itself: nested `wave-runner` Task only accepts `composer-2.5-fast`, so any other model never reaches `/implement`.
   - Cursor: `subagent_type: "issue-implementer"`, `environment: "local"` unless isolation is `cloud`, `model`: that issue's resolved slug (any slug from [references/models.md](references/models.md), including a raw slug). Prompt = handover.md implementer envelope. Set header `firewall: degraded` and run this wave's runner workflow in-process (isolation, integration, report). Do not spawn `wave-runner` to pass the model.
   - Claude Code: dispatch ONE `wave-runner` Agent with the same envelope and the translated model slug. The runner then spawns `issue-implementer` and passes each slug as `model`.
   The moment the tool returns an agent id, write it onto that issue's report row.
3. Wait for the return summary (Claude Code: the wave-runner fence; Cursor: the implementer fence, then integrate as handover.md). Foreground Task returns the result; background Task completion is delivered to this conversation. If the session would die waiting, stop after the agent-id write and tell the user to re-invoke; Phase 2 resumes.
4. Merge counts into running totals.
5. **Continue to the next wave regardless of failures inside this one.**

If a Claude Code wave-runner cannot spawn `issue-implementer`, it returns `blocked` with that diagnostic. Execute that wave's runner workflow yourself, append `firewall: degraded` to the activity log, and pass the same model slug on your own implementer dispatch. Read handover.md for the degraded path.

On Claude Code the wave-runner owns isolation, implementer dispatch, integration, and per-wave report updates. On Cursor you own that workflow so the model parameter is the one you pass.

## Phase 4 — Report schema

The report file is the contract between this skill, its wave-runners, and the human reviewer. It must contain, in this order:

1. **Header** — feature name, link to `PRD.md`, started-at, last-updated, parallelism cap, integration branch (`BASE_BRANCH`), harness, isolation, resolved `runner-model` / `implementer-model`, firewall (`intact` / `degraded`), and any preflight assumptions.
2. **Status table** — one row per scheduled issue: `ID | Title | Wave | Status | Agent ID | Worktree SHA | Integrated SHA | Started | Finished | Notes`. Statuses: `pending`, `in-progress`, `committed`, `failed`, `blocked`. Issue file at `done/` ⇔ report status `committed`. For in-place dispatches, Worktree SHA == Integrated SHA. Agent ID is the Task/Agent id used to resume that implementer (empty if never launched).
3. **Dependency graph** — fenced ASCII or mermaid block.
4. **Wave plan** — ordered list, member issues per wave, each issue annotated with implementer alias → harness slug.
5. **Activity log** — append-only, timestamped one-liners for every state transition (dispatch ids, commit SHAs, worktree paths, cherry-pick outcomes, resume attempts).
6. **Outstanding follow-ups** — aggregated from subagent reports plus integration conflicts.
7. **Resume instructions** — re-run this skill with the same feature path; Phase 2 reconciles from disk + git + agent ids.

The orchestrator owns sections 1, 3, 4, 7 plus the *initial pending* rows of section 2. Each wave-runner owns the rows in section 2 for its wave and appends to sections 5 and 6. Atomic writes only (build full content in memory, single `Write` call).

## Phase 5 — Final summary

Done when shipped/failed/blocked counts printed to chat match the status table and owned `issue-*` worktrees with no unique unintegrated `<id>:` commits are gone.

When all waves are dispatched (or an early-exit preflight happened):

1. Read the final report; recompute counts from the status table.
2. **Final cleanup**: confirm no owned `issue-*` worktrees or `<feature-slug>/issue-*` branches remain. Force-remove stragglers you own; append the cleanup to the activity log. Do not delete a worktree that still has a unique `<id>:` commit not on `BASE_BRANCH` — salvage it first.
3. Print to chat: shipped, failed, blocked, total commits, top follow-ups, path to the report file. Link any cloud implementers as `[Review](<bc-id>#changes)` when you have a `bc-` id.

## Hard rules

- **No user prompts after Phase 1 ends.** Mid-run anomalies are recorded in the report and the run continues with whatever is still actionable.
- **`/implement` implements; this skill orchestrates.** Do not write feature code yourself.
- **Local isolation is explicit `git worktree add` from `BASE_BRANCH`.** Do not trust Claude Code `isolation: "worktree"` (it branches from `origin/main`). Do not trust Cursor "isolated project copies" to start from `BASE_BRANCH` unless you created the worktree yourself and passed that path as the workspace.
- **Never dispatch parallel `issue-implementer` subagents into the same checkout.** Concurrent edits race on the index.
- **Do not auto-resolve cherry-pick conflicts.** Abort, mark `failed`, log files.
- **Do not cascade-fail.** A failed issue does not stop the wave; a failed wave does not stop the run.
- **Do not push** from the orchestrator, wave-runner, or implementer. Cloud isolation is the exception only insofar as Cursor's cloud runtime publishes the implementer's branch; still do not `git push` from our prompts. Cloud-mode **fetch of that published branch** is allowed so you can cherry-pick; see isolation.md.
- A killed session is recoverable: re-run with the same feature path. Phase 2 reconciles from the report, git, and agent ids.
