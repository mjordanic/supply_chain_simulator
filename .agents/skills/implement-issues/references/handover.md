# Handover

Three artifacts, in this order of authority: **disk report**, **git**, **fenced envelopes**. Chat prose between agents is not a contract. If memory disagrees with the report file, the report file wins; if the report disagrees with `done/` + `<id>:` commits on `BASE_BRANCH`, git wins.

## Dispatch envelope (orchestrator → wave-runner)

The prompt is a field list, not a narrative. Required keys:

```
repo_root: <abs>
feature_path: <abs>
report_path: <abs>
base_branch: <name>
wave: <n>
cap: <n>
issue_ids: [<id>, ...]
feature_slug: <slug>
harness: cursor | claude-code
isolation: inplace | worktree | cloud
implementer_models: { <id>: <harness-slug>, ... }
```

Optional: `resume_agent_ids: { <id>: <agent-id>, ... }` when Phase 2 decided to resume instead of spawning fresh.

## Dispatch envelope (wave-runner → issue-implementer)

```
workspace: <abs path, or "cloud-checkout" when isolation is cloud>
issue_id: <id>
issue_path: .scratch/<feature>/issues/<id>.md
prd_path: .scratch/<feature>/PRD.md
isolation: inplace | worktree | cloud
implement_skill: .agents/skills/implement/SKILL.md
pre_selected: true
```

The implementer also has its agent definition. The prompt does not restate `/implement`; it names the skill path and the workspace.

## Return envelope (issue-implementer → wave-runner)

The last assistant message ends with one fenced block. Parse only this block.

```report
workspace: <abs path or cloud-checkout>
issue_id: <id>
status: committed | failed | blocked
branch: <branch name>
commit_sha: <sha or null>
agent_id: <task id if known, else null>
files_changed:
  - <path>
tests_added:
  - <path>
notes: <one or two sentences>
follow_ups:
  - <item>
```

Missing fence → treat as `failed` with reason `malformed report`. Do not grep the transcript for a SHA.

## Return envelope (wave-runner → orchestrator)

```summary
wave: <N>
firewall: intact | degraded
committed:
  - id: <issue-id>
    integrated_sha: <sha on BASE_BRANCH>
    agent_id: <id or null>
failed:
  - id: <issue-id>
    reason: <one line>
blocked:
  - id: <issue-id>
    reason: <one line>
conflicts:
  - id: <issue-id>
    files: [<file>, ...]
```

Empty lists → omit the section.

## Persist before wait

Before waiting on any subagent:

1. Write `in-progress`, `Started`, isolation, and worktree path (or `cloud`) to the report row.
2. When the Task/Agent tool returns an id, write `Agent ID` on that row in the same atomic rewrite.

A dropped connection after step 2 is recoverable. A dropped connection after launch with no id write is recoverable only via git salvage.

## Resume a live agent

When a row is `in-progress` and `Agent ID` is set:

1. Cursor: Task `resume` = that id. Prompt: `Continue the assigned issue. If already finished, return only the report fence. If not, finish via /implement as in your agent definition, then return the report fence. Do not start a different issue.`
2. Claude Code: the equivalent Agent resume if the harness has it; otherwise skip to git salvage.
3. Resume success → parse `report`, integrate as if the original wait returned.
4. Resume failure (unknown id, expired, tool error) → git salvage. Do **not** launch a second implementer until salvage finishes; two implementers on one issue race.

## Git salvage (`worktree` / `inplace`)

For each owned worktree path / branch `<feature-slug>/issue-<id>`:

| Observation | Action |
|---|---|
| `<id>:` commit on the worktree branch, not on `BASE_BRANCH` | Cherry-pick it (same rules as a live return). Then remove the worktree and delete the branch. |
| `<id>:` commit already on `BASE_BRANCH` | Mark `committed`, capture SHA, remove worktree. |
| Dirty worktree, no `<id>:` commit, tracked or untracked files that look like issue work | `git -C <worktree> add -A` and commit `<id>: WIP salvage after interrupted run` on the worktree branch, then cherry-pick. Mark `blocked` if you had to salvage WIP (not `committed`). If there is nothing but report-file noise, discard. |
| Empty worktree, no unique commits | `git worktree remove --force`; delete the branch if it exists and has no unique commits. Mark `pending`. |
| `git worktree add` fails: path exists and is a registered worktree | Reuse it; do not create a second worktree for the same issue. |
| `git worktree add` fails: path exists but is **not** a worktree | If the directory is empty, `rmdir` and retry. If it has files, rename to `<path>.bak-<timestamp>` and retry; log the bak path. Never `rm -rf` a directory that is not a registered worktree of this repo. |
| `git worktree add -b` fails: branch already exists | `git worktree add <path> <feature-slug>/issue-<id>` (attach the existing branch) if that branch is not checked out elsewhere. If it is checked out in another worktree, reuse that worktree's path. |
| Worktree locked | `git worktree unlock <path>` then continue salvage/remove. |
| Leftover `index.lock` / `HEAD.lock` and no live pid | Remove the lock file, then salvage. If a git pid is alive, mark `blocked` with the pid; do not kill it. |
| Cherry-pick: empty (commit already in history) | Mark `committed` with the existing SHA. |
| Cherry-pick: conflict | `git cherry-pick --abort`. Mark `failed`. Log conflict files. Leave the worktree in place until the row is `failed` and the files are in follow-ups, then remove. |
| Nested worktree attempt | Always `cd` to the main repo root (`git rev-parse --show-toplevel` of `BASE_BRANCH`'s checkout) before `git worktree add`. |
| Cursor leftover under `.cursor/worktrees` matching `issue-<id>` | Same salvage table; do not `/delete-worktree` until unique `<id>:` commits are on `BASE_BRANCH` or recorded `failed`. |

Never `git worktree remove` a path that has a unique `<id>:` commit you have not cherry-picked or explicitly marked `failed`.

## Cloud salvage

For each `in-progress` / `blocked` row with a `bc-` Agent ID:

1. Resume the cloud agent (above).
2. If resume fails, `git fetch origin` of the recorded `branch:` (if any). If a `<id>:` commit appears, cherry-pick.
3. If neither works, mark `pending` only when you are sure no cloud VM is still running. If unsure, mark `blocked` with "possible live cloud agent `<id>`; inspect [Review](<bc-id>#changes) before re-dispatch" so Phase 3 does not start a duplicate.

## Degraded firewall

If `wave-runner` cannot spawn `issue-implementer`, or its Task tool rejects the envelope model (nested enum is only `composer-2.5-fast`), the runner returns `firewall: degraded` and **must not** implement issues itself or substitute a model.

On Cursor this is the normal path: the orchestrator dispatches `issue-implementer` itself and passes the resolved slug as `model`, then runs the wave-runner workflow in-process (same isolation, same envelopes, same serial integration, same report ownership). Append `firewall: degraded` to the header and activity log. Context will grow; that is how an arbitrary model reaches `/implement`.

## Integration (serial)

Wait for **all** implementers in the wave (or all resume attempts) before cherry-picking. Then, one issue at a time on `BASE_BRANCH`:

1. Verify ground truth. Worktree: isolation.md step 4 (`workspace_mismatch` aborts integration). Cloud: fetched `commit_sha`. Subject must start with `<id>:`.
2. `committed` → cherry-pick → record SHAs → mark `committed`.
3. `blocked` → cherry-pick the WIP commit anyway (durable scaffolding) → mark `blocked`.
4. `failed` → nothing to integrate → mark `failed`.
5. Cleanup worktree/branch after the row is updated (worktree mode).
6. Atomic report write before the next issue.

## Implementer vs runner

The implementer's hard rules live in `.agents/agents/issue-implementer.md`. The runner owns push, branch deletion, worktrees, and `implementation_report.md`.
