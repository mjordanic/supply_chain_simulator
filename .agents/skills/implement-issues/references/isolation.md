# Isolation

Three modes. The wave-runner applies them to **issue-implementers only**. The wave-runner itself always runs in this checkout so it can update the report and cherry-pick onto `BASE_BRANCH`.

| Mode | When | What the implementer sees |
|---|---|---|
| `inplace` | `cap == 1` or `--isolation inplace` | The repo root. Commit lands directly on `BASE_BRANCH`. |
| `worktree` | Claude Code when `cap > 1`; Cursor fallback when cloud preflight fails | A `git worktree` the runner created from `BASE_BRANCH`. |
| `cloud` | Cursor when `cap > 1` (default) or `--isolation cloud` | A Cursor cloud VM cloned from the **remote** `BASE_BRANCH`. |

Local Shell commands are already sandboxed by Cursor; that is not isolation between implementers. Parallel implementers need `worktree` or `cloud` or they will corrupt the index.

## `inplace`

One implementer at a time, workspace = repo root. No worktree, no cherry-pick. Worktree SHA == Integrated SHA.

## `worktree` (local checkout isolation)

This is the secure local analogue of Claude Code's worktree dispatch: each issue gets its own files, branch, and `uv` environment. It is **not** a Linux VM; it is a second Git checkout.

1. From the **repo root** (never from inside another worktree): `git worktree add <repo>/.agents/worktrees/issue-<id> -b <feature-slug>/issue-<id> <BASE_BRANCH>`.
2. **Seed gitignored skills** the parent checkout has but git will not copy: if `<repo>/.agents/skills/tdd` or `code-review` exist, `cp -R` them into the worktree's `.agents/skills/`. `/implement` is committed and is already in the worktree.
3. Dispatch `issue-implementer` with the **prompt** field `workspace:` = that absolute path (Cursor Task has no workspace/cwd parameter; isolation is this path plus the implementer's "stay inside it" rule). Claude Code: same prompt; its tools follow the process cwd after `cd`.
4. After return, **verify the commit landed in that worktree**: `git -C <worktree> log -1 --format=%s` starts with `<id>:` and `git -C <repo> status --porcelain` is empty except the report file. If the parent checkout is dirty and the worktree `HEAD` did not move, the implementer's file tools wrote to the shared workspace — mark `failed` with `workspace_mismatch`, restore the parent with `git restore` / `git clean` of non-report files, do not cherry-pick. On Cursor this is the signal to rerun the wave with `--isolation cloud` or `cap == 1`.
5. On a clean worktree return, cherry-pick onto `BASE_BRANCH` in the repo root, then `git worktree remove --force` and `git branch -D <feature-slug>/issue-<id>`.

**Create the worktree yourself.** Do not set Claude Code `isolation: "worktree"` (it branches from `origin/main`). Do not ask Cursor for an "isolated project copy" / `environment: "cloud"` as a substitute for this path — those are different modes. Do not pass `environment: "cloud"` in worktree mode.

Owned paths (create, sweep, delete only these):

- `.agents/worktrees/issue-<id>` (canonical)
- `.claude/worktrees/issue-<id>` (legacy; still salvage)
- `/tmp/worktrees/issue-<id>` (legacy)
- `.cursor/worktrees/*issue-<id>*` (Cursor-created leftovers that match this issue)

If `git worktree add` fails, follow the matching case in [handover.md](handover.md) — do not skip to `inplace` for a parallel wave.

After adding a worktree, the implementer (not the runner) runs `uv sync --group dev` as needed so `uv run pytest` works. `.cursor/worktrees.json` documents the same setup for Cursor-native worktrees; explicit `git worktree add` does not run that file automatically.

## `cloud` (Cursor VM isolation)

Each implementer is a Cursor **cloud subagent**: its own Ubuntu VM, clone, and branch. This is the container/VM option. Wave-runner and orchestrator stay local.

### Cloud preflight (Phase 0; may ask)

Done when all are true, or the user switches isolation away from `cloud`:

1. Harness is Cursor (Task accepts `environment: "cloud"`).
2. `origin` exists (`git remote get-url origin`). No fetch required.
3. `BASE_BRANCH` has an upstream (`git rev-parse --abbrev-ref @{u}` succeeds). If not, ask the user to push the branch (`git push -u origin HEAD`) — **you** do not push.
4. `HEAD` equals `@{u}` (`git rev-parse HEAD` == `git rev-parse @{u}`). Unpushed commits on `BASE_BRANCH` are invisible to cloud clones; ask the user to push or abort. This compares to the **local** remote-tracking ref — the skill does not `git fetch` in preflight. If the user pushed from another machine, they fetch themselves first.
5. `.cursor/environment.json` exists **in `HEAD`** (`git cat-file -e HEAD:.cursor/environment.json`). Cloud clones the remote commit, not the dirty worktree. If the file is local-only, ask the user to commit and push it, or abort cloud mode.

### Dispatch

- Cursor `Task`: `subagent_type: "issue-implementer"`, `environment: "cloud"`, `cloud_base_branch: BASE_BRANCH` (the **remote** branch; unpushed local-only branches fail), `model`: resolved implementer slug. `run_in_background: true` when the session is in Multitask Mode or the wave has more than one cloud implementer; otherwise foreground is fine. Prompt workspace: "this cloud checkout is the workspace; do not expect a local worktree path".

Do not create a local worktree in cloud mode. Record the Task id (`bc-…`) on the report row immediately.

### Integrate

Cloud agents commit on their VM branch and Cursor publishes that branch. After the implementer returns `status: committed` (or `blocked` with a WIP commit):

1. `git fetch origin <branch>` where `<branch>` is the `branch:` field from the `report` block (or the branch recorded on the Task result). This is the **only** git-remote read the skill allows, and only in `cloud` isolation.
2. Verify `git log -1 --format=%s <commit_sha>` starts with `<id>:`.
3. `git cherry-pick <commit_sha>` on `BASE_BRANCH`. Same conflict rules as worktree mode.
4. Do not `git push` the integration branch.

If fetch cannot see the branch, mark `failed` with "cloud branch not on origin; resume agent `<id>` or inspect [Review](<bc-id>#changes)" rather than inventing a local patch.

### Cloud resume

`bc-` ids resume with Task `resume`. A dropped local connection does **not** stop the VM. Phase 2 must try resume before launching a duplicate cloud agent (duplicates waste a VM and race on the same issue file).
