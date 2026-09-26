# Permissions

Unattended dispatch must not stop on a permission prompt. Front-load allows in Phase 0 (Claude Code) or say them once in the Phase 1 plan (Cursor).

## Cursor

No `.claude/settings.local.json` audit. Local `worktree` / `inplace` runs use Cursor's Shell sandbox; git writes may show one approval card — list `git worktree`, `git cherry-pick`, `git commit`, and `git branch -D` of `<feature-slug>/issue-*` in the plan prompt so the user can approve before Phase 2.

`cloud` isolation: implementers run on Cursor VMs. The local runner still needs `git fetch origin <implementer-branch>` and `git cherry-pick` after they return. Mention that fetch exception in the plan prompt.

Deny from our prompts regardless of harness: `git push`, `git pull`, `git clone`, `git remote` mutations, `gh`, `hub`. `git fetch` is allowed only in `cloud` isolation, and only for the implementer's published branch.

## Claude Code

Read `.claude/settings.local.json` and add any missing `permissions.allow` entries from the list below. Bundle the diff into Phase 1's wave-plan question. If the harness blocks the edit, print a copy-paste JSON block, ask the user to apply it, and abort preflight.

Keep the deny list intact.

```jsonc
// permissions.allow — minimum set for unattended dispatch
"Bash(uv run *)",
"Bash(MPLBACKEND=* uv run *)",

"Bash(git status*)",
"Bash(git log*)",
"Bash(git diff*)",
"Bash(git show*)",
"Bash(git symbolic-ref*)",
"Bash(git rev-parse*)",
"Bash(git branch*)",
"Bash(git checkout*)",
"Bash(git switch*)",
"Bash(git restore*)",
"Bash(git add*)",
"Bash(git rm*)",
"Bash(git mv*)",
"Bash(git commit*)",
"Bash(git cherry-pick*)",
"Bash(git apply*)",
"Bash(git worktree*)",
"Bash(git reflog*)",
"Bash(git cat-file*)",
"Bash(git ls-tree*)",
"Bash(git ls-files*)",
"Bash(git check-ignore*)",
"Bash(git -C *)",

"Bash(mkdir -p /tmp/*)",
"Bash(mkdir -p .claude/worktrees/*)",
"Bash(mkdir -p .agents/worktrees/*)",
"Bash(rm -rf /tmp/worktrees/*)",
"Bash(rm -rf .claude/worktrees/*)",
"Bash(rm -rf .agents/worktrees/*)",
"Bash(ls /tmp/worktrees*)",
"Bash(ls .claude/worktrees*)",
"Bash(ls .agents/worktrees*)",
"Bash(find /tmp/worktrees *)",
"Bash(find .claude/worktrees *)",
"Bash(find .agents/worktrees *)",
"Bash(cd /tmp/worktrees/*)",
"Bash(cd .claude/worktrees/*)",
"Bash(cd .agents/worktrees/*)",

"Bash(date*)",
"Bash(pwd)"
```

```jsonc
// permissions.deny — keep as the safety floor
"Bash(git push*)",
"Bash(git pull*)",
"Bash(git fetch*)",
"Bash(git remote*)",
"Bash(git clone*)",
"Bash(git ls-remote*)",
"Bash(git submodule*)",
"Bash(gh *)", "Bash(gh)",
"Bash(hub *)", "Bash(hub)"
```

On Claude Code, `cloud` isolation is unavailable, so the `git fetch*` deny stays. Cursor cloud mode does not edit this file; the fetch exception lives in the skill text, not in Claude's deny list.

Notes:

- `git worktree*` covers `add`, `remove --force`, `list`, `prune`, `unlock`.
- `rm -rf` grants are scoped to worktree directories, not a general `rm -rf *`.
- `MPLBACKEND=* uv run *` is for headless notebook re-execution.
