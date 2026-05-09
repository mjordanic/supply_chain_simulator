# ISSUES

Local issue files from `.scratch/simulator-redesign/issues/` are provided at start of context. Parse them to understand the open issues.

Each issue file has a `Status:` line on the first line. Only work on issues whose status is `ready-for-agent`. Skip `needs-triage`, `needs-info`, `ready-for-human`, and `wontfix`.

You've also been passed a file containing the last few commits. Review these to understand what work has been done.

If all `ready-for-agent` tasks are complete, output <promise>NO MORE TASKS</promise>.

# TASK SELECTION

Pick the next task. Prioritize tasks in this order:

1. Critical bugfixes
2. Development infrastructure

Getting development infrastructure like tests and types and dev scripts ready is an important precursor to building features.

3. Tracer bullets for new features

Tracer bullets are small slices of functionality that go through all layers of the system, allowing you to test and validate your approach early. This helps in identifying potential issues and ensures that the overall architecture is sound before investing significant time in development.

TL;DR - build a tiny, end-to-end slice of the feature first, then expand it out.

4. Polish and quick wins
5. Refactors

# EXPLORATION

Explore the repo.

# IMPLEMENTATION

Use /tdd to complete the task.

# FEEDBACK LOOPS

Before committing, run the feedback loops:

- `uv run pytest` to run the tests

# COMMIT

Make a git commit. The commit message must:

1. Include key decisions made
2. Include files changed
3. Blockers or notes for next iteration

# THE ISSUE

If the task is complete, move the issue file to `.scratch/simulator-redesign/issues/done/`.

If the task is not complete, add a note to the issue file with what was done.

# REPORT

After everything else (commit, issue file move, etc.), emit a single fenced ` ```report ` block as the very last thing in your final message. This is how `ralph/afk.sh` records progress across iterations so a credit-out / killed session can resume cleanly. Do NOT skip this block — even on `no-tasks` or `failed` iterations.

Format (YAML-ish, one field per line):

```report
issue_id: <id worked on, or "none">
status: committed | blocked | failed | no-progress | no-tasks
commit_sha: <full SHA, or "none">
branch: <current branch>
files_changed:
  - <path>
notes: <one short sentence — what you did, or why nothing happened>
follow_ups:
  - <item or omit list if none>
```

Status values:

- `committed` — task complete, commit landed, issue file moved to `done/`.
- `blocked` — partial work; issue updated to `needs-info` with explanation. Commit may or may not exist.
- `failed` — attempt collapsed (tests not green, etc.); nothing committed; issue left as `ready-for-agent`.
- `no-progress` — iteration produced nothing actionable for any other reason. Explain in `notes`.
- `no-tasks` — no `ready-for-agent` issues remain. Also output `<promise>NO MORE TASKS</promise>` so the loop exits.

The block must be plain text inside the fenced ` ```report ` ... ` ``` ` markers — no nested code fences, no extra prose between the markers. Keep `notes` to one line.

# FINAL RULES

ONLY WORK ON A SINGLE TASK.
