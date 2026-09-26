---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets.

Use /tdd where possible, at pre-agreed seams. Read `.agents/skills/tdd/SKILL.md` when that file exists. If it does not, still write one failing test at a public seam, then only enough code to pass it, one slice at a time.

Run typechecking regularly, single test files regularly, and the full test suite once at the end.

Once done, use /code-review to review the work. Read `.agents/skills/code-review/SKILL.md` when that file exists. If it does not, diff against the starting commit and check (a) the spec/issue and (b) AGENTS.md.

Commit your work to the current branch.

## Orchestrated run

When the caller is `/implement-issues` (no user in the loop):

- The spec is the assigned issue file plus its PRD. Do not ask where the spec is.
- Do not ask questions. Record seam choices and assumptions in the commit body. If the issue, PRD, and code cannot support a reasonable call, stop and report `blocked` to the caller instead of waiting.
- Seams come from the issue's acceptance criteria and testing notes. Write them down before the first test.
- `/code-review` fixed point is `HEAD` before your edits.
- Commit **once**, on this workspace's current branch, with the subject prefix the caller required (issue id). Include moving the issue file to `issues/done/` in that commit when the work is complete.
