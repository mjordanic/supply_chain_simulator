## Behavioral guidelines
Behavioral guidelines to reduce common LLM coding mistakes. 

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

In unattended/orchestrated runs where asking is impossible (e.g. the issue-implementer under `/implement-issues`), "ask" becomes: record the assumption in the commit body and proceed, or report `blocked` — never block waiting on a question.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.


### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## Tooling

Use `uv` for all Python tooling — `uv run`, `uv add`, `uv sync`. Never `pip`, `python -m`, or `poetry`.

## Git commits

Do not sign commits with `Co-Authored-By: Claude …` (or any other co-author trailer attributing the commit to Claude/Anthropic or OpenAI or Cursor or any other coding agent). Leave the commit message unsigned. Applies to the main thread and any subagent or skill that creates commits.

## Agent skills

### Issue tracker

Issues and PRDs live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, default strings, recorded as `Status:` lines in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Implementation orchestrator (project-local)

`/implement-issues` drives dependency-ordered, unattended implementation of
`ready-for-agent` issues in a `.scratch/<feature>/` folder. Canonical files:
`.agents/skills/implement-issues/` and `.agents/agents/{wave-runner,issue-implementer}.md`.
Cursor also loads `.cursor/skills/implement-issues` and `.cursor/agents/`.

Each issue is implemented by reading `/implement` (`.agents/skills/implement/SKILL.md`).
On Cursor the orchestrator dispatches `issue-implementer` itself so the model slug
reaches `/implement`. On Claude Code a `wave-runner` does that dispatch.
Default implementer model is `grok`. Isolation for `cap > 1` is `cloud` when
`origin` tracks the integration branch and `.cursor/environment.json` is on
that remote commit. Without a remote, or if that file is missing from `HEAD`,
isolation is `worktree`.

## Cursor Cloud specific instructions

Cloud agents boot from `.cursor/Dockerfile` and run `uv sync --group dev`
before the task. Tests are `uv run pytest`. Smoke tests do not need
`OPENAI_API_KEY`; tests marked `live` skip when it is unset. Raw M5 files are
not in the repo. A missing `data/m5/` directory is the expected case, and the
M5 notebook must skip rather than fail. Do not commit `runs/` or `data/`.

`.agents/commands/implement-issues.md` is a symlink to the skill.
`.claude/commands/implement-issues.md` and `.claude/agents/` point at the same files.

Questions are allowed only in Phases 0–1; from wave dispatch onward the run is unattended
and resumable via `implementation_report.md`. Relies on the issue-tracker layout, the five
triage labels, and `/implement` → `/tdd` + `/code-review`.

## Workflow skills

These are committed under `.agents/skills/`, same layout as
`outdoors_destinations`:
`to-spec`, `to-tickets`, `implement`, `implement-issues`, `code-review`.

`/to-spec` publishes a spec to the issue tracker. `/to-tickets` breaks it into
issues (this replaces `/to-issues`; `/to-spec` replaces `/to-prd`).
`/implement-issues` runs the `ready-for-agent` ones through `/implement`.

Other Matt Pocock skills already committed here stay in place:
`diagnose`, `grill-me`, `grill-with-docs`, `improve-codebase-architecture`,
`prototype`, `setup-matt-pocock-skills`, `tdd`, `triage`, `write-a-skill`, `zoom-out`.
