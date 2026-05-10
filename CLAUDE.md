## Tooling

Use `uv` for all Python tooling — `uv run`, `uv add`, `uv sync`. Never `pip`, `python -m`, or `poetry`.

## Git commits

Do not sign commits with `Co-Authored-By: Claude …` (or any other co-author trailer attributing the commit to Claude/Anthropic). Leave the commit message unsigned. Applies to the main thread and any subagent or skill that creates commits.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` and `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Available skills

Skills live under `.agents/skills/`. Invoke with `/skill-name`.

| Skill | When to use |
| --- | --- |
| `diagnose` | Hard bugs or performance regressions — reproduce → minimise → hypothesise → instrument → fix → regression-test |
| `grill-me` | Stress-test a plan or design via relentless interviewing |
| `grill-with-docs` | Like `grill-me` but challenges against domain model and updates `CONTEXT.md` / ADRs inline |
| `improve-codebase-architecture` | Find deepening opportunities informed by `CONTEXT.md` and `docs/adr/` |
| `prototype` | Throwaway prototype to flush out design — terminal app for logic, or UI variations |
| `setup-matt-pocock-skills` | Re-run to reconfigure issue tracker, triage labels, or domain doc layout |
| `tdd` | Build features or fix bugs with red-green-refactor loop |
| `to-issues` | Break a plan / PRD into independently-grabbable vertical-slice issues |
| `to-prd` | Turn current conversation context into a PRD on the issue tracker |
| `triage` | Move issues through the triage state machine |
| `write-a-skill` | Create a new agent skill with proper structure and bundled resources |
| `zoom-out` | Get a higher-level map of unfamiliar code — modules, callers, domain vocabulary |
