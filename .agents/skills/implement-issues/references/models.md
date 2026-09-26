# Models

Resolve an alias to a **harness slug** you will actually pass to Task (Cursor) or Agent (Claude Code). Record both the alias and the slug in the report. Pass that slug through. A nested subagent's shorter model list is not a reason to substitute or to refuse.

## Default

- **Implementer**: `grok`. Fast thinking, extra-high effort. On Cursor that is `cursor-grok-4.6-xhigh` (Grok 4.6). On Claude Code, Grok is unavailable → `opus`.
- **Runner**: `inherit` (same model as the orchestrator). Git plumbing does not need a separate pin. `--runner-model grok` is valid if you want it explicit.

`Complexity: high` does **not** bump off `grok`. The default is already the strongest Grok. Use a `Model:` line or `--implementer-model` to pin Claude/GPT.

## Aliases

| Alias | Cursor Task `model` (preferred) | Claude Code Agent `model` |
|---|---|---|
| `grok` | `cursor-grok-4.6-xhigh` | `opus` |
| `inherit` | `inherit` (or omit if the tool treats omit as inherit) | omit the parameter |
| `opus` | `claude-opus-5-thinking-high` | `opus` |
| `opus-4` | `claude-opus-4-8-thinking-high` | `opus` |
| `composer` | `composer-2.5-fast` | `haiku` |
| `gpt` | `gpt-5.6-sol-medium` | `sonnet` |
| `sonnet` | `cursor-grok-4.6-xhigh` | `sonnet` |
| `haiku` | `composer-2.5-fast` | `haiku` |

A value that already looks like a harness slug (`cursor-grok-4.6-xhigh`, `grok-4.7-high`, `claude-opus-5[effort=high]`, `opus`, or any other slug) is passed as-is. Do not rewrite it to a neighbor. Claude Code only accepts `opus` / `sonnet` / `haiku` (and omit); translate a Cursor slug through the table before a Claude Code dispatch.

## Session list

The list that matters is the **orchestrator's** Task/Agent tool, read before the first dispatch. A nested `wave-runner` Task tool that only lists `composer-2.5-fast` is not this list.

1. Map an alias to the preferred slug in the table.
2. If that slug is in the orchestrator's list, pass it.
3. If the alias is `grok` and the preferred slug is absent, pick the closest orchestrator slug in this order: any `cursor-grok*` / `grok*` with `xhigh` or `high`, then any `grok`, then `inherit`. Record that substitution in the report header.
4. A raw slug is passed unchanged even when it is absent from a nested tool's enum. Unknown alias → preflight failure. Ask in Phase 0–1.

## Passing the parameter

- Cursor `Task` on the **orchestrator**: pass the resolved slug as `model` on `issue-implementer`. That call accepts the session's model list. Do not route the model through `wave-runner`; its nested Task call rejects every slug except `composer-2.5-fast`.
- Cursor `Task` for the runner, when you still launch one: pass the resolved runner slug unless the alias is `inherit`.
- Claude Code `Agent`: pass `opus` / `sonnet` / `haiku`, or omit for inherit. Translate Cursor slugs through the table before dispatch.
- `"default"` in a per-issue map means "pass the skill default (`grok`)", not "omit and hope frontmatter matches". Always pass an explicit implementer slug after translation so Claude Code frontmatter (`color`, old `sonnet`) cannot override the grok default.

## Per-issue `Model:` line

Valid: any alias in the table, or a raw slug. Invalid: ask in Phase 1.
