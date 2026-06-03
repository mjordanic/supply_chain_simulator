# 08 — `CONTEXT.md` updates for the tuning tool

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Final documentation pass once the tuning module, CLI, and notebook are all on `main`. Three small, surgical edits to `CONTEXT.md`:

1. **New `Policy tuning study` glossary entry.** Names the artifacts (`trials.parquet`, `per_seed.parquet`, `study.json`, `holdout.parquet`, `holdout_summary.json`), the objective (mean `net_profit / initial_cash` across 16 CRN seeds at 365 ticks under log-uniform domain randomisation), the framing-1 scope (measurement instrument, not baseline-construction tool — published `OrderUpToPolicy` defaults are unchanged), and a link to ADR 0009.
2. **`CRN-paired eval` glossary entry update.** One-sentence addition noting that the single-policy half of the eval machinery is reused by the policy tuning study with a normalised-return objective.
3. **Decisions list entry for ADR 0009.** Mirrors the existing entry style for ADRs 0006, 0007, 0008. (The ADR 0008 entry was added in issue 01; this slice adds only ADR 0009.)

Keep the existing prose style. No glossary entries change beyond the three edits listed.

### Verification

After editing, re-read `CONTEXT.md` end-to-end to confirm:

- The new `Policy tuning study` entry is placed alphabetically (or in the same ordering convention used by neighbouring entries).
- The `CRN-paired eval` entry's cross-reference to the tuning module is one sentence, not a section.
- The Decisions list ends with both the ADR 0008 (from issue 01) and ADR 0009 (this slice) rows.

No tests added — this is a documentation-only slice. The acceptance check is human-readable correctness.

PRD user stories covered: 26.

## Acceptance criteria

- [ ] `CONTEXT.md` contains a new `Policy tuning study` glossary entry covering: artifact filenames, the normalised-return objective, log-uniform domain randomisation, framing-1 scope (measurement instrument), and an ADR 0009 link.
- [ ] `CONTEXT.md`'s `CRN-paired eval` glossary entry includes a one-sentence note about the single-policy reuse by the tuning study.
- [ ] `CONTEXT.md`'s Decisions list includes a row for ADR 0009 in the same style as ADRs 0006 / 0007 / 0008.
- [ ] No other `CONTEXT.md` edits are made in this slice.
- [ ] `uv run pytest` is green (no regressions; this slice changes only docs).

## Blocked by

- `07-tune-textbook-policy-notebook.md` — final documentation pass lands after every other piece of the tuning tool is on `main`.
