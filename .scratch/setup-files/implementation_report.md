# Implementation Report — setup-files

- **Feature**: setup-files
- **PRD**: [`PRD.md`](./PRD.md)
- **Started**: 2026-06-05T12:27:27Z
- **Last updated**: 2026-06-05T12:30:00Z
- **Parallelism cap**: 1 (in-place sequential)
- **Integration branch**: `claude/eloquent-hopper-Mwefh`
- **runner-model**: default
- **implementer-model**: opus (global `--implementer-model` override, applies to all issues)
- **Preflight assumptions**:
  - Feature folder chosen interactively (all `.scratch/*` share an identical clone mtime; inference was ambiguous).
  - `.claude/settings.local.json` was absent; created with the Required-Bash allow list + remote deny list during preflight.
  - No issue carries a `Complexity:` or `Model:` line; global opus override pins every implementer.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|----|-------|------|--------|--------------|----------------|---------|----------|-------|
| 01-simplify-demand-core | Simplify the demand core: remove life-cycle & freshness, relocate CRN | 1 | in-progress | — | — | 2026-06-05T12:30:00Z | — | |
| 02-run-setup-directory-end-to-end | Run a hand-authored setup directory end-to-end | 2 | pending | — | — | — | — | |
| 03-topology-scaffolder | Topology scaffolder | 3 | pending | — | — | — | — | |
| 04-llm-generator-writes-setup-files | LLM generator writes setup files (+ write_setup, dir-as-cache) | 3 | pending | — | — | — | — | |
| 05-rl-trains-against-setup-directory | RL trains against a setup directory | 3 | pending | — | — | — | — | |
| 06-tuning-optimises-against-setup-directory | Tuning optimises against a setup directory | 4 | pending | — | — | — | — | |
| 07-remove-remaining-legacy | Remove remaining legacy: Store model, World, world_loaders, dead policy classes | 5 | pending | — | — | — | — | |
| 08-docs-and-convert-examples | Docs + convert example scenarios | 5 | pending | — | — | — | — | |

## Dependency graph

```
01 ──▶ 02 ──┬──▶ 03 ─────────────┐
            ├──▶ 04 ─────────────┤
            ├──▶ 05 ──┐          │
            ├──▶ 06 ──┤          │
            │         ├──▶ 07    │
            │         │          ├──▶ 08
            └─────────┴──────────┘

07 ← {02, 05, 06}
08 ← {02, 03, 04}
```

## Wave plan

- **Wave 1**: `01-simplify-demand-core` (opus)
- **Wave 2**: `02-run-setup-directory-end-to-end` (opus)
- **Wave 3**: `03-topology-scaffolder` (opus), `04-llm-generator-writes-setup-files` (opus), `05-rl-trains-against-setup-directory` (opus)
- **Wave 4**: `06-tuning-optimises-against-setup-directory` (opus)
- **Wave 5**: `07-remove-remaining-legacy` (opus), `08-docs-and-convert-examples` (opus)

(cap=1 ⇒ issues within a wave run sequentially in the repo root, not in parallel worktrees.)

## Activity log

- 2026-06-05T12:27:27Z — Orchestrator: preflight passed; created `.claude/settings.local.json`; report initialized with 8 pending issues across 5 waves. Reconcile found 0 committed (done/ empty).
- 2026-06-05T12:30:00Z — Wave-runner: Wave 1 started; dispatching issue-implementer for 01-simplify-demand-core (model=opus).

## Outstanding follow-ups

_(none yet)_

## Resume instructions

Re-run `/implement-issues .scratch/setup-files/` with the same feature path. Phase 2 reconciles
committed work from `done/` + the integration branch's `<id>:`-prefixed commits before dispatching
anything, so a killed session resumes cleanly.
