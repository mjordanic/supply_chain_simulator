# 06 — Tuning optimises against a setup directory

Status: ready-for-agent

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Migrate the Optuna tuning stack off the legacy `world_loader` so trials evaluate against a
setup directory's `catalog` + `market`, expressed in node/edge terms.

- Tuning loads `catalog` + `market` from a setup directory (replacing `world_loader`'s
  cache/archetype/synthetic resolution). A small synthetic-setup helper may stand in when no
  setup dir is supplied.
- The per-trial graph is built programmatically in `NodeInstance`/`EdgeSpec` terms (same shape
  as RL: factory-per-active-SKU → intermediate → sink-per-active-SKU). No `StoreTemplate`.
- The old `StoreTemplate` randomisation fields move into the tuning `Config` as per-trial scale
  randomisation ranges + node defaults.
- Per-trial CRN/paired-eval determinism is preserved.

## Acceptance criteria

- [ ] Optuna trials load `catalog` + `market` from a setup directory (no `world_loader`, no `StoreTemplate`).
- [ ] The per-trial graph is constructed in `NodeInstance`/`EdgeSpec` terms with scale randomisation from the tuning `Config`.
- [ ] Per-trial CRN paired evaluation is deterministic (two identical specs ⇒ identical trajectories).
- [ ] Tuning evaluator determinism tests pass against the node/edge trial builder.

## Blocked by

- Issue 02 (setup dir loading of `catalog` + `market`, node/edge types)
