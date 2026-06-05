# 05 — RL trains against a setup directory

Status: done

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Migrate the RL stack off the legacy `world_loader` and `StoreTemplate` so it trains against a
setup directory's `catalog` + `market`, expressed entirely in node/edge terms.

- RL loads `catalog` + `market` from a setup directory (replacing `world_loader`'s
  cache/archetype/synthetic resolution). A small synthetic-setup helper may stand in when no
  setup dir is supplied.
- The per-episode graph is built programmatically in `NodeInstance`/`EdgeSpec` terms
  (factory-per-active-SKU → trainable intermediate "S" → sink-per-active-SKU). No `StoreTemplate`.
- The old `StoreTemplate` randomisation fields (`capacity_dist`, `balance_dist`, `delivery_lag`,
  `holding_rate`, `order_fee`) move into `RLConfig` as per-episode randomisation ranges + node
  defaults: per-episode capacity, opening balance, and active-SKU subset are sampled from those
  ranges, so the learned policy generalises across world scales while the world *data* stays fixed.
- The RL `OrderUpToPolicy` CRN comparison anchor and paired-eval determinism are preserved.

## Acceptance criteria

- [ ] RL training loads `catalog` + `market` from a setup directory (no `world_loader`, no `StoreTemplate`).
- [ ] The per-episode graph is constructed in `NodeInstance`/`EdgeSpec` terms; per-episode capacity, opening balance, and active-SKU subset are sampled from `RLConfig` ranges.
- [ ] CRN paired evaluation is deterministic (two identical specs ⇒ identical trajectories) and the `OrderUpToPolicy` anchor still runs.
- [ ] RL determinism/CRN tests pass against the node/edge episode builder.

## Blocked by

- Issue 02 (setup dir loading of `catalog` + `market`, node/edge types)
