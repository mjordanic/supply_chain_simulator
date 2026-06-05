# 01 — Simplify the demand core: remove life-cycle & freshness, relocate CRN

Status: done

## Parent

`.scratch/setup-files/PRD.md`

## What to build

Strip the two demand-shaping add-on layers (global product life-cycle and per-(node,product)
freshness) from the simulation core so demand becomes simply `dist × market.demand_multiplier`,
and relocate the one load-bearing job that lived alongside them (the CRN per-catalog-product
`world_rng` draw) into the runner.

`DemandSinkNode.demand_target` loses its `stage_multiplier` and freshness terms. The product
life-cycle (`lifecycle_clock`, `ItemRegistry`'s stage ownership, `ItemLifecycleParams`,
`MarketParams.stage_multipliers`) and the freshness curve (`freshness_curve`, the per-`Ware`
`freshness_alpha`/`freshness_decay`, the sink `activation_tick` bookkeeping) are removed.
`ItemRegistry` is removed entirely — its only surviving responsibility, the CRN per-catalog-product
`world_rng` draw loop (ADR 0003), moves into the runner iterating the catalog directly.

`Market` otherwise stays fully intact (seasonality, regional state, trend, cross-product
correlation, elasticity, promo, supply side, disruption events). Tests covering the deleted
features are removed, not adapted. The removed features are already documented in `TODO.md`
for a possible later re-add.

This is a refactor slice: there is no new file format yet. The existing graph-based scenario
(the chain example) is the verification vehicle.

## Acceptance criteria

- [ ] `DemandSinkNode.demand_target` computes `demand_dist.sample(rng) * market.demand_multiplier(pid, region, tick)` with no stage or freshness term.
- [ ] `freshness_curve.py`, `lifecycle_clock.py`, `ItemLifecycleParams`, and `MarketParams.stage_multipliers` are deleted; `Ware` loses its freshness fields.
- [ ] `ItemRegistry` is removed; the CRN per-catalog-product `world_rng` draw loop runs in the runner over the catalog list, preserving the existing draw order/determinism (ADR 0003).
- [ ] Tests for the deleted features (`test_freshness_curve`, `test_freshness_integration`, `test_demand_sink_freshness`, `test_lifecycle_clock`, `test_item_registry`) are removed.
- [ ] The existing graph chain scenario still runs and is deterministic (same inputs ⇒ identical run log); the suite is green.

## Blocked by

None - can start immediately.
