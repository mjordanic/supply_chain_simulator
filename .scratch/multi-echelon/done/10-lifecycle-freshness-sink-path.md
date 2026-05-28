# Lifecycle & freshness composition inside DemandSinkNode.demand_target

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Re-ground ADRs 0001–0003 (two-layer lifecycle, per-stage transitions, CRN demand sampling) inside `DemandSinkNode.demand_target`. After this slice, the lifecycle and freshness machinery lives on the demand-sink path — the legacy `Store.sample_demand`-driven path is no longer the source of truth for these dynamics.

`DemandSinkNode` gains `activation_tick: dict[str, int]` (per-product activation tick, for freshness composition).

`demand_target(tick, market, registry)`:
- `mult = market.demand_multiplier(pid, region, tick) * registry.stage_multiplier(pid) * freshness_curve.multiplier(α, β, tick - activation_tick[pid])`
- `raw = demand_dist.sample(world_rng) * mult`

**CRN preservation (load-bearing):** sample for every catalog pid every tick (mirrors ADR 0003), even when the sink isn't actively buying that pid; `world_rng` consumed in catalog-iteration order. This invariant must be enforced by a dedicated test, not inferred from end-to-end snapshots.

Ship:
- `tests/sim/test_demand_sink_freshness.py` — assertions ported from `tests/sim/test_freshness_integration.py`
- `tests/sim/test_freshness_integration.py` — reframed onto the demand-sink path
- `tests/sim/test_regression_snapshot.py` — regenerated from the graph engine on the Phase-1 chain scenario; old snapshot retired with a one-line pointer to ADR 0011

Visible verification: run the Phase-1 chain scenario with `freshness_alpha=2.0` vs `0.0` — visible hype curve on the sink's realised demand series.

## Acceptance criteria

- [ ] `DemandSinkNode.activation_tick: dict[str, int]` field exists
- [ ] `DemandSinkNode.demand_target(tick, market, registry)` composes market × stage × freshness multipliers correctly
- [ ] `world_rng` consumed for every catalog pid every tick, in catalog-iteration order, even when the sink isn't buying that pid
- [ ] `tests/sim/test_demand_sink_freshness.py` covers freshness composition assertions
- [ ] `tests/sim/test_freshness_integration.py` reframed onto the demand-sink path
- [ ] `tests/sim/test_regression_snapshot.py` regenerated from graph engine; old snapshot retired with ADR 0011 pointer
- [ ] Phase-1 scenario with `freshness_alpha=2.0` vs `0.0` produces a visibly different demand series
- [ ] `uv run pytest tests/sim -x` is green

## Blocked by

- `.scratch/multi-echelon/issues/06-phase1-policies-chain-scenario.md`
- `.scratch/multi-echelon/issues/09-multisupplier-textbook-policy-contention-scenario.md`
