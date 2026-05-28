# Demand-sinks as the demand source; Market shrinks to a multiplier engine

Status: Proposed

ADRs 0001–0003 established the demand model for the single-store era: `Market.sample_demand(pid, store, price)` drew one random base demand and composed it with lifecycle stage, per-store freshness, seasonality, promotion boost, and cross-product factors. The `Market` owned the full demand-sampling contract and was the mandatory demand source for every `Store`.

In the multi-echelon graph, demand is initiated by `DemandSinkNode` actors, not by the `Market` directly. A `DemandSinkNode` has its own `demand_dist` and is bound to a specific `product_id`; its demand target for a tick is the result of composing multiple multipliers. `Market.sample_demand` as written is structurally incompatible with this model: it takes a `store` argument for freshness and cross-product access, and it draws from `world_rng` in a way that was designed for a flat roster of `Store` objects.

**Decision — Move demand sampling into `DemandSinkNode.demand_target` and shrink `Market` to a pure multiplier engine.**

**New `Market` contract.** `Market.demand_multiplier(pid, region, tick) -> float` returns the combined float multiplier: `seasonality(pid, tick) × regional_shock(region, tick) × active_disruption_multiplier(pid, region, tick)`. It does not draw from `world_rng`; it reads the current market state that `market.tick()` already advanced. `market.tick()` continues to advance per-region demand/supply state, cycle, trend (existing math unchanged). `ItemRegistry`, `EventEngine`, `freshness_curve`, and `lifecycle_clock` are unchanged.

**`DemandSinkNode.demand_target` composition.** Each tick, a sink's demand target is:

```
demand_target = demand_dist.sample(world_rng)
              × market.demand_multiplier(pid, region, tick)
              × item_registry.stage_multiplier(pid)
              × freshness_curve.multiplier(α, β, tick − activation_tick[pid])
```

This composition re-grounds ADR 0001 (two-layer lifecycle × freshness), ADR 0002 (per-stage transition), and ADR 0003 (CRN demand for all products) onto the sink path rather than the `store` path.

**CRN preservation.** ADR 0003 requires that `world_rng` is consumed once per catalog product per tick, in catalog order, regardless of which products are active. On the sink path, this is honoured by having every sink (or the `consume_demand_sinks` orchestration) draw from `world_rng` once per catalog product each tick — including products the sink doesn't demand this tick — in catalog-iteration order. This mirrors the original "call `sample_demand` for every product in `store.inventory` every tick" loop.

**`sample_demand` retained until Phase 4.** The legacy `Store` engine uses `Market.sample_demand` and continues to work through Phase 3. `sample_demand` is deleted in Phase 4 (issue 11) when the `Store` engine is retired. In the interim both coexist: `sample_demand` for legacy callers, `demand_multiplier` for the graph engine.

Considered alternatives:

(a) **Keep `sample_demand` and have `DemandSinkNode` call it with a synthetic "store" argument.** Rejected because `sample_demand` draws from `world_rng` internally and its draw order was designed for a flat store roster. Passing a synthetic store into it would require either reimplementing the draw logic inside the method or coupling `DemandSinkNode` to `Store` internals — neither is acceptable.

(b) **Move demand sampling entirely to `ItemRegistry`.** Rejected because `ItemRegistry` manages catalog metadata and lifecycle stage, not regional market dynamics. Putting the full demand pipeline there would invert the module's purpose.

(c) **Keep `Market` as a full demand oracle, add a `DemandSinkNode`-aware overload.** Rejected because the two demand models (`store`-based and `sink`-based) would coexist indefinitely inside `Market`, coupling the module to both execution paths. The goal is a clean migration where `Market` becomes a pure multiplier source that neither path depends on for RNG draws.

## Cross-references

- ADR 0001 — Two-layer lifecycle: global PLC × per-store freshness (re-grounded on the sink path by this ADR)
- ADR 0002 — Per-stage transitions and dead stage (stage multipliers consumed by `item_registry.stage_multiplier(pid)` in the sink demand composition)
- ADR 0003 — CRN demand for all products (the one-draw-per-catalog-product-per-tick contract preserved on the sink path)
- ADR 0011 — Multi-echelon graph (`DemandSinkNode` is the node type that owns `demand_target`)
- ADR 0013 — Cash flow conservation (`income_rate` credited to sinks at `consume_demand_sinks`, separate from the demand sampling here)
- ADR 0014 — Tick phasing (`tick_world` advances market state before `consume_demand_sinks` calls `demand_multiplier`)
