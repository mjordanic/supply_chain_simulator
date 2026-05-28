# Tick phasing as upward cascade by echelon level

Status: Proposed

The single-store simulator had a simple two-phase tick: world-advance (market, events, lifecycle) then per-store observe-decide-settle. With multiple echelons, a principled scheduling rule is needed: when a demand-sink buys from an intermediate, should the intermediate have already decided its upstream orders for this tick? The cascade direction determines what information flows within a tick vs. across ticks.

**Decision — Execute one tick as a deterministic cascade of phases ordered by echelon level, sinks first, factories last.**

**Tick structure:**

1. `tick_world`: `market.tick()` → `event_engine.tick(market)` → `item_registry.tick()`. Advances shared world state: demand/supply state, active events, lifecycle stages.
2. `publish_offers`: all sellers (factories and intermediates) publish their current inventory offers to the `CentralTable` (ADR 0012).
3. **Phase cascade** `for p in 1..max_level` (sinks at level `max_level`, factories at level 0, but factories produce rather than decide):
   - Shuffle buyers at level `p` using `allocation_rng` (ADR 0016).
   - For each buyer (in shuffled order): `observe` → `decide` → `execute_buy` per order line.
4. `produce`: factories run their `FactoryPolicy.decide` and produce up to `capacity_per_tick` units, replenishing factory inventory.
5. `deliver`: scheduled `EventEngine` callbacks fire, delivering in-transit inventory to buyers.
6. `consume_demand_sinks`: each `DemandSinkNode` draws its demand target and credits `income_rate` to its cash balance.

**Physical lead time still delays delivery.** Orders placed at phase N in tick T arrive at tick `T + lead_time(supplier, buyer, pid)`, not within the same tick. The `deliver` step fires callbacks whose arrival time was set at order placement; it does not deliver orders placed in the same tick. This preserves the "goods take time to travel" invariant from the single-store era.

**Information flow within a tick.** A buyer at level `p` sees the central table as updated by all buyers at levels `< p` who have already transacted this tick. Sinks (highest level) see all intermediate-to-intermediate trades that happened earlier in the cascade. Intermediates at level `p` see factory offers at full capacity (factories haven't produced yet this tick) but may see other intermediates' prior commits. This pull-from-above ordering matches the natural demand-pull structure of most supply chains: consumer demand propagates upward, each tier reacting to what the tier above just did.

**Why sinks before intermediates, rather than the reverse?** If intermediates ordered first (push model), they would be guessing at downstream demand without observing it. The cascade from sinks upward lets each tier respond to what its downstream buyers just committed to purchasing — a demand-pull signal rather than a forecast. This is the standard operating mode in textbook supply-chain models (Silver/Pyke/Peterson chapter on multi-echelon systems).

**Why not simultaneous clearing?** Simultaneous clearing (all buyers at all levels see the tick-start snapshot and submit demand simultaneously; a global solver allocates proportionally) was considered and rejected as "Model 2" in the PRD. It loses the live-table signal (a buyer can't observe that a supplier is nearly out before deciding whether to split its order), and requires a global solver pass that couples all decisions. Sequential FCFS with shuffle is simpler and more informative for routing policies.

**Phase index convention.** Level 0 = factories. Level `max_level` = sinks. The cascade runs level `1` through `max_level` (sinks) in the buyer-decision loop; level-0 factories run separately in the `produce` step because their decision is production quantity rather than a buy order.

## Cross-references

- ADR 0011 — Multi-echelon graph (echelon levels are assigned by `compute_levels` and used as the phase index here)
- ADR 0012 — Central table + FCFS allocation (`publish_offers` and `commit` happen at the tick boundaries defined by this phasing)
- ADR 0013 — Cash flow conservation (`consume_demand_sinks` credits `income_rate` at the end of each tick's cascade)
- ADR 0016 — RNG/CRN extension: allocation sub-seed (`allocation_rng` drives the buyer shuffle at each phase of the cascade)
