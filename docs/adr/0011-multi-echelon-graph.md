# Multi-echelon graph as the core simulation model

Status: Proposed

The simulator previously modelled each actor as an independent `Store` that consumed demand from `Market.sample_demand` and placed orders into thin air. Every order was implicitly accepted in full; no upstream actor ever decremented inventory or received payment. This restricted the questions the simulator could answer — supply contention, supplier-side economics, and structural variation between echelons were all inexpressible.

**Decision — Replace the single-`Store` flat model with a directed acyclic graph of typed nodes as the canonical world representation.**

Three concrete node types share a `Node` ABC:

- `FactoryNode` — the lowest echelon (level 0). Produces exactly one product per tick at `unit_cost`. `list_price == unit_cost` by construction (ADR 0013). Holds inventory available for purchase by downstream buyers.
- `IntermediateNode` — a distribution centre, warehouse, or shop. Holds inventory of one or more products, sets selling prices, imposes per-product minimum order sizes, and accumulates pending deliveries in transit. The warehouse-vs-shop distinction is expressed via the informational-only `tags: list[str]` attribute — it never branches simulation mechanics.
- `DemandSinkNode` — the highest echelon. Generates cash at `income_rate` each tick (the only source of new cash per ADR 0013) and buys units from upstream intermediates. Each sink is bound to a single `product_id`; its `demand_dist` drives the per-tick demand target, composed with Market, lifecycle, and freshness multipliers.

**Graph structure.** `Graph(nodes, edges)` holds the validated topology. `EdgeSpec(supplier_id, buyer_id, default_lead_time, per_product_lead_time)` carries per-edge lead times with optional per-product overrides. `build_graph(nodes, edges)` raises `ValueError` on cycles, isolated nodes, or same-echelon supplier links. `compute_levels` assigns each node an echelon level via longest-path-from-any-factory; this level is the scheduling unit for the tick cascade (ADR 0014).

**Degenerate single-store case.** A three-node chain (1 `FactoryNode` → 1 `IntermediateNode` → 1 `DemandSinkNode`) is structurally equivalent to the old single-`Store` world. The RL training pipeline, textbook tuning study, and every existing scenario migrate onto the graph engine through this degenerate topology; no parallel legacy engine is maintained.

Considered alternatives:

(a) **Keep the flat `Store` model and add a supplier-buyer edge as a `Store` attribute.** Rejected because the `Store` abstraction already mixes inventory-holder, demand-consumer, and cash-holder roles. Adding supplier-buyer wiring to `Store` would deepen the coupling rather than surface the topology as a first-class concept that graph algorithms can reason about.

(b) **Use a generic node class with a `node_type` enum.** Rejected because typed subclasses (`FactoryNode`, `IntermediateNode`, `DemandSinkNode`) allow the type checker to enforce per-type invariants (e.g. factories have `produces_product_id`, sinks have `demand_dist`) at construction time rather than at runtime.

(c) **Allow cycles (e.g. lateral trade between shops).** Rejected because the tick cascade (ADR 0014) requires assignable echelon levels, which requires a DAG. If bidirectional or lateral trade becomes necessary later, a new ADR is needed to replace the cascade with a simultaneous clearing mechanism.

**Migration.** The legacy `Store` engine remains runnable through Phase 3 for backward compatibility. Phase 4 retires it: `Store`, `StoreInstance`, `StoreTemplate`, `make_stores`, and `Runner` are deleted; all scenarios migrate to `build_graph_world`; legacy tests are rewritten against the graph engine.

## Cross-references

- ADR 0012 — Central table + sequential FCFS allocation contract (the mechanism by which buyers read and commit to offers in this graph)
- ADR 0013 — Cash flow conservation (the economic constraints shared by all node types in this graph)
- ADR 0014 — Tick phasing as upward cascade by echelon level (the scheduling rule that assigns meaning to the levels computed by `compute_levels`)
- ADR 0015 — Demand-sinks as the demand source (re-grounds ADRs 0001–0003 on `DemandSinkNode.demand_target`)
- ADR 0010 — sim as base for ML layers (the graph engine replaces the `Store`-based `Simulation` as the canonical sim execution path)
