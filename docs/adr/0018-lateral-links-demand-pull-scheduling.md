# Lateral supplier links and demand-pull topological scheduling

Status: Accepted — supersedes ADR 0014; resolves ADR 0011 alternative (c). Implementation pending (until merged, `build_graph` and `Simulation.tick` remain on the tiered-cascade contract).

ADR 0014 scheduled each tick as a cascade keyed on echelon level (longest-path from any factory), shuffling same-level buyers in parallel. That breaks under two conditions: (1) a buyer depends on a **same-level supplier** (a lateral link) — settlement order then depends on the shuffle rather than the supply dependency; (2) shortcut edges make BFS-level and longest-path disagree on what counts as a "peer" link. ADR 0011 alternative (c) explicitly deferred lateral trade to "a new ADR … to replace the cascade with a simultaneous clearing mechanism." This is that ADR — and it reaches a different conclusion than (c) anticipated: a topological walk, not simultaneous clearing.

The motivating goal: a node's tier should be expressed by **price + min-order + lead-time** (so the policy chooses suppliers by price/availability), not by a rigid echelon index. Supplier choice already lives on the policy (`MultiSupplierTextbookPolicy._split_across_suppliers`); what was missing is lateral edges and a schedule that survives them.

**Decision — Replace the echelon-level cascade with a demand-pull topological walk over the full edge set, permit lateral supplier links, and replace level-based graph validation with type-based validation.**

1. **Graph shape — union-DAG.** Lateral edges (`intermediate→intermediate` at any depth) are permitted. The *union* of all edges must remain acyclic: a given pair of nodes may trade in only one direction across all products. Same-tick opposite-direction product flows (product 1 A→B while product 2 B→A) are **not** supported — model them through a shared upstream node. (See rejected "per-product DAG" below.)

2. **Validation (`validate_dag`).** Drop the BFS same-level check. Enforce **type rules** on edge endpoints instead: supplier ∈ {factory, intermediate}, buyer ∈ {intermediate, sink}, and reject **factory→sink** (flow must pass through ≥1 intermediate — the multi-echelon economics constraint). Keep the cycle, self-loop, and unreachable-node checks. `validate_dag`/`build_graph` take a `node_types: dict[str, str]` (id → `Node._node_type`) so the check needs no `Node` import — graph.py stays purely structural. These type edges were previously *silently* mishandled (a `sink→` edge buys from a seller with no published offer; a `→factory` edge is never read), so rejecting them is strictly safer.

3. **Schedule — demand-pull (reverse-topological).** Process graph-terminal buyers (sinks) first, then their suppliers, … factories last (produce step). Implemented as **Kahn's algorithm on the reversed graph**; each *ready-set* of mutually-incomparable peers is shuffled with `allocation_rng` (ADR 0016) for FCFS fairness. The topological structure is computed **once at `build_world`** (topology is static) and cached on the `Simulation`; only the ready-set shuffle re-draws each tick. A strict chain degenerates to singleton ready-sets (the shuffle is a no-op), preserving the RL degenerate-chain contract.

   Because `execute_buy` decrements the seller's on-hand inventory at sale time (allocation.py step 4b), by the time an intermediate is processed **every downstream buyer has already transacted** — so it reorders against *complete current-tick demand* and *post-sale inventory*. This aligns the code with ADR 0014's stated-but-never-implemented demand-pull intent and lets us **delete the `prev_tick_sales` one-tick-lag hack**: the observation injects `observed_sales` (complete current-tick sales, guaranteed complete by the ordering), and the confounded inventory-delta fallback is removed.

4. **Min-order-aware routing.** Buyers skip a supplier whose `min_order` they cannot meet and fall through to the next feasible supplier. This fixes silent lost sales that the relaxed graph makes common (a low-demand sink reaching a high-min-order warehouse). Applied in `_default_sink_action`; `_split_across_suppliers` cheapest-first already honours both min-order layers.

5. **Rejection log.** `AllocationResult` gains a `reason` field (`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`, `insufficient_capacity` — the binding constraint). A per-tick `rejections` stream is surfaced in the run log, each entry `{tick, buyer_id, supplier_id, pid, qty_requested, qty_filled, qty_rejected, reason}`, including `unmet_demand` entries (`supplier_id=None`) for demand no feasible supplier could fill — the true lost-sale measure once min-order fall-through is in place. Always-on.

6. **`level` is no longer a scheduling unit.** `compute_levels` (longest-path) is retained as a **display-only** hint (`inspect.py`); `node.level` carries no functional meaning (longest-path is misleading under lateral links). `tags` remain informational.

**Unification.** The duplicated cascade in `tick()` and `tick_decide_and_settle()` collapses into one `_run_demand_pull_schedule(table, current_tick)` helper used by both. The RL action-injection seam (between `tick_world` and `tick_decide_and_settle`) is unchanged.

## Rejected alternatives

- **Fully-connected marketplace** (any node buys from any node; edges as pure metadata). Rejected: the `CentralTable` already provides global visibility — the edge set is only an allow-list + lead-time/min-order carrier. Removing the allow-list reintroduces cycles *at runtime* (policies can pick A→B and B→A for the same product/tick), forcing simultaneous clearing or multi-pass settle, which loses the live-table FCFS depletion signal and the multi-echelon identity. Less robust, not more.
- **Per-product DAG** (validate/schedule per product) to allow same-tick opposite-direction product flows. Rejected for now: pushes scheduling and the holistic `decide()` interface to (node×product) granularity for a capability with no concrete current need. Revisit if specialized cross-shipping DCs become a requirement.
- **Keep push (forward-topological).** Rejected: retains the one-tick demand lag and the `prev_tick_sales` hack, and contradicts ADR 0014's documented demand-pull intent.

## Cross-references

- Supersedes ADR 0014 — Tick phasing as upward cascade by echelon level.
- Resolves ADR 0011 alternative (c) — lateral trade (with a topological walk rather than the simultaneous clearing (c) anticipated).
- Builds on ADR 0012 (central table + FCFS), ADR 0016 (allocation shuffle sub-seed), ADR 0013 (cash conservation).
