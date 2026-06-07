# PRD: Lateral supplier links + demand-pull topological scheduling

Status: ready-for-agent

Reference decision: `docs/adr/0018-lateral-links-demand-pull-scheduling.md` (supersedes ADR 0014,
resolves ADR 0011 alternative (c)). Builds on ADR 0012 (central table / FCFS), ADR 0013 (cash
conservation), ADR 0016 (allocation shuffle sub-seed). Until this lands, `build_graph` and
`Simulation.tick` remain on the tiered-cascade contract — this PRD replaces that contract.

## Problem Statement

I want to model supply chains where a node's tier is expressed by **price + min-order +
lead-time**, not by a rigid echelon index — so that the *policy* chooses which supplier to order
from based on price and availability, and so that **lateral trade** (shop→shop, warehouse→warehouse
at the same tier) is possible, the way it is in real life.

Today the graph forbids any same-level supplier link (`validate_dag` BFS check), and the tick is
scheduled as a cascade keyed on echelon level (`compute_levels`, longest-path-from-factory) with
same-level buyers shuffled in parallel. That design is both too restrictive and fragile:

- I can't author a lateral link at all — `build_graph` raises.
- The cascade breaks when a buyer depends on a same-phase supplier (settlement order then depends
  on the shuffle, not the supply dependency), and when shortcut edges make BFS-level and
  longest-path disagree on what counts as a peer.
- ADR 0014's prose argues for demand-pull ("sinks first") but the code runs *push* (level-1
  intermediates first, sinks last), so an intermediate reorders against a **one-tick-lagged**
  demand signal — papered over by a `prev_tick_sales` hack.
- When a buyer's order falls below a supplier's `min_order`, the order is **silently** rejected
  (`_default_sink_action` never checks `min_order`) — a lost sale with no record of why. This is
  rare today but becomes common the moment warehouse→sink edges are legal.
- There is no log of rejected or lost sales, so debugging why demand went unmet is guesswork.

## Solution

Replace the echelon-level cascade with a **demand-pull topological walk over the full edge set**,
allow **lateral supplier links**, and replace level-based graph validation with **type-based**
validation. From the modeller's perspective:

1. **Author lateral edges freely.** `intermediate→intermediate` edges at any depth are legal. The
   graph stays a union-DAG (no cycles); a pair of nodes trades in one direction across all
   products. Validation now rejects only what is genuinely illegal: `factory→sink` (must pass
   through ≥1 intermediate), `→factory` (factories don't buy), and `sink→*` (sinks don't sell) —
   plus cycles, self-loops, and unreachable nodes.

2. **Scheduling just works on any legal topology.** The tick processes graph-terminal buyers
   (sinks) first, then their suppliers, factories last — a reverse-topological order computed by
   Kahn's algorithm on the reversed graph, with mutually-incomparable peers shuffled by
   `allocation_rng` for FCFS fairness. Lateral links slot in as ordinary ordering constraints. A
   strict chain degenerates to the old unambiguous order.

3. **Demand signal is current, not lagged.** Because selling decrements seller inventory at sale
   time, by the time an intermediate decides, all its downstream buyers have already transacted, so
   it reorders against complete current-tick demand (`observed_sales`) and post-sale inventory. The
   `prev_tick_sales` lag hack and the confounded inventory-delta fallback are deleted.

4. **Supplier choice respects min-order.** Buyers skip a supplier whose `min_order` they can't meet
   and fall through to the next feasible supplier — so the "warehouse = bulk/cheap, shop =
   small/pricey" economics work as intended instead of producing phantom lost sales.

5. **Rejections and lost sales are logged.** Every rejection carries a reason; the run log gains a
   per-tick `rejections` stream including `unmet_demand` entries, so I can debug exactly what went
   unfilled and why.

## User Stories

1. As a modeller, I want to author an `intermediate→intermediate` (lateral) edge, so that two
   shops or two warehouses at the same tier can trade.
2. As a modeller, I want `build_graph` to accept any union-DAG of legal edge types, so that I am
   not constrained by an echelon-level rule that no longer reflects my model.
3. As a modeller, I want `factory→sink` edges rejected, so that the multi-echelon economics
   (flow through ≥1 intermediate) are preserved.
4. As a modeller, I want `→factory` and `sink→*` edges rejected with a clear error, so that
   nonsensical edges fail loudly at authoring time instead of silently no-op'ing at runtime.
5. As a modeller, I want cycles, self-loops, and unreachable nodes still rejected, so that the
   graph remains a valid union-DAG.
6. As a modeller, I want the validator to tell me *which* rule an edge violated, so that I can fix
   my topology quickly.
7. As a simulation author, I want the tick scheduled in demand-pull (reverse-topological) order, so
   that each tier reacts to its downstream's actual current-tick demand rather than a forecast.
8. As a simulation author, I want incomparable peer nodes shuffled deterministically by
   `allocation_rng`, so that no node has a permanent FCFS advantage on a scarce supplier and runs
   stay reproducible from the seed.
9. As a simulation author, I want a strict chain (factory→S→sink) to behave unambiguously
   (singleton ready-sets, no shuffle effect), so that existing degenerate scenarios are unaffected
   in shape.
10. As an intermediate-node policy, I want to observe complete current-tick sales
    (`observed_sales`) and post-sale inventory when I decide, so that my reorder uses an un-lagged
    demand signal.
11. As a maintainer, I want the `prev_tick_sales` one-tick-lag machinery removed, so that the code
    no longer carries a workaround for the old push schedule.
12. As a maintainer, I want the inventory-delta fallback removed, so that the demand signal is
    always the exact current-tick sales (possibly zero) rather than a delivery-confounded proxy.
13. As a buyer (sink or intermediate), I want to skip a supplier whose `min_order` I can't meet and
    route to the next feasible supplier, so that I don't lose a sale a pricier supplier would have
    filled.
14. As a modeller, I want a warehouse-type intermediate to use a low price + high `min_order` and a
    shop-type a higher price + low `min_order`, so that tier differences emerge purely from these
    fields with no engine branching.
15. As a debugging modeller, I want every allocation rejection recorded with a `reason`
    (`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`,
    `insufficient_capacity`), so that I can see why an order was not filled.
16. As a debugging modeller, I want genuine lost sales recorded as `unmet_demand` entries (demand
    no feasible supplier could fill), so that I can distinguish avoided rejections from real lost
    demand.
17. As a debugging modeller, I want the rejection log surfaced per-tick in the run log alongside
    `node_orders`/`node_pending`, so that I can inspect it with the existing tooling.
18. As an RL practitioner, I want the `tick_world` / `tick_decide_and_settle` seam unchanged, so
    that the RL env and its action-injection shim keep working without modification.
19. As an RL practitioner, I want the degenerate three-node training chain to remain valid under the
    new schedule, so that training/eval pipelines still run.
20. As a maintainer, I want the duplicated cascade in `tick()` and `tick_decide_and_settle()`
    collapsed into one shared `_run_demand_pull_schedule` helper, so that the two paths cannot
    drift.
21. As a maintainer, I want the topological structure computed once at `build_world` and cached,
    with only the per-tick ready-set shuffle re-drawing `allocation_rng`, so that scheduling is
    cheap and the CRN draw cadence is preserved.
22. As an inspector, I want `compute_levels` retained only as a display-only longest-path hint, so
    that `inspect.py` can still show a depth number while nothing functional depends on it.
23. As a maintainer, I want `node.level` to carry no functional meaning, so that the misleading
    "echelon level" under lateral links doesn't drive any behavior.
24. As a researcher running CRN-paired eval, I want both arms to share the identical new schedule,
    so that paired comparisons stay valid even though absolute golden outputs shift.
25. As a maintainer, I want graph.py to remain free of `src.sim` imports (types passed as a
    `dict[str, str]`), so that it stays a pure, test-from-anywhere structural module.

## Implementation Decisions

**Graph shape — union-DAG.** Lateral `intermediate→intermediate` edges (any depth) are permitted.
The union of all edges must remain acyclic; a pair of nodes trades in only one direction across all
products. Same-tick opposite-direction *per-product* flows are **not** supported (model via a shared
upstream).

**Validation (`validate_dag`).** Drop the BFS same-level check. Add **type rules** on edge
endpoints: supplier ∈ {factory, intermediate}, buyer ∈ {intermediate, sink}, reject `factory→sink`.
Keep cycle (3-colour DFS), self-loop, and unreachable-node checks. `validate_dag` / `build_graph`
gain a `node_types: dict[str, str]` parameter (id → `Node._node_type`, values `"factory"` /
`"intermediate"` / `"demand_sink"`); `build_world` constructs it from `scenario.nodes`. No `Node`
import enters graph.py. Errors name the violated rule.

**Reverse-topological scheduler (deep module in graph.py).** A pure function builds the schedule
structure from `(nodes, edges)`: in-degree over the **reversed** edge set so that a node is "ready"
only once all its downstream buyers are processed (sinks/graph-terminals start ready). The structure
yields successive ready-sets of mutually-incomparable nodes. A separate seam applies the
`allocation_rng` shuffle to each ready-set to produce the per-tick total order. The structure is
deterministic and unit-testable without a `Simulation`; the shuffle is injected, not embedded.

**Unified walk (`_run_demand_pull_schedule` in runner.py).** One helper replaces the duplicated
buyer loops in `tick()` and `tick_decide_and_settle()`. It: walks ready-sets in demand-pull order
(shuffling each via `allocation_rng`), runs each node's observe→decide→`execute_buy` per line, then
the factory `produce` step. The topological structure is computed once in `build_world` and cached
on `Simulation`; only the shuffle re-draws each tick. The RL action-injection seam (between
`tick_world` and `tick_decide_and_settle`) is unchanged.

**Demand signal.** Sales are accumulated during the walk into `Simulation._tick_sales`. When an
intermediate is processed, the observation injects `observed_sales = self._tick_sales.get(node.id,
{})` — guaranteed complete by the ordering. The two policy read-sites
(`MultiSupplierTextbookPolicy.decide`, `_SingleSupplierAdapter.decide`) rename
`prev_tick_sales` → `observed_sales` and map it into `observation["sales"]`. The inventory-delta
fallback is removed. The prev-tick capture/swap in the runner is removed.

**Min-order-aware routing.** `_default_sink_action` skips a supplier when the quantity it would
route there is below that supplier's `min_order`, falling through to the next-cheapest feasible
supplier. `_split_across_suppliers` cheapest-first **needs no change**: it delegates to
`_default_cheapest_first_strategy`, which computes `effective_min = max(offer.min_order,
buyer_min_order_floor)` and skips a supplier both pre-allocation (`remaining < effective_min`) and
post-clamp (`qty < effective_min`) — both min-order layers are honoured (verified). The single
confirmed silent-lost-sale gap is the policy-less `_default_sink_action` default path.

**Rejection log.** `AllocationResult` gains `reason: str | None`, set at the binding constraint in
`execute_buy` (`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`,
`insufficient_capacity`). `Simulation` accumulates a per-tick `_tick_rejections: list[dict]`
(reset each tick), surfaced in the run log as `ticks[i]["rejections"]`; each entry is
`{tick, buyer_id, supplier_id, pid, qty_requested, qty_filled, qty_rejected, reason}`. Sink-level
`unmet_demand` entries (`supplier_id=None`, `qty_rejected = demand_target − total_filled`) record
genuine lost sales. Always-on (no flag).

**`level` demotion.** `compute_levels` (longest-path) is retained as a display-only hint consumed
only by `inspect.py`. `node.level` is set but functionally inert. The level-bucket scheduling and
the `Simulation.levels` scheduling dict are removed.

**Out-of-scope by decision (recorded in ADR 0018):** fully-connected marketplace, per-product DAG
scheduling, forward-topological (push) order, and routing-layer *skip* tracing (only actual
rejections / lost sales are logged).

## Testing Decisions

Good tests assert **external behavior**, not implementation details: given a topology / inputs, a
function returns the right order / verdict / log — not how it iterated. Lean on the pure deep
modules so most coverage needs no `Simulation`. Prior art: `tests/sim/test_graph_runner_chain.py`
(end-to-end Runner on a chain), existing graph-validation tests, and the `execute_buy` clamp-path
unit tests from ADR 0012.

Modules to test (all four areas confirmed in scope):

1. **graph.py pure modules.**
   - *Reverse-topo scheduler:* a strict chain yields singleton ready-sets (shuffle is a no-op);
     a lateral edge A→B forces A after B; mutually-incomparable peers appear in a seed-deterministic
     shuffled order; sinks (graph-terminals) are first, factories last; identical topology + seed →
     identical order (CRN), different seed → different peer order.
   - *Type validation:* `factory→sink`, `→factory`, `sink→*` each rejected with the rule named;
     `intermediate→intermediate` (lateral) accepted; cycles, self-loops, and unreachable nodes
     still rejected; `factory→intermediate` and `intermediate→sink` accepted.
2. **allocation + routing.**
   - `AllocationResult.reason` is correct for each clamp path (no offer, below min-order, stock,
     cash, capacity), including the binding constraint when a clamp drives the fill to zero.
   - `_default_sink_action` skips a too-small (below `min_order`) cheapest supplier and falls
     through to the next feasible one; asserts no silent lost sale when a pricier shop can fill.
3. **Rejection log (Runner, end-to-end).**
   - The run log's per-tick `rejections` stream contains entries with correct
     quantities/reasons for an engineered shortage; `unmet_demand` entries appear with the right
     residual when no feasible supplier can fill demand.
4. **Demand-pull integration (Runner, end-to-end, lateral graph).**
   - On a graph with a lateral link, `observed_sales` reflects current-tick demand (an intermediate
     reorders in response to the same tick's downstream purchase); cash conservation holds
     (ADR 0013); the run is bit-deterministic from the seed.

## Out of Scope

- Fully-connected marketplace / removing the edge allow-list (rejected in ADR 0018).
- Per-product DAG validation and per-product (node×product) scheduling for same-tick
  opposite-direction product flows (rejected for now; revisit if specialized cross-shipping DCs are
  needed).
- Forward-topological (push) scheduling.
- Tracing routing-layer *skips* (proactive supplier skips that successfully route elsewhere) — only
  actual rejections and genuine lost sales are logged.
- A toggle to disable the rejection log (add later only if a long run bloats the log).
- Re-tuning RL agents or refreshing tuning studies; absolute golden outputs and CRN-paired uplift
  numbers will shift and are expected to be regenerated, but that regeneration is not part of this
  PRD.
- Reworking `Market`, lifecycle, or freshness (untouched).

## Further Notes

- **ADR 0014's prose contradicts its code — confirmed, and it matters.** Verified in
  `runner.py`: the cascade is `for p in range(1, max_level + 1)` over `compute_levels`
  (longest-path-from-factory, factory=0), so on a legal chain `factory(0)→warehouse(1)→shop(2)→
  sink(3)` it processes **warehouses first, sinks last** — *push*, not the demand-pull the ADR 0014
  prose claims. Consequence for the implementer: **the demand-pull switch is a behavior change, not
  a pure refactor.** Today an intermediate at level *p* reorders before its downstream (levels
  *p+1…*) transact this tick, so it can only see *last* tick's demand — which is exactly why the
  `prev_tick_sales` lag hack exists (`runner.py` ~178–183, 271). Reversing to sinks-first changes
  the realised numbers, so expect golden/regression outputs to shift (not just shuffle shape).
  ADR 0018 is authoritative.
- **`_split_across_suppliers` min-order behavior — verified correct, NOT a gap.** It delegates to
  `_default_cheapest_first_strategy` (`policy.py` ~2156–2210), which honours both layers via
  `effective_min = max(offer.min_order, buyer_min_order_floor)`, skipping a supplier pre-allocation
  (`remaining < effective_min`) and post-clamp (`qty < effective_min`). The CONTEXT.md claim holds.
  The **only** confirmed silent-lost-sale path is the policy-less `_default_sink_action` default
  (ADR 0018 decision 4 / "Min-order-aware routing" above).
- **Why the prev-tick hack can die:** `execute_buy` step 4b decrements seller on-hand inventory at
  sale time, so under demand-pull a seller's post-sale inventory and complete sales are both known
  by the time it decides.
- **CRN note:** the *shape* of the shuffles changes (ready-sets vs. level buckets), so previously
  cached golden outputs won't match. Paired comparisons remain valid because both arms use the
  identical new schedule.
- CONTEXT.md has been updated with forward-pointers (`Graph`, `Phase cascade`, `Allocation`, new
  `Rejection log` term) tagged "decided, not yet implemented"; flip those to present-tense as the
  code lands.
