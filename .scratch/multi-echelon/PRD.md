# PRD: Multi-echelon graph as the core simulation model

Status: ready-for-agent

Related plan: `/Users/mislavjordanic/.claude/plans/enumerated-wandering-eich.md` (phased implementation roadmap)

Related ADRs (will be authored alongside the implementation):
- ADR 0011 — Multi-echelon graph as the core sim model (supersedes single-`Store` semantics)
- ADR 0012 — Central table + sequential FCFS allocation contract
- ADR 0013 — Cash flow conservation across nodes
- ADR 0014 — Tick phasing as upward cascade by echelon level
- ADR 0015 — Demand-sinks as the demand source; `Market` shrinks to a multiplier engine (re-grounds ADRs 0001–0003 onto the sink path)
- ADR 0016 — RNG/CRN extension: the `allocation` sub-seed stream

## Problem Statement

The simulator today is single-echelon. A run can contain multiple `Store`s, but they do not trade with one another — each `Store` independently consumes demand drawn from `Market.sample_demand` and places orders into thin air (the order is an anonymous lambda callback scheduled by `EventEngine` that materialises inventory after a lead time, with no supplier on the other side ever decrementing). This restricts the questions the simulator can answer:

1. **No supply contention.** Every order is implicitly accepted in full. A research question like "what happens when two retailers compete for limited warehouse inventory?" cannot be expressed, and any policy that should reason about supplier reliability has no signal to learn from.
2. **No supplier-side economics.** Cash flow ends at the store boundary. Goods materialise at a fixed `unit_cost` without an upstream actor whose finite capacity, pricing power, or production cost matters.
3. **No structural variation between echelons.** A "warehouse vs. shop" question has no representation: every actor in the world is structurally a `Store` with `Market`-driven demand.
4. **The textbook reorder-policy family and the RL agent have no notion of supplier choice.** `OrderUpToPolicy` and friends emit `order[pid] = qty` with the supplier implicit. A policy that wants to split an order across two warehouses, fall back when one is short, or enforce per-supplier minimum order sizes cannot be expressed in the current `Policy.decide()` contract.

These limits cap the methodological contributions the simulator can support — supply-chain RL papers worth publishing on this codebase (per `models_canvas.md`) increasingly require multi-echelon mechanics.

## Solution

Rewrite `src/sim/` so a **directed acyclic graph of typed nodes** is the world. Three node types — `FactoryNode`, `IntermediateNode`, `DemandSinkNode` — share a `Node` ABC. Factories produce one product each in exchange for cash at manufacturing cost. Intermediates hold inventory, set prices, and route orders across multiple upstream suppliers. Demand-sinks generate cash at an `income_rate`, demand units at a `demand_dist`, and buy from intermediates according to their attached policy.

Within one tick, decisions cascade upward by echelon level — phase 1: demand-sinks buy from intermediates; phase 2: intermediates order from upstream intermediates; phase N: factories produce. Within each phase, buyers are shuffled (from a new deterministic `allocation_rng`) and execute orders sequentially. Each buyer reads a **live, globally visible central table** of `(seller_id, pid) → {available_qty, list_price, min_order, fill_rate_recent}` and returns a multi-supplier-split order `order[pid] = [(supplier_id, qty), ...]`. Suppliers decrement inventory live as allocations land; buyers see partial fills and the rolling fill-rate signal at next observe. Physical delivery still takes lead-time ticks, but lead time now lives per-edge with optional per-product overrides.

Cash flows between nodes (buyer−/seller+) at the moment of allocation. Demand-sinks are the only source of new cash; factories absorb cash at zero margin (price == manufacturing cost). Holding cost and order fee still go to the void. The existing CRN/seed contract is preserved and extended with one new sub-seed (`allocation`) that drives the per-phase buyer shuffle.

The single-store case becomes a degenerate three-node graph (1 factory → 1 shop → 1 demand-sink), so the RL training pipeline, the textbook tuning study, and every existing scenario migrate onto the new engine rather than running on a parallel one. The `TextbookReorderPolicy` family migrates under a new `IntermediatePolicy` ABC, gaining a multi-supplier routing extension. `OrderUpToPolicy` remains the canonical CRN comparison anchor.

## User Stories

1. As a simulator user, I want each actor in the world to be a typed node in a directed graph, so that supplier-buyer relationships are first-class and not implicit.
2. As a scenario author, I want to declare a `FactoryNode` with `produces_product_id` and `unit_cost`, so that I can model the entry point of inventory into the chain.
3. As a scenario author, I want to declare an `IntermediateNode` with `carried_products`, `capacity`, and free-form `tags`, so that I can model warehouses, shops, distribution centres, or any other tier of the chain without coding a new class per tier.
4. As a scenario author, I want to declare a `DemandSinkNode` bound to one product with a `demand_dist` and an `income_rate`, so that consumer groups with different behaviours and budgets can be authored side-by-side.
5. As a scenario author, I want to attach an edge `(supplier, buyer)` with `default_lead_time` and optional `per_product_lead_time` overrides, so that distance and route reliability are expressible in the topology rather than buried in each store.
6. As a scenario author, I want multiple factories to be able to produce the same product, so that supply competition is a first-class feature of the model and not something I have to fake.
7. As a scenario author, I want graph construction to fail loudly on cycles, unreachable nodes, or same-level supplier links, so that invalid topologies cannot silently run.
8. As a policy author writing an `IntermediatePolicy`, I want my `decide(observation, central_table)` to return `order[pid] = [(supplier_id, qty), ...]`, so that I can split an order across suppliers, prefer the cheapest one, fall back when stock is short, and enforce per-(buyer, supplier) minimum-order discipline.
9. As a policy author, I want the central table I read during `decide()` to reflect prior buyers' allocations within this phase, so that my routing decisions account for live supply depletion rather than a stale tick-start snapshot.
10. As a policy author, I want my per-(buyer, supplier) minimum-order configuration honoured by my own decision logic, and supplier-imposed minimums honoured by the allocator, so that the two layers of min-order constraint do not silently collide.
11. As a policy author writing a `DemandSinkPolicy`, I want a default greedy-cheapest-feasible implementation provided out of the box, so that I do not need to write a buyer strategy before I can run a graph end-to-end.
12. As a policy author writing a `FactoryPolicy`, I want a `StaticFactoryPolicy(capacity_per_tick)` shipped, so that I can stand up a graph without yet authoring production economics.
13. As a researcher migrating the textbook policies, I want `OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, and `PeriodicReorderPolicy` to be re-rooted on a `MultiSupplierTextbookPolicy(IntermediatePolicy)` base, so that the published textbook math (rate estimator, safety horizon, two-pass fair-share allocator) is preserved while the order-emission step is upgraded to multi-supplier splits.
14. As a researcher, I want the default `_split_across_suppliers` strategy to be cheapest-first with both min-order layers honoured, and pluggable via a `routing_strategy` constructor kwarg, so that I can A/B alternative routing rules without subclassing.
15. As a researcher running CRN-paired evaluations, I want `OrderUpToPolicy` to remain the canonical comparison anchor on the graph engine, so that uplift numbers are comparable across the single-store legacy and the multi-echelon era.
16. As a researcher relying on the seeding contract, I want the existing three determinism invariants to continue to hold post-rewrite, so that paired evaluations are still bit-identical across worlds: (a) same `world_seed` yields identical market, event, lifecycle, *and* buyer-shuffle sequences; (b) same `(node_template, init_seed)` yields identical step-0 node state regardless of attached policy; (c) policy swap on one node does not perturb `world_rng` or `allocation_rng`.
17. As a researcher, I want a new `allocation` sub-seed added to `_SUB_SEED_PARAMS` and derived via the existing `_derive_seed(seed, purpose)` formula, so that the buyer shuffle is reproducible and orthogonal to the existing four sub-seeds.
18. As a simulator maintainer, I want one tick to execute as a deterministic cascade of phases ordered by echelon level (sinks → intermediates upward → factories produce), so that information flow inside one business day matches the natural pull of demand up the chain.
19. As a simulator maintainer, I want physical lead time still to delay delivery between phases (orders placed in phase N arrive at `current_tick + lead_time`, not within the same tick), so that the cascade does not collapse the supply chain into instantaneous fulfilment.
20. As a simulator maintainer, I want cash transfers between nodes (buyer−, seller+) executed at allocation time, so that a buyer cannot claim more inventory than it can pay for, and so that the existing `order_cost` accounting timing carries over unchanged.
21. As a simulator maintainer, I want factory cash flow to be zero-margin by construction (price == `unit_cost`), so that the factory balance is a no-op book-keeping field and cash conservation reduces to "sinks create, ops destroys, the chain transfers."
22. As a simulator maintainer, I want demand-sinks to be the only place new cash enters the system, with `income_rate` as the single knob, so that scale of consumer purchasing power is a first-class scenario lever.
23. As a simulator maintainer, I want `Market.sample_demand` deleted and replaced with `Market.demand_multiplier(pid, region, tick)`, so that the old "Market drives demand at every store" path is fully retired and ItemRegistry / EventEngine / lifecycle survive intact.
24. As a simulator maintainer, I want lifecycle stage and freshness curve composition (ADRs 0001–0003) re-grounded inside `DemandSinkNode.demand_target`, so that CRN demand sampling across the catalog (one draw per product per tick) continues to hold on the new engine.
25. As an RL trainer, I want `RLEnv.reset` to build a degenerate three-node graph (1 factory → 1 trainable intermediate → 1 sink) per episode, so that single-store RL training continues to work and serves as the simplest entry point onto the new engine.
26. As an RL trainer, I want my action decoder to emit per-supplier splits, so that when an episode later runs against a graph with multiple upstream suppliers, the encoder/decoder pair generalises without a second rewrite.
27. As an RL trainer, I want my observation tensor extended with a `central_table_snapshot[product_slot]` block (supplier_count, min_price, mean_lead_time, mean_fill_rate), so that my policy can learn from supplier-side signal.
28. As a tuning user, I want `MultiSupplierTextbookPolicy` searchable in the existing Optuna study with new tunables `per_supplier_min_order_floor` and `routing_strategy`, so that tuning continues to find textbook headroom on the new engine.
29. As a tuning user, I want the per-trial CRN tuple expanded to include the `allocation` sub-seed, so that paired comparisons across trials remain bit-identical on the world stream.
30. As a CLI user, I want `uv run python main.py scenarios/<file>.py` to continue to work post-migration on a graph-shaped scenario, so that my muscle memory survives.
31. As a notebook reader, I want `Scenario.nodes_df()` and `Scenario.edges_df()` available alongside the existing `catalog_df()`/`market_df()` views, so that inspecting a graph scenario in `notebooks/03-inspect_scenario.ipynb` does not require reading source.
32. As a notebook reader, I want a converted `notebooks/01-quickstart.ipynb` that walks a 3-node chain end-to-end, so that the simplest path through the new engine is documented before contention or RL.
33. As a `WorldBuilder` user, I want a `world_to_graph(world, *, sink_density)` helper that synthesises a default topology from the existing `World.store_templates`, so that LLM-generated worlds continue to feed scenarios without re-prompting the LLM for graph topology.
34. As a release maintainer, I want the legacy `Store` engine deleted in one well-marked phase (Phase 4 of the plan) rather than left in place forever as a parallel module, so that the codebase does not carry two simulators on its back.
35. As a release maintainer, I want `Policy` kept as a one-release alias of `NodePolicy`, so that grep-friendly migration is possible without immediate breakage.
36. As a release maintainer, I want the unaffected modules (`event_engine.py`, `item_registry.py`, `distributions.py`, `freshness_curve.py`, `lifecycle_clock.py`, `metrics.py`, `agents/ppo.py`, notebooks 04–07) explicitly listed and not touched, so that the migration's blast radius is contained and reviewable.
37. As a research-direction owner, I want supply contention to be a first-class question the simulator can answer (two retailers, one shared warehouse, finite stock, shuffled FCFS), so that the publication bets in `models_canvas.md` that depend on this mechanic (PerceiverShop variants under contention, diffusion policies under disruption with finite supply) become tractable.

## Implementation Decisions

### Node hierarchy

- Three subclasses on a shared `Node` ABC: `FactoryNode`, `IntermediateNode`, `DemandSinkNode`. Warehouse vs shop is a free-form `tags: list[str]` attribute on `IntermediateNode` — informational only, never branches mechanics.
- `Node` carries `id: str`, `region: str`, `policy: NodePolicy | None`, `init_seed: int`, computed `level: int`. Per-class fields:
  - `FactoryNode`: `produces_product_id: str`, `unit_cost: float`, `capacity_per_tick: int | float | Distribution`, `inventory: int`, `list_price: float` (always == `unit_cost` per ADR 0013).
  - `IntermediateNode`: `carried_products: set[str]`, `capacity: int | float | Distribution`, `tags: list[str]`, `inventory: dict[str, int]`, `pending: dict[str, dict[str, int]]` (per supplier × pid in-transit), `list_prices: dict[str, float]`, `min_order_imposed: dict[str, int]` (server-side rejection threshold per product).
  - `DemandSinkNode`: `product_id: str`, `demand_dist: Distribution`, `income_rate: float`, `cash: float`, `activation_tick: dict[str, int]` (per-product, for freshness composition).

### Graph

- `Graph(nodes, edges)` validates DAG-only, computes `levels` (longest path from any factory), exposes `suppliers_of(buyer_id)`, `buyers_of(supplier_id)`, `lead_time(supplier, buyer, pid)`. Authoring helper `build_graph(nodes, edges) -> Graph` raises on cycles, unreachable nodes, same-level supplier links.
- `EdgeSpec(supplier_id, buyer_id, default_lead_time, per_product_lead_time: dict | None)` — lead time per-edge with optional per-product override. Today's `Store.delivery_lags` becomes a fallback.

### Central table

- `CentralTable.rows: dict[(seller_id, pid), Offer]` where `Offer = {available_qty, list_price, min_order, fill_rate_recent}`.
- Sellers (factories + intermediates) `publish(seller_id, pid, offer)` at the start of each tick; allocator `commit(seller_id, pid, qty)` decrements `available_qty` live and updates the rolling qty-weighted EMA on `fill_rate_recent` (window = 10 ticks).
- Global visibility — `snapshot_for_buyer(pid)` returns every `(seller_id, Offer)` for that product. Topology-restricted views explicitly out of scope.

### Allocation

- `allocation.execute_buy(buyer, supplier, pid, qty_requested, table, cash_ledger) -> AllocationResult`:
  - Reject when `qty_requested < min(supplier-imposed min_order, buyer-side policy min_order)` per ADR 0012's two-layer rule.
  - Clamp by live `available_qty`, by `buyer.cash / list_price` (allocation-time payment per ADR 0013), by remaining buyer capacity.
  - `table.commit(...)`, debit buyer cash, credit supplier cash, schedule delivery callback at `current_tick + lead_time(supplier, buyer, pid)`.
  - Return `AllocationResult(qty_filled, qty_rejected, cash_paid)`.
- `shuffle_buyers(buyers, allocation_rng)` — single deterministic shuffle per phase using the `allocation_rng` stream.

### Tick phasing

- One tick: `tick_world` (market multiplier + event_engine + item_registry) → `publish_offers` → for `p in 1..max_level`: shuffle buyers at level `p`, per buyer observe → decide → execute_buy per line → produce (factories) → deliver (scheduled callbacks fire) → consume_demand_sinks.
- Physical lead time still delays delivery between phases. Orders placed at phase N arrive at `current_tick + lead_time`, not within the same tick.

### Scenario shape

- `Scenario` gains `nodes: list[NodeInstance]` and `edges: list[EdgeSpec]`. `NodeInstance(node, init_seed, policy)`; `policy` omitted from JSON (mirrors `StoreInstance`).
- Phase 0 keeps both `stores` (legacy, optional) and `nodes` (new, optional) with an `is_graph` property; Phase 4 promotes `nodes`/`edges` to required and deletes `stores`/`StoreInstance`/`StoreTemplate`/`make_stores`.
- `Scenario.nodes_df()` and `Scenario.edges_df()` join the existing DataFrame inspection methods.

### Policy interface

- `NodePolicy(ABC)` carries `policy_seed: int | None`, `policy_rng: Random`. Three type-paired subclass ABCs:
  - `FactoryPolicy.decide(obs_factory) -> {"produce_qty": int, "list_price": float}` (list_price always == unit_cost in default impl).
  - `IntermediatePolicy.decide(obs_intermediate, central_table) -> {"order": {pid: [(supplier_id, qty), ...]}, "list_price": {pid: float}, "min_order_imposed": {pid: int}}`.
  - `DemandSinkPolicy.decide(obs_sink, central_table) -> {"buy": [(supplier_id, qty), ...]}` for the sink's bound product.
- Shipped Phase 1 concrete policies: `StaticFactoryPolicy(capacity_per_tick)`, `DefaultDemandSinkPolicy` (greedy cheapest-feasible), `IntermediatePolicy.SingleSupplierAdapter` wrapping the existing `TextbookReorderPolicy` to one supplier.
- Phase 2 lifts `OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy` onto `MultiSupplierTextbookPolicy(IntermediatePolicy)` with `_split_across_suppliers(pid, qty_total, suppliers, central_table)`. Default routing: cheapest-first, both min-order layers honoured, partial fills surfaced via `fill_rate_recent`. Pluggable via `routing_strategy` ctor kwarg.

### Cash flow

- Demand-sinks: `cash += income_rate` each tick. Spent on `buy` allocations at allocation time. Unspent cash accumulates.
- Inter-node trades: buyer balance `−= qty_filled × list_price`, seller balance `+= qty_filled × list_price`. Executed at `execute_buy`.
- Factories: `unit_cost == list_price` ⇒ balance is a no-op. Modelled cleanly as "cash absorbed by manufacturing." Holding cost and fixed order fee continue to go to the void (unchanged from today).

### Demand-sink mechanics

- Each sink bound to one `product_id`. Each tick: `demand_target = demand_dist.sample(world_rng) × Market.demand_multiplier(pid, region, tick) × ItemRegistry.stage_multiplier(pid) × freshness_curve.multiplier(α, β, tick − activation_tick[pid])`.
- Policy converts `demand_target` into `[(supplier_id, qty), ...]` based on current central-table snapshot and `cash`. Unmet demand (insufficient supplier inventory or insufficient cash) = lost sale.
- CRN preservation: world_rng consumed in catalog-iteration order for every catalog pid every tick (mirrors ADR 0003), even when the sink isn't actively buying that pid.

### Market

- `Market.sample_demand` is deleted. `Market.demand_multiplier(pid, region, tick) -> float` bundles seasonality + regional shock + active disruption multipliers.
- `Market.tick()` continues to advance per-region demand/supply state, cycle, trend (existing math unchanged).
- `ItemRegistry`, `EventEngine`, `freshness_curve`, `lifecycle_clock` are unchanged.

### RNG / CRN contract

- New sub-seed: `"allocation": (0xA24B_AED4, 0x0000_0007)` in `_SUB_SEED_PARAMS`. Derived via existing `_derive_seed(seed, purpose)` formula.
- `allocation_rng = Random(_derive_seed(world_seed, "allocation"))` constructed in `build_world` and consumed by `shuffle_buyers` at each phase.
- Three determinism invariants enforced by tests: world stream bit-identical across runs with same `world_seed`; step-0 node state bit-identical across runs with same `(template, init_seed)` regardless of policy; policy swap on one node does not perturb `world_rng` or `allocation_rng`.

### Deep-module extraction

These modules carry rich behaviour behind small interfaces and are tested in isolation; they will see the bulk of the test surface:

- **`Graph`** — DAG validation, level computation, topology queries. Pure structural logic over `(nodes, edges)`. Stable interface (`validate_dag`, `compute_levels`, `suppliers_of`, `buyers_of`, `lead_time`).
- **`CentralTable`** — `publish` / `commit` / `snapshot_for_buyer`. Rolling EMA fill-rate. Small interface; rich semantics around live mutation during a phase.
- **`allocation.execute_buy`** — single-call FCFS allocation primitive. Pure given mocked buyer / supplier / table. Encapsulates all the clamping logic and the payment + delivery-scheduling side effects.
- **`MultiSupplierTextbookPolicy._split_across_suppliers`** — pure routing function over `(qty, suppliers, central_table)`. Encapsulates min-order discipline + price ordering + partial-fill arithmetic.

Shallow / orchestration modules (covered by integration tests, not deep unit tests): `Node` subclasses, `GraphSimulation.tick`, `GraphRunner.run`, `Market`.

### Phasing

The work ships in 7 phases per the plan. Each phase ends runnable, with its own smoke scenario and test set. Phase 0 lands skeleton + ADRs additively; Phase 1 runs a 3-node chain; Phase 2 adds multi-supplier contention; Phase 3 grounds lifecycle/freshness on the sink path; Phase 4 retires the legacy `Store` engine; Phase 5 re-integrates RL; Phase 6 re-integrates tuning; Phase 7 ships docs / notebooks / ADR ratification.

## Testing Decisions

### What makes a good test here

- Test external behaviour, not implementation details. A test should describe a contract a future reader can rely on, and should fail loudly if the contract is broken — even by a refactor that preserves the contract differently.
- Determinism invariants are non-negotiable. The three CRN guarantees (world stream identity, init-state independent of policy, policy swap orthogonal to world/allocation streams) must be enforced by dedicated tests that pin the invariant directly, not inferred from end-to-end run snapshots.
- Cash conservation is a budget property of the whole graph; assert it at the system level (sum of demand-sink-created cash == sum of inventory-value-flowing-through-graph + holding+fee-to-void + final balances), not via per-node accounting.
- Prefer many small unit tests on the deep modules (`Graph`, `CentralTable`, `allocation.execute_buy`, `_split_across_suppliers`) over many integration tests on `GraphRunner` — the deep modules are where the complexity lives and where regressions hide.

### Which modules will be tested

**Deep-module unit tests (highest priority):**
- `Graph` — DAG validation rejects cycles / same-level supplier links / unreachable nodes; `compute_levels` matches longest-path-from-any-factory; topology queries (`suppliers_of`, `buyers_of`, `lead_time`) honour per-edge overrides. → `tests/sim/test_graph_dag.py`
- `CentralTable` — `publish` overwrites; `commit` decrements `available_qty` live and never below zero; `fill_rate_recent` EMA tracks qty-weighted partial fills; `snapshot_for_buyer` reflects post-commit state. → `tests/sim/test_central_table.py`
- `allocation.execute_buy` — every clamp path (inventory-limited, cash-limited, capacity-limited, min-order-rejected by either layer); payment debits buyer and credits supplier by the same amount; delivery callback scheduled at `current_tick + lead_time`. → `tests/sim/test_allocation_fcfs.py`
- `MultiSupplierTextbookPolicy._split_across_suppliers` — cheapest-first ordering; both min-order layers reject sub-min lines; partial-fill arithmetic when no single supplier covers the requested qty; pluggable `routing_strategy` invoked when supplied. → `tests/sim/test_multisupplier_routing.py`

**Determinism / CRN tests (load-bearing):**
- `tests/sim/test_graph_determinism_chain.py` — three invariants stated above, on the Phase-1 chain scenario.
- `tests/sim/test_allocation_determinism.py` — buyer-shuffle sequence reproducible from `world_seed`; policy swap on one buyer does not perturb the shuffle.
- `tests/sim/test_determinism.py` — existing legacy tests rewritten to graph-level invariants in Phase 2.

**Construction / shape tests:**
- `tests/sim/test_node_construction.py` — per-class init via `init_seed` is bit-identical regardless of attached policy (extends the existing `test_two_stores_same_init_seed_identical_step0` invariant to all three node types).
- `tests/sim/test_scenario_graph_roundtrip.py` — `Scenario` with `nodes` / `edges` survives `to_dict` / `from_dict` round-trip; `NodeInstance.policy` omitted as specified.

**Integration / smoke tests:**
- `tests/sim/test_graph_runner_chain.py` — Phase-1 chain runs 50 ticks; cash conservation holds; no negative inventory anywhere.
- `tests/sim/test_demand_sink_freshness.py` — lifecycle + freshness composition holds on the demand-sink path (assertions ported from the existing `test_freshness_integration.py`).
- `tests/sim/test_regression_snapshot.py` — regenerated from the graph engine on the Phase-1 scenario; old snapshot retired with a pointer to ADR 0011.

**Rewritten downstream test suites (Phases 5–6):**
- `tests/rl/test_eval_crn.py` — CRN tuple expanded to include the `allocation` sub-seed.
- `tests/rl/test_env_smoke.py`, `tests/rl/test_episode_sampler.py`, `tests/rl/test_encoders.py`, `tests/rl/test_two_scale_eval_smoke.py`.
- `tests/tuning/test_evaluator.py`, `tests/tuning/test_search_spaces.py`.

### Prior art in the codebase

- The determinism-invariant style is established in `tests/sim/test_determinism.py` — same shape generalises to graph-level invariants in Phase 2.
- The two-pass fair-share allocator's unit-test style in `tests/sim/test_textbook_helpers.py` is the model for `_split_across_suppliers` unit tests: many small deterministic cases on a pure function.
- The CRN-paired integration test style in `tests/rl/test_eval_crn.py` carries over directly; only the CRN tuple shape changes.

## Out of Scope

- **Cycles in the graph.** DAG-only by design (ADR 0014's cascade phasing requires assignable echelon levels). A node that buys from a peer at the same echelon level is not expressible. If such topologies become important later, a follow-up ADR is needed.
- **Topology-restricted central-table visibility.** Every node sees every offer in this PRD. A "you only know your upstream suppliers" model is a possible future extension.
- **Two-tick request/response allocation (Model 3).** Order → upstream-mailbox → response → buyer takes 2+ ticks in that model. We chose Model 1b (sequential FCFS, live routing within a tick) per the grilling decision; Model 3 is explicitly out.
- **Snapshot-at-phase-start allocation (Model 2).** Deterministic fair-share against a fixed snapshot was considered and rejected in favour of live FCFS with shuffle. Re-opening this is a new ADR.
- **Factory margin > 0.** Factories charge exactly `unit_cost` per ADR 0013. Studying factory pricing power is out of scope for this PRD; if needed, ADR 0013 can be amended to add a `factory_margin` field.
- **Holding / order fee routing to a service-provider node.** Both continue to go to the void as today. Strict cash conservation across the entire system is not a goal.
- **Order escrow at order time (vs. payment at allocation).** Buyer pays at allocation, not at order submission. The "buyer reserves cash up-front" semantics are out of scope.
- **LLM-authored graph topology.** `WorldBuilder` continues to author `catalog` + `market` + `store_templates`; graph topology is synthesised by `world_to_graph(world, *, sink_density)`. Letting the LLM author the topology directly is a future ADR.
- **Removal of `HeuristicPolicy`.** Phase 4 deletes it. Anyone needing the kitchen-sink heuristic for parameter-demo purposes should branch before Phase 4 lands.
- **Per-node-type observation schema documentation in `CONTEXT.md`.** The schemas are documented inline at the call sites; the `CONTEXT.md` updates in Phase 7 cover the domain concepts, not the observation tensor shapes.
- **Backward compatibility for serialised legacy `Scenario` JSON.** Once Phase 4 lands, old `stores`-shaped JSON cannot be loaded. The migration path is "rebuild from the scenario authoring script."

## Further Notes

- **The plan file `/Users/mislavjordanic/.claude/plans/enumerated-wandering-eich.md` is the authoritative phased roadmap.** This PRD is the user-stories and decisions view; the plan file is the implementation view. Both should stay in sync — when a decision in this PRD changes, the plan file is updated.
- **The work is large.** ~45–50 affected Python files. Phase 0 is intentionally a no-behaviour-change skeleton so each subsequent phase can land independently, with green tests, on the trunk branch (`multi-echelon` or main as appropriate).
- **The textbook policy migration is the load-bearing piece.** Phase 1 introduces the single-supplier adapter to validate that the rate-estimate / safety-horizon math survives the new tick phasing. Phase 2 lifts it to multi-supplier routing. `OrderUpToPolicy` must remain the canonical CRN comparison anchor through every phase.
- **`models_canvas.md` motivates several research directions that depend on this rewrite** — PerceiverShop variants under supply contention, diffusion policies for inventory + pricing under disruption with finite upstream supply, the OctoSupply cross-world pretraining angle. None of these become possible until the graph engine lands.
- **Per-issue breakdown will follow.** This PRD will be sliced into independently-grabbable vertical-slice issues under `.scratch/multi-echelon/issues/` via the `/to-issues` skill once the PRD is approved.
