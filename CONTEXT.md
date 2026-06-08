# Domain & Architecture Glossary

## Domain concepts

**Node** (`Node` ABC, `src/sim/node.py`)
The fundamental actor in the supply-chain graph. Three concrete subtypes share a `Node` ABC:
`FactoryNode`, `IntermediateNode`, and `DemandSinkNode`. Each node carries `id: str`,
`region: str`, `init_seed: int`, `policy: NodePolicy | None`, and the computed `level: int`
(set by `Graph.compute_levels` after construction; `None` until set). Since ADR 0018 `level` is a
**display-only** longest-path hint — it is no longer a scheduling unit. Construction consumes only
the per-instance `init_rng` (seeded from `init_seed`), so two nodes built from the same
`(NodeSubclass, init_seed)` start step-0 bit-identical regardless of which policy is later
attached. Nodes are not frozen dataclasses — the simulation engine mutates runtime fields
(`inventory`, `cash`, `pending`) in-place during each tick. See ADR 0011.

**FactoryNode** (`FactoryNode(Node)`, `src/sim/node.py`)
Produces one product per tick in exchange for cash at manufacturing cost. Fields:
`produces_product_id: str`, `unit_cost: float`, `capacity_per_tick: int | float | Distribution`,
`inventory: int`, `list_price: float` (always == `unit_cost` per ADR 0013; zero-margin by
construction). Factories are the entry point for inventory into the chain.

**IntermediateNode** (`IntermediateNode(Node)`, `src/sim/node.py`)
Holds multi-product inventory, sets list prices, routes orders across multiple upstream suppliers.
Represents warehouses, shops, distribution centres, or any mid-chain tier — the distinction is
expressed via free-form `tags: list[str]` (informational only, never branches mechanics). Fields:
`carried_products: set[str]`, `capacity: int | float | Distribution`, `inventory: dict[str, int]`,
`pending: dict[str, dict[str, int]]` (per-supplier × pid in-transit), `list_prices: dict[str, float]`,
`min_order_imposed: dict[str, int]` (server-side rejection threshold per product), `cash: float`.

**DemandSinkNode** (`DemandSinkNode(Node)`, `src/sim/node.py`)
The only source of new cash in the system. Bound to one `product_id`. Each tick: earns
`income_rate` cash, then demands units from `demand_dist.sample(world_rng) ×
Market.demand_multiplier(pid, region, tick)`. Policy converts `demand_target` into
`[(supplier_id, qty), ...]` based on the current central-table snapshot and available cash.
Unmet demand (insufficient supplier inventory or cash) = lost sale. CRN preservation: `world_rng`
is consumed in catalog-iteration order for every catalog pid every tick, even when the sink is
not actively buying that pid. See ADR 0015.

**Graph** (`Graph`, `src/sim/graph.py`)
Validated directed acyclic graph of typed nodes. Constructed via `build_graph(nodes, edges,
node_types=...)`, which raises `ValueError` on cycles, self-loops, unreachable nodes, or edges
that break the **type rules** (ADR 0018): supplier ∈ {factory, intermediate}, buyer ∈
{intermediate, sink}, and `factory→sink` is rejected (flow must pass through ≥1 intermediate).
**Lateral `intermediate→intermediate` edges at any depth are legal** — only the *union* of all
edges must stay acyclic (a node pair trades in one direction across all products; same-tick
opposite-direction product flows are not supported). `node_types: dict[str, str]` (id →
`Node._node_type`) keeps the check structural without importing `Node`. Computes display-only
`levels` via longest-path-from-any-source. Topology queries: `suppliers_of(buyer_id)`,
`buyers_of(supplier_id)`, `lead_time(supplier, buyer, pid)`. `build_demand_pull_schedule(graph)`
returns the reverse-topological ready-sets used by the scheduler. Pure structural logic — no
imports from other `src.sim` modules; safe to import from tests. See ADR 0011 and ADR 0018.

**EdgeSpec** (`EdgeSpec`, `src/sim/graph.py`)
Immutable directed supply edge descriptor. Fields: `supplier_id: str`, `buyer_id: str`,
`default_lead_time: int`, `per_product_lead_time: dict[str, int] | None`. Lead time per-edge with
optional per-product override via `lead_time_for(pid)`.

**CentralTable** (`CentralTable`, `src/sim/central_table.py`)
Live, globally visible offer book. Rows: `dict[(seller_id, pid), Offer]` where
`Offer = {available_qty, list_price, min_order, fill_rate_recent}`. At the start of each tick all
sellers `publish(seller_id, pid, offer)`; as allocations land the allocator calls
`commit(seller_id, pid, qty)` which decrements `available_qty` live (never below zero) and updates
the rolling qty-weighted EMA on `fill_rate_recent` (window = 10 ticks).
`snapshot_for_buyer(pid)` returns every `(seller_id, Offer)` for that product, reflecting prior
buyers' allocations within this phase — so a buyer's routing decisions account for live supply
depletion rather than a stale tick-start snapshot. See ADR 0012.

**Allocation** (`allocation.execute_buy`, `src/sim/allocation.py`)
Single-call FCFS allocation primitive. `execute_buy(buyer, supplier, pid, qty_requested, table,
cash_ledger) -> AllocationResult` encapsulates: two-layer min-order rejection (supplier-imposed and
buyer-side policy min-order, per ADR 0012), clamping by live `available_qty` / buyer cash / buyer
remaining capacity, `table.commit(...)`, cash transfer (buyer− / seller+), and delivery callback
scheduling at `current_tick + lead_time`. Returns `AllocationResult(qty_filled, qty_rejected,
cash_paid, reason)` — the `reason` field names the binding constraint on a rejection (ADR 0018).
Buyer routing is min-order-aware: a supplier whose `min_order` can't be met is skipped and the
buyer falls through to the next feasible supplier instead of taking a silent lost sale.
`shuffle_buyers(buyers, allocation_rng)` performs one deterministic per-phase buyer
shuffle using the `allocation_rng` stream. See ADR 0012, ADR 0016, and ADR 0018.

**Rejection log** (run-log `ticks[i]["rejections"]`, ADR 0018)
Always-on per-tick stream for debugging rejected/lost sales. Each entry
`{tick, buyer_id, supplier_id, pid, qty_requested, qty_filled, qty_rejected, reason}`; `reason` ∈
{`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`,
`insufficient_capacity`, `unmet_demand`}. `unmet_demand` entries (`supplier_id=None`) record demand
that no feasible supplier could fill — the true lost-sale measure once min-order fall-through avoids
the avoidable rejections.

**Demand-pull schedule** (tick phasing, `src/sim/runner.py`, ADR 0018)
One tick executes as a **demand-pull topological walk** — Kahn's algorithm on the reversed graph
(sinks first, factories last): `tick_world` (market multiplier + event_engine + item_registry) →
`publish_offers` → walk each ready-set of mutually-incomparable peers in reverse-topological order,
shuffling the ready-set with `allocation_rng` for FCFS fairness: per-buyer observe → decide →
`execute_buy` per line → deliver (scheduled callbacks fire) → factory produce → consume demand
sinks. Physical lead time still delays delivery — orders placed at tick N arrive at
`current_tick + lead_time`, not within the same tick. Because `execute_buy` decrements seller
inventory at sale time, by the time an intermediate is processed every downstream buyer has already
transacted, so it reorders against *complete current-tick* demand (`observed_sales`) — there is no
`prev_tick_sales` one-tick lag. This survives lateral links, where echelon-level ordering would
not. The schedule structure (`build_demand_pull_schedule`) is computed once at `build_world`
(topology is static) and cached on the `Simulation`; only the ready-set shuffle re-draws each tick.
A strict chain degenerates to singleton ready-sets (the shuffle is a no-op). See ADR 0018.

**Policy** (`NodePolicy` ABC, `src/sim/policy.py`)
Decision logic attached to a Node. Three type-paired subclass ABCs: `FactoryPolicy.decide(obs_factory)`,
`IntermediatePolicy.decide(obs_intermediate, central_table)`,
`DemandSinkPolicy.decide(obs_sink, central_table)`. Each Policy instance owns a private
`policy_rng` seeded from its `policy_seed` kwarg, kept disjoint from `world_rng` and
`allocation_rng` so policy choice never perturbs world or allocation stochasticity. `Policy` is
kept as a one-release alias of `NodePolicy`. Shipped concrete policies: `StaticFactoryPolicy(capacity_per_tick)`,
`DefaultDemandSinkPolicy` (greedy cheapest-feasible), `SingleSupplierAdapter` (wraps a
`TextbookReorderPolicy` to one supplier), `MultiSupplierTextbookPolicy` (lifts the four textbook
variants to multi-supplier routing with `_split_across_suppliers`). See ADR 0006, ADR 0011.

**TextbookReorderPolicy family** (`MultiSupplierTextbookPolicy(IntermediatePolicy)` abstract base + 4 concrete classes, `src/sim/policy.py`)
Family of four reorder policies sharing one base class, re-rooted on `IntermediatePolicy` for the
graph engine. The shared machinery: censored-sales rate estimator over a `delivery_lag`-length
rolling window, two-pass fair-share allocator across capacity and cash pools, cash-budget
pilot-order cold-start, opt-in adaptive safety stock. Concrete variants: `OrderUpToPolicy` (s,S),
`ReorderPointPolicy` (s,Q), `PeriodicOrderUpToPolicy` (R,S), `PeriodicReorderPolicy` (R,s,S).
Multi-supplier routing via `_split_across_suppliers(pid, qty_total, suppliers, central_table)`:
default strategy is cheapest-first with both min-order layers honoured, pluggable via
`routing_strategy` constructor kwarg. `OrderUpToPolicy` remains the canonical CRN comparison
anchor for RL evaluation. See ADR 0006, ADR 0008.

**Market** (`Market`, `src/sim/market.py`)
Regional economic environment shared across nodes. Holds per-region demand/supply state, seasonal
cycle, and trend drift. Provides `demand_multiplier(pid, region, tick) -> float` — bundles
seasonality + regional shock + active disruption multipliers. `Market.tick()` advances per-region
demand/supply state, cycle, trend. One Market per simulation run. See ADR 0015.

**Item** / **Ware** / **SKU**
A product in the catalog. `Ware` is the static record (`product_id P####`, `name`, `category`,
`base_price`, `unit_cost`, `seasonality`, `related_products`) plus optional per-product lifecycle
and freshness authoring fields. The simulator samples demand per catalog product every tick for
CRN cleanliness (ADR 0003).

**EventEngine**
Generates stochastic disruption events (natural_disaster, economic_crisis, political_unrest,
pandemic, technological_breakthrough) that shift regional demand/supply. Also dispatches scheduled
callbacks for order arrivals.

**Order**
In the graph engine, an order is a list of `(supplier_id, qty)` tuples produced by an
`IntermediatePolicy.decide()` call. No explicit Order object — delivery is a scheduled callback
via EventEngine at `current_tick + lead_time`.

**Observation**
A dict of visible state handed to a Policy each tick. For `IntermediatePolicy`, also includes the
live `CentralTable` so routing decisions account for supply depletion by earlier buyers in the
same tick, plus `observed_sales` — the node's complete current-tick sales (guaranteed complete by
the demand-pull ordering, ADR 0018), which replaced the old `prev_tick_sales` one-tick lag. Built
by `Simulation._observe_node()`.

**Active subset**
The K (default 5) product ids drawn per episode for RL training. Frozen for the episode duration.
The assortment is encoded via slot-shuffled observations. Distinct from the full catalog
(world_rng demand draws happen for every catalog product to preserve CRN cleanliness). See ADR 0003.

**Run Log**
Dict produced by `Runner.run()`. Top-level keys: `n_steps` (int), `ticks` (list of per-tick node
snapshots with `node_cash`, `node_inventory`, `node_pending`, `node_orders`, and the always-on
`rejections` stream — see **Rejection log**), `global` (market/event/lifecycle time-series).
Consumed by `DataExporter`.

**CRN-paired eval**
Evaluation protocol where the RL policy and `OrderUpToPolicy` run on bit-identical
`(world_seed, init_seed, capacity, balance, active_subset, slot_permutation, allocation_seed)`
tuples — Common Random Numbers. The `allocation` sub-seed (ADR 0016) is included in the CRN
tuple. Uplift is computed paired per seed and averaged across 32 held-out seeds. See ADR 0003,
ADR 0006, ADR 0016.

**Policy tuning study** (`src/tuning/`)
Optuna-based hyperparameter tuner for `MultiSupplierTextbookPolicy` and subclasses. New tunables:
`per_supplier_min_order_floor` (int[0,10]) and `routing_strategy` (Categorical). Per-trial CRN
tuple expanded to include the `allocation` sub-seed for bit-identical paired comparisons on the
graph engine. The objective remains mean `net_profit / initial_cash`. Artifacts land at
`runs/tuning/<study_name>/` as `trials.parquet` + `per_seed.parquet` + `study.json`. See ADR 0009.

**Run model — "policies attached to nodes"**
Each Node has its own Policy (or `None`). A run binds a list of `NodeInstance(node, init_seed,
policy)` triples for graph-mode scenarios. Use patterns the triple list expresses:
- **Homogeneous fleet** (robustness): repeat one policy class across varying `(node, init_seed)`
  per node. One scenario, multiple nodes each with distinct seeds.
- **CRN A/B comparison**: two sub-graphs sharing the same `init_seed` structure, one per policy
  variant. Bit-identical step-0 state, different policies — paired evaluation.
- **Mixed roster**: just write the node-instance list. No regime abstraction above the list.

**RL Env** (`src/rl/env.py`)
Gymnasium-compatible environment wrapping the graph engine. `reset()` builds a degenerate
three-node graph: one `FactoryNode` → one trainable `IntermediateNode` ("S") → one
`DemandSinkNode` per active SKU. `step(action)` advances one tick via the two-phase tick API
(`tick_world()` / `tick_decide_and_settle()`), with the action injected through an
`RLIntermediatePolicy` shim. The observation tensor is extended with a
`central_table_snapshot[product_slot]` block (supplier_count, min_price, mean_lead_time,
mean_fill_rate) — 4 features per product slot, N_PER_SKU = 18. The action decoder emits
per-supplier splits. See ADR 0007, ADR 0011.

**RL Episode**
One `reset()`-to-terminated pass through the RL Env. Fixed at 180 ticks. Episode return = sum of
per-tick `net_profit` from the `IntermediateNode` ("S")'s cash delta. Capacity and opening balance
sampled log-uniformly per episode.

**Scenario** (`Scenario`, `src/sim/scenario.py`)
The reproducible inputs to a run. Fields: `nodes: list[NodeInstance]` (required), `edges:
list[EdgeSpec]` (required), `catalog: list[Ware]`, `market: MarketParams`, `disruption:
DisruptionParams`, `item_lifecycle: ItemLifecycleParams`, `n_steps: int`, `start_date: datetime`,
`world_seed: int`. Authoring helper `make_nodes(triples)` where `triples: list[tuple[Node, int,
Policy]]`. Inspection methods: `catalog_df()`, `market_df()`, `disruption_df()`, `lifecycle_df()`,
`summary_df()`, `nodes_df()`, `edges_df()`. See ADR 0011.

**Setup directory**
A folder containing `catalog.csv` + `setup.yaml`. The canonical on-disk representation of a
scenario. `load_setup(setup_dir)` in `src/sim/setup_io.py` parses the two files and returns a
`Scenario`. `write_catalog_and_market(catalog, market, setup_dir)` writes the LLM-generated
catalog and market block. The directory doubles as a cache for the LLM generator —
`WorldBuilder.build_setup(n_items, setup_dir)` is a no-op when `catalog.csv` already exists.

**Slot-shuffled observation**
Observation tensor where K active SKUs occupy K slots with a per-episode slot-to-SKU permutation,
preventing the agent from associating slot position with product identity. See ADR 0004.

---

## Architecture concepts

**Scenario** (`src/sim/scenario.py`)
Flat typed dataclass holding a complete experiment: `catalog` (`list[Ware]`), `market`
(`MarketParams`), `disruption` (`DisruptionParams`), `item_lifecycle` (`ItemLifecycleParams`),
`nodes` (`list[NodeInstance]`), `edges` (`list[EdgeSpec]`), `n_steps`, `start_date`, `world_seed`.
Authoring helper `make_nodes(triples)` where `triples: list[tuple[Node, int, Policy]]`. DataFrame
inspection methods: `catalog_df()`, `market_df()`, `disruption_df()`, `lifecycle_df()`,
`summary_df()`, `nodes_df()`, `edges_df()` — `nodes_df()` and `edges_df()` are the canonical way
to inspect a graph scenario.

**Distribution** (`src/sim/distributions.py`)
Abstract base class with one method, `sample(rng) -> Any`, and five concrete implementations:
`Constant(value)`, `Uniform(lo, hi)`, `Normal(mean, std)`, `Choice(options)`,
`LogUniform(lo, hi)`. Used for domain-randomisation parameters (per-episode capacity, opening
balance) and node construction fields. Typed, comparable, JSON-serialisable.

**Simulation** (`src/sim/runner.py`)
Mutable bundle returned by `build_world(scenario, *, policy_overrides=None)`. Holds `(scenario,
world_rng, allocation_rng, item_registry, market, event_engine, graph, nodes)` and exposes a
two-phase tick API:

- `tick_world() → list[WorldEvent]` — phase 1: advances market, events, and item lifecycle;
  publishes all seller offers to the CentralTable; returns active events.
- `tick_decide_and_settle(active_events) → TickResult` — phase 2: the demand-pull walk
  (reverse-topological ready-sets, shuffled per tick) → per-buyer observe → decide → `execute_buy`
  per line → factory produce → deliver callbacks → sink demand. RL action-injection happens in the
  seam between phases.
- `tick() → TickResult` — convenience composing both phases.

`allocation_rng` is a new `Random` stream derived via `_derive_seed(world_seed, "allocation")`
(ADR 0016). `build_world` deep-copies nodes for multi-call safety.

**Runner** (`src/sim/runner.py`)
Owns the simulation loop. `Runner(scenario).run() → dict` builds all subsystems via `build_world`
and runs the observe → decide → advance → log loop for `scenario.n_steps` ticks. Maintains the
four-stream RNG split (`world_rng`, `allocation_rng`, one `policy_rng` per Policy, one `init_rng`
per NodeInstance) that makes CRN comparison correct. Returns a run log with keys `n_steps`,
`ticks`, `global`.

**DataExporter** (`src/sim/data_exporter.py`)
Consumes a `Scenario` and a completed run log and writes: parquet time-series, parquet static
node/product tables, JSON config snapshot and run log, and a regional supply/demand PNG. Layout:
`config/scenario.json`, `data/run_log.json`, `data/products.parquet`, `data/nodes.parquet`
(node table), `data/timeseries.parquet`, `reports/overview.png`. Public methods:
`export_all(output_folder)` plus per-artifact saves.

**Episode sampler** (`src/sim/episode_sampler.py`)
Canonical home for `EpisodeSpec`, `sample_episode(...)`, and default-params factories. Sub-seed
derivation: `sub = (episode_seed * PRIME + OFFSET) & 0xFFFF_FFFF`, one `(PRIME, OFFSET)` pair per
purpose. Five sub-seeds for the graph-engine era: `assortment`, `capacity`, `balance`, `world`,
and `allocation` (ADR 0016). RL adds a 6th `slot` stream in `src.rl.episode_sampler`.

**Metrics** (`src/sim/metrics.py`)
Canonical home for `RunSlice`, `aggregate_episode(run_slice) -> dict[str, float]`, and five KPI
helpers. Pure data + math; no runtime deps on `src.tuning` or `src.rl`. For graph-engine runs,
`net_profit` is computed from the tracked `IntermediateNode` ("S")'s cumulative `cash_delta`.

**WorldBuilder** (`src/llm/world_builder.py`)
LLM-driven generator producing a `(catalog, market)` pair. Three stages: taxonomy → catalog →
market domain params. The canonical entry point is `WorldBuilder.build_setup(n_items, setup_dir)`,
which persists `catalog.csv` and the `market:` block of `setup.yaml` and acts as a dir-as-cache.
Topology, policies, and disruption params are the modeller's domain, not the generator's.

**Setup IO** (`src/sim/setup_io.py`)
`load_setup(setup_dir)` parses `catalog.csv` + `setup.yaml` and returns a `Scenario`.
`write_catalog_and_market(catalog, market, setup_dir)` persists the LLM-generated catalog and
market block for later `load_setup` calls. `load_or_build_setup(setup_dir, build_fn, ...)` wraps
`build_setup` with a dir-as-cache check.

**Topology scaffolder** (`src/sim/topology_scaffolder.py`)
`scaffold_topology(wares, spec) -> dict` generates a starter `{"nodes": [...], "edges": [...]}`
YAML-ready dict from a `list[Ware]` and a `ScaffoldSpec` (shop count, sink density, region,
capacities, lead times). Called by `main.py scaffold`.

---

## Decisions

- [ADR 0001](docs/adr/0001-two-layer-lifecycle.md) — Lifecycle is two-layer: global PLC × per-node freshness curve.
- [ADR 0002](docs/adr/0002-per-stage-transitions-and-dead-stage.md) — Per-stage transition probabilities and a `dead` stage with terminal-by-default cycle.
- [ADR 0003](docs/adr/0003-crn-demand-for-all-products.md) — Demand is sampled for every catalog product each tick, even when inactive (CRN cleanliness).
- [ADR 0004](docs/adr/0004-rl-training-env.md) — RL training env: randomised assortment, slot-shuffled observations, hidden market state, frozen assortment within episode.
- [ADR 0005](docs/adr/0005-demand-relative-action-decoding.md) — Action decoder expresses order quantity in lead-times-of-demand. *Superseded by ADR 0007.*
- [ADR 0006](docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family replaces the heuristic baseline as the RL comparison anchor.
- [ADR 0007](docs/adr/0007-rl-scale-invariance-package.md) — RL scale-invariance package: order-up-to action decoder, demand-units inventory feature, log-uniform domain randomisation.
- [ADR 0008](docs/adr/0008-safety-horizons-as-fraction-of-delivery-lag.md) — Reparameterise textbook safety horizons as fractions of delivery lag (fix per-SKU mis-scaling).
- [ADR 0009](docs/adr/0009-policy-hyperparameter-tuning-tool.md) — Policy hyperparameter tuning tool (Optuna-based).
- [ADR 0010](docs/adr/0010-sim-as-base-for-ml-layers.md) — `src/sim/` is the canonical home for rollout primitives; `src/tuning/` and `src/rl/` are sibling consumers.
- [ADR 0011](docs/adr/0011-multi-echelon-graph.md) — Multi-echelon DAG of typed nodes replaces the single-`Store` flat model. **Accepted.**
- [ADR 0012](docs/adr/0012-central-table-fcfs-allocation.md) — Central table + sequential FCFS allocation with live offer-book mutation. **Accepted.**
- [ADR 0013](docs/adr/0013-cash-flow-conservation.md) — Cash flow conservation: sinks create, ops destroy, inter-node trades transfer; factories are zero-margin. **Accepted.**
- [ADR 0014](docs/adr/0014-tick-phasing-cascade.md) — Tick phasing as upward cascade by echelon level (sinks → intermediates → factories). **Accepted — superseded by ADR 0018.**
- [ADR 0015](docs/adr/0015-demand-sinks-market-multiplier.md) — Demand-sinks own demand sampling; `Market` shrinks to a multiplier engine (`demand_multiplier`). **Accepted.**
- [ADR 0016](docs/adr/0016-allocation-rng-sub-seed.md) — `allocation` sub-seed added to CRN seeding contract; drives deterministic per-phase buyer shuffle. **Accepted.**
- [ADR 0017](docs/adr/0017-setup-files-as-deterministic-input.md) — Setup files as deterministic input. **Accepted.**
- [ADR 0018](docs/adr/0018-lateral-links-demand-pull-scheduling.md) — Lateral supplier links + demand-pull topological scheduling (Kahn on reversed graph); type-based validation; min-order-aware routing; rejection log. **Accepted — supersedes ADR 0014.**
