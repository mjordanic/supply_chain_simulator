# `src/sim/` — Simulator core

The discrete-event, **multi-echelon** supply-chain simulator: a validated DAG of typed
nodes (factories → intermediates → demand sinks), a live central offer book,
FCFS allocation, market dynamics, item life-cycle, freshness curves, and the data
exporter that persists run results. Every other sub-package (`src/llm/`,
`src/tuning/`, `src/rl/`) builds on top of the primitives defined here.

## Contents

1. [Layout](#layout)
2. [Concepts](#concepts)
3. [Running a scenario](#running-a-scenario)
4. [Authoring a graph scenario](#authoring-a-graph-scenario)
5. [Output layout](#output-layout)
6. [Policy reference](#policy-reference)
7. [Example scenarios](#example-scenarios)
8. [Common Random Numbers and reproducibility](#common-random-numbers-and-reproducibility)
9. [Visualizing a run](#visualizing-a-run)
10. [Further reading](#further-reading)

## Layout

```
src/sim/
  node.py                Node ABC + FactoryNode / IntermediateNode / DemandSinkNode
  graph.py               EdgeSpec + Graph (DAG validation, levels, topology queries) + build_graph
  central_table.py       CentralTable live offer book + Offer (fill-rate EMA)
  allocation.py          execute_buy (FCFS primitive) + shuffle_buyers + AllocationResult
  policy.py              NodePolicy ABCs + concrete node policies + TextbookReorderPolicy family
  runner.py              Runner + Simulation / build_world / TickResult (two-phase tick API)
  market.py              regional demand/supply; demand_multiplier(pid, region, tick)
  event_engine.py        stochastic disruptions + scheduled delivery callbacks
  item_registry.py       catalog + per-item global lifecycle stage + freshness params
  lifecycle_clock.py     pure advance_stage() over [introduction, growth, maturity, decline, dead]
  freshness_curve.py     pure m(τ) = 1 + α · exp(−τ / β) multiplier
  observation.py         per-node observation builders handed to policies each tick
  distributions.py       Constant / Uniform / Normal / Choice / LogUniform
  metrics.py             RunSlice + aggregate_episode + KPI helpers (shared with tuning + RL)
  episode_sampler.py     EpisodeSpec + sample_episode + default-params factories (5 sub-seeds)
  scenario.py            Scenario dataclass + load_catalog() / make_nodes() helpers
  world.py               World dataclass (catalog + market + store_templates) + world_to_graph
  world_loader.py        cache_path → archetype → synthetic fallback resolver
  data_exporter.py       parquet + JSON + PNG writer
```

## Concepts

- **Node** (`Node` ABC) — the actor in the supply-chain graph. Three concrete subtypes:
  - **`FactoryNode`** — produces one product per tick at `unit_cost`; sells at `list_price` == `unit_cost` (zero-margin, ADR 0013). Echelon level 0, the entry point of inventory.
  - **`IntermediateNode`** — warehouse / DC / shop. Holds multi-product `inventory`, sets `list_prices`, imposes per-product `min_order_imposed`, routes orders across multiple upstream suppliers, tracks `pending` (per-supplier × pid in-transit). Warehouse-vs-shop is a free-form `tags` label, never branches mechanics.
  - **`DemandSinkNode`** — the only source of new cash. Bound to one `product_id`; each tick earns `income_rate`, computes a demand target, and buys from intermediates. Unmet demand (no supplier stock or no cash) is a lost sale.

  Nodes are *not* frozen — the engine mutates `inventory` / `cash` / `pending` in place. Construction consumes only the per-node `init_rng`, so two nodes built from the same `(subclass, init_seed)` start bit-identical regardless of attached policy. See ADR 0011.
- **Graph** + **EdgeSpec** — `build_graph(node_ids, edges)` validates a DAG (raises on cycles, unreachable nodes, or same-level supplier links) and computes echelon `levels` via longest-path-from-any-factory. `EdgeSpec(supplier_id, buyer_id, default_lead_time, per_product_lead_time=None)` is a directed supply edge with an optional per-product lead-time override. Topology queries: `suppliers_of`, `buyers_of`, `lead_time`.
- **CentralTable** + **Offer** — a live, globally-visible offer book. At the start of each tick every seller `publish`es an `Offer(available_qty, list_price, min_order, fill_rate_recent)`; as allocations land the allocator `commit`s, decrementing `available_qty` live and updating a qty-weighted EMA on `fill_rate_recent` (10-tick window). `snapshot_for_buyer(pid)` reflects prior buyers' allocations *within the same phase*, so routing accounts for live supply depletion. See ADR 0012.
- **Allocation** — `execute_buy(buyer, supplier, pid, qty_requested, table, cash_ledger) -> AllocationResult` is the single FCFS primitive: two-layer min-order rejection (supplier-imposed + buyer-side policy floor), clamp by live `available_qty` / buyer cash / buyer remaining capacity, `table.commit`, cash transfer, and a delivery callback scheduled at `current_tick + lead_time`. Physical lead time still delays arrival. `shuffle_buyers(buyers, allocation_rng)` is the deterministic per-phase buyer shuffle. See ADR 0012, ADR 0016.
- **Phase cascade (tick phasing)** — one tick is a deterministic upward cascade by echelon level: `tick_world` (market multiplier → events → item lifecycle) → publish all offers → for each level p from 1..max: shuffle buyers → per-buyer observe → decide → `execute_buy` per line → factory produce → fire delivery callbacks → consume demand sinks. Demand pulls up the chain in natural order. See ADR 0014.
- **Two-layer lifecycle** — every product has a global PLC stage in `[introduction, growth, maturity, decline, dead]` advanced by `LifecycleClock` against per-stage `stage_change_probs`; on top, every `(sink, product)` pair has a freshness curve `m(τ) = 1 + α · exp(−τ / β)` keyed on `DemandSinkNode.activation_tick`. Composed multiplicatively with `Market.demand_multiplier` inside `DemandSinkNode.demand_target`. See ADR 0001.
- **Four-stream RNG split** — `world_rng` (market, events, lifecycle, demand draws), `allocation_rng` (per-phase buyer shuffle, ADR 0016), `policy_rng` (one per policy instance), and `init_rng` (one per node, step-0 state) never share state. This is what makes CRN comparison correct.
- **DataExporter** — consumes the run log + scenario and writes parquet/JSON/PNG under `data/<scenario_stem>/`.

For full domain definitions see [`CONTEXT.md`](../../CONTEXT.md).

## Running a scenario

`main.py` (at the repo root) is a thin CLI: it imports a Python file by path, expects a top-level `scenario` symbol of type `Scenario`, and dispatches to `Runner` and `DataExporter`.

```bash
# Default output: data/<file_stem>/
uv run python main.py scenarios/example_homogeneous.py

# Custom output folder
uv run python main.py scenarios/example_paired_comparison.py --output /tmp/run
```

Each example scenario is also independently runnable:

```bash
uv run python scenarios/example_chain_three_node.py
```

Errors `main.py` emits on a bad scenario path:

- `main.py: load_scenario_from_path: scenario file not found: <path>` — file does not exist
- `main.py: load_scenario_from_path: <path> does not expose a top-level \`scenario\` attribute` — module loaded but no `scenario =` at module level
- `main.py: load_scenario_from_path: <path>.scenario is <type>, expected Scenario` — wrong type

`Runner` requires a **graph-mode** scenario (`scenario.is_graph == True`, i.e. `nodes` is non-empty). The legacy `stores`-based fields remain on `Scenario` for backward compatibility but are ignored by the engine.

## Authoring a graph scenario

A scenario file is a Python module that constructs a `Scenario` with `nodes` + `edges` and binds it to the name `scenario` at module level.

**1. Catalog.** `load_catalog(items)` accepts a list of dicts and assigns stable `P{i:04d}` ids. Every dict can additionally set per-`Ware` lifecycle / freshness / stock overrides; if omitted, the corresponding default on `ItemLifecycleParams` applies:

```python
from src.sim.scenario import load_catalog

catalog = load_catalog([
    {
        "name": "Widget A",
        "category": "Widgets",
        "related_products": [],
        "base_price": 20.0,
        "unit_cost": 12.0,
        "seasonality": "all_season",
        # Optional per-Ware overrides (any/all may be omitted):
        # "init_stage": "maturity",
        # "stage_change_probs": {"introduction": 0.02, "growth": 0.005,
        #                        "maturity": 0.001, "decline": 0.005, "dead": 0.003},
        # "freshness_alpha": 0.0,        # 0 = staple, no hype curve
        # "freshness_decay": 30.0,
        # "init_stock_share": 2.0,       # weight for initial stock allocation
    },
])
pid = catalog[0].product_id   # "P0000"
```

**2. Market, disruption, lifecycle.** Flat dataclasses; stochastic fields hold `Distribution` objects (`Constant`, `Uniform`, `Normal`, `Choice`, `LogUniform`). For a single-product chain the market can be a no-op (flat multipliers):

```python
from src.sim.distributions import Constant, Normal
from src.sim.scenario import DisruptionParams, ItemLifecycleParams, MarketParams

_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
lifecycle = ItemLifecycleParams(
    stages=_STAGES,
    init_stage="maturity",
    # Per-current-stage transition table consumed by LifecycleClock.advance_stage.
    # All-zero ⇒ products stay at init_stage (strict-terminal lifecycle).
    default_stage_change_probs={s: 0.0 for s in _STAGES},
    default_freshness_alpha=0.0,   # 0 collapses the freshness curve to identically 1
    default_freshness_decay=1.0,
    default_init_stock_share=1.0,
)
disruption = DisruptionParams(
    event_prob=0.0, types=["natural_disaster"], regions=["US"],
    severity=Constant(0.0), duration=Constant(1),
)
market = MarketParams(
    cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
    peak_factor=1.0, off_factor=1.0, season_months={}, regions=["US"],
    correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
    stage_multipliers={"introduction": 1.0, "growth": 1.0, "maturity": 1.0,
                       "decline": 1.0, "dead": 0.0},
    price_elasticity=0.0, promo_multiplier=1.0,
    demand_factor_min=0.1, supply_factor_min=0.01,
    cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
    trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
    base_demand=Constant(10),
)
```

**3. Build the nodes.** Construct typed `Node` instances and attach a policy to each:

```python
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy, IntermediatePolicy, StaticFactoryPolicy,
)

factory = FactoryNode(
    id="factory-1", region="US", init_seed=1,
    produces_product_id=pid, unit_cost=12.0, capacity_per_tick=50,
    inventory=100, list_price=12.0, cash=0.0,
)
factory.policy = StaticFactoryPolicy(capacity_per_tick=50, unit_cost=12.0, policy_seed=1)

shop = IntermediateNode(
    id="shop-1", region="US", init_seed=2,
    carried_products={pid}, capacity=500, tags=["shop"],
    inventory={pid: 20}, pending={}, list_prices={pid: 20.0},
    min_order_imposed={pid: 0}, cash=500.0,
)
# SingleSupplierAdapter wraps a textbook policy to one named supplier.
shop.policy = IntermediatePolicy.SingleSupplierAdapter(
    supplier_id="factory-1", cover_horizon_ticks=14, safety_lead_pct_of_lag=1/3,
    delivery_lag=2, unit_cost=12.0, list_price_out=20.0, policy_seed=2,
)

sink = DemandSinkNode(
    id="sink-1", region="US", init_seed=3,
    product_id=pid, demand_dist=Normal(mean=10, std=2), income_rate=200.0, cash=1000.0,
)
sink.policy = DefaultDemandSinkPolicy(policy_seed=3)
```

**4. Wire edges.** `EdgeSpec` is a directed supply edge with a per-edge lead time:

```python
from src.sim.graph import EdgeSpec

edges = [
    EdgeSpec(supplier_id="factory-1", buyer_id="shop-1", default_lead_time=2),
    EdgeSpec(supplier_id="shop-1",    buyer_id="sink-1", default_lead_time=1),
]
```

Multi-supplier topologies are just more edges: give two factories an edge into the same shop and an `IntermediatePolicy` that routes across both (e.g. `OrderUpToPolicy`, which lifts the textbook rule to multi-supplier routing).

**5. Final assembly.** Bind to `scenario` via `NodeInstance` (or the `make_nodes(triples)` helper). `stores=[]` keeps the legacy field empty:

```python
from datetime import datetime
from src.sim.scenario import NodeInstance, Scenario

scenario = Scenario(
    catalog=catalog, market=market, disruption=disruption, item_lifecycle=lifecycle,
    stores=[],
    nodes=[
        NodeInstance(node=factory, init_seed=1, policy=factory.policy),
        NodeInstance(node=shop,    init_seed=2, policy=shop.policy),
        NodeInstance(node=sink,    init_seed=3, policy=sink.policy),
    ],
    edges=edges,
    n_steps=180, start_date=datetime(2024, 1, 1), world_seed=42,
)
```

`make_nodes([(node, init_seed, policy), ...])` is the declarative shortcut — the triple list *is* the roster. CRN A/B comparisons are expressed by repeating the same `(node, init_seed)` structure with different policies; robustness sweeps by varying `init_seed`; mixed rosters by writing the literal list. There is no regime abstraction above it.

Save under `scenarios/my_run.py` and run `uv run python main.py scenarios/my_run.py`.

### From an LLM-built world

`world_to_graph(world, *, sink_density=1.0)` synthesises a default factory → intermediate → sink topology from a `World.store_templates`, so LLM-generated catalogs feed graph scenarios without re-prompting. `sink_density` (in `(0, 1]`) controls how many catalog products get a demand-sink node. See `scenarios/example_llm_world_offline.py`.

## Output layout

`Runner(scenario).run()` returns a run log with top-level keys `n_steps`, `ticks`, `global`:

- `ticks` — one entry per tick: `tick`, `node_cash`, `node_inventory` (per-node `{pid: qty}`; factories use `{"_total": qty}`), `node_pending` (per-node `{pid: in_transit}` summed across suppliers), `node_orders`.
- `global` — `time` (`simulation_step`, `simulation_date`), `market_supply` / `market_demand` per region, and `products` (resolved freshness / lifecycle per pid).

`DataExporter(scenario, run_log).export_all(output_folder)` writes (relative to `--output`, default `data/<stem>/`):

```
data/<stem>/
  config/
    scenario.json            full Scenario.to_json() (policies excluded — Python objects)
  data/
    run_log.json             the run log above, datetimes → ISO strings
    products.parquet         static catalog with resolved freshness_alpha/decay + init_stock_share
    stores.parquet           static node table (node_id, node_type, region, init_seed, policy_type)
    timeseries.parquet       per-step × per-node × per-product table
  reports/
    overview.png             regional supply/demand chart
```

`timeseries.parquet` columns: `simulation_step`, `simulation_date`, `store_id` (node id), `product_id`, `inventory`, `demand`, `sales`, `order_quantity`, `outstanding_orders`, `promotion_status`, `active_status`, `price`, `revenue`, `total_cost`, `holding_cost`, `profit`. For **graph-mode** runs the exporter currently populates `inventory` per `(node, product)` from the per-tick node snapshots; the richer per-product economic columns are `None` (they were produced by the retired per-store engine and are kept in the schema for backward compatibility). Cash and P&L per node are available from `run_log["ticks"][t]["node_cash"]`.

## Policy reference

### Policy hierarchy

`NodePolicy` is the base ABC. Three type-paired subclass ABCs match the three node types:

| ABC | `decide` signature | attached to |
| --- | --- | --- |
| `FactoryPolicy` | `decide(obs_factory)` | `FactoryNode` |
| `IntermediatePolicy` | `decide(obs_intermediate, central_table)` | `IntermediateNode` |
| `DemandSinkPolicy` | `decide(obs_sink, central_table)` | `DemandSinkNode` |

Each policy owns a private `policy_rng` seeded from its `policy_seed` kwarg, disjoint from `world_rng` and `allocation_rng` so policy choice never perturbs the world or allocation streams. `Policy` is kept as a one-release alias of `NodePolicy`; `HeuristicPolicy` is a **tombstone** that raises on construction (the heuristic baseline was retired in favour of the textbook family — ADR 0006).

### Shipped node policies

| class | base | role |
| --- | --- | --- |
| `StaticFactoryPolicy(*, capacity_per_tick, unit_cost, policy_seed=None)` | `FactoryPolicy` | produce a fixed quantity each tick, publish at `unit_cost` |
| `DefaultDemandSinkPolicy(*, policy_seed=None)` | `DemandSinkPolicy` | greedy: buy the demand target from the cheapest feasible direct suppliers first |
| `IntermediatePolicy.SingleSupplierAdapter(*, supplier_id, ...)` | `IntermediatePolicy` | route a textbook reorder policy to a single named supplier |
| `MultiSupplierTextbookPolicy` (+ 4 concrete) | `IntermediatePolicy` | textbook reorder rules with multi-supplier routing |
| `RLIntermediatePolicy` | `IntermediatePolicy` | shim that replays a pre-decoded action injected by the RL stack |

### `TextbookReorderPolicy` family (`MultiSupplierTextbookPolicy`)

Four textbook inventory rules sharing one base class: a censored-sales rate estimator over a `delivery_lag`-length rolling window, a two-pass fair-share allocator across the capacity and cash pools, a cash-budget "pilot order" cold-start, and opt-in adaptive safety stock. All four emit pure inventory decisions (`price[pid] = base_price`, no activations/promotions). Multi-supplier routing via `_split_across_suppliers` honours both min-order layers; the default strategy is cheapest-first, pluggable via the `routing_strategy` kwarg (`"cheapest_first"` / `"fill_rate_weighted"`).

Shared `MultiSupplierTextbookPolicy` kwargs:

| kwarg | default | meaning |
| --- | --- | --- |
| `policy_seed` | `None` | seed for `policy_rng` |
| `cover_horizon_ticks` | 14 | cycle length driving `S − s` for order-up-to (EOQ result, independent of lead time) |
| `safety_lead_pct_of_lag` | 1/3 | fraction of per-edge lead time used as the safety horizon: `s = (lag + pct × lag) × rate` |
| `delivery_lag` | 2 | fallback lead time the inner policy assumes for pilot sizing |
| `unit_cost` | 1.0 | upstream unit cost (pilot-order budgeting) |
| `list_price_out` | 0.0 | the node's own downstream selling price |
| `per_supplier_min_order_floor` | 0 | buyer-side min order per line (the second min-order layer) |
| `routing_strategy` | `None` | `None`/`"cheapest_first"` or `"fill_rate_weighted"` |

The four concrete variants:

| class | rule | extra kwarg | notes |
| --- | --- | --- | --- |
| `OrderUpToPolicy` | (s,S) continuous review | — | the **CRN comparison anchor** for RL |
| `ReorderPointPolicy` | (s,Q) continuous review | `Q` (default rate-derived) | fixed-order-quantity variant |
| `PeriodicOrderUpToPolicy` | (R,S) periodic review | `review_interval` (ticks) | order-up-to with a calendar trigger |
| `PeriodicReorderPolicy` | (R,s,S) periodic review | `review_interval` (ticks) | (s,S) gated by a calendar trigger |

### Writing a custom policy

Subclass the ABC matching the node it drives and implement `decide`:

```python
from collections.abc import Mapping
from typing import Any

from src.sim.policy import IntermediatePolicy


class MyIntermediatePolicy(IntermediatePolicy):
    def decide(self, observation: Mapping[str, Any], central_table) -> dict[str, Any]:
        orders = []
        for pid in observation["carried_products"]:
            # central_table.snapshot_for_buyer(pid) → [(supplier_id, Offer), ...]
            # reflecting live supply depletion by earlier buyers this phase.
            offers = central_table.snapshot_for_buyer(pid)
            if offers:
                supplier_id, _offer = min(offers, key=lambda so: so[1].list_price)
                orders.append((supplier_id, pid, 0))   # (supplier, product, qty)
        return {"orders": orders, "list_prices": observation["list_prices"]}
```

All stochastic choices should consume `self.policy_rng` so policy decisions never perturb the world stream.

## Example scenarios

Graph-engine examples (no API key needed — synthetic catalogs):

**`scenarios/example_chain_three_node.py`** — minimal `factory → shop → sink` chain, single product, 180 ticks. The smallest end-to-end graph; the right starting point for reading the engine. Writes to `runs/example_chain_three_node/`.

```bash
uv run python scenarios/example_chain_three_node.py
```

**`scenarios/example_two_factories_two_shops.py`** — 2 factories + 2 shops + 2 sinks. Both shops source from both factories via `OrderUpToPolicy` (cheapest-first routing). Cheap-but-slow `f-lo` depletes under demand bursts, dropping its `fill_rate_recent` — the contention signal the central table surfaces. Writes to `runs/example_two_factories_two_shops/`.

```bash
uv run python scenarios/example_two_factories_two_shops.py
```

**`scenarios/example_homogeneous.py`** — three "stores", each expanded to a `factory → shop → 5 sinks` sub-graph sharing one policy config but distinct seeds. The right starting point for population-level evaluation of a single policy.

```bash
uv run python main.py scenarios/example_homogeneous.py
```

**`scenarios/example_paired_comparison.py`** — paired CRN A/B: the same `(node, init_seed)` sub-graphs run `OrderUpToPolicy` vs `PeriodicOrderUpToPolicy` on bit-identical world data. The right starting point for a variance-reduced policy comparison.

```bash
uv run python main.py scenarios/example_paired_comparison.py
```

**`scenarios/example_llm_world_offline.py`** — the LLM pipeline driven by a `CannedClient` (no API key), then `world_to_graph` to synthesise the topology. Demonstrates the LLM-world → graph path deterministically.

```bash
uv run python main.py scenarios/example_llm_world_offline.py
```

> **Note.** `scenarios/example_llm_world.py` and the batch `scenarios/llm_world_{20,100,250,1000}.py` still use the legacy `stores`/`HeuristicPolicy` path (`is_graph == False`) and are **not** runnable on the current graph engine. Use `example_llm_world_offline.py` for the LLM → graph route; the large batch scenarios are pending migration.

## Common Random Numbers and reproducibility

Four independent random streams, four independent seeds:

| Seed | Stream | Used for |
|---|---|---|
| `Scenario.world_seed` | `world_rng` | market dynamics, disruption events, lifecycle transitions, demand draws |
| derived `_derive_seed(world_seed, "allocation")` | `allocation_rng` | per-phase buyer shuffle (ADR 0016) |
| `NodePolicy.policy_seed` | `policy_rng` | policy decisions only (one stream per policy instance) |
| `NodeInstance.init_seed` | `init_rng` | per-node step-0 state |

Consequences:

- Replaying the same `Scenario` produces the same world trajectory.
- Two nodes constructed from the same `(subclass, init_seed)` pair start step 0 bit-identical.
- Swapping a policy on a scenario does not perturb the world or allocation streams, so per-policy outcome differences come from policy decisions alone — the property the paired-comparison and tuning/RL evaluators rely on.

`build_world` deep-copies nodes so the same node templates can be reused across calls without mutation aliasing.

## Visualizing a run

`notebooks/04a-deep_dive_active_only.ipynb` is a single-store deep-dive over a run's parquet + run-log artifacts: world view, store financials, decision summary, and per-product cards over the active SKUs. The four headline panels below are rendered from the canned dataset at `data/llm_world_250/` via `uv run python scripts/render_readme_simulator_images.py`.

**Market supply / demand per region.** The market dynamics every node sees, with disruption windows shaded by event type. This is the world stream held fixed across paired-CRN comparisons.

![Market supply and demand per region](../../docs/images/sim_market_supply_demand.png)

**Equity composition + cumulative P&L.** Stacked equity (cash + inventory at cost + outstanding orders at cost) on the left axis; cumulative P&L on the right.

![Equity composition and cumulative P&L](../../docs/images/sim_equity_composition.png)

**Revenue vs cost per step.** Per-tick revenue (above zero) decomposed against order cost and holding cost (below zero); the black line is realised step P&L.

![Revenue vs cost per step](../../docs/images/sim_revenue_vs_cost.png)

**Per-product card — `P0333`.** Top panel: on-hand inventory, outstanding orders, and order-quantity bars against capacity. Bottom panel: sampled demand vs realised sales; the red band is unmet demand (a stockout).

![Per-product card for P0333](../../docs/images/sim_product_P0333.png)

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md) — domain and architecture glossary
- ADRs [0011](../../docs/adr/0011-multi-echelon-graph.md)–[0016](../../docs/adr/0016-allocation-rng-sub-seed.md) — the multi-echelon design decisions
