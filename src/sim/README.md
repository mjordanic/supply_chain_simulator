# `src/sim/` — Simulator core

The discrete-event, **multi-echelon** supply-chain simulator: a validated DAG of typed nodes
(factories → intermediates → demand sinks), a live central offer book, FCFS allocation, market
dynamics, and the data exporter that persists run results. Every other sub-package
(`src/llm/`, `src/tuning/`, `src/rl/`) builds on top of the primitives defined here.

## Contents

1. [Layout](#layout)
2. [Concepts](#concepts)
3. [Running a setup directory](#running-a-setup-directory)
4. [Authoring a setup directory by hand](#authoring-a-setup-directory-by-hand)
5. [Real-data demand replay (M5)](#real-data-demand-replay-m5)
6. [Output layout](#output-layout)
7. [Policy reference](#policy-reference)
8. [Common Random Numbers and reproducibility](#common-random-numbers-and-reproducibility)
9. [Visualizing a run](#visualizing-a-run)
10. [Further reading](#further-reading)

## Layout

```
src/sim/
  node.py                Node ABC + FactoryNode / IntermediateNode / DemandSinkNode
  replay_demand_sink.py  ReplayDemandSinkNode (demand from an observed series — real-data replay)
  graph.py               EdgeSpec + Graph (DAG validation, levels, topology queries) + build_graph
  central_table.py       CentralTable live offer book + Offer (fill-rate EMA)
  allocation.py          execute_buy (FCFS primitive) + shuffle_buyers + AllocationResult
  policy.py              NodePolicy ABCs + concrete node policies + TextbookReorderPolicy family
  runner.py              Runner + Simulation / build_world / TickResult (two-phase tick API)
  market.py              regional demand/supply; demand_multiplier(pid, region, tick)
  flat_world.py          flat_world() — market/lifecycle params with multiplier chain ≡ 1.0 (pure replay)
  event_engine.py        stochastic disruptions + scheduled delivery callbacks
  observation.py         per-node observation builders handed to policies each tick
  distributions.py       Constant / Uniform / Normal / Choice / LogUniform
  metrics.py             RunSlice + aggregate_episode + KPI helpers (shared with tuning + RL)
  episode_sampler.py     EpisodeSpec + sample_episode + default-params factories (5 sub-seeds)
  scenario.py            Scenario dataclass + load_catalog() / make_nodes() helpers
  setup_io.py            load_setup / write_catalog_and_market / load_or_build_setup
  topology_scaffolder.py scaffold_topology (nodes/edges block generator)
  policy_registry.py     string-name → Policy factory (used by setup_io)
  data_exporter.py       parquet + JSON + PNG writer
  inspect.py             run-log → tidy DataFrames (node / global / per-product, equity)
```

## Concepts

- **Node** (`Node` ABC) — the actor in the supply-chain graph. Three concrete subtypes:
  - **`FactoryNode`** — produces one product per tick at `unit_cost`; sells at `list_price` == `unit_cost` (zero-margin, ADR 0013). The entry point of inventory into the chain.
  - **`IntermediateNode`** — warehouse / DC / shop. Holds multi-product `inventory`, sets `list_prices`, imposes per-product `min_order_imposed`, routes orders across multiple upstream suppliers, tracks `pending` (per-supplier × pid in-transit). Warehouse-vs-shop is a free-form `tags` label, never branches mechanics.
  - **`DemandSinkNode`** — the only source of new cash. Bound to one `product_id`; each tick earns `income_rate`, computes a demand target, and buys from intermediates. Unmet demand (no supplier stock or no cash) is a lost sale.
  - **`ReplayDemandSinkNode`** (`replay_demand_sink.py`) — a `DemandSinkNode` whose per-tick demand comes from an observed series instead of `demand_dist` — the real-data replay mode. The full market multiplier chain still applies on top, and the node burns the same `world_rng` draws as the stochastic sink, so mixed replay/stochastic graphs stay CRN-clean. See [Real-data demand replay (M5)](#real-data-demand-replay-m5).

  Nodes are *not* frozen — the engine mutates `inventory` / `cash` / `pending` in place. Construction consumes only the per-node `init_rng`, so two nodes built from the same `(subclass, init_seed)` start bit-identical regardless of attached policy. See ADR 0011.
- **Graph** + **EdgeSpec** — `build_graph(node_ids, edges, node_types=...)` validates a DAG and computes display-only `levels` via longest-path-from-any-source. Validation is **type-based** (ADR 0018): supplier ∈ {factory, intermediate}, buyer ∈ {intermediate, sink}, `factory → sink` rejected (flow must pass through ≥1 intermediate); plus the cycle / self-loop / unreachable-node checks. Lateral `intermediate → intermediate` edges at any depth are legal — only the *union* of all edges must stay acyclic (a node pair may trade in one direction across all products). `level` is a display hint, not a scheduling unit. `EdgeSpec(supplier_id, buyer_id, default_lead_time, per_product_lead_time=None)` is a directed supply edge with an optional per-product lead-time override. Topology queries: `suppliers_of`, `buyers_of`, `lead_time`.
- **CentralTable** + **Offer** — a live, globally-visible offer book. At the start of each tick every seller `publish`es an `Offer(available_qty, list_price, min_order, fill_rate_recent)`; as allocations land the allocator `commit`s, decrementing `available_qty` live and updating a qty-weighted EMA on `fill_rate_recent` (10-tick window). `snapshot_for_buyer(pid)` reflects prior buyers' allocations *within the same phase*, so routing accounts for live supply depletion. See ADR 0012.
- **Allocation** — `execute_buy(buyer, supplier, pid, qty_requested, table, cash_ledger) -> AllocationResult` is the single FCFS primitive: two-layer min-order rejection (supplier-imposed + buyer-side policy floor), clamp by live `available_qty` / buyer cash / buyer remaining capacity, `table.commit`, cash transfer, and a delivery callback scheduled at `current_tick + lead_time`. Physical lead time still delays arrival. `AllocationResult` carries a `reason` field naming the binding constraint on a rejection (`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`, `insufficient_capacity`). Buyer routing is **min-order-aware**: a supplier whose `min_order` can't be met is skipped and the buyer falls through to the next feasible supplier rather than silently losing the sale. `shuffle_buyers(buyers, allocation_rng)` is the deterministic per-phase buyer shuffle. See ADR 0012, ADR 0016, ADR 0018.
- **Demand-pull schedule (tick phasing)** — one tick is a reverse-topological walk (sinks first, factories last) via Kahn's algorithm on the reversed graph: `tick_world` (market multiplier → events) → publish all offers → walk each ready-set of incomparable peers (shuffled with `allocation_rng` for FCFS fairness): per-buyer observe → decide → `execute_buy` per line → fire delivery callbacks → factory produce → consume demand sinks. Because `execute_buy` decrements seller stock at sale time, by the time an intermediate is processed its downstream buyers have already transacted, so it reorders against *complete current-tick* demand (`observed_sales`) — there is no one-tick demand lag. The schedule structure is computed once at `build_world` (topology is static); only the ready-set shuffle re-draws each tick. This survives lateral links. See ADR 0018.
- **Four-stream RNG split** — `world_rng` (market, events, demand draws), `allocation_rng` (per-phase buyer shuffle, ADR 0016), `policy_rng` (one per policy instance), and `init_rng` (one per node, step-0 state) never share state. This is what makes CRN comparison correct.
- **DataExporter** — consumes the run log + scenario and writes parquet/JSON/PNG under `data/<setup-dir-name>/`.

For full domain definitions see [`CONTEXT.md`](../../CONTEXT.md).

## Running a setup directory

`main.py` (at the repo root) is a thin CLI with two subcommands.

**`run`** — load a setup directory and simulate:

```bash
# Default output: data/<setup-dir-name>/
uv run python main.py run setups/three_node_chain

# Custom output folder
uv run python main.py run setups/three_node_chain --output /tmp/run
```

**`scaffold`** — generate a starter `nodes:`/`edges:` block from a catalog CSV and write (or merge)
it into a target `setup.yaml`:

```bash
uv run python main.py scaffold setups/my_run/catalog.csv --out setups/my_run/setup.yaml
```

## Authoring a setup directory by hand

A setup directory contains two files (plus `demand_series.parquet` when the setup uses
`sink_replay` nodes — see [Real-data demand replay (M5)](#real-data-demand-replay-m5)):

```
my_run/
  catalog.csv             one row per SKU
  setup.yaml              run, market, disruption, nodes, edges
  demand_series.parquet   (only for sink_replay nodes) tidy (series_id, tick, qty) demand
```

**`catalog.csv`** — columns: `product_id`, `name`, `category`, `base_price`, `unit_cost`,
`seasonality`, `related_products` (optional), `init_stock_share` (optional).

```csv
product_id,name,category,base_price,unit_cost,seasonality
P0001,Widget A,widgets,12.50,5.00,all_season
P0002,Widget B,widgets,15.00,6.00,peak
```

**`setup.yaml`** — a structured document with these top-level keys:

```yaml
run:
  n_steps: 180
  start_date: "2024-01-01"
  world_seed: 42

market:
  cycle_len: 365
  cycle_amp: 0.0
  init_demand: 1.0
  # ... (see setups/three_node_chain/setup.yaml for a full example)

disruption:
  event_prob: 0.0
  types: [natural_disaster]
  regions: [US]
  severity: {kind: constant, value: 0.0}
  duration: {kind: constant, value: 1}

nodes:
  - id: factory-1
    type: factory           # factory | intermediate | demand_sink | sink_replay
    region: US
    produces_product_id: P0001
    unit_cost: 5.0
    capacity_per_tick: 50
    inventory: 100
    list_price: 5.0
    cash: 0.0
    policy:
      name: static_factory
      params:
        target_inventory: 200

  - id: shop-1
    type: intermediate
    region: US
    carried_products: [P0001]
    capacity: 500
    tags: [shop]
    inventory: {P0001: 20}
    list_prices: {P0001: 12.50}
    min_order_imposed: {P0001: 0}
    cash: 500.0
    policy:
      name: order_up_to
      params:
        cover_horizon_ticks: 14

  - id: sink-1
    type: demand_sink
    region: US
    product_id: P0001
    demand_dist: {kind: normal, mean: 10.0, std: 2.0}
    income_rate: 200.0
    cash: 1000.0
    policy:
      name: default_demand_sink
      params: {}

edges:
  - supplier: factory-1
    buyer: shop-1
    lead_time: 2
  - supplier: shop-1
    buyer: sink-1
    lead_time: 1
```

**Available policy names** (`name:` in the `policy:` block):

| name | node type | class |
|---|---|---|
| `static_factory` | factory | `StaticFactoryPolicy` |
| `default_demand_sink` | demand_sink | `DefaultDemandSinkPolicy` |
| `order_up_to` | intermediate | `OrderUpToPolicy` |
| `reorder_point` | intermediate | `ReorderPointPolicy` |
| `periodic_order_up_to` | intermediate | `PeriodicOrderUpToPolicy` |
| `periodic_reorder` | intermediate | `PeriodicReorderPolicy` |
| `single_supplier` | intermediate | `IntermediatePolicy.SingleSupplierAdapter` |

**Distribution kinds** (used in `demand_dist`, `capacity_per_tick`, `severity`, `duration`, …):

| kind | params | example |
|---|---|---|
| `constant` | `value` | `{kind: constant, value: 10.0}` |
| `uniform` | `lo`, `hi` | `{kind: uniform, lo: 5.0, hi: 15.0}` |
| `normal` | `mean`, `std` | `{kind: normal, mean: 10.0, std: 2.0}` |
| `choice` | `options` | `{kind: choice, options: [5, 10, 15]}` |
| `log_uniform` | `lo`, `hi` | `{kind: log_uniform, lo: 100, hi: 10000}` |

See `setups/three_node_chain/` and `setups/two_factories_two_shops/` for complete hand-authored examples.

## Real-data demand replay (M5)

The simulator can replay observed real-world demand instead of sampling it. Three pieces
work together (ADR 0020):

- **`ReplayDemandSinkNode`** (`replay_demand_sink.py`) — a `DemandSinkNode` subclass whose
  per-tick base demand is `series[tick]` instead of a `demand_dist` draw. The full
  multiplier chain (market × seasonal × regional) still applies on top, so synthetic
  what-ifs — promos, disruptions, elasticity — compose on replayed demand unchanged.
  *Pure* replay is achieved by authoring the multiplier chain flat, not by a node flag:
  **`flat_world(regions)`** (`flat_world.py`) returns `MarketParams` + `ItemLifecycleParams`
  whose chain is bit-exactly 1.0 on every tick. CRN invariant: the replay sink burns
  exactly the same `world_rng` draws as the stochastic sink (one per catalog item per
  tick), so mixing replay and stochastic sinks in one graph leaves every other RNG
  stream bit-identical.
- **`sink_replay` setup nodes** — replay sinks are authorable in `setup.yaml`. The node
  references a `series_id` in `demand_series.parquet` (tidy `(series_id, tick, qty)`),
  which must sit next to `setup.yaml` and cover at least `n_steps` ticks:

  ```yaml
  - id: sink-FOODS_3_090-CA_1
    type: sink_replay
    region: CA
    product_id: FOODS_3_090
    series_id: FOODS_3_090_CA_1
    income_rate: 1000000.0
    cash: 0.0
  ```

- **`PriceReplayPolicy`** (`policy.py`) — opt-in wrapper around any `IntermediatePolicy`
  that delegates ordering decisions to the inner policy unchanged and overwrites the
  `list_price` slice with observed per-tick price arrays. Ordering is bit-identical with
  and without the wrapper; products without a price array keep the inner policy's price.
  Constructed programmatically (`PriceReplayPolicy(inner, prices, n_steps)`) — it is not
  in the policy registry.

The **M5 dataset adapter** (`src/datasets/m5.py`) produces all of this from the raw
Kaggle M5 (Walmart) files in one call: it slices items/stores/dates, runs a data-quality
report, and emits a complete setup directory — shops, `sink_replay` nodes, catalog, and
the demand/price/calendar parquets. Upstream topology (factories, DCs) is the scenario
author's job. Full reference: [`src/datasets/README.md`](../datasets/README.md);
end-to-end walkthrough: [`notebooks/m5_replay_example.ipynb`](../../notebooks/m5_replay_example.ipynb).

## Output layout

`Runner(scenario).run()` returns a run log with top-level keys `n_steps`, `ticks`, `global`:

- `ticks` — one entry per tick: `tick`, `node_cash`, `node_inventory` (per-node `{pid: qty}`; factories use `{"_total": qty}`), `node_pending` (per-node `{pid: in_transit}` summed across suppliers), `node_orders`, `node_flows`, `purchases` (the flow log, see below), and `rejections` (see below).
- `global` — `time` (`simulation_step`, `simulation_date`), `market_supply` / `market_demand` per region, and `events`.
- `node_flows` / `purchases` (per-tick flow log, ADR 0019) — `node_flows` is one record per `(node, pid)`: `sales` (units sold as supplier), `demand` (units requested of it; a sink's exogenous `demand_target`), `price` (decision-time `list_price`; `null` for sinks), and `stockout` (decision-time on-hand == 0). `purchases` is one record per `(buyer, supplier, pid)`: `qty_filled` and `cash_paid` (= `qty_filled × supplier list_price`). `node_orders` is derived from these purchase rows. Build tidy DataFrames with `inspect.flow_frame` / `inspect.purchase_frame`.
- `rejections` (per-tick, always-on, ADR 0018) — an observability stream for unfilled or partially-filled orders. Each entry is `{tick, buyer_id, supplier_id, pid, qty_requested, qty_filled, qty_rejected, reason}`, where `reason` ∈ {`no_offer`, `below_min_order`, `insufficient_stock`, `insufficient_cash`, `insufficient_capacity`}. Demand that no feasible supplier could fill is recorded as an `unmet_demand` entry (`supplier_id=None`) — the true lost-sale measure once min-order fall-through removes the avoidable rejections.

`DataExporter(scenario, run_log).export_all(output_folder)` writes (relative to `--output`, default `data/<setup-dir-name>/`):

```
data/<setup-dir-name>/
  config/
    catalog.csv              verbatim copy of input catalog
    setup.yaml               verbatim copy of input setup
    scenario.json            run metadata snapshot
  data/
    run_log.json             the run log above, datetimes → ISO strings
    products.parquet         static catalog table
    nodes.parquet            static node table (node_id, node_type, region, init_seed, policy_type)
    timeseries.parquet       per-step × per-node × per-product table
  reports/
    overview.png             regional supply/demand chart
```

## Policy reference

### Policy hierarchy

`NodePolicy` is the base ABC. Three type-paired subclass ABCs match the three node types:

| ABC | `decide` signature | attached to |
| --- | --- | --- |
| `FactoryPolicy` | `decide(obs_factory)` | `FactoryNode` |
| `IntermediatePolicy` | `decide(obs_intermediate, central_table)` | `IntermediateNode` |
| `DemandSinkPolicy` | `decide(obs_sink, central_table)` | `DemandSinkNode` |

Each policy owns a private `policy_rng` seeded from its `policy_seed` kwarg, disjoint from
`world_rng` and `allocation_rng` so policy choice never perturbs the world or allocation streams.
`Policy` is kept as a one-release alias of `NodePolicy`.

### Shipped node policies

| class | base | role |
| --- | --- | --- |
| `StaticFactoryPolicy(*, capacity_per_tick, unit_cost, policy_seed=None)` | `FactoryPolicy` | produce a fixed quantity each tick, publish at `unit_cost` |
| `DefaultDemandSinkPolicy(*, policy_seed=None)` | `DemandSinkPolicy` | greedy: buy the demand target from the cheapest feasible direct suppliers first |
| `IntermediatePolicy.SingleSupplierAdapter(*, supplier_id, ...)` | `IntermediatePolicy` | route a textbook reorder policy to a single named supplier |
| `MultiSupplierTextbookPolicy` (+ 4 concrete) | `IntermediatePolicy` | textbook reorder rules with multi-supplier routing |
| `PriceReplayPolicy(inner, prices, n_steps)` | `IntermediatePolicy` | wrapper: inner policy orders, observed daily prices overwrite `list_price` |
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

Custom policies can be wired into a setup directory by registering them in
`src/sim/policy_registry.py` or constructing a `Scenario` programmatically via the Python API.

## Common Random Numbers and reproducibility

Four independent random streams, four independent seeds:

| Seed | Stream | Used for |
|---|---|---|
| `Scenario.world_seed` | `world_rng` | market dynamics, disruption events, demand draws |
| derived `_derive_seed(world_seed, "allocation")` | `allocation_rng` | per-phase buyer shuffle (ADR 0016) |
| `NodePolicy.policy_seed` | `policy_rng` | policy decisions only (one stream per policy instance) |
| `NodeInstance.init_seed` | `init_rng` | per-node step-0 state |

Consequences:

- Replaying the same `Scenario` produces the same world trajectory.
- Two nodes constructed from the same `(subclass, init_seed)` pair start step 0 bit-identical.
- Swapping a policy on a scenario does not perturb the world or allocation streams, so per-policy outcome differences come from policy decisions alone — the property the paired-comparison and tuning/RL evaluators rely on.

`build_world` deep-copies nodes so the same node templates can be reused across calls without mutation aliasing.

## Visualizing a run

`notebooks/02-run-and-inspect.ipynb` dissects a run's outputs: run-log anatomy, per-tier cash/inventory/equity, market supply/demand, per-product time-series, and the exported parquet/JSON/PNG artifacts.

**Market supply / demand per region.** The market dynamics every node sees, with disruption windows shaded by event type. This is the world stream held fixed across paired-CRN comparisons.

![Market supply and demand per region](../../docs/images/sim_market_supply_demand.png)

**Equity composition + cumulative P&L.** Stacked equity (cash + inventory at cost + outstanding orders at cost) on the left axis; cumulative P&L on the right.

![Equity composition and cumulative P&L](../../docs/images/sim_equity_composition.png)

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md) — domain and architecture glossary
- ADRs [0011](../../docs/adr/0011-multi-echelon-graph.md)–[0018](../../docs/adr/0018-lateral-links-demand-pull-scheduling.md) — the multi-echelon design decisions (latest: lateral links + demand-pull scheduling)
