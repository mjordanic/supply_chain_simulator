# Supply Chain Simulator

A small, hackable **multi-echelon** supply-chain simulator. A scenario is a validated directed
acyclic graph of typed nodes — factories produce, intermediate nodes (warehouses / shops) hold
inventory and route orders across multiple upstream suppliers, and demand sinks generate the only
new cash in the system — all sharing one stochastic world (regional supply/demand, seasonal cycles,
and disruption events). A live central offer book lets buyers route against real-time supplier
availability; orders settle through a first-come-first-served allocator and arrive after a physical
lead time. Each node runs its own decision policy. Demand can be drawn from stochastic
distributions or replayed from real-world sales data (e.g. the Kaggle M5 / Walmart dataset — see
[`src/sim/README.md`](src/sim/README.md#real-data-demand-replay-m5)).

The simulator ships with a family of textbook inventory policies — `OrderUpToPolicy` (s,S),
`ReorderPointPolicy` (s,Q), and two periodic variants, all lifted to multi-supplier routing — that
serve as ready-made baselines. Three add-ons build on top: an **LLM world generator** that drafts
a realistic catalog and market from a domain prompt, a **hyperparameter tuner** built on Optuna,
and a **PPO reinforcement-learning stack** that trains a continuous-control policy against the
textbook baseline using Common Random Numbers.

It is a demo project — the goal is to be readable and easy to extend, not production-grade.

![Market supply and demand per region](docs/images/sim_market_supply_demand.png)

## Quickstart

```bash
uv sync                                               # install
uv run python main.py run setups/three_node_chain     # run the minimal example
uv run pytest                                         # tests
```

Outputs land under `data/three_node_chain/` (parquet + JSON + PNG). Neither example setup
requires an API key.

## Two-stage workflow

Scenarios live in **setup directories** — plain folders containing two files:

| File | Contents |
|---|---|
| `catalog.csv` | One row per SKU: `product_id`, `name`, `category`, `base_price`, `unit_cost`, `seasonality` |
| `setup.yaml` | `run:`, `market:`, `disruption:`, `nodes:`, `edges:` blocks |

**Stage 1 — prepare a setup directory** (pick one):

```bash
# Option A: author it by hand (copy setups/three_node_chain as a starting point)

# Option B: scaffold the nodes/edges block from an existing catalog CSV
uv run python main.py scaffold my_catalog.csv --out setups/my_run/setup.yaml

# Option C: let the LLM draft the catalog and market (needs OPENAI_API_KEY)
python - <<'EOF'
from src.llm.world_builder import WorldBuilder
from src.llm.openai_client import OpenAIClient
catalog, market = WorldBuilder("fashion_retail", OpenAIClient()).build_setup(
    n_items=50, setup_dir="setups/fashion_retail"
)
EOF
# Then fill in nodes/edges (run scaffold on the generated catalog.csv, then edit)
```

**Stage 2 — run:**

```bash
uv run python main.py run setups/my_run
uv run python main.py run setups/my_run --output /tmp/my_run_out
```

## Determinism and A/B comparisons

Every run is fully reproducible from `world_seed` in `setup.yaml`. To compare two policies
on bit-identical worlds, copy the setup directory and change only the `policy:` block on the
node(s) of interest — `world_seed`, `market:`, `disruption:`, and `nodes:` initial state stay
identical, so any outcome difference is attributable to the policy alone.

## Bundled setup directories

```
setups/
  three_node_chain/        minimal factory → shop → sink, 1 product, 30 ticks
  two_factories_two_shops/ 2 factories + 2 shops + 2 sinks; both shops compete for
                           inventory from a cheap-but-slow and a premium-but-fast factory
```

Run either:

```bash
uv run python main.py run setups/three_node_chain
uv run python main.py run setups/two_factories_two_shops
```

## What's inside

### Simulator and policies (`src/sim/`)

The simulator core. A scenario bundles a product catalog, a market, stochastic disruption events,
and a graph of typed nodes (`FactoryNode` → `IntermediateNode` → `DemandSinkNode`) wired by
`EdgeSpec` supply edges, each node running its own decision policy. Every tick runs as a
demand-pull topological walk (sinks first, factories last): sellers publish offers to a live
central table, buyers observe and decide in supply-dependency order, the FCFS allocator settles
trades, and deliveries arrive after their lead time. Lateral `intermediate → intermediate` edges
(e.g. a warehouse sourcing from a peer warehouse) are supported, as long as the union of all edges
stays acyclic. Custom
policies subclass the `NodePolicy` ABC matching their node type and override `decide`. Four
textbook inventory rules ship with the repo — `OrderUpToPolicy` (s,S), `ReorderPointPolicy` (s,Q),
and two periodic variants, all with multi-supplier routing — ready to drop in as baselines.

![Equity composition and cumulative P&L](docs/images/sim_equity_composition.png)

Full reference: [`src/sim/README.md`](src/sim/README.md)

### LLM world generator (`src/llm/`)

A pipeline that drafts a coherent catalog and market from a single domain prompt like
`"fashion_retail"` or `"sports_cars"`. Each stage is a schema-validated LLM call; the output is
persisted as `catalog.csv` + the `market:` block of `setup.yaml` (the setup directory acts as a
local cache — repeat calls with the same directory skip the LLM). Topology, policies, and run
parameters are the modeller's domain.

**`sports_cars_100`** — 102 items across 7 categories.

| id | name | category | base | cost | season |
|---|---|---|---:|---:|---|
| P0000 | Apex Vector R8 Track Coupe | Track-Ready Coupes | 128,900 | 91,500 | summer |
| P0024 | Meridian Grand V12 Tourer | Grand Tourers | 168,900 | 121,000 | fall/winter |
| P0042 | Briarwood Convertible GT | Roadsters & Convertibles | 112,400 | 79,700 | spring/summer |
| P0059 | Inferno X Supercar | Exotic Supercars | 315,000 | 224,000 | summer |
| P0073 | Spectra V8 Sport Sedan | Performance Sedans | 82,900 | 58,600 | all_season |

**`fashion_retail_250`** — 336 items across 6 categories.

| id | name | category | base | cost | season |
|---|---|---|---:|---:|---|
| P0000 | Women's Essential Crewneck Tee | Women's Apparel | 24.00 | 8.00 | spring/summer |
| P0086 | Men's Classic Oxford Shirt | Men's Apparel | 54.00 | 20.00 | all_season |
| P0151 | Men's Classic Derby Shoes | Footwear | 98.00 | 38.00 | all_season |
| P0192 | Women's Leather Tote Bag | Accessories | 118.00 | 46.00 | all_season |
| P0241 | Kids' Graphic Tee Pack | Kids' Apparel | 24.00 | 7.50 | spring/summer |

(Sample excerpts from generated worlds — the full files are not committed; the generation
command is in [`src/llm/README.md`](src/llm/README.md#example).)

Full reference: [`src/llm/README.md`](src/llm/README.md)

### Hyperparameter tuning (`src/tuning/`)

An Optuna-based hyperparameter search for any policy. Each trial runs the policy across a fixed
set of seeded episodes spanning two orders of magnitude in node capacity, optimising mean profit
per opening dollar. The top winners are re-checked on a separate held-out seed set with bootstrap
confidence intervals. All four textbook policies have ready-made search spaces; custom policies
need a ~10-line callback.

![Pareto front: profit vs service level](docs/images/tuning_pareto_front.png)

Full reference: [`src/tuning/README.md`](src/tuning/README.md)

### Reinforcement learning (`src/rl/`)

A PPO training stack with a **shared-weight per-product policy** (ADR 0021): one policy
manages any catalog size up to `K_max = 32`, with K sampled per episode. The agent learns
continuous pricing and ordering decisions; **implicit assortment** — ordering zero for a
product is stopping it — removes the need for a discrete carry/drop head. A deterministic
Arbiter reconciles the joint proposal against node capacity and the cash budget before
orders reach the engine. Per-episode node capacity is sampled across two orders of
magnitude so one trained policy covers a wide size range. Evaluation pairs it head-to-head
against the textbook baseline on bit-identical worlds (CRN) — any uplift is the policy, not
seed luck.

![RL vs OrderUpToPolicy KPIs](docs/images/rl_vs_baseline_kpis.png)

Full reference: [`src/rl/README.md`](src/rl/README.md)

## Repo layout

```
src/
  sim/         core simulator + textbook policies
  datasets/    M5 (Kaggle) → setup-directory adapter
  llm/         LLM catalog and market generator
  tuning/      Optuna-based hyperparameter search
  rl/          PPO training stack
setups/        runnable example setup directories
notebooks/     exploration + analysis notebooks
data/          scenario outputs
runs/          tuning + RL training artifacts
docs/images/   figures used in READMEs
tests/         pytest suite
```

## Further reading

- [CONTEXT.md](CONTEXT.md) — domain and architecture glossary
