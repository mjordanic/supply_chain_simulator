# PRD: Rewrite the notebook suite for the multi-echelon graph engine

Status: ready-for-agent

Related ADRs:
- [ADR 0010](../../docs/adr/0010-sim-as-base-for-ml-layers.md) — `src/sim/` is the canonical home for rollout primitives (the new run-log inspection module lands here)
- [ADR 0011](../../docs/adr/0011-multi-echelon-graph.md) — Multi-echelon DAG of typed nodes replaces the single-`Store` flat model
- [ADR 0012](../../docs/adr/0012-central-table-fcfs-allocation.md) — Central table + sequential FCFS allocation (the contention topology showcases this)
- [ADR 0014](../../docs/adr/0014-tick-phasing-cascade.md) — Tick phasing as an upward cascade by echelon level (the level metadata drives topology layout)
- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family (the four policies compared in 06)
- [ADR 0003](../../docs/adr/0003-crn-demand-for-all-products.md), [ADR 0007](../../docs/adr/0007-rl-scale-invariance-package.md), [ADR 0009](../../docs/adr/0009-policy-hyperparameter-tuning-tool.md) — RL eval + tuning add-ons surfaced by notebooks 07–09

## Problem Statement

The multi-echelon refactor (ADRs 0011–0016) replaced the single-`Store` flat model with a validated DAG of typed nodes (`FactoryNode` → `IntermediateNode` → `DemandSinkNode`), a live `CentralTable`, an FCFS allocator, and an upward tick cascade by echelon level. The run-log shape changed accordingly: a completed run now returns `{n_steps, ticks, global}` where each tick carries `node_cash`, `node_inventory` (`{_total, <pid>: qty}`), `node_pending`, and `node_orders` — there is no longer a top-level `stores` key.

The `notebooks/` folder did not fully follow. Of the eleven notebooks:

1. **Four are stale and crash on graph-engine data.** `00-check_simulated_data`, `04-deep_dive_per_product`, `04a-deep_dive_active_only`, and the RL pair `05/06/07` were last touched on the pre-refactor model. `00` reads `run_log['stores'][sid]['balance']`, a key that no longer exists. `06-compare_rl_vs_baseline` calls `observation_dim(config)` (the signature is now `observation_dim(K_active: int)`) and points at the `fashion_run` checkpoint, whose 74-dim input layer predates the central-table observation block (the current Actor expects 94 dims), so the checkpoint cannot even be loaded.

2. **Filenames no longer describe contents.** A previous slice rewrote four notebooks in place without renaming them, so `01-openai_world_builder.ipynb` now actually contains a "3-node chain quickstart", and the numbering tells the reader nothing about the reading order.

3. **The headline capability of the refactor — multiple topologies — is shown nowhere.** Every example scenario tops out at three echelon levels (factory → intermediate → sink). There is no notebook that builds, draws, runs, and compares different graph shapes, even though that is the single most important thing the multi-echelon engine added.

4. **The `DataExporter` only writes the `IntermediateNode` tier** to `timeseries.parquet`; factory and sink cash/inventory live only in the in-memory run-log. Notebooks that read the parquet (the old `00`) therefore cannot inspect the full chain even after being patched.

The user wants the notebook suite to once again do what notebooks are for in this repo: let a reader **inspect** generated worlds/scenarios/runs interactively, **experiment** with different inputs, and **showcase** what the simulator and its policies can do — now including different topologies — with every cell running and every figure rendering.

## Solution

Refactor `notebooks/` into a coherent, fully-renumbered ten-notebook suite, all on the graph engine, all verified to execute end-to-end with embedded plots. Introduce one small deep module (`src/sim/inspect.py`) that converts an in-memory run-log into tidy per-tier DataFrames, so the inspection/showcase notebooks stay thin and the parsing is unit-tested once rather than duplicated.

All notebooks **run simulations live** (`Runner(scenario).run()`, sub-2-second runs) and read the in-memory `run_log` rather than pre-exported parquet, so they always reflect current code and can see **every** echelon tier (factory, warehouse/DC, shop, sink). The `DataExporter` is **not** changed.

The final suite (old → new):

| New notebook | Source | Purpose |
| --- | --- | --- |
| `00-build-or-load-world` | old 01 | LLM world generation; the OpenAI client is constructed **lazily inside the `build_fn`** so a warm `fashion_retail_20` cache runs with no API key. |
| `01-inspect-world` | old 02 | Walk every `World` inspection surface (catalog, market, store templates) and derive a graph scenario via `world_to_graph`. |
| `02-inspect-scenario` | old 03 | Inspect a graph-mode `Scenario` (`nodes_df()`, `edges_df()`, catalog/market/lifecycle/disruption frames) before running. |
| `03-run-and-inspect-simulation` | old 00 | Live run of `llm_world_20`; reconstruct per-tier cash, inventory value, equity composition, and P&L from `run_log['ticks']` across **all** tiers. |
| `04-deep-dive-per-product` | merge 04 + 04a | Per-product card grid (lifecycle stage, freshness, season, demand/sales/inventory/orders) with an `ACTIVE_ONLY` flag switching between every product and active-only. |
| `05-topology-gallery` | **NEW** | Build, draw, run, and compare five topologies (below). The headline deliverable. |
| `06-policy-comparison` | from `example_paired_comparison` | The four textbook policies (`OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, `PeriodicReorderPolicy`) CRN-paired on the single-shop `fashion_retail_20` world; profit-per-dollar and service-level comparison. |
| `07-tune-textbook-policy` | old 08 | Optuna tuning tutorial + analysis; reads `runs/tuning/order_up_to_v2_retail`. |
| `08-monitor-rl-training` | old 05 | Reads TensorBoard event files from the graph-engine RL run (`runs/rl_ppo`). |
| `09-rl-vs-baseline` | merge 06 + 07 | CRN-paired RL-vs-`OrderUpToPolicy` evaluation: aggregate KPI/per-seed section **plus** a per-product single-seed overlay. Uses the compatible `rl_ppo` 5000-step checkpoint, framed honestly as smoke-trained. |

The `05-topology-gallery` features five graph shapes, each drawn with `networkx.multipartite_layout` keyed on echelon level (available as `Simulation.levels` / `Node.level` after `build_world`), run live, and compared on KPIs:

1. **Linear chain (3-tier)** — `1F → 1S → 1 sink`. The reference shape; explains echelon levels and the tick cascade.
2. **Multi-supplier contention** — `2F → 2S → 2 sinks` with shared suppliers; demonstrates the central table + FCFS allocator under contention (`fill_rate_recent` drops, cheapest-first routing).
3. **Deep 4-tier (new)** — `factory → warehouse/DC → shops → sinks` (four echelon levels). The genuine multi-echelon demonstration that no current scenario shows.
4. **Fan-out fleet** — few factories → several shops → many per-product sinks (the `homogeneous` / `llm_world_20` shape).
5. **Two disconnected sub-graphs** — two independent factory→shop→sink chains sharing no nodes in one `Scenario`; confirmed accepted by `build_graph` (validation rejects cycles/same-level links/nodes unreachable-from-any-factory, **not** disconnected components).

Topologies 1, 2, and the fleet reuse the existing `scenarios/*.py`; the 4-tier and disconnected graphs are built by **inline `build_*()` helper functions** inside the gallery notebook (no new `scenarios/` files, no `DataExporter` involvement).

Conventions for the whole suite: `%matplotlib inline` everywhere (renders on GitHub's static viewer and executes cleanly headless — so what is verified equals what ships); modest figure DPI; **executed outputs committed** so plots render on GitHub without running.

Doc sync: `CONTEXT.md` and `README.md` reference notebooks by their old names/numbers (e.g. `03-inspect_scenario`, `04a-deep_dive_active_only`, the `01/02/05/06/08` walkthrough links). These references are updated to the new lineup as part of this work.

## User Stories

1. As a new reader of the repo, I want the notebooks numbered in a sensible reading order with filenames that match their contents, so that I can follow world-building → inspection → simulation → topologies → policies → add-ons without guessing.
2. As a reader, I want every notebook to run top-to-bottom with no cell errors on the current code, so that I can trust the suite reflects the multi-echelon engine.
3. As a reader browsing on GitHub, I want each notebook to show its embedded plots without running anything, so that I can evaluate the simulator's capabilities at a glance.
4. As an experimenter, I want notebooks to build and run scenarios live and read the in-memory run-log, so that editing a parameter and re-running a cell immediately reflects in every figure.
5. As an analyst, I want to inspect cash, inventory value, and P&L for **every** echelon tier (factory, warehouse, shop, sink), not just the shop tier, so that I can reason about where money and stock sit in the whole chain.
6. As someone learning the engine, I want a notebook that builds an LLM world (or loads it from cache) without needing an API key when the world is already cached, so that I can run it offline.
7. As someone learning the engine, I want to walk every `World` inspection surface and then derive a graph scenario from it, so that I understand how a generated world becomes a runnable graph.
8. As someone learning the engine, I want to inspect a graph `Scenario` via `nodes_df()` and `edges_df()` before running it, so that I can confirm the topology and node configuration are what I intended.
9. As an inventory researcher, I want a per-product deep-dive that shows lifecycle stage, freshness multiplier, seasonal multiplier, demand, sales, inventory, and orders per SKU, so that I can audit individual product behaviour.
10. As an inventory researcher, I want the deep-dive to toggle between all products and active-only with one flag, so that I can switch between a full audit and a focused view without a second notebook.
11. As a supply-chain modeller, I want a topology gallery that draws each graph as a level-layered diagram, so that I can see the echelon structure at a glance.
12. As a supply-chain modeller, I want to see a linear 3-tier chain, so that I have a minimal reference for echelon levels and the tick cascade.
13. As a supply-chain modeller, I want to see a multi-supplier contention topology where two shops share two factories, so that I can observe the central table, FCFS allocation, and `fill_rate_recent` drops under contention.
14. As a supply-chain modeller, I want to see a genuine 4-tier `factory → warehouse → shops → sinks` graph, so that the multi-echelon capability is demonstrated beyond three levels.
15. As a supply-chain modeller, I want to see a fan-out fleet (few factories, several shops, many per-product sinks), so that I understand scale and one shop serving a full catalog.
16. As a supply-chain modeller, I want to run two fully disconnected sub-graphs inside one simulation, so that I can confirm independent chains coexist in a single world and compare their trajectories.
17. As a supply-chain modeller, I want each topology run live and compared on KPIs (e.g. system cash growth, service level, profit per dollar), so that I can see how graph shape affects outcomes.
18. As a policy researcher, I want the four textbook policies compared head-to-head on one shared world using Common Random Numbers, so that any difference is the policy and not seed luck.
19. As a policy researcher, I want the textbook comparison to run on the rich 20-SKU fashion world (seasonality, lifecycle, freshness), so that the censored-sales rate estimator is meaningfully exercised.
20. As a policy researcher, I want the comparison reported as profit-per-opening-dollar and a service-level proxy, so that I can rank the variants on the same metrics the tuner optimises.
21. As an RL researcher, I want a notebook that reads the graph-engine training run's TensorBoard scalars and plots the learning, loss, and CRN-eval curves inline, so that I can monitor training offline.
22. As an RL researcher, I want a notebook that loads a graph-engine-compatible checkpoint and runs the CRN-paired evaluation harness against `OrderUpToPolicy`, so that I can measure paired uplift.
23. As an RL researcher, I want the RL-vs-baseline notebook to show both aggregate KPIs/per-seed uplift and a per-product single-seed overlay, so that I can inspect the comparison at two resolutions in one place.
24. As an RL researcher, I want the notebook to state plainly that the shipped checkpoint is smoke-trained (so uplift is ~0 or negative) rather than implying a tuned win, so that I am not misled about RL performance.
25. As a maintainer, I want a single tested helper that turns a run-log into tidy per-tier DataFrames, so that the three inspection notebooks share one correct parser instead of three drifting copies.
26. As a maintainer, I want `CONTEXT.md` and `README.md` notebook references updated to the new names, so that the docs don't point at deleted files.
27. As a maintainer, I want the LLM-world notebook to fail with a clear, actionable message when the cache is cold and no API key is present, rather than an opaque construction error.
28. As a contributor, I want the new topologies defined as readable inline builder functions in the gallery notebook, so that I can see exactly how to compose a graph programmatically.

## Implementation Decisions

- **New deep module `src/sim/inspect.py`** (aligns with ADR 0010). Pure functions over the in-memory run-log, no plotting, no I/O:
  - `node_timeseries_df(run_log, scenario) -> DataFrame` — long form `(tick, node_id, node_type, region, level, cash, inventory_total, pending_total, orders_total)`, reconstructed from `run_log['ticks'][*]['node_cash' | 'node_inventory' | 'node_pending' | 'node_orders']` joined to node metadata from `scenario.nodes`.
  - `global_timeseries_df(run_log) -> DataFrame` — `(tick, market_supply, market_demand, …)` from `run_log['global']`.
  - `per_product_df(run_log, scenario, node_id) -> DataFrame` — per-(tick, pid) inventory/orders for one node, expanding the per-pid keys in `node_inventory` (excluding the `_total` synthetic key).
  - An equity helper (cash + inventory-at-cost + outstanding-at-cost) usable per node, valuing inventory against catalog `unit_cost`.
  These are the deep, testable surface; everything else in the notebooks is presentation.
- **Notebooks run live**, never read pre-exported parquet for time-series. `DataExporter` and its parquet schema are unchanged. (The `timeseries.parquet` shop-only limitation is the reason; fixing the exporter is out of scope.)
- **Full renumber + rename** to the ten-notebook lineup in the table above. Old files are deleted/merged; new files carry descriptive names.
- **Backend:** `%matplotlib inline` in every notebook (replacing the old `%matplotlib widget`), matching the convention the already-rewritten notebooks adopted.
- **Committed outputs:** every notebook is executed end-to-end and committed with embedded outputs; figure DPI/size kept modest to bound repo growth.
- **`00-build-or-load-world`** constructs `OpenAIClient` (and `WorldBuilder`) **lazily inside the `build_fn` passed to `load_or_build_world`**, so a warm cache (`data/worlds/fashion_retail_20/`) never constructs a client and needs no key; a cold cache without a key raises a clear, actionable message.
- **Topology gallery** builds the new 4-tier and disconnected graphs with inline `build_*()` helpers; reuses `scenarios/example_two_factories_two_shops.py` (contention), the 3-node chain, and the fleet shape. Drawing uses `networkx.multipartite_layout` keyed on echelon level (`Simulation.levels` / `Node.level`, populated by `build_world`). Disconnected graphs are valid: `build_graph` rejects cycles, same-level supplier links, and nodes unreachable from any factory — not disconnected components.
- **`06-policy-comparison`** evaluates the four textbook variants on the cached single-shop `fashion_retail_20` world, CRN-paired (shared `world_seed`/`init_seed`/allocation seed), reporting mean `net_profit / initial_cash` and a service-level proxy.
- **RL notebooks point at the graph-engine run `runs/rl_ppo`** (checkpoint `actor_step0000005000.pt`, 94-dim input, loads into the current `Actor`). `runs/fashion_run` (74-dim, pre-refactor) is **not** used for evaluation. The eval harness entry point is `src.rl.eval.evaluate`; observation/action sizing uses `observation_dim(K_active)` / `action_dim(K_active)` (int argument). RL notebooks expose the experiment name and checkpoint path as top-of-notebook variables.
- **Verification** is part of the deliverable: execute every notebook headless via `jupyter nbconvert --to notebook --execute` (loading `.env` so the lazy-client path is covered by a warm cache), and require zero cell errors and a rendered figure for every plotting cell.
- **Doc sync:** update notebook references in `CONTEXT.md` and `README.md` to the new names. No ADR is created — this is a docs/UX change, not a hard-to-reverse architectural trade-off.

## Testing Decisions

- **What makes a good test here:** assert on the external behaviour of `src/sim/inspect.py` — given a small, hand-constructed (or tiny-run) run-log + scenario, the returned DataFrames have the expected columns, one row per `(tick, node)` (or `(tick, node, pid)`), correct `node_type`/`level` joins, correct `_total`-key handling (the synthetic `_total` is excluded from per-product expansion and used for the tier total), and equity = cash + inventory-at-cost + outstanding-at-cost. Do **not** assert on private parsing internals or on exact float trajectories of a full run.
- **Modules tested:** `src/sim/inspect.py` only. The notebooks themselves are "tested" by the headless execution gate (every cell runs, every figure renders), not by pytest.
- **Prior art:** follow the existing `tests/sim/` style (e.g. the determinism/chain tests under `tests/sim/`), using a minimal scenario (the 3-node chain) run for a handful of ticks as the fixture, or a hand-written run-log dict matching the documented `{n_steps, ticks, global}` shape.
- The full notebook suite executing without error is the acceptance gate for the notebook deliverables; the `src/sim/inspect.py` unit tests are the acceptance gate for the shared parser.

## Out of Scope

- Changing `DataExporter` or the parquet schema (e.g. exporting factory/sink tiers). Notebooks read the in-memory run-log instead.
- Training a real (non-smoke) RL policy on the graph engine. The RL notebooks use the existing compatible 5000-step checkpoint and are framed accordingly; a longer training run is deferred.
- New reusable `scenarios/*.py` files for the 4-tier and disconnected topologies. These stay inline in the gallery notebook.
- `jupytext` or any new authoring dependency; notebooks are assembled with the already-present `nbformat`.
- Any change to engine behaviour, policies, the RL stack, or the tuner. This PRD only consumes those surfaces.
- Interactive (`ipympl`/`widget`) plotting. Standardised on static inline output.

## Further Notes

- Default world for `03`, `04`, and `06` is the cached `fashion_retail_20` / `llm_world_20` scenario (20 SKUs, single shop, 500 ticks) — small enough for a full per-product card grid, rich enough (seasonality, lifecycle, freshness) to exercise policy machinery. The user explicitly approved using "llm fashion world 20".
- Run-log shape (authoritative for the parser): `run_log = {n_steps: int, ticks: [{tick, node_cash: {nid: float}, node_inventory: {nid: {_total: int, <pid>: int}}, node_pending: {nid: {...}}, node_orders: {nid: {...}}}], global: {time, market_supply, market_demand, products, events}}`.
- `08-monitor-rl-training` reads TensorBoard scalars (format is env-agnostic). The richest curves on disk are `runs/fashion_run` (1M-step, old env); the graph-engine run `runs/rl_ppo` is short. The notebook defaults to `rl_ppo` for consistency with `09`, with the experiment name as an editable top variable so a future longer run drops in without edits.
- networkx 3.6.1 and nbformat/nbconvert/nbclient/ipympl are already installed; no environment changes are needed.
- A previous slice (commit `8c99e04`) already rewrote the contents of the notebooks now becoming `00/01/02/07`; this PRD finalises their names/numbers and re-verifies them alongside the rewrites and the new gallery.
