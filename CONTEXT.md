# Domain & Architecture Glossary

## Domain concepts

**Store** (`Store`, `src/sim/store.py`)
A retail store agent. Tracks inventory per SKU, cash balance, pending shipments, active assortment, prices, promotions, and an `activation_tick` map (per-SKU last-activation timestamp that drives the **freshness curve**). Delegates every decision to an attached Policy. Construction consumes only the per-instance `init_rng` (seeded from `StoreInstance.init_seed`), so two stores built from the same `(template, init_seed)` start step 0 bit-identical regardless of which Policy is later attached.

**Policy** (`Policy` ABC, `src/sim/policy.py`)
Decision logic attached to a Store. Each tick it receives an Observation and returns an Action dict with four keys: `order`, `price`, `activate`, `deactivate`. Each Policy instance owns a private `policy_rng` seeded from its `policy_seed` kwarg, kept disjoint from `world_rng` so policy choice never perturbs world stochasticity. Two implementation families: (a) `HeuristicPolicy` — the 20+ kwarg rule-based showcase that drives every knob the policy framework exposes (dynamic pricing, promotions, slow-mover deactivation, periodic catalog review); retained only for `scenarios/example_*` parameter-demo purposes. (b) The **TextbookReorderPolicy** family — `OrderUpToPolicy` (s,S), `ReorderPointPolicy` (s,Q), `PeriodicOrderUpToPolicy` (R,S), `PeriodicReorderPolicy` (R,s,S) — pure textbook inventory rules with flat pricing and no catalog rotation. `OrderUpToPolicy` is the canonical CRN comparison anchor for RL evaluation. See ADR 0006.

**TextbookReorderPolicy family** (`TextbookReorderPolicy` abstract base + 4 concrete classes, `src/sim/policy.py`)
Family of four reorder policies sharing one base class. The shared machinery: censored-sales rate estimator over a `delivery_lag`-length rolling window, two-pass fair-share allocator across the capacity and cash pools, cash-budget "pilot order" cold-start (default `opening_budget_pct=0.50` of opening cash, split evenly across the K active SKUs, fired once per never-observed SKU), and an opt-in adaptive safety stock that bumps `safety_lead_ticks` while the recent window contains stockout ticks (default off via `stockout_safety_bonus_ticks=0`). Levels are expressed in demand-units: `s = (delivery_lag + safety_lead_ticks) × rate`, `S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate`. Scale-invariance in capacity is a structural property — the same defaults work across store sizes from corner shop to flagship, mirroring the demand-relative action decoder of the RL framework (ADR 0007, superseding ADR 0005). All four variants emit `activate=[]` / `deactivate=[]` / `promotions={}` and `price[pid] = base_price[pid]` (no cost floor); the policies are pure inventory rules with exogenous pricing. See ADR 0006.

**Market**
Regional economic environment shared across Stores. Holds per-region demand/supply state, seasonal cycle, and trend drift. Drives per-SKU demand via `sample_demand()`. One Market per simulation run.

**Item** / **Ware** / **SKU**
A product in the catalog. `Ware` is the static record loaded from config (product_id `P####`, name, category, base_price, unit_cost, seasonality, related_products) plus per-product lifecycle/freshness authoring fields: `init_stage`, `stage_change_probs` (dict keyed by current stage), `freshness_alpha`, `freshness_decay`, and `init_stock_share` weight. Each is optional; when omitted, the corresponding default on `ItemLifecycleParams` applies. `Item` is the live runtime object held by `ItemRegistry` — owns the **global lifecycle stage** and its transition logic only; the **per-(store, product) freshness curve** is owned by `Store`.

**Lifecycle stage**
One of `[introduction, growth, maturity, decline, dead]` — the global PLC position of a product across the market, not per-store. Each stage maps to a `stage_multipliers` factor on baseline demand; transitions are stochastic, governed by the per-stage `stage_change_probs` table on `ItemLifecycleParams` (overridable per `Ware`). `dead` carries `stage_multipliers["dead"] = 0.05` (trickle demand); `dead → introduction` defaults to ~0.003 per tick (~1-year mean comeback) — products *can* re-launch, rarely. Set to 0 to make `dead` strictly terminal. See ADR 0002.

**Freshness curve**
Per-(store, product) demand multiplier `m(τ) = 1 + α · exp(−τ / β)` where `τ` is ticks since the product was last activated in this store. Composes multiplicatively with the global stage multiplier and seasonal/promo factors in `Market.sample_demand`. Resets to `τ = 0` on every `Store.activate_item` event; when `freshness_alpha = 0` (staples / commodities), the multiplier is identically `1`. Models newness hype and novelty wear-out — independent of, and orthogonal to, the global PLC. See ADR 0001.

**ItemRegistry**
Catalog of all Items. Owns the **global lifecycle stage** for every catalog SKU (one stage per product across all stores) and its stochastic transitions (`tick()`, advanced from `world_rng`). Provides `stage(sku)` and `related(sku)` lookups used by the Observation builder and by `Market.sample_demand`. The per-(store, product) freshness curve is *not* owned here — it lives on `Store`.

**EventEngine**
Generates stochastic disruption events (natural_disaster, economic_crisis, political_unrest, pandemic, technological_breakthrough) that shift regional demand/supply. Also dispatches scheduled callbacks for order arrivals.

**Order**
A purchase of `quantity` units of a SKU placed by a Store. Submitted via EventEngine; arrives after an adjusted lead time derived from current market supply. No explicit Order object — lifecycle is a scheduled lambda callback.

**Observation**
A dict of visible state handed to a Policy each tick. Contains inventory, prices, sales history, pending orders, promotions, market events, and product lifecycle info. Built by `Runner._observe()`.

**Action**
A dict returned by a Policy each tick. Keys: `order` (qty per SKU), `price` (price per SKU), `activate` (SKU list), `deactivate` (SKU list).

**Active subset**
The K (default 5) product ids drawn uniformly without replacement per episode from the catalog universe; the only SKUs the RL agent makes pricing and ordering decisions for during that episode. The assortment is frozen for the episode's duration — activate/deactivate decisions are disabled. Distinct from the store's full inventory, which includes all catalog products (demand is sampled for every catalog product each tick to preserve CRN cleanliness; see ADR 0003). Varies independently across episodes via the episode RNG.

**Run Log**
Nested dict accumulating all state across the simulation run. Top-level keys: `global` (market/event/lifecycle time-series), `stores` (per-store per-product metrics), `actions`, `observations`. Produced by `Runner.run()` (`src/sim/runner.py`) and consumed by `DataExporter`.

**CRN-paired eval**
Evaluation protocol where the RL policy and `OrderUpToPolicy` (the (s,S) textbook reorder policy) are run on bit-identical `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` tuples — Common Random Numbers. Uplift is computed paired per seed (RL return minus baseline return on the same world trajectory) and averaged across a fixed held-out set of 32 seeds disjoint from the training seed space. Paired comparison is more powerful than comparing independent runs because world variance cancels; any observed difference is attributable to the policy alone. See ADR 0003 for the CRN demand-sampling guarantee that makes this property hold within a single simulation run; see ADR 0006 for why (s,S) is the canonical comparison anchor over the previous heuristic baseline.

**Run model — "policies attached to stores"**
Each Store has its own Policy. A run binds an explicit list of `(StoreTemplate, init_seed, Policy)` triples and executes them in one shared world. There is one authoring helper, `make_stores(triples)`; no `homogeneous` / `paired` regime split. Use patterns the triple list expresses:
- **Single policy across diverse stores** (robustness): repeat one Policy across varying `(template, init_seed)`. Variance-reduced estimate of that policy across heterogeneous conditions.
- **Multiple policies on identical stores** (CRN comparison): repeat the same `(template, init_seed)` with different Policies. Same world shocks, bit-identical step-0 store — paired evaluation. Generalises to k-way comparison by listing k triples per shared seed.
- **Mixed roster** (e.g. policies `A,A,B,C,D` across templates `T1,T2,T3,T4,T4,T4`): just write the triple list. Authoring is purely declarative; the data model imposes no "regime" abstraction above the list.

**Store Template**
A reusable, deterministic store specification: profile (capacity, balance, lead time, holding rate), region, `init_active_count` and optional explicit `init_active_products` list, and an `init_freshness` mode (`"baseline"` for established stores — initial active SKUs skip the hype window; `"fresh"` for grand-opening scenarios — initial active SKUs start at `τ = 0` with full hype). The `init_seed` is *not* on the template; it lives on `StoreInstance`, so one template can spawn multiple replicates with distinct seeds (noise sweep) or shared seeds (CRN duplicates). Two stores instantiated from the same `(template, init_seed)` are bit-identical at step 0 regardless of which Policy is attached.

**RL Env**
Gymnasium-compatible environment wrapping the simulator's `Market` / `EventEngine` / `ItemRegistry` / `Store` subsystems in step-by-step semantics. Each `reset()` produces a fresh `Scenario` via the episode sampler, building a single store with a freshly sampled active subset, capacity, opening balance, and slot permutation. Each `step(action)` advances one tick using the same sequence as `Runner.run()`, with the action injected through an `RLPolicy` shim. The action decoder expresses order quantity as an order-up-to target in lead-times of expected demand: `target_lt × effective_rate − inventory_position`, where `effective_rate = max(rolling_5_mean_sales, base_demand_prior)` and `target_lt` is selected by the action (centred at 15 lead-times when `action = 0`). The encoder carries inventory in two frames: capacity-units (slot 0, useful for pricing) and demand-units (slot 13, useful for ordering, saturating at 30 lead-times of cover). Per-episode capacity and opening balance are sampled from log-uniform distributions spanning two orders of magnitude (`LogUniform(100, 10_000)` capacity, `LogUniform(10_000, 1_000_000)` balance), so a single trained policy covers corner-shop to flagship store sizes. See ADR 0007 (supersedes ADR 0005) and ADR 0004.

**RL Episode**
One `reset()`-to-terminated pass through the RL Env. Fixed at 180 ticks (half-year at daily resolution). Distinct from "Scenario" in the batch-runner sense — an RL episode is disposable and re-sampled from scratch on every `reset()`, whereas a Scenario is an authored experiment artifact. Episode return equals the sum of per-tick balance deltas over all 180 ticks.

**Scenario**
The reproducible inputs to a run. Includes the catalog, market init/cycle parameters, disruption parameters, the list of (Store Template, Policy) pairings, and a `world_seed` that determines all market and event stochasticity. The same Scenario replayed twice yields the same world trajectory; only the policies' divergent actions create different per-store outcomes.

**Slot-shuffled observation**
Observation tensor where the K active SKUs occupy K slots, with the slot-to-SKU mapping permuted per episode. The permutation is drawn from the episode RNG at reset time and is fixed for the episode. The encoder places the SKU at `active_subset[slot_perm[i]]` into slot `i`; the decoder applies the inverse permutation to map action slot `i` back to the corresponding product id. This prevents the agent from associating slot position with product identity, forcing it to learn from features rather than position. See ADR 0004.

---

## Architecture concepts

**Scenario** (`src/sim/scenario.py`)
Flat typed dataclass holding a complete experiment: `catalog` (`list[Ware]`), `market` (`MarketParams`), `disruption` (`DisruptionParams`), `item_lifecycle` (`ItemLifecycleParams`), `stores` (`list[StoreInstance]`), `n_steps`, `start_date`, `world_seed`. JSON-serialisable via `to_json()` / `from_json()` — except `StoreInstance.policy`, which is intentionally left out: scenarios are world artifacts, while policies are wired up at experiment-authoring time. Authoring helpers in the same module: `load_catalog(items)` and `make_stores(triples)` where `triples: list[tuple[StoreTemplate, int, Policy]]`. There is no other store-list helper — duplication of `(template, init_seed)` across triples is how CRN comparison is expressed; varying `init_seed` is how robustness sweeps are expressed.

The canonical World→Scenario derivation is `Scenario.from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed)` — it copies `catalog` and `market` from the `World` artifact and fills the remaining fields from the caller. The six DataFrame inspection methods are the canonical way to examine a scenario before running it: `catalog_df()`, `stores_df()`, `market_df()`, `disruption_df()`, `lifecycle_df()`, `summary_df()`. `stores_df()` exposes `policy_class` populated from live `Policy` instances; when the scenario is reconstructed from JSON via `Scenario.from_json()` (historical-run audit path), `policy_class` is `None` by design — policies are not serialised. Use `load_scenario_from_path(path)` to obtain a live `Scenario` with policies attached from a scenario script. See `notebooks/03-inspect_scenario.ipynb` for a worked walkthrough of all six views.

**Distribution** (`src/sim/distributions.py`)
Abstract base class with one method, `sample(rng) -> Any`, and five concrete implementations: `Constant(value)`, `Uniform(lo, hi)`, `Normal(mean, std)`, `Choice(options)`, `LogUniform(lo, hi)`. `LogUniform` draws from a log-uniform distribution (uniform in log-space; requires `lo > 0` and `hi > lo`) and is used for domain-randomisation parameters that span orders of magnitude, such as per-episode capacity (`LogUniform(100, 10_000)`) and opening balance (`LogUniform(10_000, 1_000_000)`). Replaces the lambda-as-config idiom from the deleted system; instances are typed, comparable, and JSON-serialisable. Stochastic fields on `MarketParams`, `DisruptionParams`, `StoreTemplate`, etc. accept either a scalar or a `Distribution` and are sampled against an injected RNG (`world_rng` for market/events, `init_rng` for store construction).

**Runner** (`src/sim/runner.py`)
Owns the simulation loop. `Runner(scenario).run() → dict` builds all subsystems from a `Scenario` and runs the observe → decide → advance → log loop. Maintains the three-stream RNG split (`world_rng` from `Scenario.world_seed`; one `policy_rng` per `Policy` instance from its `policy_seed`; one `init_rng` per `StoreInstance` from its `init_seed`) that makes Common Random Numbers comparison correct.

**DataExporter** (`src/sim/data_exporter.py`)
Consumes a `Scenario` and a completed run log and writes all outputs: parquet time-series, parquet static product/store tables, JSON config snapshot and run log, and a regional supply/demand PNG. Public methods: `export_all(output_folder)` plus per-artifact `save_scenario_json`, `save_run_log_json`, `save_products_parquet`, `save_stores_parquet`, `save_timeseries_parquet`, `save_overview_plot`. Each method creates its own subdirectory under `output_folder` (`config/`, `data/`, `reports/`).

**WorldBuilder** (`src/llm/world_builder.py`)
LLM-driven world generator producing a `World(catalog, market, store_templates)` artifact for an archetype string (e.g. `"fashion_retail"`). Four stages: taxonomy (LLM) → catalog (deterministic Python skeleton allocator + LLM naming, including per-`Ware` lifecycle and freshness parameters per category) → store templates (LLM, per region — including `init_active_products` rosters and `init_freshness` mode) → market domain params (LLM authors the domain slice; math defaults are hand-set). Each LLM call is parsed through a Pydantic schema in `src/llm/schemas.py`; on `ValidationError`, the failure is fed back into the next prompt for self-correction (default 3 retries). Consumed by scenario authors who want a generated catalog/market instead of hand-coding one. Policies, disruption parameters, and seeds remain author-supplied.

Worlds are persisted to `data/worlds/<name>/world.json`. The JSON payload includes a `_meta` block with fields `archetype`, `n_items`, `model`, `builder_version`, and `built_at`. The canonical script entry point is `load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False)` — it returns a cached `World` on a hit and, on a miss, prints a warning and prompts for interactive consent before invoking `build_fn` (raises `LLMBuildAbortedError` if the user declines or the context is non-interactive). Pass `auto_confirm=True` for CI or offline scripts. See `notebooks/02-inspect_world.ipynb` for a worked walkthrough of `World` DataFrame views (`catalog_df`, `store_templates_df`, `market_df`, `meta_df`).

---

## Decisions

- [ADR 0001](docs/adr/0001-two-layer-lifecycle.md) — Lifecycle is two-layer: global PLC × per-store freshness curve.
- [ADR 0002](docs/adr/0002-per-stage-transitions-and-dead-stage.md) — Per-stage transition probabilities and a `dead` stage with terminal-by-default cycle.
- [ADR 0003](docs/adr/0003-crn-demand-for-all-products.md) — Demand is sampled for every catalog product each tick, even when inactive (CRN cleanliness).
- [ADR 0004](docs/adr/0004-rl-training-env.md) — RL training env: randomised assortment, slot-shuffled observations, hidden market state, frozen assortment within episode.
- [ADR 0005](docs/adr/0005-demand-relative-action-decoding.md) — Action decoder expresses order quantity in lead-times-of-demand, not fraction-of-shelf-space. *Superseded by ADR 0007.*
- [ADR 0006](docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family replaces the heuristic baseline as the RL comparison anchor.
- [ADR 0007](docs/adr/0007-rl-scale-invariance-package.md) — RL scale-invariance package: order-up-to action decoder, demand-units inventory feature, log-uniform domain randomisation.
