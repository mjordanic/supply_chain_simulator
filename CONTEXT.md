# Domain & Architecture Glossary

## Domain concepts

**Store** (`Store`, `src/sim/store.py`)
A retail store agent. Tracks inventory per SKU, cash balance, pending shipments, active assortment, prices, promotions, and an `activation_tick` map (per-SKU last-activation timestamp that drives the **freshness curve**). Delegates every decision to an attached Policy. Construction consumes only the per-instance `init_rng` (seeded from `StoreInstance.init_seed`), so two stores built from the same `(template, init_seed)` start step 0 bit-identical regardless of which Policy is later attached.

**Policy** (`Policy` ABC, `BaselinePolicy`, `src/sim/policy.py`)
Decision logic attached to a Store. Each tick it receives an Observation and returns an Action dict with four keys: `order`, `price`, `activate`, `deactivate`. Each Policy instance owns a private `policy_rng` seeded from its `policy_seed` kwarg, kept disjoint from `world_rng` so policy choice never perturbs world stochasticity. Currently one implementation: `BaselinePolicy` — threshold-driven reorder, dynamic pricing, and catalog rotation, configured entirely by kwargs hyperparameters.

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

**Run Log**
Nested dict accumulating all state across the simulation run. Top-level keys: `global` (market/event/lifecycle time-series), `stores` (per-store per-product metrics), `actions`, `observations`. Produced by `Runner.run()` (`src/sim/runner.py`) and consumed by `DataExporter`.

**Run model — "policies attached to stores"**
Each Store has its own Policy. A run binds an explicit list of `(StoreTemplate, init_seed, Policy)` triples and executes them in one shared world. There is one authoring helper, `make_stores(triples)`; no `homogeneous` / `paired` regime split. Use patterns the triple list expresses:
- **Single policy across diverse stores** (robustness): repeat one Policy across varying `(template, init_seed)`. Variance-reduced estimate of that policy across heterogeneous conditions.
- **Multiple policies on identical stores** (CRN comparison): repeat the same `(template, init_seed)` with different Policies. Same world shocks, bit-identical step-0 store — paired evaluation. Generalises to k-way comparison by listing k triples per shared seed.
- **Mixed roster** (e.g. policies `A,A,B,C,D` across templates `T1,T2,T3,T4,T4,T4`): just write the triple list. Authoring is purely declarative; the data model imposes no "regime" abstraction above the list.

**Store Template**
A reusable, deterministic store specification: profile (capacity, balance, lead time, holding rate), region, `init_active_count` and optional explicit `init_active_products` list, and an `init_freshness` mode (`"baseline"` for established stores — initial active SKUs skip the hype window; `"fresh"` for grand-opening scenarios — initial active SKUs start at `τ = 0` with full hype). The `init_seed` is *not* on the template; it lives on `StoreInstance`, so one template can spawn multiple replicates with distinct seeds (noise sweep) or shared seeds (CRN duplicates). Two stores instantiated from the same `(template, init_seed)` are bit-identical at step 0 regardless of which Policy is attached.

**Scenario**
The reproducible inputs to a run. Includes the catalog, market init/cycle parameters, disruption parameters, the list of (Store Template, Policy) pairings, and a `world_seed` that determines all market and event stochasticity. The same Scenario replayed twice yields the same world trajectory; only the policies' divergent actions create different per-store outcomes.

---

## Architecture concepts

**Scenario** (`src/sim/scenario.py`)
Flat typed dataclass holding a complete experiment: `catalog` (`list[Ware]`), `market` (`MarketParams`), `disruption` (`DisruptionParams`), `item_lifecycle` (`ItemLifecycleParams`), `stores` (`list[StoreInstance]`), `n_steps`, `start_date`, `world_seed`. JSON-serialisable via `to_json()` / `from_json()` — except `StoreInstance.policy`, which is intentionally left out: scenarios are world artifacts, while policies are wired up at experiment-authoring time. Authoring helpers in the same module: `load_catalog(items)` and `make_stores(triples)` where `triples: list[tuple[StoreTemplate, int, Policy]]`. There is no other store-list helper — duplication of `(template, init_seed)` across triples is how CRN comparison is expressed; varying `init_seed` is how robustness sweeps are expressed.

The canonical World→Scenario derivation is `Scenario.from_world(world, *, disruption, item_lifecycle, stores, n_steps, start_date, world_seed)` — it copies `catalog` and `market` from the `World` artifact and fills the remaining fields from the caller. The six DataFrame inspection methods are the canonical way to examine a scenario before running it: `catalog_df()`, `stores_df()`, `market_df()`, `disruption_df()`, `lifecycle_df()`, `summary_df()`. `stores_df()` exposes `policy_class` populated from live `Policy` instances; when the scenario is reconstructed from JSON via `Scenario.from_json()` (historical-run audit path), `policy_class` is `None` by design — policies are not serialised. Use `load_scenario_from_path(path)` to obtain a live `Scenario` with policies attached from a scenario script. See `notebooks/03-inspect_scenario.ipynb` for a worked walkthrough of all six views.

**Distribution** (`src/sim/distributions.py`)
Abstract base class with one method, `sample(rng) -> Any`, and four concrete implementations: `Constant(value)`, `Uniform(lo, hi)`, `Normal(mean, std)`, `Choice(options)`. Replaces the lambda-as-config idiom from the deleted system; instances are typed, comparable, and JSON-serialisable. Stochastic fields on `MarketParams`, `DisruptionParams`, `StoreTemplate`, etc. accept either a scalar or a `Distribution` and are sampled against an injected RNG (`world_rng` for market/events, `init_rng` for store construction).

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
