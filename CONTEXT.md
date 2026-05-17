# Domain & Architecture Glossary

## Domain concepts

**Store** (`Store`, `src/sim/store.py`)
A retail store agent. Tracks inventory per SKU, cash balance, pending shipments, active assortment, prices, promotions, and an `activation_tick` map (per-SKU last-activation timestamp that drives the **freshness curve**). Delegates every decision to an attached Policy. Construction consumes only the per-instance `init_rng` (seeded from `StoreInstance.init_seed`), so two stores built from the same `(template, init_seed)` start step 0 bit-identical regardless of which Policy is later attached.

**Policy** (`Policy` ABC, `src/sim/policy.py`)
Decision logic attached to a Store. Each tick it receives an Observation and returns an Action dict with four keys: `order`, `price`, `activate`, `deactivate`. Each Policy instance owns a private `policy_rng` seeded from its `policy_seed` kwarg, kept disjoint from `world_rng` so policy choice never perturbs world stochasticity. Two implementation families: (a) `HeuristicPolicy` — the 20+ kwarg rule-based showcase that drives every knob the policy framework exposes (dynamic pricing, promotions, slow-mover deactivation, periodic catalog review); retained only for `scenarios/example_*` parameter-demo purposes. (b) The **TextbookReorderPolicy** family — `OrderUpToPolicy` (s,S), `ReorderPointPolicy` (s,Q), `PeriodicOrderUpToPolicy` (R,S), `PeriodicReorderPolicy` (R,s,S) — pure textbook inventory rules with flat pricing and no catalog rotation. `OrderUpToPolicy` is the canonical CRN comparison anchor for RL evaluation. See ADR 0006.

**TextbookReorderPolicy family** (`TextbookReorderPolicy` abstract base + 4 concrete classes, `src/sim/policy.py`)
Family of four reorder policies sharing one base class. The shared machinery: censored-sales rate estimator over a `delivery_lag`-length rolling window, two-pass fair-share allocator across the capacity and cash pools, cash-budget "pilot order" cold-start (default `opening_budget_pct=0.50` of opening cash, split evenly across the K active SKUs, fired once per never-observed SKU), and an opt-in adaptive safety stock that bumps the effective safety horizon while the recent window contains stockout ticks (default off via `stockout_safety_bonus_pct_of_lag=0.0`). Levels are expressed in demand-units; the safety horizon is computed per pid as a fraction of that pid's delivery lag: `effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])`, then `s = (delivery_lag + effective_safety_ticks) × rate`, `S = s + cover_horizon_ticks × rate`. Default `safety_lead_pct_of_lag = 2/3` reproduces ADR 0006's behaviour at the canonical `lag=3` test scale. Scale-invariance in capacity is a structural property — the same defaults work across store sizes from corner shop to flagship — and per-SKU correctness under varying lead times is preserved by the ratio form (ADR 0008). All four variants emit `activate=[]` / `deactivate=[]` / `promotions={}` and `price[pid] = base_price[pid]` (no cost floor); the policies are pure inventory rules with exogenous pricing. See ADR 0006 and ADR 0008.

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
Evaluation protocol where the RL policy and `OrderUpToPolicy` (the (s,S) textbook reorder policy) are run on bit-identical `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` tuples — Common Random Numbers. Uplift is computed paired per seed (RL return minus baseline return on the same world trajectory) and averaged across a fixed held-out set of 32 seeds disjoint from the training seed space. Paired comparison is more powerful than comparing independent runs because world variance cancels; any observed difference is attributable to the policy alone. The single-policy half of this machinery — world construction via `build_world` (`src/sim/runner.py`), step-by-step rollout via `Simulation.tick()`, and projection of per-tick active-subset traces into a `RunSlice` (`src/sim/metrics.py`) then `aggregate_episode` (`src/sim/metrics.py`) — is also reused by the **Policy tuning study** with a normalised-return objective. See ADR 0003 for the CRN demand-sampling guarantee that makes this property hold within a single simulation run; see ADR 0006 for why (s,S) is the canonical comparison anchor over the previous heuristic baseline.

**Policy tuning study** (`src/tuning/`)
Optuna-based hyperparameter tuner for any `Policy` subclass. Scoped to "how much profit headroom exists above the published textbook defaults" (framing 1, ADR 0009) — not per-world tuning of the canonical RL baseline, which ADR 0006 explicitly rejected. A study consists of N trials (default 150), each running the policy on a fixed CRN seed set (default 16 search seeds, disjoint from training and from the two-scale eval ranges) at log-uniform domain randomisation across capacity and balance. The trial-level objective is mean `net_profit / initial_cash`, dimensionless and scale-comparable. Per-seed un-normalised KPIs (net profit, service level, stockout rate, turnover, revenue, mean-price-pct-of-MSRP, plus the seed's capacity and initial cash) are stashed in `trial.user_attrs` so the result notebook can reconstruct Pareto fronts, parameter-sensitivity scatters, and per-capacity-bucket robustness slices without re-running the study. Bundled trial-callback factories cover the four textbook variants; custom policies write their own factory. Artifacts land at `runs/tuning/<study_name>/` as `trials.parquet` (one row per trial) + `per_seed.parquet` (one row per (trial, seed)) + `study.json` (metadata + provenance). The `confirm_top_k()` entry point re-evaluates the search winners on a disjoint 32-seed held-out set and writes `holdout.parquet` + `holdout_summary.json`. Rollout and sampling consume sim primitives directly: episodes are sampled via `src.sim.episode_sampler.sample_episode` (through the `src.tuning.episode` Config-adapter), worlds are loaded via `src.sim.world_loader.load_world` (through the `src.tuning.world_loader` Config-adapter), rollout runs via `build_world` + `Simulation.tick()` (`src/sim/runner.py`), and KPIs are computed via `RunSlice` + `aggregate_episode` (`src/sim/metrics.py`). The tuning module has no dependency on `src.rl`. See ADR 0009.

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

The canonical attach pattern for spec-based callers that hold `policy=None` on `StoreInstance` is `build_world(scenario, policy_overrides=[my_policy, ...])` — see the **Simulation** entry below.

**Distribution** (`src/sim/distributions.py`)
Abstract base class with one method, `sample(rng) -> Any`, and five concrete implementations: `Constant(value)`, `Uniform(lo, hi)`, `Normal(mean, std)`, `Choice(options)`, `LogUniform(lo, hi)`. `LogUniform` draws from a log-uniform distribution (uniform in log-space; requires `lo > 0` and `hi > lo`) and is used for domain-randomisation parameters that span orders of magnitude, such as per-episode capacity (`LogUniform(100, 10_000)`) and opening balance (`LogUniform(10_000, 1_000_000)`). Replaces the lambda-as-config idiom from the deleted system; instances are typed, comparable, and JSON-serialisable. Stochastic fields on `MarketParams`, `DisruptionParams`, `StoreTemplate`, etc. accept either a scalar or a `Distribution` and are sampled against an injected RNG (`world_rng` for market/events, `init_rng` for store construction).

**Simulation** (`src/sim/runner.py`)
Mutable bundle returned by `build_world(scenario, *, policy_overrides=None)`. Holds `(scenario, world_rng, item_registry, market, event_engine, stores)` and exposes a two-phase tick API:

- `tick_world() → list[WorldEvent]` — phase 1: advances market, events, and item lifecycle; returns active events.
- `tick_decide_and_settle(active_events) → TickResult` — phase 2: per-store observe → decide → dispatch orders → sample demand and settle accounting. RL action-injection happens in the seam: encode obs from post-`tick_world` state, run actor, decode, set pending action on `RLPolicy`, then call this method.
- `tick() → TickResult` — convenience composing both phases; equivalent to `tick_decide_and_settle(tick_world())`.

`TickResult` is a frozen dataclass `(actions: dict[int, dict], demand_traces: dict[int, dict], active_events: list[WorldEvent])`.

`build_world(scenario, *, policy_overrides=None)` is the canonical attach pattern for spec-based callers (tuning, RL eval, RL env) whose specs carry `policy=None`. When `policy_overrides` is supplied, store `i` is built with `policy_overrides[i]`; override wins when `StoreInstance.policy` is also set. Scenario-authoring callers leave `policy_overrides=None`.

**Runner** (`src/sim/runner.py`)
Owns the simulation loop. `Runner(scenario).run() → dict` builds all subsystems (now via `build_world`) and runs the observe → decide → advance → log loop. Maintains the three-stream RNG split (`world_rng` from `Scenario.world_seed`; one `policy_rng` per `Policy` instance from its `policy_seed`; one `init_rng` per `StoreInstance` from its `init_seed`) that makes Common Random Numbers comparison correct. The public API is unchanged; the body now composes `build_world` + `Simulation.tick()` per step plus the existing log-collection helpers.

**DataExporter** (`src/sim/data_exporter.py`)
Consumes a `Scenario` and a completed run log and writes all outputs: parquet time-series, parquet static product/store tables, JSON config snapshot and run log, and a regional supply/demand PNG. Public methods: `export_all(output_folder)` plus per-artifact `save_scenario_json`, `save_run_log_json`, `save_products_parquet`, `save_stores_parquet`, `save_timeseries_parquet`, `save_overview_plot`. Each method creates its own subdirectory under `output_folder` (`config/`, `data/`, `reports/`).

**World** (`src/sim/world.py`)
The persisted simulation artifact produced by `WorldBuilder` and consumed by scenario authoring. A `World(catalog, market, store_templates, meta)` dataclass that carries the three structured fields needed to author a `Scenario`: `catalog` (`list[Ware]`), `market` (`MarketParams`), and `store_templates` (`dict[str, StoreTemplate]`). `meta` carries provenance: `archetype`, `n_items`, `model`, `builder_version`, and `built_at`.

Worlds are persisted to `data/worlds/<name>/world.json`. `World.to_json()` / `World.from_json()` handle serialisation; the on-disk format is stable. Four DataFrame inspection methods are available for notebook use: `catalog_df()`, `store_templates_df()`, `market_df()`, `meta_df()`. `from src.llm.world_builder import World` continues to work as a transitional re-export during migration. See `notebooks/02-inspect_world.ipynb` for a worked walkthrough.

**Episode sampler** (`src/sim/episode_sampler.py`)
Canonical home for the shared `(scenario, active_subset)` sampler, the 4-stream seed split (assortment / capacity / balance / world), and the three public default-params factories. Exposes:

- `EpisodeSpec(scenario, active_subset)` — frozen dataclass. No `slot_permutation` — that is an RL-observation-encoder concern (ADR 0004, "Slot-shuffled observation").
- `sample_episode(catalog, base_template, *, K_active, episode_length, capacity_dist, balance_dist, episode_seed, market_params=None, disruption_params=None, lifecycle_params=None, start_date=None) → EpisodeSpec` — Config-agnostic (kwargs, not a Config object); safe to call from multiple threads.
- `default_market_params()`, `default_disruption_params()`, `default_lifecycle_params()` — public factories consumed by `sample_episode`'s `is None` fallbacks and by `src.sim.world_loader`'s synthetic-fallback path.

Sub-seed derivation: `sub = (episode_seed * PRIME + OFFSET) & 0xFFFF_FFFF`, one `(PRIME, OFFSET)` pair per purpose. RL adds a 5th `slot` stream in `src.rl.episode_sampler`. Tuning's Config-adapter (`src.tuning.episode`) unpacks `TuningConfig` and forwards primitives here. The historical `TuningEpisodeSpec` name is a transitional alias for `EpisodeSpec`.

**World loader** (`src/sim/world_loader.py`)
Canonical home for the `cache_path → archetype → synthetic fallback` resolution logic that both `src/tuning/` and `src/rl/train.py` previously duplicated. Exposes a single public function:

`load_world(*, archetype, cache_path=None, delivery_lag, holding_rate, order_fee, K_active, synthetic_fallback=True, synthetic_catalog_size=100) → tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]`

Resolution order:
1. Explicit `cache_path` — loaded unconditionally when the file exists.
2. `data/worlds/<archetype>/world.json` — auto-lookup by archetype label.
3. Synthetic fallback — `synthetic_catalog_size`-item catalog, no LLM required (raises `FileNotFoundError` when `synthetic_fallback=False`).

For loaded worlds, `disruption_params.regions` is adjusted to match `world.market.regions` (the same fix both old callers applied). For the synthetic fallback, both `market_params` and `disruption_params` are `None` so `sample_episode` uses its own defaults.

Config-adapters: `src.tuning.world_loader.load_world(config)` and `src.rl.train._load_world_catalog_and_template(config)` are now thin shims (~10 lines each) that unpack their respective Config objects and call this function.

**Metrics** (`src/sim/metrics.py`)
Canonical home for `RunSlice` (frozen per-tick active-subset trace dataclass), `aggregate_episode(run_slice) -> dict[str, float]`, and the five KPI helpers (`service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, `profit_decomposition`). Pure data + math; no runtime deps on `src.tuning`, `src.rl`, or `src.llm`. Consumed by `src.tuning.rollout.run_policy_episode` and `src.rl.eval.evaluate`; both projects compute identical KPIs from identical state-machine outputs, so any apparent divergence reflects policy behaviour, not metric drift. See ADR 0010.

**WorldBuilder** (`src/llm/world_builder.py`)
LLM-driven world generator that produces a `World` artifact for an archetype string (e.g. `"fashion_retail"`). Four stages: taxonomy (LLM) → catalog (deterministic Python skeleton allocator + LLM naming, including per-`Ware` lifecycle and freshness parameters per category) → store templates (LLM, per region — including `init_active_products` rosters and `init_freshness` mode) → market domain params (LLM authors the domain slice; math defaults are hand-set). Each LLM call is parsed through a Pydantic schema in `src/llm/schemas.py`; on `ValidationError`, the failure is fed back into the next prompt for self-correction (default 3 retries). Consumed by scenario authors who want a generated catalog/market instead of hand-coding one. Policies, disruption parameters, and seeds remain author-supplied.

The canonical script entry point is `load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False)` — it returns a cached `World` on a hit and, on a miss, prints a warning and prompts for interactive consent before invoking `build_fn` (raises `LLMBuildAbortedError` if the user declines or the context is non-interactive). Pass `auto_confirm=True` for CI or offline scripts.

---

## Decisions

- [ADR 0001](docs/adr/0001-two-layer-lifecycle.md) — Lifecycle is two-layer: global PLC × per-store freshness curve.
- [ADR 0002](docs/adr/0002-per-stage-transitions-and-dead-stage.md) — Per-stage transition probabilities and a `dead` stage with terminal-by-default cycle.
- [ADR 0003](docs/adr/0003-crn-demand-for-all-products.md) — Demand is sampled for every catalog product each tick, even when inactive (CRN cleanliness).
- [ADR 0004](docs/adr/0004-rl-training-env.md) — RL training env: randomised assortment, slot-shuffled observations, hidden market state, frozen assortment within episode.
- [ADR 0005](docs/adr/0005-demand-relative-action-decoding.md) — Action decoder expresses order quantity in lead-times-of-demand, not fraction-of-shelf-space. *Superseded by ADR 0007.*
- [ADR 0006](docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family replaces the heuristic baseline as the RL comparison anchor.
- [ADR 0007](docs/adr/0007-rl-scale-invariance-package.md) — RL scale-invariance package: order-up-to action decoder, demand-units inventory feature, log-uniform domain randomisation.
- [ADR 0008](docs/adr/0008-safety-horizons-as-fraction-of-delivery-lag.md) — Reparameterise textbook safety horizons as fractions of delivery lag (fix per-SKU mis-scaling).
- [ADR 0009](docs/adr/0009-policy-hyperparameter-tuning-tool.md) — Policy hyperparameter tuning tool (Optuna-based).
- [ADR 0010](docs/adr/0010-sim-as-base-for-ml-layers.md) — `src/sim/` is the canonical home for rollout primitives (metrics, episode sampler, world loader, World artifact, two-phase tick API); `src/tuning/` and `src/rl/` are sibling consumers.
