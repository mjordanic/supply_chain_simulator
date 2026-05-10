# Retail Market Simulator

Multi-agent retail market simulation. Each store runs its own decision policy in a shared world (regional demand/supply, item lifecycle, stochastic disruption events). Scenarios are reproducible by construction: the world, per-policy, and per-store random streams are kept separate so two runs that differ only in policy share the same world trajectory. Outputs are parquet, JSON, and PNG.

## Contents

1. [Quickstart](#quickstart)
2. [Project layout](#project-layout)
3. [Concepts](#concepts)
4. [Running a scenario](#running-a-scenario)
5. [Output layout](#output-layout)
6. [Authoring a scenario](#authoring-a-scenario)
7. [Example scenarios](#example-scenarios)
8. [Common Random Numbers and reproducibility](#common-random-numbers-and-reproducibility)
9. [LLM world builder](#llm-world-builder)
10. [Testing](#testing)
11. [Further reading](#further-reading)

## Quickstart

```bash
uv sync
uv run python main.py scenarios/example_homogeneous.py
uv run pytest
```

The first run writes parquet/JSON/PNG artifacts under `data/example_homogeneous/`.

## Project layout

```
main.py                         CLI shim: load scenario file, dispatch to Runner + DataExporter
scenarios/                      runnable example Scenario modules
  example_homogeneous.py        single policy, three stores varying init_seed
  example_paired_comparison.py  paired CRN A/B between two BaselinePolicy variants
  example_llm_world.py          12-item LLM world; needs OPENAI_API_KEY on first run
  example_llm_world_offline.py  same pipeline driven by a CannedClient (no network)
  llm_world_100.py              100-item LLM world, 3 stores
  llm_world_1000.py             1000-item LLM world, 5 stores, 2000-step run
src/sim/                        core simulator
  scenario.py                   Scenario dataclass + load_catalog() / make_stores() helpers
  runner.py                     simulation loop (observe → decide → advance → log)
  store.py                      Store with full accounting (delegates step-0 setup to StoreInitializer)
  store_initializer.py          pure init_store_state seam (bit-identity contract)
  policy.py                     Policy ABC + BaselinePolicy + NoopPolicy
  market.py                     regional demand/supply (composes stage × freshness × season × promo × cross)
  event_engine.py               stochastic disruptions + scheduled deliveries
  item_registry.py              catalog + per-item lifecycle/freshness/stock-share state
  lifecycle_clock.py            pure advance_stage() over [introduction, growth, maturity, decline, dead]
  freshness_curve.py            pure m(τ) = 1 + α · exp(−τ/β) multiplier
  distributions.py              Constant / Uniform / Normal / Choice
  data_exporter.py              parquet + JSON + PNG writer
src/llm/                        LLM world builder
  world_builder.py              pipeline orchestrator + World artifact + load_or_build_world cache
  schemas.py                    Pydantic schemas (Taxonomy, Catalog, Correlations, FreshnessSet,
                                StoreTemplateList, MarketDomain, plus RelatedRef / ItemRelations /
                                ItemFreshness / SeasonWindow / TaxonomyCategory / StoreTemplateSpec)
  prompts.py                    prompt templates with retry support (six stages)
  openai_client.py              OpenAI structured-output client (Protocol + concrete OpenAIClient)
src/forecasting/                optional ARIMA(1,1,1) sales predictor (not wired into the runner)
tests/                          pytest suite
```

## Concepts

- **Scenario** — frozen experiment inputs: catalog, market, disruption, item lifecycle, list of `(template, init_seed, policy)` store instances, `n_steps`, `start_date`, `world_seed`.
- **StoreTemplate** — reusable store profile (region, capacity, balance, lead time, fees, …) plus an optional `init_active_products` roster and an `init_freshness` mode (`"baseline"` for established stores, `"fresh"` for grand-opening). The pair `(template, init_seed)` is the bit-identical step-0 contract: two stores built from the same pair start identical regardless of attached policy.
- **Policy** — `BaselinePolicy` is the supplied implementation. It takes kwargs hyperparameters (`promo_threshold`, `promo_discount`, `review_interval`, …) plus its own `policy_seed`.
- **Two-layer lifecycle** — every product has a global stage in `[introduction, growth, maturity, decline, dead]` advanced by `LifecycleClock` against the per-stage `stage_change_probs` table; on top of that, every `(store, product)` pair has a freshness curve `m(τ) = 1 + α · exp(−τ/β)` that resets on each `Store.activate_item`. See [docs/adr/0001-two-layer-lifecycle.md](docs/adr/0001-two-layer-lifecycle.md) and [docs/adr/0002-per-stage-transitions-and-dead-stage.md](docs/adr/0002-per-stage-transitions-and-dead-stage.md).
- **RNG split** — `world_rng` (market, events, item lifecycle) and `policy_rng` (policy decisions) and `init_rng` (per-store initial state) never share state. This is what lets policies be compared on identical worlds.
- **DataExporter** — consumes the run log and writes parquet/JSON/PNG under `data/<scenario_stem>/`.

For full domain definitions see [CONTEXT.md](CONTEXT.md).

## Running a scenario

`main.py` is a thin CLI: it imports a Python file by path, expects a top-level `scenario` symbol of type `Scenario`, and dispatches to `Runner` and `DataExporter`.

```bash
# Default output: data/<file_stem>/
uv run python main.py scenarios/example_homogeneous.py

# Custom output folder
uv run python main.py scenarios/example_paired_comparison.py --output /tmp/run
```

Each example scenario is also independently runnable:

```bash
uv run python scenarios/example_homogeneous.py
```

Errors `main.py` emits on a bad scenario path:

- `main.py: scenario file not found: <path>` — file does not exist
- `main.py: <path> does not expose a top-level `scenario` symbol` — module loaded but no `scenario =` at module level
- `main.py: <path>.scenario is <type>, expected Scenario` — wrong type

## Output layout

Every run produces (relative to `--output`, default `data/<stem>/`):

```
data/<stem>/
  config/
    scenario.json            full Scenario.to_json() (policies excluded — they are Python objects)
  data/
    run_log.json             per-step log with ISO-formatted datetimes
    products.parquet         static catalog (product_id, name, category, prices, seasonality)
    stores.parquet           static store metadata (template_id, region, init_seed, policy_type)
    timeseries.parquet       per-step × per-store × per-product metrics
  reports/
    overview.png             regional supply/demand chart
```

`timeseries.parquet` columns: `simulation_step`, `simulation_date`, `store_id`, `product_id`, `inventory`, `demand`, `sales`, `order_quantity`, `outstanding_orders`, `promotion_status`, `active_status`, `price`, `revenue`, `total_cost`, `holding_cost`, `profit`.

## Authoring a scenario

A scenario file is a Python module that constructs a `Scenario` and binds it to the name `scenario` at module level. The pieces:

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
    {
        "name": "Widget B",
        "category": "Widgets",
        "related_products": [["Widget A", 0.5]],   # cross-product correlation in [0, 1]
        "base_price": 30.0,
        "unit_cost": 18.0,
        "seasonality": "all_season",
    },
])
```

**2. Market, disruption, lifecycle.** Flat dataclasses; stochastic fields hold `Distribution` objects (`Constant`, `Uniform`, `Normal`, `Choice`):

```python
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import DisruptionParams, ItemLifecycleParams, MarketParams

market = MarketParams(
    cycle_len=365,
    cycle_amp=0.1,
    init_demand=100.0,
    init_supply=100.0,
    peak_factor=1.2,
    off_factor=0.7,
    season_months={"all_season": list(range(1, 13))},
    regions=["US"],
    correlation=0.7,
    trend_update_interval=20,
    min_value=20.0,
    max_value=200.0,
    stage_multipliers={"introduction": 0.7, "growth": 1.5, "maturity": 1.0, "decline": 0.2, "dead": 0.05},
    price_elasticity=-1.5,
    promo_multiplier=1.0,
    demand_factor_min=0.1,
    demand_divisor=100.0,
    supply_factor_min=0.01,
    supply_divisor=100.0,
    demand_range=(80.0, 120.0),
    cross_inv_lo=0.3,
    cross_inv_hi=0.7,
    cross_factor_range=(0.3, 1.6),
    trend=Constant(1.0),
    demand_shock=Normal(0.0, 5.0),
    supply_shock=Normal(0.0, 5.0),
    base_demand=Uniform(2, 8),
)

disruption = DisruptionParams(
    event_prob=0.05,
    types=["natural_disaster", "economic_crisis"],
    regions=["US"],
    severity=Constant(1.0),
    duration=Constant(3),
)

_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
lifecycle = ItemLifecycleParams(
    stages=_STAGES,
    init_stage="maturity",
    # Per-current-stage transition table consumed by LifecycleClock.advance_stage.
    # Set every entry to 0.0 to disable transitions; set "dead" to 0.0 specifically
    # for strict-terminal lifecycle. Per-Ware Ware.stage_change_probs override.
    default_stage_change_probs={s: 0.0 for s in _STAGES},
    # Catalog-wide freshness defaults; per-Ware Ware.freshness_alpha / freshness_decay override.
    # alpha=0 collapses the curve to identically 1 (no hype effect).
    default_freshness_alpha=0.0,
    default_freshness_decay=1.0,
    # Catalog-wide weight for initial-stock allocation across the active set.
    # Per-Ware Ware.init_stock_share overrides; default 1.0 ⇒ even split.
    default_init_stock_share=1.0,
)
```

**3. Store template and policy.** A template is a reusable spec; scalar fields can be replaced with a `Distribution` to randomize across stores at construction time. `init_active_products` (optional) is an explicit roster of product ids to activate at step 0; when omitted, the initializer falls back to `init_rng.sample(catalog, init_active_count)`. `init_freshness` selects between `"baseline"` (initial active SKUs skip the hype window — established store) and `"fresh"` (initial active SKUs start at `τ = 0` — grand-opening).

```python
from src.sim.policy import BaselinePolicy
from src.sim.scenario import StoreTemplate

template = StoreTemplate(
    id="standard",
    region="US",
    capacity=200,
    init_balance=10_000.0,
    init_stock_pct=0.4,
    delivery_lag=2,
    holding_rate=0.005,
    order_fee=10.0,
    init_active_count=3,
    # Optional explicit roster — overrides random sampling when set:
    # init_active_products=["P0000", "P0002", "P0004"],
    # Default "baseline" (established store); use "fresh" for grand-opening.
    init_freshness="baseline",
)

policy = BaselinePolicy(
    policy_seed=1000,
    min_qty=1,
    init_qty_factor=0.3,
    promo_threshold=0.4,
    promo_discount=0.7,
    review_interval=10,
    target_active_count=4,
)
```

`BaselinePolicy` accepts ~20 kwargs in total (promo length range and cooldown, history window, slow-sales limit, etc.); see `src/sim/policy.py` for the full list.

**4. Wire stores together.** One declarative helper, `make_stores(triples)`, covers every case. The triple list *is* the roster — there is no regime abstraction above it:

- **Robustness sweep** (one policy across diverse stores): repeat one policy with varying `init_seed` and/or template.
- **Paired CRN comparison** (two policies on bit-identical world data): repeat the same `(template, init_seed)` with two different policies. Pair `i` lives at indices `2i` and `2i+1`.
- **k-way CRN comparison**: repeat the same `(template, init_seed)` with `k` different policies — same shape, no API change.
- **Mixed roster** (e.g. `A, A, B, C, D` across `T1, T2, T3, T4, T4, T4`): just write the literal triple list.

**5. Final assembly.** Bind to `scenario`:

```python
from datetime import datetime
from src.sim.scenario import Scenario, make_stores

scenario = Scenario(
    catalog=catalog,
    market=market,
    disruption=disruption,
    item_lifecycle=lifecycle,
    stores=make_stores([
        (template, 1, policy),
        (template, 2, policy),
        (template, 3, policy),
    ]),
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)
```

Save under `scenarios/my_run.py` and run `uv run python main.py scenarios/my_run.py`.

## Example scenarios

**`scenarios/example_homogeneous.py`** — three stores, one shared `BaselinePolicy`, distinct `init_seed`s. The right starting point for population-level evaluation of a single policy.

```bash
uv run python main.py scenarios/example_homogeneous.py
```

**`scenarios/example_paired_comparison.py`** — two pairs (4 stores) running an aggressive vs. conservative `BaselinePolicy` on bit-identical world data. Aggressive promotes earlier with bigger markdowns (`promo_threshold=0.30`, `promo_discount=0.6`); conservative promotes only on heavy stock with gentler markdowns (`promo_threshold=0.70`, `promo_discount=0.85`). The right starting point for a variance-reduced policy A/B test.

```bash
uv run python main.py scenarios/example_paired_comparison.py
```

**`scenarios/example_llm_world.py`** — 12-item fashion-retail world built by the LLM, three stores. First run needs `OPENAI_API_KEY` and caches the world to `data/worlds/fashion_retail_12/world.json`; subsequent runs load locally.

```bash
OPENAI_API_KEY=... uv run python main.py scenarios/example_llm_world.py
```

**`scenarios/example_llm_world_offline.py`** — the same pipeline driven by a `CannedClient` that pops pre-built Pydantic payloads. Useful for inspecting payload shapes and running deterministically with no API key.

```bash
uv run python main.py scenarios/example_llm_world_offline.py
```

**`scenarios/llm_world_100.py`** / **`scenarios/llm_world_1000.py`** — larger LLM-built worlds (100 and 1000 items). Same cache + consent model as `example_llm_world.py`; the 1000-item scenario also rescales the LLM-authored template for the larger catalog.

## Common Random Numbers and reproducibility

Three independent random streams, three independent seeds:

| Seed | Stream | Used for |
|---|---|---|
| `Scenario.world_seed` | `world_rng` | market dynamics, disruption events, item lifecycle transitions |
| `Policy.policy_seed` | `policy_rng` | policy decisions only (one stream per policy instance) |
| `StoreInstance.init_seed` | `init_rng` | step-0 SKU activation and stock allocation |

Consequences:

- Replaying the same `Scenario` produces the same world trajectory.
- Two stores constructed from the same `(template, init_seed)` pair start step 0 bit-identical — the integration tests assert this on the paired-comparison example.
- Swapping a policy on a scenario does not perturb the world stream, so per-policy outcome differences come from policy decisions alone.

## LLM world builder

`src/llm/world_builder.py` generates a runnable `World(catalog, market, store_templates)` from a domain archetype string (e.g. `"fashion_retail"`, `"grocery"`, `"pharmacy"`). Use it when you want a thematically coherent catalog and market without hand-coding one; you still author the policy, disruption parameters, item-lifecycle parameters, and seeds.

`load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False)` is the canonical entry point — it returns a cached `World` on hit and, on a miss, prints a warning and prompts for interactive consent before invoking `build_fn` (raises `LLMBuildAbortedError` if the user declines or the context is non-interactive). Pass `auto_confirm=True` for CI / offline scripts.

### The stages

1. **`build_market_domain_params()`** — one LLM call. The LLM authors a domain-meaningful slice (`cycle_len`, `peak_factor`, `off_factor`, `init_demand`, `init_supply`, `season_months`, `regions`, `price_elasticity`); `WorldBuilder` merges that with hand-set math defaults (clamps, divisors, distribution objects, lifecycle stage multipliers including `dead`) defined in `world_builder.py:_MARKET_MATH_DEFAULTS` to produce a complete `MarketParams`. Sets the region list consumed by the templates call.
2. **`build_taxonomy()`** — one LLM call. Returns a `Taxonomy` with categories and `target_share` weights for the archetype. Invoked transitively by `sample_catalog`.
3. **`sample_catalog(n)`** — four sub-steps:
   1. (Python) `allocate_skeletons(n, taxonomy)` deterministically distributes `n` slots across taxonomy categories proportional to `target_share`.
   2. (LLM) one `catalog_prompt` call names and prices the slots (`name`, `category`, `base_price`, `unit_cost`, `seasonality`).
   3. (LLM) **chunked** `correlations_prompt` calls author `related_products` against the explicit name list — chunk size `correlations_chunk_size` (default `50`); cross-chunk references resolve through the prompt's candidate-name list. Dangling / self / duplicate refs are dropped at the boundary rather than triggering schema retries.
   4. (LLM) **chunked** `freshness_prompt` calls author per-item `freshness_alpha` / `freshness_decay`; chunk size `freshness_chunk_size` (default `50`). Items the LLM omits or misnames keep `Ware` default `None` so `ItemRegistry` falls back to `ItemLifecycleParams` defaults.

   Output is a `list[Ware]` with stable `P{i:04d}` ids assigned by `load_catalog`. Per-`Ware` lifecycle (`init_stage`, `stage_change_probs`) and `init_stock_share` fields are **not** authored by the LLM — `ItemRegistry` always falls back to `ItemLifecycleParams` defaults for them.
4. **`build_store_templates()`** — one LLM call. Returns `dict[str, StoreTemplate]` keyed by template id (e.g. `"flagship"`, `"standard"`, `"outlet"`). Each template carries operating parameters plus an `init_active_count` and an `init_freshness` mode (`"baseline"` for established stores — initial SKUs skip the hype window; `"fresh"` for grand-opening — initial SKUs enter at full hype). The starting roster (`init_active_products`) is **not** LLM-authored; `init_store_state` random-samples `init_active_count` SKUs from the catalog at store-construction time.

`build(n_items)` runs the stages in order **market → catalog (taxonomy + naming + correlations + freshness) → templates** — the market call sets the region list that the templates call consumes. Each stage caches its result on the instance, so calling `build()` twice does not re-hit the LLM.

Total LLM calls for `build(n_items)`: `2 + ceil(n / correlations_chunk_size) + ceil(n / freshness_chunk_size) + 1` (market, taxonomy, catalog naming, chunked correlations, chunked freshness, templates). For `n=100` with default chunk sizes that's ~6 calls; for `n=1000` it's ~46.

### Schemas and validation

Every LLM payload is parsed through a Pydantic model in `src/llm/schemas.py`. Validators currently enforced:

- `CatalogItem.base_price > unit_cost` (positive margin), `base_price > 0`, `unit_cost >= 0`
- `CatalogItem.seasonality` is a closed enum (`spring`, `summer`, `fall`, `winter`, `spring/summer`, `fall/winter`, `all_season`)
- `RelatedRef.correlation ∈ [0, 1]`
- `ItemRelations` `related` references that don't resolve against the catalog (or are self-references / within-item duplicates) are pruned post-parse — the schema does not reject them, the boundary sanitiser in `WorldBuilder.sample_catalog` does
- `ItemFreshness.alpha >= 0` and `ItemFreshness.decay > 0` (strictly positive so the consumer never divides by zero)
- `TaxonomyCategory.target_share ∈ (0, 1]`
- `StoreTemplateList` template ids are unique; each `StoreTemplateSpec.init_freshness` is the `InitFreshness` enum (`"baseline"` or `"fresh"`); operating fields are non-negative and `init_stock_pct ∈ [0, 1]`
- `MarketDomain.price_elasticity < 0` (demand must fall with price); `MarketDomain.season_months` is a list of `{name, months}` records with unique names (not a free-form dict — OpenAI strict-output mode rejects open-ended object keys)

Per-`Ware` lifecycle (`init_stage`, `stage_change_probs`) and `init_stock_share` fields exist on `Ware` for hand-authored catalogs but are **not** present in any LLM schema — they always fall back to `ItemLifecycleParams` defaults on LLM-built catalogs.

### Retry on schema failure

If the LLM response fails Pydantic validation, the validation error string is prepended to the next prompt so the model can self-correct. Default retry budget is 3 attempts per stage, configurable via `WorldBuilder(max_retries=...)`. Other exceptions (transport errors, refusals) propagate immediately.

### Setup

The shipped client uses OpenAI's structured outputs API
(`client.beta.chat.completions.parse`):

- Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAIClient(...)`.
- Default model: `gpt-5.4-mini` (see `OpenAIClient.__init__`). Override via `OpenAIClient(model="…")`.
- Pass a custom `OPENAI_BASE_URL` only when routing through a proxy / compatible endpoint — a stale value pointing at localhost is the most common source of `APIConnectionError`.

### Example

```python
from src.llm.openai_client import OpenAIClient
from src.llm.world_builder import WorldBuilder

client = OpenAIClient()                  # reads OPENAI_API_KEY
builder = WorldBuilder(archetype="fashion_retail", client=client)
world = builder.build(n_items=15)

# world.catalog          → list[Ware]
# world.market           → MarketParams (LLM domain slice + math defaults)
# world.store_templates  → dict[str, StoreTemplate]
```

### Wiring it into a Scenario

```python
from datetime import datetime
from src.sim.policy import BaselinePolicy
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    Scenario,
    make_stores,
)
from src.sim.distributions import Constant

template = next(iter(world.store_templates.values()))   # or pick by key
policy = BaselinePolicy(policy_seed=1000)

_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]

scenario = Scenario(
    catalog=world.catalog,
    market=world.market,
    disruption=DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=world.market.regions,
        severity=Constant(1.0),
        duration=Constant(3),
    ),
    item_lifecycle=ItemLifecycleParams(
        stages=_STAGES,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in _STAGES},
    ),
    stores=make_stores([(template, 1 + i, policy) for i in range(5)]),
    n_steps=100,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)
```

### Test seam

`LLMClient` is a `Protocol` (`src/llm/openai_client.py`), so tests inject a fake client implementing `structured_completion(*, system, user, schema)` rather than hitting the network. See `tests/llm/` for examples.

## Testing

```bash
uv run pytest
```

The suite covers the simulator subsystems plus an integration tier:

- `tests/test_examples_and_cli.py` exercises both example scenarios end-to-end and asserts that paired triples (same `(template, init_seed)`, different policies) produce step-0-identical pairs and that `main.py` writes every artifact in [Output layout](#output-layout).
- A regression snapshot test pins a fixed scenario's outputs against a saved baseline so unintended changes to the simulation math show up immediately.

## Further reading

- [CONTEXT.md](CONTEXT.md) — domain and architecture glossary
