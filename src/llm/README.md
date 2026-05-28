# `src/llm/` — LLM world generator

An LLM-driven pipeline that drafts a complete `World(catalog, market, store_templates)` artifact from a single domain archetype string like `"fashion_retail"` or `"sports_cars"`. Use it when you want a thematically coherent catalog and market without hand-coding one; you still author the policy, disruption parameters, item-lifecycle parameters, and seeds.

## Contents

1. [Layout](#layout)
2. [The four stages](#the-four-stages)
3. [Sample catalogs](#sample-catalogs)
4. [Schemas and validation](#schemas-and-validation)
5. [Retry on schema failure](#retry-on-schema-failure)
6. [Setup](#setup)
7. [Example](#example)
8. [Wiring into a Scenario](#wiring-into-a-scenario)
9. [Test seam](#test-seam)

## Layout

```
src/llm/
  world_builder.py   pipeline orchestrator + load_or_build_world cache
                     (re-exports `World` from src/sim/world.py)
  schemas.py         Pydantic schemas (Taxonomy, Catalog, Correlations, FreshnessSet,
                     StoreTemplateList, MarketDomain, plus RelatedRef / ItemRelations /
                     ItemFreshness / SeasonWindow / TaxonomyCategory / StoreTemplateSpec)
  prompts.py         prompt templates with retry support (six stages)
  openai_client.py   OpenAI structured-output client (Protocol + concrete OpenAIClient)
```

`load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False)` is the canonical entry point — it returns a cached `World` on hit and, on a miss, prints a warning and prompts for interactive consent before invoking `build_fn` (raises `LLMBuildAbortedError` if the user declines or the context is non-interactive). Pass `auto_confirm=True` for CI / offline scripts.

## The four stages

`WorldBuilder(archetype, client).build(n_items)` runs four stages in order **market → catalog (taxonomy + naming + correlations + freshness) → templates** — the market call sets the region list that the templates call consumes. Each stage caches its result on the instance, so calling `build()` twice does not re-hit the LLM.

1. **`build_market_domain_params()`** — one LLM call. The LLM authors a domain-meaningful slice (`cycle_len`, `peak_factor`, `off_factor`, `init_demand`, `init_supply`, `season_months`, `regions`, `price_elasticity`); `WorldBuilder` merges that with hand-set math defaults (clamps, divisors, distribution objects, lifecycle stage multipliers including `dead`) defined in `world_builder.py:_MARKET_MATH_DEFAULTS` to produce a complete `MarketParams`. Sets the region list consumed by the templates call.
2. **`build_taxonomy()`** — one LLM call. Returns a `Taxonomy` with categories and `target_share` weights for the archetype. Invoked transitively by `sample_catalog`.
3. **`sample_catalog(n)`** — four sub-steps:
   1. (Python) `allocate_skeletons(n, taxonomy)` deterministically distributes `n` slots across taxonomy categories proportional to `target_share`.
   2. (LLM) one `catalog_prompt` call names and prices the slots (`name`, `category`, `base_price`, `unit_cost`, `seasonality`).
   3. (LLM) **chunked** `correlations_prompt` calls author `related_products` against the explicit name list — chunk size `correlations_chunk_size` (default `50`); cross-chunk references resolve through the prompt's candidate-name list. Dangling / self / duplicate refs are dropped at the boundary rather than triggering schema retries.
   4. (LLM) **chunked** `freshness_prompt` calls author per-item `freshness_alpha` / `freshness_decay`; chunk size `freshness_chunk_size` (default `50`). Items the LLM omits or misnames keep `Ware` default `None` so `ItemRegistry` falls back to `ItemLifecycleParams` defaults.

   Output is a `list[Ware]` with stable `P{i:04d}` ids assigned by `load_catalog`. Per-`Ware` lifecycle (`init_stage`, `stage_change_probs`) and `init_stock_share` fields are **not** authored by the LLM — `ItemRegistry` always falls back to `ItemLifecycleParams` defaults for them.
4. **`build_store_templates()`** — one LLM call. Returns `dict[str, StoreTemplate]` keyed by template id (e.g. `"flagship"`, `"standard"`, `"outlet"`). Each template carries operating parameters plus an `init_active_count` and an `init_freshness` mode (`"baseline"` for established stores — initial SKUs skip the hype window; `"fresh"` for grand-opening — initial SKUs enter at full hype). The starting roster (`init_active_products`) is **not** LLM-authored; `init_store_state` random-samples `init_active_count` SKUs from the catalog at store-construction time.

Total LLM calls for `build(n_items)`: `2 + ceil(n / correlations_chunk_size) + ceil(n / freshness_chunk_size) + 1` (market, taxonomy, catalog naming, chunked correlations, chunked freshness, templates). For `n=100` with default chunk sizes that's ~6 calls; for `n=1000` it's ~46.

## Sample catalogs

Two cached worlds ship in `data/worlds/`. Inspect them in `notebooks/02-inspect_world.ipynb` via `world.catalog_df()`.

### `sports_cars_100` (102 items across 7 categories)

| id | name | category | base | cost | season |
|---|---|---|---:|---:|---|
| P0000 | Apex Vector R8 Track Coupe | Track-Ready Coupes | 128,900 | 91,500 | summer |
| P0001 | Vortex RS Sprint Coupe | Track-Ready Coupes | 119,500 | 84,200 | spring/summer |
| P0002 | Carbonline GT3 Aero Coupe | Track-Ready Coupes | 136,800 | 97,200 | summer |
| P0024 | Meridian Grand V12 Tourer | Grand Tourers | 168,900 | 121,000 | fall/winter |
| P0025 | Regal Milesport GT | Grand Tourers | 154,500 | 110,800 | all_season |
| P0026 | Auric Continental S Tourer | Grand Tourers | 162,300 | 116,500 | fall/winter |
| P0042 | Briarwood Convertible GT | Roadsters & Convertibles | 112,400 | 79,700 | spring/summer |
| P0043 | Mariposa Open-Air Roadster | Roadsters & Convertibles | 104,800 | 74,100 | summer |
| P0044 | Sunflare Turbo Spider | Roadsters & Convertibles | 118,200 | 83,400 | spring/summer |
| P0059 | Inferno X Supercar | Exotic Supercars | 315,000 | 224,000 | summer |
| P0060 | Nebula 12C Hypercar | Exotic Supercars | 428,500 | 305,000 | summer |
| P0061 | Sable Apex MR | Exotic Supercars | 367,900 | 262,100 | summer |
| P0073 | Spectra V8 Sport Sedan | Performance Sedans | 82,900 | 58,600 | all_season |
| P0074 | Granite S4 Performance Sedan | Performance Sedans | 76,400 | 54,100 | all_season |
| P0075 | Vertex M50 Sport Sedan | Performance Sedans | 88,950 | 63,100 | all_season |
| P0087 | Swiftline 2.0T Coupe | Compact Sport Coupes | 31,900 | 22,300 | all_season |
| P0088 | Pacer GT Turbo | Compact Sport Coupes | 28,600 | 19,800 | all_season |
| P0089 | Kestrel S Hatch Coupe | Compact Sport Coupes | 33,450 | 23,600 | all_season |
| P0098 | Heritage Circuit 911 RS | Special Editions & Collector Models | 214,900 | 152,800 | fall |
| P0099 | Founders Series V12 Coupe | Special Editions & Collector Models | 248,600 | 176,900 | fall |
| P0100 | Anniversary Apex 50 | Special Editions & Collector Models | 193,700 | 137,500 | fall |

### `fashion_retail_250` (336 items across 6 categories)

The `related_products` column shows the LLM-authored cross-product correlations: top-3 partner items with correlation weight (∈ [0, 1]) in parentheses. Fashion-only — the `sports_cars_100` catalog above omits this column for table-width reasons.

| id | name | category | base | cost | season | related_products |
|---|---|---|---:|---:|---|---|
| P0000 | Women's Essential Crewneck Tee | Women's Apparel | 24.00 | 8.00 | spring/summer | Women's Stretch Skinny Jeans (0.82), Women's Knit Cardigan (0.68), Women's Tailored Blazer (0.42) |
| P0001 | Women's Ribbed Tank Top | Women's Apparel | 22.00 | 7.50 | summer | Women's High-Rise Bike Shorts (0.66), Women's Stretch Skinny Jeans (0.52), Women's Cropped Hoodie (0.41) |
| P0002 | Women's Relaxed Linen Button-Up Shirt | Women's Apparel | 42.00 | 15.00 | spring/summer | Women's Wide-Leg Trousers (0.63), Women's Paperbag Waist Shorts (0.58), Women's Pleated Midi Skirt (0.47) |
| P0086 | Men's Classic Oxford Shirt | Men's Apparel | 54.00 | 20.00 | all_season | Men's Tapered Suit Pants (0.74), Men's Tailored Blazer (0.68), Men's Leather Belt (0.45) |
| P0087 | Men's Slim Fit Chino Pants | Men's Apparel | 59.00 | 22.00 | all_season | Men's Classic Oxford Shirt (0.62), Men's Piqué Golf Shirt (0.56), Men's Dress Chinos (0.41) |
| P0088 | Men's Performance Polo | Men's Apparel | 39.00 | 14.00 | spring/summer | Men's Straight-Leg Khaki Pants (0.51), Men's Canvas Low-Top Sneakers (0.46), Men's Leather Boat Shoes (0.34) |
| P0151 | Men's Classic Derby Shoes | Footwear | 98.00 | 38.00 | all_season | Men's Slim Wool Trousers (0.58), Men's Tailored Dress Shirt (0.54), Men's Wool Blend Overcoat (0.33) |
| P0152 | Women's Everyday Ballet Flats | Footwear | 76.00 | 29.00 | all_season | Women's Pleated Midi Skirt (0.47), Women's Straight-Leg Ponte Pants (0.44), Women's Knit Cardigan (0.38) |
| P0153 | Men's Canvas Low-Top Sneakers | Footwear | 68.00 | 25.00 | spring/summer | Men's Relaxed Fit Jeans (0.62), Men's Cotton Crewneck T-Shirt (0.57), Men's Denim Jacket (0.41) |
| P0192 | Women's Leather Tote Bag | Accessories | 118.00 | 46.00 | all_season | Women's Tailored Blazer (0.48), Women's Straight-Leg Ponte Pants (0.43), Women's Sunglasses (0.42) |
| P0193 | Men's Reversible Leather Belt | Accessories | 42.00 | 14.50 | all_season | Men's Slim Fit Chino Pants (0.51), Men's Tailored Dress Shirt (0.46), Men's Dress Chinos (0.44) |
| P0194 | Women's Straw Sun Hat | Accessories | 36.00 | 11.50 | summer | Women's Resort Straw Tote (0.66), Women's Sunglasses (0.57), Women's Espadrille Wedges (0.41) |
| P0241 | Kids' Graphic Tee Pack | Kids' Apparel | 24.00 | 7.50 | spring/summer | Kids' Denim Overalls (0.48), Kids' Jogger Sweatpants (0.34), Kids' Denim Jacket (0.22) |
| P0242 | Kids' Denim Overalls | Kids' Apparel | 38.00 | 13.00 | all_season | Kids' Graphic Tee Pack (0.58), Kids' Long Sleeve Tee (0.41), Kids' Denim Jacket (0.24) |
| P0243 | Kids' Fleece Zip Hoodie | Kids' Apparel | 34.00 | 11.50 | fall/winter | Kids' Jogger Sweatpants (0.72), Kids' Graphic Sweatshirt (0.55), Kids' Knit Beanie (0.31) |
| P0292 | Kids' Cotton Briefs 6-Pack | Intimates & Sleepwear | 18.00 | 5.50 | all_season | Kids' Training Bra (0.33), Kids' School Socks 5-Pack (0.31), Kids' Pajama Onesie (0.24) |
| P0293 | Kids' Training Bra | Intimates & Sleepwear | 22.00 | 6.50 | all_season | Kids' Cotton Briefs 6-Pack (0.38), Women's Wireless T-Shirt Bra (0.00) |
| P0294 | Women's Seamless Briefs 3-Pack | Intimates & Sleepwear | 24.00 | 7.00 | all_season | Women's Wireless T-Shirt Bra (0.61), Women's Lace Bralette (0.45), Women's Seamless Camisole (0.39) |

## Schemas and validation

Every LLM payload is parsed through a Pydantic model in `src/llm/schemas.py`. Validators currently enforced:

- `CatalogItem.base_price > unit_cost` (positive margin), `base_price > 0`, `unit_cost >= 0`
- `CatalogItem.seasonality` is a closed enum (`spring`, `summer`, `fall`, `winter`, `spring/summer`, `fall/winter`, `all_season`)
- `RelatedRef.correlation ∈ [0, 1]`
- `ItemRelations.related` references that don't resolve against the catalog (or are self-references / within-item duplicates) are pruned post-parse — the schema does not reject them, the boundary sanitiser in `WorldBuilder.sample_catalog` does
- `ItemFreshness.alpha >= 0` and `ItemFreshness.decay > 0` (strictly positive so the consumer never divides by zero)
- `TaxonomyCategory.target_share ∈ (0, 1]`
- `StoreTemplateList` template ids are unique; each `StoreTemplateSpec.init_freshness` is the `InitFreshness` enum (`"baseline"` or `"fresh"`); operating fields are non-negative and `init_stock_pct ∈ [0, 1]`
- `MarketDomain.price_elasticity < 0` (demand must fall with price); `MarketDomain.season_months` is a list of `{name, months}` records with unique names (not a free-form dict — OpenAI strict-output mode rejects open-ended object keys)

Per-`Ware` lifecycle (`init_stage`, `stage_change_probs`) and `init_stock_share` fields exist on `Ware` for hand-authored catalogs but are **not** present in any LLM schema — they always fall back to `ItemLifecycleParams` defaults on LLM-built catalogs.

## Retry on schema failure

If the LLM response fails Pydantic validation, the validation error string is prepended to the next prompt so the model can self-correct. Default retry budget is 3 attempts per stage, configurable via `WorldBuilder(max_retries=...)`. Other exceptions (transport errors, refusals) propagate immediately.

## Setup

The shipped client uses OpenAI's structured outputs API (`client.beta.chat.completions.parse`):

- Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAIClient(...)`.
- Default model: `gpt-5.4-mini` (see `OpenAIClient.__init__`). Override via `OpenAIClient(model="…")`.
- Pass a custom `OPENAI_BASE_URL` only when routing through a proxy / compatible endpoint — a stale value pointing at localhost is the most common source of `APIConnectionError`.

## Example

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

## Wiring into a Scenario

The LLM produces a `World` (catalog + market + store templates); `world_to_graph` (`src/sim/world.py`) synthesises a default multi-echelon topology from it — one `factory → shop → sink-per-product` sub-graph per store template. It returns `{"node_instances": [...], "edges": [...]}` with policies left unset, so attach a policy to each node by type before assembling the `Scenario`:

```python
from datetime import datetime
from src.sim.distributions import Constant
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy, OrderUpToPolicy, StaticFactoryPolicy,
)
from src.sim.scenario import DisruptionParams, ItemLifecycleParams, Scenario
from src.sim.world import world_to_graph

topology = world_to_graph(world, sink_density=1.0)   # every product gets a sink

for i, ni in enumerate(topology["node_instances"]):
    node = ni.node
    if isinstance(node, FactoryNode):
        ni.policy = node.policy = StaticFactoryPolicy(
            capacity_per_tick=node.capacity_per_tick, unit_cost=node.unit_cost)
    elif isinstance(node, IntermediateNode):
        ni.policy = node.policy = OrderUpToPolicy(policy_seed=1000 + i)
    elif isinstance(node, DemandSinkNode):
        ni.policy = node.policy = DefaultDemandSinkPolicy(policy_seed=2000 + i)

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
    stores=[],
    nodes=topology["node_instances"],
    edges=topology["edges"],
    n_steps=100,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)
```

`scenarios/example_llm_world_offline.py` runs this exact path end-to-end with a `CannedClient` (no API key).

## Test seam

`LLMClient` is a `Protocol` (`src/llm/openai_client.py`), so tests inject a fake client implementing `structured_completion(*, system, user, schema)` rather than hitting the network. See `tests/llm/` for examples. `scenarios/example_llm_world_offline.py` drives the same pipeline with a `CannedClient` that pops pre-built Pydantic payloads — useful for inspecting payload shapes and running deterministically with no API key.
