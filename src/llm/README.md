# `src/llm/` — LLM catalog and market generator

An LLM-driven pipeline that drafts a coherent catalog and market from a single domain archetype
string like `"fashion_retail"` or `"sports_cars"`. The output is persisted as `catalog.csv` and
the `market:` block of `setup.yaml` inside a setup directory, which doubles as a local cache —
repeat calls with the same directory skip the LLM entirely. Topology, policies, disruption
parameters, and run parameters are the modeller's domain.

## Contents

1. [Layout](#layout)
2. [The three pipeline stages](#the-three-pipeline-stages)
3. [Sample catalogs](#sample-catalogs)
4. [Schemas and validation](#schemas-and-validation)
5. [Retry on schema failure](#retry-on-schema-failure)
6. [Setup](#setup)
7. [Example](#example)
8. [Test seam](#test-seam)

## Layout

```
src/llm/
  world_builder.py   pipeline orchestrator + build_setup (dir-as-cache entry point)
  schemas.py         Pydantic schemas (Taxonomy, Catalog, Correlations, FreshnessSet,
                     MarketDomain, plus RelatedRef / ItemRelations / ItemFreshness /
                     SeasonWindow / TaxonomyCategory)
  prompts.py         prompt templates with retry support (five stages)
  openai_client.py   OpenAI structured-output client (Protocol + concrete OpenAIClient)
```

`WorldBuilder.build_setup(n_items, setup_dir)` is the canonical entry point. It returns
`(catalog, market)` and, on a cache miss, runs the LLM pipeline and writes `catalog.csv` plus
the `market:` block of `setup.yaml` to the directory. On a cache hit it loads and returns the
existing files without any LLM calls.

## The three pipeline stages

`WorldBuilder(archetype, client).build_setup(n_items, setup_dir)` runs three stages in order:

1. **`build_market_domain_params()`** — one LLM call. The LLM authors a domain-meaningful slice
   (`cycle_len`, `peak_factor`, `off_factor`, `init_demand`, `init_supply`, `season_months`,
   `regions`, `price_elasticity`); `WorldBuilder` merges that with hand-set math defaults to
   produce a complete `MarketParams`. Sets the region list consumed by later stages.
2. **`build_taxonomy()`** — one LLM call. Returns a `Taxonomy` with categories and
   `target_share` weights for the archetype. Invoked transitively by `sample_catalog`.
3. **`sample_catalog(n)`** — four sub-steps:
   1. (Python) `allocate_skeletons(n, taxonomy)` deterministically distributes `n` slots across
      taxonomy categories proportional to `target_share`.
   2. (LLM) one `catalog_prompt` call names and prices the slots (`name`, `category`,
      `base_price`, `unit_cost`, `seasonality`).
   3. (LLM) **chunked** `correlations_prompt` calls author `related_products` against the
      explicit name list — chunk size `correlations_chunk_size` (default `50`); cross-chunk
      references resolve through the prompt's candidate-name list. Dangling / self / duplicate
      refs are dropped at the boundary rather than triggering schema retries.
   4. (LLM) **chunked** `freshness_prompt` calls author per-item `freshness_alpha` /
      `freshness_decay`; chunk size `freshness_chunk_size` (default `50`). Items the LLM omits
      or misnames keep `Ware`'s default `None` for those fields.

   Output is a `list[Ware]` with stable `P{i:04d}` ids assigned by `load_catalog`.

Total LLM calls for `build_setup(n_items, setup_dir)`:
`2 + ceil(n / correlations_chunk_size) + ceil(n / freshness_chunk_size)`
(market, taxonomy, catalog naming, chunked correlations, chunked freshness). For `n=100` with
default chunk sizes that's ~6 calls; for `n=1000` it's ~46.

## Sample catalogs

The excerpts below illustrate what the pipeline produces for two archetypes. The full worlds
are **not** committed (`data/` is git-ignored) — regenerate them with the command in
[Example](#example). The only committed, key-free sample is
`notebooks/example_catalogs/fashion_retail/` (8 items + a `market:` block), which notebook
`01-generate-with-llm.ipynb` uses as its offline fallback.

### `sports_cars_100` (102 items across 7 categories)

| id | name | category | base | cost | season |
|---|---|---|---:|---:|---|
| P0000 | Apex Vector R8 Track Coupe | Track-Ready Coupes | 128,900 | 91,500 | summer |
| P0001 | Vortex RS Sprint Coupe | Track-Ready Coupes | 119,500 | 84,200 | spring/summer |
| P0002 | Carbonline GT3 Aero Coupe | Track-Ready Coupes | 136,800 | 97,200 | summer |
| P0024 | Meridian Grand V12 Tourer | Grand Tourers | 168,900 | 121,000 | fall/winter |
| P0059 | Inferno X Supercar | Exotic Supercars | 315,000 | 224,000 | summer |
| P0073 | Spectra V8 Sport Sedan | Performance Sedans | 82,900 | 58,600 | all_season |
| P0087 | Swiftline 2.0T Coupe | Compact Sport Coupes | 31,900 | 22,300 | all_season |
| P0098 | Heritage Circuit 911 RS | Special Editions & Collector Models | 214,900 | 152,800 | fall |

### `fashion_retail_250` (336 items across 6 categories)

| id | name | category | base | cost | season |
|---|---|---|---:|---:|---|
| P0000 | Women's Essential Crewneck Tee | Women's Apparel | 24.00 | 8.00 | spring/summer |
| P0086 | Men's Classic Oxford Shirt | Men's Apparel | 54.00 | 20.00 | all_season |
| P0151 | Men's Classic Derby Shoes | Footwear | 98.00 | 38.00 | all_season |
| P0192 | Women's Leather Tote Bag | Accessories | 118.00 | 46.00 | all_season |
| P0241 | Kids' Graphic Tee Pack | Kids' Apparel | 24.00 | 7.50 | spring/summer |
| P0292 | Kids' Cotton Briefs 6-Pack | Intimates & Sleepwear | 18.00 | 5.50 | all_season |

## Schemas and validation

Every LLM payload is parsed through a Pydantic model in `src/llm/schemas.py`. Validators currently enforced:

- `CatalogItem.base_price > unit_cost` (positive margin), `base_price > 0`, `unit_cost >= 0`
- `CatalogItem.seasonality` is a closed enum (`spring`, `summer`, `fall`, `winter`, `spring/summer`, `fall/winter`, `all_season`)
- `RelatedRef.correlation ∈ [0, 1]`
- `ItemRelations.related` references that don't resolve against the catalog (or are self-references / within-item duplicates) are pruned post-parse — the schema does not reject them, the boundary sanitiser in `WorldBuilder.sample_catalog` does
- `ItemFreshness.alpha >= 0` and `ItemFreshness.decay > 0` (strictly positive so the consumer never divides by zero)
- `TaxonomyCategory.target_share ∈ (0, 1]`
- `MarketDomain.price_elasticity < 0` (demand must fall with price); `MarketDomain.season_months` is a list of `{name, months}` records with unique names (not a free-form dict — OpenAI strict-output mode rejects open-ended object keys)

## Retry on schema failure

If the LLM response fails Pydantic validation, the validation error string is prepended to the
next prompt so the model can self-correct. Default retry budget is 3 attempts per stage,
configurable via `WorldBuilder(max_retries=...)`. Other exceptions (transport errors, refusals)
propagate immediately.

## Setup

The shipped client uses OpenAI's structured outputs API (`client.beta.chat.completions.parse`):

- Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAIClient(...)`.
- Default model: `gpt-5.4-mini` (see `OpenAIClient.__init__`). Override via `OpenAIClient(model="…")`.
- Pass a custom `OPENAI_BASE_URL` only when routing through a proxy / compatible endpoint — a stale value pointing at localhost is the most common source of `APIConnectionError`.

## Example

Generate a catalog and market, persist to a setup directory, then scaffold nodes/edges:

```python
from src.llm.world_builder import WorldBuilder
from src.llm.openai_client import OpenAIClient

builder = WorldBuilder(archetype="fashion_retail", client=OpenAIClient())

# On first call: runs the LLM pipeline and writes catalog.csv + market block.
# On subsequent calls: loads from disk, no LLM calls.
catalog, market = builder.build_setup(n_items=50, setup_dir="setups/fashion_retail")

# catalog  → list[Ware]
# market   → MarketParams
```

The same generation as a one-shot shell command — e.g. to create the 200-item setup used by
the RL training quickstart in [`src/rl/README.md`](../rl/README.md#quickstart):

```bash
export OPENAI_API_KEY=sk-...
uv run python - <<'EOF'
from src.llm.world_builder import WorldBuilder
from src.llm.openai_client import OpenAIClient

WorldBuilder(archetype="fashion_retail", client=OpenAIClient()).build_setup(
    n_items=200, setup_dir="setups/fashion_retail"
)
EOF
```

For RL training that directory is complete as-is — `src.rl.train --setup-dir` reads only
`catalog.csv` and the `market:` block. The scaffold step below is needed only when you want
to run the setup through `main.py run` (which requires `nodes:`/`edges:`).

Then either hand-author the `nodes:` and `edges:` blocks in `setup.yaml`, or use the scaffolder:

```bash
uv run python main.py scaffold setups/fashion_retail/catalog.csv \
  --out setups/fashion_retail/setup.yaml \
  --shop-count 2 --sink-density 0.5
```

Run it:

```bash
uv run python main.py run setups/fashion_retail
```

## Test seam

`LLMClient` is a `Protocol` (`src/llm/openai_client.py`), so tests inject a fake client
implementing `structured_completion(*, system, user, schema)` rather than hitting the network.
See `tests/llm/` for examples.
