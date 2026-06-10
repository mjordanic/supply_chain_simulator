# M5 Dataset Adapter

`src/datasets/m5.py` — converts raw Kaggle M5 files into a complete, runnable
supply-chain simulator scenario directory in one function call.

## What it does

The M5 dataset records daily unit sales for 3,049 Walmart items across 10
stores and 1,941 days. This adapter:

1. Parses the three raw M5 CSV files (sales, sell prices, calendar).
2. Slices by item IDs, store IDs, and a date window.
3. Expands weekly Walmart prices to daily series with forward/back-fill.
4. Produces a per-(item, store) data-quality report.
5. Emits a complete setup directory loadable by `load_setup` — `catalog.csv`,
   `setup.yaml`, `demand_series.parquet`, `prices.parquet`, `calendar.parquet`,
   and `quality_report.parquet`.

**Declared semantics:** observed M5 sales = true demand. The M5 series records
units sold, which can be stockout-censored. The quality report's `longest_zero_run`
column flags suspected out-of-stock periods so you can pick high-quality slices.

## Downloading the raw files from Kaggle

The raw files are not included in this repository (redistribution rights
unclear). Download them yourself:

```
kaggle competitions download -c m5-forecasting-accuracy
```

Or visit <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>
and download manually.

Extract the archive. You need these three files:

| File | Typical name after extraction |
|------|-------------------------------|
| Sales data | `sales_train_validation.csv` or `sales_train_evaluation.csv` |
| Sell prices | `sell_prices.csv` |
| Calendar | `calendar.csv` |

Place them in any directory, e.g. `data/m5/`. The `data/` directory is
git-ignored, so the files will not be committed.

## Running the adapter

```python
from datetime import date
from src.datasets.m5 import load_m5_slice, emit_m5_setup_dir

# 1. Ingest a slice (small example: 2 items, 1 store, 30 days).
m5 = load_m5_slice(
    data_dir="data/m5/",
    item_ids=["HOBBIES_1_001", "FOODS_3_090"],
    store_ids=["CA_1"],
    start_date=date(2016, 1, 29),
    end_date=date(2016, 2, 27),
)

# 2. (Optional) inspect the quality report before committing to this slice.
from src.datasets.m5 import quality_report
print(quality_report(m5))

# 3. Emit the setup directory (shops + replay sinks + artifacts).
emit_m5_setup_dir(m5, data_dir="data/m5/", setup_dir="data/m5_scenario/")

# 4. Load the scenario and run it.
from src.sim.setup_io import load_setup
from src.sim.runner import Runner

scenario = load_setup("data/m5_scenario/")
log = Runner(scenario).run()
```

The emitted setup directory is a standard ADR 0017 setup dir. You can attach
your own upstream nodes (factories, DCs) and policies to it, exactly as shown
in the example notebook.

## Emitted topology

`emit_m5_setup_dir` writes shops and replay sinks only — no upstream nodes
(factories, distribution centres). Upstream topology is the scenario author's
job. The example notebook (`notebooks/m5_replay_example.ipynb`) shows how to
wire up a minimal upstream and run end-to-end.

- One **shop node** per store (region = M5 state, e.g. `CA`).
- One **replay-demand sink** per (item, store) pair, wired to the shop.

Catalog parameters:

| Field | Derivation |
|-------|-----------|
| `base_price` | Median observed daily price for the item across all stores in the slice |
| `unit_cost` | `unit_cost_fraction × base_price` (default 40 %; override via parameter) |
| `category` | M5 `dept_id` (e.g. `HOBBIES_1`) |
| `seasonality` | `"default"` (inert under a flat-world market) |

Sink `income_rate` is set high enough that the affordability cap never binds —
the simulated cash mechanics will not silently censor replayed demand.

## Example notebook

`notebooks/m5_replay_example.ipynb` — end-to-end walkthrough: raw files →
quality report → multi-echelon CA scenario → run → exact-replay assertion.

## Unit-cost fiction

`unit_cost` is derived as a fraction of `base_price` (default 40 %). This is
an invented cost assumption — M5 does not contain cost data. The fraction is
explicit and overridable via `emit_m5_setup_dir(..., unit_cost_fraction=0.4)`.
