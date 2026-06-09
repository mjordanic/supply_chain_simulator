# `src/tuning/` — Policy hyperparameter tuning

An Optuna-based hyperparameter tuner for any `Policy` subclass. It frames tuning as a measurement
instrument — *how much profit headroom exists above the published textbook defaults?* — rather than
per-world baseline construction. Trials run the policy on a fixed CRN seed set with log-uniform
domain randomisation across capacity and balance; the objective is mean `net_profit / initial_cash`
(dimensionless, scale-comparable across two orders of magnitude of node size).

## Contents

1. [Layout](#layout)
2. [Quickstart](#quickstart)
3. [Output layout](#output-layout)
4. [Programmatic use](#programmatic-use)
5. [Deploying a tuned policy in a multi-echelon graph](#deploying-a-tuned-policy-in-a-multi-echelon-graph)
6. [Analysis notebook](#analysis-notebook)
7. [CRN guarantee](#crn-guarantee)
8. [CLI flags](#cli-flags)

## Layout

```
src/tuning/
  config.py         TuningConfig — every knob the study reads (n_trials, seed offsets,
                    distributions, optional setup_dir)
  episode.py        Config-adapter — unpacks TuningConfig and calls src/sim/episode_sampler.sample_episode
  rollout.py        run_policy_episode(policy, spec) — one episode on the graph engine
                    (factory(s) → IntermediateNode "S" → sink-per-product) via build_world + Simulation.tick
  evaluator.py      evaluate_policy_normalised(factory, specs) — single-policy CRN evaluator
  search_spaces.py  bundled trial-callback factories: order_up_to_space, reorder_point_space,
                    periodic_order_up_to_space, periodic_reorder_space
  study.py          run_study(...) + confirm_top_k(...) + `python -m src.tuning.study` CLI
```

Custom policies write their own ~10-line trial-callback factory (`f(trial) -> Policy`) and pass
it to `run_study`; the four bundled factories cover the textbook variants.

The tuner runs on the **multi-echelon graph engine**: each trial builds a degenerate
`factory → IntermediateNode("S") → sink-per-product` graph from the catalog and attaches the
candidate `MultiSupplierTextbookPolicy` to node `S`. The four bundled search spaces tune the
shared textbook knobs plus two routing tunables — `per_supplier_min_order_floor` (int `[0, 10]`,
the buyer-side min order per line) and `routing_strategy` (categorical: `cheapest_first` /
`fill_rate_weighted`) — alongside each variant's own parameters (`Q`, `review_interval`, …). They
surface as extra columns in `trials.parquet`.

## Quickstart

Run a 150-trial study against a synthetic catalog (no API key, no world cache):

```bash
uv run python -m src.tuning.study \
  --policy order_up_to \
  --trials 150 \
  --study-name order_up_to_v1
```

Run against a setup directory (catalog and market from `setups/my_run/`):

```bash
uv run python -m src.tuning.study \
  --policy order_up_to \
  --trials 150 \
  --study-name order_up_to_v1 \
  --setup-dir setups/my_run
```

Each trial evaluates on 16 CRN seeds (default; `--n-search-seeds`) at 365-tick episodes
(`--episode-length`). Sequential runtime is ~40–80 min on a laptop. To skip the holdout phase,
pass `--skip-holdout`. All flags:

```bash
uv run python -m src.tuning.study --help
```

## Output layout

Artifacts land under `runs/tuning/<study-name>/`:

```
runs/tuning/<study-name>/
  trials.parquet         one row per trial: kpis + sampled params
  per_seed.parquet       long format: one row per (trial, seed) with per-seed KPIs
  study.json             study metadata + search-space spec + best trial + git SHA
  holdout.parquet        top-K winners + published default re-evaluated on holdout seeds
  holdout_summary.json   bootstrap CIs + headline paired uplift (tuned − default)
```

`trials.parquet` columns: `trial_id`, `state`, `datetime_start`, `datetime_complete`,
`duration_s`, `mean_normalised_return`, `mean_net_profit`, `mean_service_level`,
`mean_stockout_rate`, `mean_inventory_turnover`, `mean_revenue`, `mean_mean_price_pct_of_msrp`,
plus one column per sampled hyperparameter.

`per_seed.parquet` columns: `trial_id`, `seed`, `capacity`, `initial_cash`, `net_profit`,
`normalised_return`, `service_level`, `stockout_rate`, `inventory_turnover`, `revenue`,
`mean_price_pct_of_msrp`.

The Parquet files load with just `pandas` + `pyarrow` — no Optuna import needed at read time.

## Programmatic use

```python
from src.tuning import TuningConfig, order_up_to_space, run_study, confirm_top_k
from src.tuning import make_synthetic_catalog

config = TuningConfig(n_trials=150, n_search_seeds=16)
catalog = make_synthetic_catalog(config.K_catalog)

study = run_study(
    order_up_to_space,
    catalog=catalog,
    tuning_config=config,
    study_name="order_up_to_v1",
)

summary = confirm_top_k(
    study,
    order_up_to_space,
    catalog=catalog,
    tuning_config=config,
    study_dir="runs/tuning/order_up_to_v1",
)

print(summary["headline_uplift"])  # paired (tuned − default) on holdout
```

With a setup directory:

```python
from src.tuning import load_catalog_and_market_from_setup

catalog, market = load_catalog_and_market_from_setup("setups/my_run")
config = TuningConfig(n_trials=10, n_search_seeds=4, episode_length=60)
study = run_study(
    order_up_to_space,
    catalog=catalog,
    tuning_config=config,
    study_name="demo",
    output_dir="/tmp/demo_tuning",
    market_params=market,
)
```

## Deploying a tuned policy in a multi-echelon graph

Tuning optimises a single `IntermediateNode` in the degenerate 3-node
`factory → "S" → sink-per-product` graph described above — **never your authored multi-echelon
scenario**. But the result is just an `OrderUpToPolicy` (an ordinary `IntermediatePolicy`), so
deploying it into a real multi-echelon graph is a one-liner: **attach it to whichever node you
want via `Runner`'s `policy_overrides`; every other node keeps its own policy.** Rebuild the
tuned policy from the study's `best_params` with the same search-space factory (this maps the
categorical `routing_strategy` string back to its callable, exactly as `confirm_top_k` does):

```python
import optuna
from src.sim.setup_io import load_setup
from src.sim.runner import Runner
from src.tuning import order_up_to_space

# `study` from run_study(...), or best_params loaded from study.json on disk.
tuned = order_up_to_space(optuna.trial.FixedTrial(study.best_params))

scenario = load_setup("setups/two_factories_two_shops")  # f-lo,f-hi → shop-1,shop-2 → sink-1,sink-2
runner = Runner(scenario, policy_overrides={"shop-1": tuned})  # shop-2 keeps its setup.yaml policy
run_log = runner.run()
```

The tuned policy was searched across log-uniform capacity/balance randomisation and a randomised
SKU assortment (ADR 0009), so it is node-agnostic — the same kwargs drop onto a small shop or a
flagship warehouse. It still decides only from *local* observation (its own inventory / cash /
sales plus the central-table offers from its direct suppliers); there is no joint cross-node
optimisation. `notebooks/05-tune-a-policy.ipynb` demonstrates this end-to-end. (A *trained RL*
actor is not portable this way yet — see `TODO.md` §8.)

## Analysis notebook

`notebooks/05-tune-a-policy.ipynb` walks through a study: optimisation trajectory,
tuned-vs-default headline with bootstrap CIs, parameter-sensitivity scatters, and the best-trial
parameters. It runs a small live demo study end-to-end so the API is exercised without committing
to the full 40–80-minute run.

Headline outputs from a 150-trial study:

**Pareto front — profit vs service level.** Each dot is one trial; red dots are non-dominated;
the gold star is the search winner.

![Pareto front: profit vs service level](../../docs/images/tuning_pareto_front.png)

**Hyperparameter importance (fANOVA).** `safety_lead_pct_of_lag` dominates; `cover_horizon_ticks`
is secondary; `stockout_safety_bonus_pct_of_lag` is near-noise.

![Hyperparameter importance](../../docs/images/tuning_param_importance.png)

**Scale robustness — per-capacity-bucket mean net profit.** Tuned vs published default on the
32 held-out seeds, bucketed into small / medium / large by sampled capacity.

![Per-capacity improvement](../../docs/images/tuning_per_capacity.png)

## CRN guarantee

Every trial in a study evaluates on the same eval-specs list, built once before `study.optimize`
is called and closed over by the trial objective. Two trials sampling identical kwargs on identical
seeds therefore produce bit-identical trajectories — differences between trials are attributable to
the policy kwargs alone. The per-trial CRN tuple includes the graph engine's `allocation` sub-seed
(ADR 0016) so the deterministic per-phase buyer shuffle is held fixed across trials too. The search
seeds (`seed_offset = 12_000_000`) and holdout seeds (`holdout_seed_offset = 13_000_000`) are
disjoint from each other and from the RL training and eval ranges.

## CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--policy` | required | One of `order_up_to`, `reorder_point`, `periodic_order_up_to`, `periodic_reorder` |
| `--study-name` | required | Output sub-directory under `--output-dir` |
| `--trials` | 150 | Number of Optuna trials |
| `--n-search-seeds` | 16 | CRN seeds for the search phase |
| `--n-holdout-seeds` | 32 | CRN seeds for the holdout phase |
| `--seed-offset` | 12_000_000 | First search-phase seed |
| `--sampler-seed` | 42 | Optuna sampler seed |
| `--top-k` | 5 | Number of top trials re-evaluated on holdout |
| `--episode-length` | 365 | Ticks per episode (one calendar year) |
| `--setup-dir` | None | Setup directory; catalog and market loaded from it when set |
| `--output-dir` | runs/tuning | Root output directory |
| `--skip-holdout` | off | Write only the three search artifacts; skip `confirm_top_k` |
