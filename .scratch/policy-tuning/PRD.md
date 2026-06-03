# PRD: Policy hyperparameter tuning tool + textbook-safety reparameterisation

Status: ready-for-agent

Related ADRs:
- [ADR 0006](../../docs/adr/0006-textbook-reorder-policy-family.md) — Textbook reorder-policy family (establishes the canonical baseline this PRD tunes against)
- [ADR 0008](../../docs/adr/0008-safety-horizons-as-fraction-of-delivery-lag.md) — Reparameterise textbook safety horizons as fractions of delivery lag (prerequisite for tuning)
- [ADR 0009](../../docs/adr/0009-policy-hyperparameter-tuning-tool.md) — Policy hyperparameter tuning tool (Optuna-based)

## Problem Statement

The textbook reorder-policy family (ADR 0006) ships with four hand-picked defaults on `OrderUpToPolicy`: `cover_horizon_ticks=10`, `safety_lead_ticks=2`, `opening_budget_pct=0.50`, `stockout_safety_bonus_ticks=0`. These defaults were chosen from textbook references and confirmed to produce a profitable trajectory at the canonical CLI scale (`delivery_lag=3`). Two follow-up problems are open:

1. **The published defaults are not measured against a search.** Nobody knows how much profit is being left on the table relative to the best achievable `OrderUpToPolicy` kwargs across the same log-uniform domain randomisation the RL training sees. The RL paired-uplift number in `notebooks/06-compare_rl_vs_baseline.ipynb` is the difference between RL and the published defaults; whether RL is beating "the best (s,S) can do" or "the textbook defaults" is not currently distinguishable.

2. **Future custom policies (the user's stated long-term goal — "a tool for tuning hyperparameters of policies", plural) have nowhere to plug in.** A bespoke tuning script tied to `OrderUpToPolicy` would not compose with a new policy added next quarter. The codebase needs a tuning module with a stable interface — pass in a `Policy` factory and a search space, get back a study artifact.

A subordinate problem surfaces during design: `safety_lead_ticks` is an absolute integer applied uniformly across SKUs, but `Store.observe()` already exposes per-pid `delivery_lags` and the policy already reads them per pid. On any catalog with heterogeneous lead times (LLM-built worlds where per-category lag is authored independently), `safety_lead_ticks=2` translates to a 200 %-of-lag safety on a `lag=1` SKU and a 20 %-of-lag safety on a `lag=10` SKU — exactly backwards from the textbook newsvendor formula. The bug is latent on hand-authored worlds with uniform lag, surfaces on LLM worlds, and becomes load-bearing the moment a tuner is asked to search over the parameter.

## Solution

Two work bundles, landed in order:

**Bundle A — Reparameterise textbook safety horizons as fractions of delivery lag (ADR 0008).** Rename `safety_lead_ticks` → `safety_lead_pct_of_lag` (float) and `stockout_safety_bonus_ticks` → `stockout_safety_bonus_pct_of_lag` (float). Compute per-pid inside `TextbookReorderPolicy.decide()` by multiplying the ratio by that pid's `delivery_lag`. Keep `cover_horizon_ticks` absolute (EOQ cycle length is independent of lead time). Hard rename — no compatibility alias. New defaults are `safety_lead_pct_of_lag = 2/3 ≈ 0.667` and `stockout_safety_bonus_pct_of_lag = 0.0`, producing bit-identical trajectories at the canonical `lag=3` scale. Class docstring gains an "EOQ vs safety stock" theoretical block explaining the choice of denominators. All call sites migrate (~6 files: `src/sim/policy.py`, `src/rl/eval.py`, `src/rl/train.py`, notebooks 06 and 07, regression-snapshot regen, tests).

**Bundle B — Optuna-based policy tuning tool at `src/tuning/` + demonstration notebook (ADR 0009).** Four-file module: `config.py` (`TuningConfig` dataclass), `evaluator.py` (single-policy CRN evaluator returning normalised return + KPI dict — the deep, testable module), `study.py` (`run_study()` + `confirm_top_k()` entry points + artifact serialisation), `search_spaces.py` (four bundled trial-callback factories for the four textbook policies). Pattern A (trial-callback) API. Objective is mean `net_profit / initial_cash` across 16 CRN seeds (disjoint from training and from the two-scale eval ranges) at 365 ticks per episode under log-uniform capacity + balance domain randomisation. Optuna `TPESampler` with fixed seed; sequential trials; no pruner; in-memory study with file-based artifact dump at the end. Three artifact files per study at `runs/tuning/<study_name>/`: `trials.parquet`, `per_seed.parquet`, `study.json` — plus `holdout.parquet` + `holdout_summary.json` from the top-K re-evaluation. New optional dependency: `optuna`.

The accompanying notebook `notebooks/08-tune_textbook_policy.ipynb` follows framing α (tutorial + analysis hybrid): a short inline cell runs a ~10-trial demo study live so the API is visible, then the remaining sections load `runs/tuning/order_up_to_v1/` from disk. The shipped notebook contains *no* committed full-study fixture — the user must run the full optimisation themselves (one CLI invocation, ~40–80 minutes) before the headline analysis cells show meaningful results. Cells that depend on the on-disk study artifact display a friendly message and the CLI invocation to populate the directory when the file is missing.

The result is a measurement instrument that answers "how much headroom does `OrderUpToPolicy` have above its published defaults?" without changing what the published defaults are. The canonical RL baseline trajectory at the standard CLI scale is bit-identical to pre-PRD behaviour; the trajectory on heterogeneous-lag worlds differs (this is the per-SKU bug being fixed).

## User Stories

1. As an RL researcher reading `notebooks/06-compare_rl_vs_baseline.ipynb`, I want to know whether the published `OrderUpToPolicy` defaults are anywhere near the best achievable (s,S) kwargs on the same evaluation distribution, so that the paired uplift I report distinguishes "RL beats the textbook defaults" from "RL beats the best textbook policy".
2. As an inventory researcher with a custom `Policy` subclass, I want a tuning module that accepts a `Policy` factory and an Optuna search space without forcing me to write the eval loop, so that adding a new policy and tuning it is a ten-line script rather than a copy-paste of `src/rl/eval.py`.
3. As an inventory researcher who wants to compare the four textbook variants on equal footing, I want bundled trial-callback factories for `OrderUpToPolicy`, `ReorderPointPolicy`, `PeriodicOrderUpToPolicy`, and `PeriodicReorderPolicy`, so that running a study on any of them is one import.
4. As a scenario author whose catalog has heterogeneous lead times across SKUs (e.g. an LLM-built fashion world where shoes have lag 7 and ready-made apparel has lag 2), I want `safety_lead_pct_of_lag` to produce a per-SKU safety horizon that scales *with* lead time, so that fast-supply SKUs aren't over-protected and slow-supply SKUs aren't under-protected.
5. As a scenario author running the existing CLI template (uniform `delivery_lag=3`), I want the new `safety_lead_pct_of_lag=2/3` default to produce a bit-identical trajectory to the old `safety_lead_ticks=2` default, so that the migration does not silently shift the RL CRN baseline.
6. As a CI maintainer, I want existing `OrderUpToPolicy(safety_lead_ticks=2)` call sites to fail loudly at construction time (TypeError) rather than silently accept and ignore the old kwarg, so that the migration is auditable from grep.
7. As an RL researcher running `src/rl/eval.py::evaluate`, I want the `evaluate(rl_policy_fn, baseline_policy_factory, …)` injection signature to remain unchanged after the rename, so that the RL eval contract is preserved.
8. As an RL researcher running `src/rl/train.py`, I want the baseline factory body to migrate to `OrderUpToPolicy()` with no explicit kwargs (defaults only) and produce the same training-loop behaviour at the standard scale, so that the change is a one-line edit.
9. As the maintainer of `tests/sim/test_regression_snapshot.py`, I want the pinned trajectory regenerated against the new kwarg names and confirmed bit-identical at the standard scale, so that any rounding flip on other lead times surfaces as a deliberate review item rather than silent drift.
10. As a tool user running my first study, I want `python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1` to be a single CLI invocation that produces `runs/tuning/order_up_to_v1/` with all four artifact files, so that I do not need to write Python to use the tool.
11. As a tool user inspecting a study after the fact, I want `pd.read_parquet("runs/tuning/order_up_to_v1/trials.parquet")` and `pd.read_parquet(".../per_seed.parquet")` to load cleanly into pandas without an Optuna dependency at read time, so that downstream analysis is portable.
12. As a notebook reader opening `notebooks/08-tune_textbook_policy.ipynb` for the first time, I want a short inline demo cell that runs a ~10-trial study live in ~1–2 minutes, so that I can see the tuning API execute without committing to the full 40–80-minute run.
13. As the same notebook reader, I want the remaining analysis cells to load from `runs/tuning/order_up_to_v1/` when present and display a clear "run `python -m src.tuning.study ...` to populate" message when absent, so that the notebook degrades gracefully rather than erroring.
14. As a Pareto-front-curious researcher, I want the notebook's section 8 to read all 150 trials' `(mean_service_level, mean_net_profit)` pairs from `per_seed.parquet`, plot a scatter, and highlight the Pareto front, so that I can see *which* tuned points trade profit for service level vs. which dominate on both.
15. As a scale-robustness sceptic, I want the notebook's section 9 to bucket `per_seed.parquet` rows by capacity (small / medium / large via the LogUniform sample) and show tuned-vs-default per bucket, so that the "scale-invariant" claim of ADR 0006 is empirically stress-tested.
16. As a parameter-sensitivity analyst, I want the notebook's section 6 to scatter `(param value, mean_normalised_return)` for each of the 4 tunables, coloured by trial order, so that I can see where TPE concentrated and which dimensions matter most.
17. As an Optuna user, I want the notebook's section 7 to compute `optuna.importance.get_param_importances` on a reconstructed in-memory study, so that fANOVA-style importance bars are available without persisting an `optuna_dashboard`-compatible SQLite file.
18. As a study runner, I want `study.json` to capture the study name, n_trials, n_search_seeds, episode_length, sampler_seed, the four search-space ranges, the best trial id, the best params dict, the start/finish datetimes, and the git SHA at run time, so that any artifact directory is self-describing for provenance auditing.
19. As a study runner wanting confidence in the search winner, I want `confirm_top_k(study, k=5, n_seeds=32)` to re-evaluate the top-5 search trials on a 32-seed held-out set disjoint from the 16 search seeds, write `holdout.parquet`, and produce a one-line summary in `holdout_summary.json` comparing the held-out tuned mean to the published default's held-out mean, so that I can quote a defensible final number that wasn't selected on the search seeds.
20. As a future custom-policy author, I want to write my own trial-callback factory inline (calling `trial.suggest_int`, `trial.suggest_float`, `trial.suggest_categorical` directly) and pass it to `run_study(policy_space=...)`, so that the four bundled factories are convenient shortcuts but not the only path.
21. As a maintainer worried about creating a redundant baseline, I want the tuning tool to be presented as a *measurement instrument* (framing 1, ADR 0009) and explicitly *not* as a way to construct the canonical RL CRN baseline, so that the ADR 0006 commitments are preserved.
22. As a developer adding a fifth bundled space factory in a follow-up, I want `src/tuning/search_spaces.py` to be a small file with one factory per textbook policy following an obvious convention, so that adding `eoq_space` or `base_stock_space` is a tens-of-lines patch.
23. As a developer testing the tuning evaluator in isolation, I want `src/tuning/evaluator.py` to expose a pure function `evaluate_policy_normalised(policy_factory, eval_specs, config) -> dict[str, Any]` that takes pre-built `EpisodeSpec` objects, so that a unit test can call it on a 2-seed synthetic episode list without launching Optuna.
24. As a developer worried about world-trajectory determinism, I want every tuning trial on the same seed to produce a bit-identical world trajectory regardless of which kwargs Optuna sampled, so that any observed difference in net profit across trials is attributable to the policy alone (CRN guarantee at the trial level).
25. As a user trying to fit a study in a CI window, I want `TuningConfig` to expose all the major dials (`n_trials`, `n_search_seeds`, `n_holdout_seeds`, `episode_length`, `seed_offset`, `sampler_seed`) so that I can shrink a study for smoke-testing.
26. As a `CONTEXT.md` reader unfamiliar with the codebase, I want a new "Policy tuning study" glossary entry that names the artifacts, the objective, and the framing-1 scope, so that I can understand the tool's purpose without reading ADR 0009 first.
27. As a developer worried about transitive dependency bloat, I want `optuna` added via `uv add optuna` (single direct dependency, pulls in scipy + sqlalchemy + a handful of small deps), so that the addition is auditable and easy to remove if the tool turns out unused.
28. As a tester writing `tests/tuning/`, I want the test directory's structure to mirror the source structure (`test_config.py`, `test_evaluator.py`, `test_study.py`, `test_search_spaces.py`) so that locating tests is mechanical.
29. As a Bundle-A migrator, I want to verify after the rename that `OrderUpToPolicy()` (defaults), `OrderUpToPolicy(safety_lead_pct_of_lag=0.0)`, and `OrderUpToPolicy(safety_lead_pct_of_lag=3.0)` all construct cleanly and produce monotonic-in-safety inventory trajectories on a synthetic test, so that the new parameter behaves intuitively across its range.
30. As a CRN-paired-eval consumer, I want `OrderUpToPolicy` after Bundle A to continue consuming *only* `policy_rng` (never `world_rng` or `init_rng`), so that the CRN-disjoint property of paired evaluation is preserved.

## Implementation Decisions

### Bundle ordering and atomicity

Bundle A (rename) lands first as a single PR. Bundle B (tuning tool) lands second as a second PR, after Bundle A is on `main`. Rationale: Bundle B's search-space definitions reference the post-Bundle-A kwarg names (`safety_lead_pct_of_lag`), so landing them out of order would require an interim shim. Atomic PRs avoid this.

### Bundle A — Reparameterisation

**Signature change on `TextbookReorderPolicy.__init__`:**

Before (ADR 0006):
```python
def __init__(
    self,
    *,
    policy_seed: int | None = None,
    cover_horizon_ticks: int = 10,
    safety_lead_ticks: int = 2,
    opening_budget_pct: float = 0.50,
    stockout_safety_bonus_ticks: int = 0,
    min_qty: int = 0,
) -> None:
```

After (ADR 0008):
```python
def __init__(
    self,
    *,
    policy_seed: int | None = None,
    cover_horizon_ticks: int = 10,
    safety_lead_pct_of_lag: float = 2 / 3,
    opening_budget_pct: float = 0.50,
    stockout_safety_bonus_pct_of_lag: float = 0.0,
    min_qty: int = 0,
) -> None:
```

No compatibility alias for the renamed kwargs.

**Per-pid computation inside `decide()`:**

```
effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
effective_bonus_ticks  = round(stockout_safety_bonus_pct_of_lag × delivery_lag[pid])  # when stockout-detected
s = (delivery_lag[pid] + effective_safety_ticks + effective_bonus_ticks) × rate
S = s + cover_horizon_ticks × rate
```

The two shared helpers in `src/sim/policy.py` (`_estimate_rate`, `_allocate_two_pass_fair_share`) are unchanged.

**Docstring on `TextbookReorderPolicy`:**

The class docstring gains a top-level block titled "Why these denominators" with three paragraphs:
- EOQ result for cycle length (`T* = √(2K/(D·h))`) — lead time does not appear, so `cover_horizon_ticks` is absolute.
- Textbook newsvendor result for safety stock (`SS = z·σ·√L`) — safety scales with lead time, so the safety horizons are expressed as fractions of `delivery_lag`.
- Linear approximation (`SS ≈ k·L·rate`) preserves the qualitative invariant without introducing a new mechanism; matches the rest of the policy's `ticks × rate` structure.

**Migration call sites (~6 files):**

- `src/sim/policy.py` — `TextbookReorderPolicy.__init__`, `TextbookReorderPolicy.decide()` body, docstring.
- `src/rl/eval.py` — type-hint imports only (no explicit kwarg passing in current code).
- `src/rl/train.py` — type-hint imports only.
- `notebooks/06-compare_rl_vs_baseline.ipynb` — `baseline_factory()` body if it overrides; markdown referencing "safety lead ticks".
- `notebooks/07-rl_vs_baseline_per_product.ipynb` — same.
- `tests/sim/test_regression_snapshot.py` — regenerate the pinned trajectory; assert bit-identical at the standard CLI `lag=3` scale.

Any test that instantiates `OrderUpToPolicy(safety_lead_ticks=…)` or `OrderUpToPolicy(stockout_safety_bonus_ticks=…)` flips to the new kwarg names. Grep is the authoritative migration tool.

### Bundle B — Tuning tool

**Module layout `src/tuning/`:**

```
src/tuning/
├── __init__.py        # public API exports
├── config.py          # TuningConfig dataclass
├── study.py           # run_study(), confirm_top_k(), artifact serialisation, __main__ CLI
├── evaluator.py       # evaluate_policy_normalised() — deep module
└── search_spaces.py   # order_up_to_space(trial), reorder_point_space(trial), periodic_order_up_to_space(trial), periodic_reorder_space(trial)
```

**Deep module: `evaluator.py`.**

```python
def evaluate_policy_normalised(
    policy_factory: Callable[[], Policy],
    eval_specs: list[EpisodeSpec],
    *,
    config: RLConfig,
) -> dict[str, Any]:
    """Run a single policy on eval_specs, return normalised mean + per-seed KPI lists.

    Returns a dict with keys:
        mean_normalised_return: float  # mean of (net_profit / initial_cash) across seeds
        per_seed_net_profit: list[float]
        per_seed_initial_cash: list[float]
        per_seed_capacity: list[float]
        per_seed_service_level: list[float]
        per_seed_stockout_rate: list[float]
        per_seed_inventory_turnover: list[float]
        per_seed_revenue: list[float]
        per_seed_mean_price_pct_of_msrp: list[float]
        mean_*: float for each per_seed_* key
    """
```

This is the testable seam: takes pre-built `EpisodeSpec` objects, returns a flat dict. No Optuna, no I/O. Internally reuses `_build_world()` from `src/rl/eval.py` (lift it to a shared private helper if needed; the existing function is suitable for module-internal reuse).

**Public API in `src/tuning/__init__.py`:**

```python
from src.tuning.config import TuningConfig
from src.tuning.evaluator import evaluate_policy_normalised
from src.tuning.search_spaces import (
    order_up_to_space,
    reorder_point_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
)
from src.tuning.study import run_study, confirm_top_k

__all__ = [
    "TuningConfig",
    "evaluate_policy_normalised",
    "order_up_to_space",
    "reorder_point_space",
    "periodic_order_up_to_space",
    "periodic_reorder_space",
    "run_study",
    "confirm_top_k",
]
```

**`TuningConfig` dataclass (`config.py`):**

```python
@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 150
    n_search_seeds: int = 16
    n_holdout_seeds: int = 32
    episode_length: int = 365
    seed_offset: int = 12_000_000   # disjoint from training (default 0) and from two-scale eval (10_000_000 + 1_000_000)
    holdout_seed_offset: int = 13_000_000
    sampler_seed: int = 42
    top_k_for_holdout: int = 5
```

All fields immutable, all defaults documented.

**`run_study()` signature (`study.py`):**

```python
def run_study(
    policy_space: Callable[[optuna.Trial], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_name: str,
    output_dir: str = "runs/tuning",
) -> optuna.Study:
```

Returns the live in-memory `optuna.Study` so the caller (notebook section 2 demo cell) can inspect it without re-reading from disk. Side-effect: writes `trials.parquet`, `per_seed.parquet`, `study.json` under `{output_dir}/{study_name}/`. The Optuna study itself uses `storage=None` (`InMemoryStorage`).

**`confirm_top_k()` signature:**

```python
def confirm_top_k(
    study: optuna.Study,
    policy_space: Callable[[optuna.Trial], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_dir: str,  # absolute path under runs/tuning/<study_name>/
) -> dict[str, Any]:
```

Loads the top-`tuning_config.top_k_for_holdout` trials by `study.best_trial`-style rank, re-instantiates each policy via `policy_space` and `optuna.trial.FixedTrial(params)`, runs them on `tuning_config.n_holdout_seeds` CRN seeds starting at `tuning_config.holdout_seed_offset`, plus runs the published default (`OrderUpToPolicy()` with no kwargs) on the same seeds for comparison. Writes `holdout.parquet` (one row per (top-k policy + default, seed) = `(top_k + 1) × n_holdout_seeds` rows) + `holdout_summary.json` (mean ± bootstrap-CI for each, plus the headline "tuned vs default" pair).

**Bundled trial-callback factories (`search_spaces.py`):**

```python
def order_up_to_space(trial: optuna.Trial) -> Policy:
    from src.sim.policy import OrderUpToPolicy
    return OrderUpToPolicy(
        cover_horizon_ticks=trial.suggest_int("cover_horizon_ticks", 1, 30),
        safety_lead_pct_of_lag=trial.suggest_float("safety_lead_pct_of_lag", 0.0, 3.0),
        opening_budget_pct=trial.suggest_float("opening_budget_pct", 0.05, 0.95),
        stockout_safety_bonus_pct_of_lag=trial.suggest_float("stockout_safety_bonus_pct_of_lag", 0.0, 2.0),
    )
```

`reorder_point_space`, `periodic_order_up_to_space`, `periodic_reorder_space` follow the same shape, with each variant's extra kwargs (`Q` for `ReorderPointPolicy`, `review_interval` for the periodic variants) added.

**Artifact schemas:**

`trials.parquet` columns: `trial_id, state, datetime_start, datetime_complete, duration_s, mean_normalised_return, mean_net_profit, mean_service_level, mean_stockout_rate, mean_inventory_turnover, mean_revenue, mean_mean_price_pct_of_msrp` + one column per search-space parameter (`cover_horizon_ticks`, `safety_lead_pct_of_lag`, `opening_budget_pct`, `stockout_safety_bonus_pct_of_lag`).

`per_seed.parquet` columns: `trial_id, seed, capacity, initial_cash, net_profit, normalised_return, service_level, stockout_rate, inventory_turnover, revenue, mean_price_pct_of_msrp`. Long format — one row per (trial, seed) → `n_trials × n_search_seeds` rows total.

`study.json` fields: `study_name`, `n_trials`, `n_search_seeds`, `n_holdout_seeds`, `episode_length`, `seed_offset`, `holdout_seed_offset`, `sampler_seed`, `search_space` (dict of param name → `{type, low, high}`), `best_trial_id`, `best_params`, `best_value`, `started_at`, `finished_at`, `git_sha` (subprocess `git rev-parse HEAD`; empty string if unavailable), `policy_class_name` (string introspected from the factory).

`holdout.parquet` columns: same as `per_seed.parquet` plus `label` (one of `"tuned_rank_1"`, …, `"tuned_rank_5"`, `"default"`).

`holdout_summary.json` fields: `tuned_best_label`, `tuned_best_mean_net_profit`, `tuned_best_ci_low`, `tuned_best_ci_high`, `default_mean_net_profit`, `default_ci_low`, `default_ci_high`, `headline_uplift`, `headline_uplift_ci_low`, `headline_uplift_ci_high`, `n_holdout_seeds`.

**CLI entry point:**

`src/tuning/study.py` has a `__main__` block:

```
uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1
```

Flags: `--policy {order_up_to|reorder_point|periodic_order_up_to|periodic_reorder}` (selects which bundled factory), `--trials <int>` (overrides `TuningConfig.n_trials`), `--study-name <str>` (required, used in the output path), `--world <archetype>` (default `rl_train`), `--n-search-seeds`, `--n-holdout-seeds`, `--seed-offset`, `--sampler-seed`, `--top-k`, `--skip-holdout` (default false). The CLI invocation also calls `confirm_top_k` by default; `--skip-holdout` lets a smoke run skip the 32-seed re-evaluation.

**Notebook structure (`notebooks/08-tune_textbook_policy.ipynb`):**

Section layout (10 sections per the design session):
1. Setup & motivation (imports + framing-1 link to ADR 0009)
2. The tool in one screen + live 10-trial demo
3. Load pre-run study from `runs/tuning/order_up_to_v1/` (graceful-failure cell if absent)
4. Optimization trajectory plot
5. Headline: tuned vs published default (paired CRN on held-out seeds)
6. Parameter sensitivity scatters (4-panel)
7. Param importances (replay trials into in-memory Optuna study, call `optuna.importance.get_param_importances`)
8. Pareto front: profit vs service level (scatter + frontier highlight)
9. Scale robustness: per-capacity-bucket paired bar
10. Closing markdown

No committed full-study fixture. The shipped `.ipynb` includes executed outputs from when the implementing agent ran the study locally; future readers either inherit those outputs or re-run after producing their own artifacts.

**Optuna dependency:**

`uv add optuna` in Bundle B. Direct dep only; transitive deps (alembic, colorlog, packaging, PyYAML, scipy, sqlalchemy, tqdm) are accepted as small. No `optuna-dashboard` — no SQLite, no dashboard server.

**Determinism / CRN invariants:**

- The same `seed_offset` + same `n_search_seeds` produce identical `eval_specs` across all trials of all studies (the EpisodeSpec list is built once at the start of `run_study()` and reused for every trial's `evaluate_policy_normalised` call).
- `optuna.samplers.TPESampler(seed=config.sampler_seed)` makes the TPE proposal sequence deterministic.
- `Policy` instances built by trial-callback factories never consume `world_rng` or `init_rng`; their `policy_seed` is fixed at the default (None → 0 inside `Policy.__init__`) so two trials sampling identical params produce identical trajectories on identical seeds.

### CONTEXT.md updates

1. `TextbookReorderPolicy family` glossary entry: kwarg renames (`safety_lead_ticks` → `safety_lead_pct_of_lag`, `stockout_safety_bonus_ticks` → `stockout_safety_bonus_pct_of_lag`), per-pid computation, default-equivalence note at `lag=3`, ADR 0008 cross-reference.
2. `CRN-paired eval` glossary entry: one-sentence addition that the single-policy half of the machinery is reused by the policy tuning study with a normalised-return objective.
3. New `Policy tuning study` glossary entry: framing-1 scope, normalised objective, log-uniform domain randomisation, artifact layout, link to ADR 0009.
4. Decisions list: ADR 0008 and ADR 0009 entries.

## Testing Decisions

### What makes a good test here

Test contracts, not internals. The tuning module's external contract is "given a policy factory and a search space, produce reproducible artifacts that capture the search-space best on a fixed CRN seed set." A test that asserts "the artifact files exist and parse cleanly" is testing the contract; a test that asserts "TPESampler was instantiated with seed=42" is testing the implementation. Prefer the former.

The deep module `evaluator.py::evaluate_policy_normalised` is the seam where most real testing leverage lives — it is pure-ish (takes EpisodeSpecs, returns a dict; no Optuna, no I/O), so synthetic 2-seed unit tests are cheap and informative. The `study.py` orchestration is mostly plumbing; smoke-test that a 3-trial study produces the right artifact files with the right schema.

### Test modules

**`tests/sim/` extensions for Bundle A (Reparameterisation):**

1. `tests/sim/test_textbook_policy.py::test_safety_pct_of_lag_per_sku` — Construct an `OrderUpToPolicy` on a synthetic catalog with two SKUs at `lag=1` and `lag=10`; drive a Store through 20 ticks of constant demand; assert `s` for the lag-10 SKU is roughly 5× the `s` for the lag-1 SKU (10·rate + 6.7·rate vs 1·rate + 0.67·rate ≈ 16.7·rate vs 1.67·rate); confirms per-SKU correctness.
2. `tests/sim/test_textbook_policy.py::test_safety_pct_of_lag_default_equivalence` — Construct `OrderUpToPolicy()` (defaults) on the canonical `lag=3` template; drive a Store through 50 ticks; assert the resulting `decide()` output dict is bit-identical to the pre-migration `OrderUpToPolicy(safety_lead_ticks=2)` baseline (this test is its own snapshot; if it ever drifts, the rounding policy in the new code changed).
3. `tests/sim/test_textbook_policy.py::test_no_safety_lead_ticks_kwarg` — `OrderUpToPolicy(safety_lead_ticks=2)` raises `TypeError` (hard rename, no compatibility alias).
4. `tests/sim/test_regression_snapshot.py` — regenerated pinned trajectory; the existing assertion harness is unchanged.

**`tests/tuning/` for Bundle B (Tuning tool):**

5. `tests/tuning/test_evaluator.py::test_evaluate_policy_normalised_2_seeds` — Build 2 `EpisodeSpec` objects with known seeds; call `evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs, config=RLConfig())`; assert the returned dict has all expected keys, all `per_seed_*` lists have length 2, `mean_normalised_return == mean(per_seed_normalised_return)`, and re-calling on the same inputs returns bit-identical numbers (CRN determinism).
6. `tests/tuning/test_evaluator.py::test_normalisation_is_dimensionless` — Build two specs with capacity ratio 10× and balance ratio 10×; assert that on a no-op policy (returns zero orders → profit is purely from initial cash decay), the per-seed `normalised_return` is approximately equal between the two specs (within 1 % relative) — confirming the normalisation actually cancels scale.
7. `tests/tuning/test_search_spaces.py::test_order_up_to_space_constructs` — Call `order_up_to_space(optuna.trial.FixedTrial({"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.5, "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0}))`; assert it returns an `OrderUpToPolicy` instance with the expected kwargs.
8. `tests/tuning/test_search_spaces.py::test_all_four_factories_construct` — Same for the other three bundled factories.
9. `tests/tuning/test_study.py::test_run_study_smoke_3_trials` — Run a 3-trial study with `n_search_seeds=2`, `episode_length=10`, write artifacts to a tmp dir, assert the three files (`trials.parquet`, `per_seed.parquet`, `study.json`) exist and have the expected schemas (column names + row counts).
10. `tests/tuning/test_study.py::test_run_study_artifacts_load_with_pandas_only` — After running the smoke study, re-open the three artifact files using only `pandas` and `json` (no `optuna` import); assert the schemas are usable without Optuna at read time.
11. `tests/tuning/test_study.py::test_confirm_top_k_smoke` — Run a 5-trial study with `n_holdout_seeds=2`; call `confirm_top_k(study, top_k=2, ...)`; assert `holdout.parquet` exists with 2×(2+1) = 6 rows (2 top-k + 1 default × 2 seeds) and `holdout_summary.json` has the documented fields.
12. `tests/tuning/test_config.py::test_seed_offsets_are_disjoint` — Assert `TuningConfig.seed_offset` is disjoint from `RLConfig.eval_seed_offset` and from the two-scale eval ranges (a one-line invariant test that protects against accidentally tuning on training seeds).

**Existing test that needs an audit (no expected change):**

13. `tests/rl/test_eval_crn.py` — Existing CRN self-comparison tests. Re-run after Bundle A to confirm the rename did not break the eval contract. No assertion changes.

### Prior art

- `tests/sim/test_baseline_policy.py` (post-ADR-0006 renamed to `test_heuristic_policy.py`) and `tests/sim/test_textbook_policy.py` — observation-shape contract tests, kwargs round-trip, RNG-isolation checks. The Bundle A tests follow the same shape.
- `tests/rl/test_eval_crn.py` — CRN determinism harness: same world_seed produces bit-identical trajectories. The Bundle B `test_run_study_smoke_3_trials` reuses this pattern at the study level.
- `src/rl/eval.py` `_build_world` and `_run_baseline` — internal reuse target for `src/tuning/evaluator.py::evaluate_policy_normalised`. The eval module's single-policy rollout loop is suitable for lifting into a private shared helper if needed.

### Tests not written

- No direct test of `TPESampler` behaviour. Optuna is a tested dependency; we trust it produces a reasonable proposal distribution.
- No test that the *best* trial after a 150-trial run beats the published default. The PRD does not claim "tuning will find a better config" — it claims "the tool exists and produces measurable artifacts." A passing-result test on a real run would couple the test suite to a stochastic outcome.
- No test for `optuna.importance.get_param_importances`. Computed in the notebook only; covered by Optuna's own test suite.
- No notebook-execution test. Notebook execution gates have been brittle in this repo; the analysis cells are exercised via the unit tests on the underlying functions.

## Out of Scope

- **Per-world or per-scale tuning** (framings 2 and 3 in ADR 0009). Recorded explicitly in the ADR as "not what v1 ships". A future framing-2 capability would add a `--scenario` flag to the CLI and a per-world config dimension; not in this PRD.
- **Multi-objective Optuna (NSGA-II).** The objective is single-scalar mean normalised return. Pareto fronts are reconstructed *post-hoc* in the notebook from `per_seed.parquet`; the optimiser is not asked to find them.
- **Pruning.** No `MedianPruner` or `HyperbandPruner` in v1. Adding a pruner requires intermediate-value reporting per seed inside a trial, restructuring the eval loop. Deferred until the study budget grows past the sub-hour range.
- **Parallel trials (`n_jobs > 1`).** The simulator subsystems are not audited for thread safety. Per-trial parallelism stays sequential in v1. The natural place to add concurrency later is *within* a trial (the 16 seeds are embarrassingly parallel); deferred.
- **SQLite-backed Optuna study.** No `optuna-dashboard`, no mid-study resume. If a long study crashes, re-run. Swapping in SQLite is a one-line change at the `optuna.create_study()` call site when needed.
- **Tuning the other three textbook variants.** Bundled factories ship for all four (`order_up_to_space`, `reorder_point_space`, `periodic_order_up_to_space`, `periodic_reorder_space`), but the notebook example targets `OrderUpToPolicy` only — it is the canonical RL CRN anchor. Studies on the other three variants are a follow-up activity.
- **Committed full-study fixture in the repo.** Per user direction: `runs/` is gitignored; the user wants every reader to execute the full optimisation themselves before seeing real headline numbers. The notebook ships with a tiny inline demo and graceful "run the CLI first" messaging on the analysis cells.
- **Auto-update of published defaults if the tuner finds a better config.** Out of scope — that would require a follow-up ADR amending or superseding ADR 0006's published defaults, and a re-run of every downstream CRN comparison. The tuning study produces a *measurement*; the ADR 0006 published defaults stay published.
- **Tuning `min_qty` on the textbook policies.** Per ADR 0006 framing, `min_qty=0` is textbook-pure. Including it would let Optuna introduce non-textbook behaviour and cloud the framing-1 story.
- **A `tuning` skill or CLI helper outside `src/tuning/`.** The skill list in `CLAUDE.md` is not extended.

## Further Notes

- The whole tuning module is sized so that its objective function (`evaluate_policy_normalised`) is *the* deep, reusable seam. If a future need arises to evaluate any single policy on any CRN seed set (e.g. for ablation studies on the textbook family that are not Optuna-driven), `evaluate_policy_normalised` is the function to call. The Optuna orchestration in `study.py` is a thin layer on top.
- The `confirm_top_k()` flow doubles as a defensive measure against TPE overfitting to the 16 search seeds: if the top-k re-evaluation on 32 disjoint seeds shows the search winner does not generalise, the headline number is the *re-evaluation* mean, not the search mean. Both are recorded for transparency.
- Per the user's design-session preferences: "no SQLite, file-based artifacts only"; "no committed full-study fixture in the notebook"; "EOQ theoretical block in the docstring"; "hard rename, no backward-compat alias".
- The split between `evaluator.py` and `study.py` is deliberate: `evaluator.py` is the deep module (rarely changes; the most-tested) and `study.py` is the shallow orchestration layer (changes when the Optuna API surface evolves). Reviewers should expect the bulk of test coverage in `tests/tuning/test_evaluator.py`.
- ADR 0008 and ADR 0009 are independent landings: 0008 fixes a bug in the existing policy and 0009 introduces a new module. Bundling them in this single PRD reflects that the same agent will implement both in a two-PR sequence; their commit history is intentionally separable.
- The `optuna` dependency footprint is ~7 transitive packages (alembic, colorlog, packaging, PyYAML, scipy, sqlalchemy, tqdm). `scipy` is the largest of these; if the project ever wants to avoid `scipy`, a pure-NumPy TPE implementation is feasible but out of scope.
- The CLI invocation `uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1` is the canonical user-facing entry point; the notebook documents this as the prerequisite for the analysis sections.
