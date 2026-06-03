# 03 — Bundled search-space factories for the four textbook variants

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add `src/tuning/search_spaces.py` with four trial-callback factories — one per textbook policy variant. Each factory takes an `optuna.Trial` and returns a fully-constructed `Policy` instance with the tunables sampled from the documented ranges. These are the Pattern-A factories that `run_study(policy_space=...)` will accept.

Factories:

```python
def order_up_to_space(trial: optuna.Trial) -> Policy:
    from src.sim.policy import OrderUpToPolicy
    return OrderUpToPolicy(
        cover_horizon_ticks=trial.suggest_int("cover_horizon_ticks", 1, 30),
        safety_lead_pct_of_lag=trial.suggest_float("safety_lead_pct_of_lag", 0.0, 3.0),
        opening_budget_pct=trial.suggest_float("opening_budget_pct", 0.05, 0.95),
        stockout_safety_bonus_pct_of_lag=trial.suggest_float("stockout_safety_bonus_pct_of_lag", 0.0, 2.0),
    )

def reorder_point_space(trial: optuna.Trial) -> Policy: ...
def periodic_order_up_to_space(trial: optuna.Trial) -> Policy: ...
def periodic_reorder_space(trial: optuna.Trial) -> Policy: ...
```

The three additional factories follow the same shape, each adding their variant-specific kwargs (`Q` for `ReorderPointPolicy`, `review_interval` for the periodic variants). `min_qty` is NOT tuned on any variant (per ADR 0009 framing-1: `min_qty=0` is textbook-pure and tuning it would let Optuna introduce non-textbook behaviour).

Extend `src/tuning/__init__.py` to re-export the four factories:

```python
from src.tuning.search_spaces import (
    order_up_to_space,
    reorder_point_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
)
```

This file is intentionally small and uniform — a future custom-policy author who wants to add a fifth factory should be able to do so in a tens-of-lines patch following the obvious convention.

### Tests added

In `tests/tuning/test_search_spaces.py`:

- `test_order_up_to_space_constructs` — Call `order_up_to_space(optuna.trial.FixedTrial({"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.5, "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0}))`; assert it returns an `OrderUpToPolicy` instance with the expected kwargs.
- `test_reorder_point_space_constructs` — Same for `ReorderPointPolicy` (including `Q`).
- `test_periodic_order_up_to_space_constructs` — Same for `PeriodicOrderUpToPolicy` (including `review_interval`).
- `test_periodic_reorder_space_constructs` — Same for `PeriodicReorderPolicy` (including both extra kwargs).
- `test_all_factories_use_documented_ranges` — Iterate the four factories with a `FixedTrial` set to each range boundary (low / high) for each tunable; assert the construction succeeds and the kwarg is set to the boundary value. Guards against silent range-narrowing.

PRD user stories covered: 3, 20, 22.

## Acceptance criteria

- [ ] `src/tuning/search_spaces.py` exports `order_up_to_space`, `reorder_point_space`, `periodic_order_up_to_space`, `periodic_reorder_space`.
- [ ] All four factories accept an `optuna.Trial`-compatible object and return a `Policy` subclass instance.
- [ ] `from src.tuning import order_up_to_space, reorder_point_space, periodic_order_up_to_space, periodic_reorder_space` works.
- [ ] No factory tunes `min_qty` on any variant.
- [ ] `order_up_to_space` uses kwarg names `cover_horizon_ticks`, `safety_lead_pct_of_lag`, `opening_budget_pct`, `stockout_safety_bonus_pct_of_lag` (i.e. the post-Bundle-A names).
- [ ] All five tests in `tests/tuning/test_search_spaces.py` pass.
- [ ] `uv run pytest tests/tuning/` is green.

## Blocked by

- `01-reparameterise-safety-horizons.md` — `order_up_to_space` calls `OrderUpToPolicy(safety_lead_pct_of_lag=...)`, the renamed kwarg.
- `02-tuning-module-skeleton-and-evaluator.md` — extends `src/tuning/__init__.py` exports.
