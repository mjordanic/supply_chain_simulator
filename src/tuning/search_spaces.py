"""Bundled trial-callback factories for the four textbook policy variants.

Each factory accepts an ``optuna.Trial`` (or any trial-compatible object,
e.g. ``optuna.trial.FixedTrial``) and returns a fully-constructed ``Policy``
instance with all tunables sampled from their documented ranges.

Factory summary:

    order_up_to_space(trial)          -> OrderUpToPolicy
    reorder_point_space(trial)        -> ReorderPointPolicy
    periodic_order_up_to_space(trial) -> PeriodicOrderUpToPolicy
    periodic_reorder_space(trial)     -> PeriodicReorderPolicy

Design notes:

- ``min_qty`` is NOT tuned on any variant.  Per ADR 0009 framing-1,
  ``min_qty=0`` is textbook-pure; tuning it would let Optuna introduce
  non-textbook behaviour and cloud the framing-1 story.

- All shared tunables use the same names and ranges across factories so
  that cross-variant comparisons are straightforward (same axis = same
  meaning).

- A future custom-policy author who wants to add a fifth factory should
  follow the obvious convention: one function, one ``optuna.Trial`` arg,
  one ``Policy`` return.  The function body is typically 5-10 lines.

Tunable ranges (``COVER_HORIZON_TICKS`` and the constants below):

    cover_horizon_ticks               int         [1, 30]  — EOQ cycle in ticks
    safety_lead_pct_of_lag            float       [0.0, 3.0]
    stockout_safety_bonus_pct_of_lag  float       [0.0, 2.0]
    Q                                 int         [1, 30]  — ReorderPoint fixed qty
    review_interval                   int         [1, 14]  — periodic cadence in ticks

Pinned (not tuned):

    opening_budget_pct           float  OPENING_BUDGET_PCT
    per_supplier_min_order_floor int    MIN_ORDER_FLOOR
    routing_strategy             cheapest-first (``routing_strategy=None``)
"""

from __future__ import annotations

import optuna

from src.sim.policy import MIN_ORDER_FLOOR, OPENING_BUDGET_PCT

# Inclusive bounds. ``suggest_*`` calls below are the only place a trial
# learns them.
COVER_HORIZON_TICKS = (1, 30)
SAFETY_LEAD_PCT_OF_LAG = (0.0, 3.0)
STOCKOUT_SAFETY_BONUS_PCT_OF_LAG = (0.0, 2.0)
REORDER_QUANTITY = (1, 30)
REVIEW_INTERVAL_TICKS = (1, 14)


def _sample_shared(trial: optuna.Trial) -> dict:
    """Sample the tunables shared across all four factories."""
    return {
        "cover_horizon_ticks": trial.suggest_int(
            "cover_horizon_ticks", *COVER_HORIZON_TICKS
        ),
        "safety_lead_pct_of_lag": trial.suggest_float(
            "safety_lead_pct_of_lag", *SAFETY_LEAD_PCT_OF_LAG
        ),
        "stockout_safety_bonus_pct_of_lag": trial.suggest_float(
            "stockout_safety_bonus_pct_of_lag", *STOCKOUT_SAFETY_BONUS_PCT_OF_LAG
        ),
    }


def order_up_to_space(trial: optuna.Trial) -> "Policy":  # noqa: F821
    """(s,S) continuous-review policy factory.

    Samples the tunables for ``OrderUpToPolicy`` and returns a fully-
    constructed instance.

    Search space:

        cover_horizon_ticks               int   [1, 30]
        safety_lead_pct_of_lag            float [0.0, 3.0]
        stockout_safety_bonus_pct_of_lag  float [0.0, 2.0]
    """
    from src.sim.policy import OrderUpToPolicy

    shared = _sample_shared(trial)
    return OrderUpToPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=OPENING_BUDGET_PCT,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        per_supplier_min_order_floor=MIN_ORDER_FLOOR,
        routing_strategy=None,
    )


def reorder_point_space(trial: optuna.Trial) -> "Policy":  # noqa: F821
    """(s,Q) continuous-review policy factory.

    Samples the shared tunables plus the ``Q`` (fixed reorder quantity)
    parameter for ``ReorderPointPolicy``.

    Search space:

        cover_horizon_ticks               int   [1, 30]
        safety_lead_pct_of_lag            float [0.0, 3.0]
        stockout_safety_bonus_pct_of_lag  float [0.0, 2.0]
        Q                                 int   [1, 30]
    """
    from src.sim.policy import ReorderPointPolicy

    shared = _sample_shared(trial)
    return ReorderPointPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=OPENING_BUDGET_PCT,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        Q=trial.suggest_int("Q", *REORDER_QUANTITY),
        per_supplier_min_order_floor=MIN_ORDER_FLOOR,
        routing_strategy=None,
    )


def periodic_order_up_to_space(trial: optuna.Trial) -> "Policy":  # noqa: F821
    """(R,S) periodic-review policy factory.

    Samples the shared tunables plus the ``review_interval`` cadence
    parameter for ``PeriodicOrderUpToPolicy``.

    Search space:

        cover_horizon_ticks               int   [1, 30]
        safety_lead_pct_of_lag            float [0.0, 3.0]
        stockout_safety_bonus_pct_of_lag  float [0.0, 2.0]
        review_interval                   int   [1, 14]
    """
    from src.sim.policy import PeriodicOrderUpToPolicy

    shared = _sample_shared(trial)
    return PeriodicOrderUpToPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=OPENING_BUDGET_PCT,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        review_interval=trial.suggest_int("review_interval", *REVIEW_INTERVAL_TICKS),
        per_supplier_min_order_floor=MIN_ORDER_FLOOR,
        routing_strategy=None,
    )


def periodic_reorder_space(trial: optuna.Trial) -> "Policy":  # noqa: F821
    """(R,s,S) periodic-review policy factory.

    Samples the shared tunables plus both the ``Q`` (fixed reorder
    quantity) and ``review_interval`` cadence parameters for
    ``PeriodicReorderPolicy``.

    Search space:

        cover_horizon_ticks               int   [1, 30]
        safety_lead_pct_of_lag            float [0.0, 3.0]
        stockout_safety_bonus_pct_of_lag  float [0.0, 2.0]
        review_interval                   int   [1, 14]
    """
    from src.sim.policy import PeriodicReorderPolicy

    shared = _sample_shared(trial)
    return PeriodicReorderPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=OPENING_BUDGET_PCT,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        review_interval=trial.suggest_int("review_interval", *REVIEW_INTERVAL_TICKS),
        per_supplier_min_order_floor=MIN_ORDER_FLOOR,
        routing_strategy=None,
    )
