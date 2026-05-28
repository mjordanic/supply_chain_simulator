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

Tunable ranges:

    cover_horizon_ticks               int         [1, 30]  — EOQ cycle in ticks
    safety_lead_pct_of_lag            float       [0.0, 3.0]
    stockout_safety_bonus_pct_of_lag  float       [0.0, 2.0]
    Q                                 int         [1, 30]  — ReorderPoint fixed qty
    review_interval                   int         [1, 14]  — periodic cadence in ticks
    per_supplier_min_order_floor      int         [0, 10]  — buyer-side min order per line
    routing_strategy                  Categorical ["cheapest_first",
                                                   "fill_rate_weighted"]

Pinned (not tuned):

    opening_budget_pct           float 0.8  — stores already start with stock
"""

from __future__ import annotations

import optuna

_OPENING_BUDGET_PCT_PINNED = 0.8

# Routing strategy choices (Categorical).
_ROUTING_CHOICES = ["cheapest_first", "fill_rate_weighted"]


def _sample_shared(trial: optuna.Trial) -> dict:
    """Sample the tunables shared across all four factories."""
    return {
        "cover_horizon_ticks": trial.suggest_int("cover_horizon_ticks", 1, 30),
        "safety_lead_pct_of_lag": trial.suggest_float("safety_lead_pct_of_lag", 0.0, 3.0),
        "stockout_safety_bonus_pct_of_lag": trial.suggest_float(
            "stockout_safety_bonus_pct_of_lag", 0.0, 2.0
        ),
        "per_supplier_min_order_floor": trial.suggest_int(
            "per_supplier_min_order_floor", 0, 10
        ),
        "routing_strategy": trial.suggest_categorical(
            "routing_strategy", _ROUTING_CHOICES
        ),
    }


def _resolve_routing(strategy_name: str):
    """Return the routing callable for ``strategy_name``, or ``None`` for default.

    ``"cheapest_first"`` maps to the ``MultiSupplierTextbookPolicy`` built-in
    (pass ``routing_strategy=None`` to use it).  ``"fill_rate_weighted"`` uses
    the fill-rate EMA from the ``CentralTable`` to weight allocations.
    """
    if strategy_name == "cheapest_first":
        return None  # default behaviour
    if strategy_name == "fill_rate_weighted":
        from src.sim.policy import MultiSupplierTextbookPolicy
        return MultiSupplierTextbookPolicy._routing_fill_rate_weighted
    return None


def order_up_to_space(trial: optuna.Trial) -> "Policy":  # noqa: F821
    """(s,S) continuous-review policy factory.

    Samples the tunables for ``OrderUpToPolicy`` and returns a fully-
    constructed instance.

    Search space:

        cover_horizon_ticks               int   [1, 30]
        safety_lead_pct_of_lag            float [0.0, 3.0]
        stockout_safety_bonus_pct_of_lag  float [0.0, 2.0]
        per_supplier_min_order_floor      int   [0, 10]
        routing_strategy                  Categorical ["cheapest_first",
                                                       "fill_rate_weighted"]
    """
    from src.sim.policy import OrderUpToPolicy

    shared = _sample_shared(trial)
    return OrderUpToPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=_OPENING_BUDGET_PCT_PINNED,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        per_supplier_min_order_floor=shared["per_supplier_min_order_floor"],
        routing_strategy=_resolve_routing(shared["routing_strategy"]),
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
        per_supplier_min_order_floor      int   [0, 10]
        routing_strategy                  Categorical ["cheapest_first",
                                                       "fill_rate_weighted"]
    """
    from src.sim.policy import ReorderPointPolicy

    shared = _sample_shared(trial)
    return ReorderPointPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=_OPENING_BUDGET_PCT_PINNED,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        Q=trial.suggest_int("Q", 1, 30),
        per_supplier_min_order_floor=shared["per_supplier_min_order_floor"],
        routing_strategy=_resolve_routing(shared["routing_strategy"]),
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
        per_supplier_min_order_floor      int   [0, 10]
        routing_strategy                  Categorical ["cheapest_first",
                                                       "fill_rate_weighted"]
    """
    from src.sim.policy import PeriodicOrderUpToPolicy

    shared = _sample_shared(trial)
    return PeriodicOrderUpToPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=_OPENING_BUDGET_PCT_PINNED,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        review_interval=trial.suggest_int("review_interval", 1, 14),
        per_supplier_min_order_floor=shared["per_supplier_min_order_floor"],
        routing_strategy=_resolve_routing(shared["routing_strategy"]),
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
        per_supplier_min_order_floor      int   [0, 10]
        routing_strategy                  Categorical ["cheapest_first",
                                                       "fill_rate_weighted"]
    """
    from src.sim.policy import PeriodicReorderPolicy

    shared = _sample_shared(trial)
    return PeriodicReorderPolicy(
        cover_horizon_ticks=shared["cover_horizon_ticks"],
        safety_lead_pct_of_lag=shared["safety_lead_pct_of_lag"],
        opening_budget_pct=_OPENING_BUDGET_PCT_PINNED,
        stockout_safety_bonus_pct_of_lag=shared["stockout_safety_bonus_pct_of_lag"],
        review_interval=trial.suggest_int("review_interval", 1, 14),
        per_supplier_min_order_floor=shared["per_supplier_min_order_floor"],
        routing_strategy=_resolve_routing(shared["routing_strategy"]),
    )
