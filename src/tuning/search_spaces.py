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

Tunable ranges (``COVER_HORIZON_TICKS`` and the constants below — the
transition log samples these same bounds and does not invent its own):

    cover_horizon_ticks               int         [1, 30]  — EOQ cycle in ticks
    safety_lead_pct_of_lag            float       [0.0, 3.0]
    stockout_safety_bonus_pct_of_lag  float       [0.0, 2.0]
    Q                                 int         [1, 30]  — ReorderPoint fixed qty
    review_interval                   int         [1, 14]  — periodic cadence in ticks

Pinned (not tuned), shared with the replenishment planner:

    opening_budget_pct           float  OPENING_BUDGET_PCT
    per_supplier_min_order_floor int    MIN_ORDER_FLOOR
    routing_strategy             cheapest-first (``routing_strategy=None``)
"""

from __future__ import annotations

import random

import optuna

from src.sim.policy import MIN_ORDER_FLOOR, OPENING_BUDGET_PCT

# Inclusive bounds. ``suggest_*`` calls below are the only place a trial
# learns them. The transition-log mixture draws through those calls.
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


def order_up_to_from_params(params: dict) -> "Policy":  # noqa: F821
    """Rebuild the ``order_up_to_space`` policy for a finished trial."""
    return order_up_to_space(optuna.trial.FixedTrial(dict(params)))


def maximise_order_up_to(objective, *, n_trials: int, sampler_seed: int = 0) -> tuple[dict, float]:
    """Maximise ``objective(order_up_to_space(trial))`` on one fixed scenario."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )

    def _objective(trial: optuna.Trial) -> float:
        return float(objective(order_up_to_space(trial)))

    study.optimize(_objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)


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


class _UniformTrial:
    """Uniform ``suggest_int`` / ``suggest_float`` without an Optuna study.

    The transition log calls the space factories through this so the
    ranges stay inside those factories. A repeated name returns the same
    draw, matching ``Trial``.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.params: dict = {}

    def suggest_int(self, name: str, low: int, high: int, *args, **kwargs) -> int:
        del args, kwargs
        if name not in self.params:
            self.params[name] = self._rng.randint(int(low), int(high))
        return int(self.params[name])

    def suggest_float(self, name: str, low: float, high: float, *args, **kwargs) -> float:
        del args, kwargs
        if name not in self.params:
            self.params[name] = float(low) + (float(high) - float(low)) * self._rng.random()
        return float(self.params[name])


def sample_cover_horizon_ticks(seed: int) -> int:
    """One uniform draw of ``cover_horizon_ticks`` from the shared space."""
    trial = _UniformTrial(random.Random(seed))
    return int(_sample_shared(trial)["cover_horizon_ticks"])


def sample_textbook_policy(behavior: str, seed: int):
    """Build one textbook policy by calling that family's space factory.

    ``behavior`` is ``order_up_to``, ``reorder_point``,
    ``periodic_order_up_to``, or ``periodic_reorder``. The return is the
    policy and the sampled hyperparameters (not the pinned cold-start
    fraction, floor, or routing).
    """
    spaces = {
        "order_up_to": order_up_to_space,
        "reorder_point": reorder_point_space,
        "periodic_order_up_to": periodic_order_up_to_space,
        "periodic_reorder": periodic_reorder_space,
    }
    try:
        factory = spaces[behavior]
    except KeyError as exc:
        raise ValueError(f"unknown textbook behavior {behavior!r}") from exc
    trial = _UniformTrial(random.Random(seed))
    policy = factory(trial)
    return policy, dict(trial.params)
