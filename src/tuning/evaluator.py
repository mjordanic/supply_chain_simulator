"""Single-policy CRN evaluator for hyperparameter tuning studies.

``evaluate_policy_normalised`` is the testable seam at the heart of the
tuning module. It accepts pre-built ``TuningEpisodeSpec`` objects and
returns a flat dict of KPIs — no Optuna import, no file I/O. The
orchestration layer (``study.py``) calls this once per trial; unit tests
call it directly with synthetic 2-seed episode lists.

Design notes
------------
- Self-contained — does not import anything from the RL package. The
  world-construction and per-tick rollout primitives live in
  ``src.tuning.rollout``.
- CRN tuple: four disjoint RNG streams per episode, all derived from
  ``episode_seed`` via ``_derive_seed``:

    assortment_seed  — which K products are active
    world_seed       — market / event / lifecycle draws (world_rng)
    init_seed        — step-0 store state initialisation (init_rng)
    allocation_seed  — buyer-shuffle order in the graph cascade
                       (allocation_rng = Random(_derive_seed(world_seed,
                       "allocation"))).  Added in issue 13 to reflect the
                       graph engine's per-phase shuffle introduced in
                       issue 05 (ADR 0016).

  Policy instances built by the factory consume *only* ``policy_rng``
  (seeded independently). Two trials sampling identical params on
  identical seeds produce bit-identical trajectories.
- Normalised return is ``net_profit / initial_cash``. This cancels per-
  episode scale (capacity × unit cost), making the objective comparable
  across the log-uniform capacity distribution.
"""

from __future__ import annotations

from typing import Any, Callable

from src.sim.policy import Policy
from src.tuning.episode import TuningEpisodeSpec
from src.tuning.rollout import run_policy_episode  # noqa: F401 (public re-export)


def evaluate_policy_normalised(
    policy_factory: Callable[[], Policy],
    eval_specs: list[TuningEpisodeSpec],
) -> dict[str, Any]:
    """Run a single policy on eval_specs; return normalised mean + per-seed KPI lists.

    Parameters
    ----------
    policy_factory:
        Zero-argument callable returning a fresh ``Policy`` instance.
        Called once per ``TuningEpisodeSpec`` so per-policy RNG is reset
        between seeds.
    eval_specs:
        Pre-built list of ``TuningEpisodeSpec`` objects.

    Returns
    -------
    dict with keys:
        mean_normalised_return: float
            Mean of ``net_profit / initial_cash`` across all seeds.
        per_seed_*: list[float] — per-seed KPI lists.
        mean_*: float — mean of each per-seed list.
    """
    per_seed_net_profit: list[float] = []
    per_seed_initial_cash: list[float] = []
    per_seed_capacity: list[float] = []
    per_seed_service_level: list[float] = []
    per_seed_stockout_rate: list[float] = []
    per_seed_inventory_turnover: list[float] = []
    per_seed_revenue: list[float] = []
    per_seed_mean_price_pct_of_msrp: list[float] = []

    for spec in eval_specs:
        policy = policy_factory()
        metrics = run_policy_episode(policy, spec)

        initial_cash = float(spec.balance)
        capacity = float(spec.capacity)

        per_seed_net_profit.append(metrics["net_profit"])
        per_seed_initial_cash.append(initial_cash)
        per_seed_capacity.append(capacity)
        per_seed_service_level.append(metrics["service_level"])
        per_seed_stockout_rate.append(metrics["stockout_rate"])
        per_seed_inventory_turnover.append(metrics["inventory_turnover"])
        per_seed_revenue.append(metrics["revenue"])
        per_seed_mean_price_pct_of_msrp.append(metrics["mean_price_pct_of_msrp"])

    def _mean(vals: list[float]) -> float:
        return sum(vals) / max(1, len(vals))

    per_seed_normalised_return = [
        p / max(1e-9, c)
        for p, c in zip(per_seed_net_profit, per_seed_initial_cash)
    ]
    mean_normalised_return = _mean(per_seed_normalised_return)

    return {
        "mean_normalised_return": mean_normalised_return,
        "per_seed_net_profit": per_seed_net_profit,
        "per_seed_initial_cash": per_seed_initial_cash,
        "per_seed_capacity": per_seed_capacity,
        "per_seed_service_level": per_seed_service_level,
        "per_seed_stockout_rate": per_seed_stockout_rate,
        "per_seed_inventory_turnover": per_seed_inventory_turnover,
        "per_seed_revenue": per_seed_revenue,
        "per_seed_mean_price_pct_of_msrp": per_seed_mean_price_pct_of_msrp,
        "mean_net_profit": _mean(per_seed_net_profit),
        "mean_initial_cash": _mean(per_seed_initial_cash),
        "mean_capacity": _mean(per_seed_capacity),
        "mean_service_level": _mean(per_seed_service_level),
        "mean_stockout_rate": _mean(per_seed_stockout_rate),
        "mean_inventory_turnover": _mean(per_seed_inventory_turnover),
        "mean_revenue": _mean(per_seed_revenue),
        "mean_mean_price_pct_of_msrp": _mean(per_seed_mean_price_pct_of_msrp),
    }


__all__ = ["evaluate_policy_normalised"]
