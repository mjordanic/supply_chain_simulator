"""Single-policy CRN evaluator for hyperparameter tuning studies.

``evaluate_policy_normalised`` is the testable seam at the heart of the
tuning module.  It accepts pre-built ``EpisodeSpec`` objects and returns a
flat dict of KPIs — no Optuna import, no file I/O.  The orchestration
layer (``study.py``) calls this function once per trial; the unit tests
call it directly with synthetic 2-seed episode lists.

Design notes
------------
- The function reuses the internal ``_build_world`` and the baseline
  rollout loop from ``src.rl.eval`` rather than duplicating the
  world-construction machinery.  The public ``evaluate`` in
  ``src.rl.eval`` is unchanged (it remains the paired CRN function).
- Policy instances built by the factory consume *only* ``policy_rng``;
  ``world_rng`` and ``init_rng`` are CRN-disjoint per the existing eval
  contract.  Two trials sampling identical params on identical seeds
  produce bit-identical trajectories.
- Normalised return is ``net_profit / initial_cash``.  This cancels out
  the per-episode scale (capacity × unit cost), making the objective
  comparable across the log-uniform capacity distribution used during
  training.
"""

from __future__ import annotations

from typing import Any, Callable

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import EpisodeSpec
from src.rl.eval import _build_world, _dispatch_orders, _process_demand
from src.rl.metrics import RunSlice, aggregate_episode
from src.sim.policy import Policy


def evaluate_policy_normalised(
    policy_factory: Callable[[], Policy],
    eval_specs: list[EpisodeSpec],
    *,
    config: RLConfig,
) -> dict[str, Any]:
    """Run a single policy on eval_specs; return normalised mean + per-seed KPI lists.

    The function calls ``policy_factory()`` once per spec so each seed
    gets a fresh policy instance with reset RNG state — the same CRN
    contract honoured by the paired-eval baseline arm.

    Parameters
    ----------
    policy_factory:
        Zero-argument callable returning a fresh ``Policy`` instance.
        Called once per ``EpisodeSpec`` so per-policy RNG is reset between
        seeds.
    eval_specs:
        Pre-built list of ``EpisodeSpec`` objects.  Typically produced by
        :func:`src.rl.eval.build_eval_seeds` with a ``seed_offset`` drawn
        from ``TuningConfig.seed_offset``.
    config:
        ``RLConfig`` instance.  Only ``episode_length`` (via
        ``spec.scenario.n_steps``) is consumed here; the config is
        forwarded to ``_build_world`` for completeness.

    Returns
    -------
    dict with keys:
        mean_normalised_return: float
            Mean of ``net_profit / initial_cash`` across all seeds.
        per_seed_net_profit: list[float]
        per_seed_initial_cash: list[float]
        per_seed_capacity: list[float]
        per_seed_service_level: list[float]
        per_seed_stockout_rate: list[float]
        per_seed_inventory_turnover: list[float]
        per_seed_revenue: list[float]
        per_seed_mean_price_pct_of_msrp: list[float]
        mean_net_profit: float
        mean_initial_cash: float
        mean_capacity: float
        mean_service_level: float
        mean_stockout_rate: float
        mean_inventory_turnover: float
        mean_revenue: float
        mean_mean_price_pct_of_msrp: float
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
        metrics = _run_single_policy(policy, spec)

        initial_cash = float(spec.scenario.stores[0].template.init_balance)
        capacity = float(spec.scenario.stores[0].template.capacity)

        per_seed_net_profit.append(metrics["net_profit"])
        per_seed_initial_cash.append(initial_cash)
        per_seed_capacity.append(capacity)
        per_seed_service_level.append(metrics["service_level"])
        per_seed_stockout_rate.append(metrics["stockout_rate"])
        per_seed_inventory_turnover.append(metrics["inventory_turnover"])
        per_seed_revenue.append(metrics["revenue"])
        per_seed_mean_price_pct_of_msrp.append(metrics["mean_price_pct_of_msrp"])

    n = len(eval_specs)

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


# ---------------------------------------------------------------------------
# Private: single-episode rollout (reuses world-construction from eval.py)
# ---------------------------------------------------------------------------


def _run_single_policy(
    policy: Policy,
    spec: EpisodeSpec,
) -> dict[str, float]:
    """Roll out ``policy`` on ``spec`` for one full episode.

    Uses the same tick order as ``_run_baseline`` in ``src.rl.eval``:
    world tick → observe → decide → dispatch → demand.  Collects per-tick
    traces into a ``RunSlice`` and returns ``aggregate_episode`` metrics.
    """
    _, item_registry, market, event_engine, store = _build_world(spec)
    store.policy = policy

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    episode_length = spec.scenario.n_steps

    for _tick in range(episode_length):
        # 1–3: world ticks (same order as Runner and eval._run_baseline)
        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        # 4–5: observe and decide
        obs_dict = store.observe(market.current_step(), item_registry)
        action = store.decide(obs_dict)

        # 6–7: dispatch orders and settle demand
        _dispatch_orders(store, market, event_engine, action)
        _process_demand(store, market, action)

        # Collect per-tick traces for active SKUs
        tick_sales = [store.sales.get(pid, 0) for pid in spec.active_subset]
        tick_demand = [store.demand.get(pid, 0) for pid in spec.active_subset]
        tick_inv = [store.inventory.get(pid, 0) for pid in spec.active_subset]
        tick_price = [store.prices.get(pid, 0.0) for pid in spec.active_subset]
        tick_msrp = [store.base_prices.get(pid, 1.0) for pid in spec.active_subset]
        tick_rev = [store.revenue.get(pid, 0.0) for pid in spec.active_subset]
        tick_hold = [store.holding_cost.get(pid, 0.0) for pid in spec.active_subset]
        tick_order_cost = [
            action.get("order", {}).get(pid, 0) * store.costs.get(pid, 0.0)
            for pid in spec.active_subset
        ]
        tick_fee: list[float] = []
        for pid in spec.active_subset:
            ordered = action.get("order", {}).get(pid, 0)
            tick_fee.append(float(store.order_fee) if ordered > 0 else 0.0)

        run_slice.sales.append(tick_sales)
        run_slice.demand.append(tick_demand)
        run_slice.inventory.append(tick_inv)
        run_slice.price.append(tick_price)
        run_slice.msrp.append(tick_msrp)
        run_slice.revenue.append(tick_rev)
        run_slice.holding_cost.append(tick_hold)
        run_slice.order_cost.append(tick_order_cost)
        run_slice.order_fee.append(tick_fee)

    return aggregate_episode(run_slice)


__all__ = ["evaluate_policy_normalised"]
