"""Single-episode rollout primitives for tuning studies.

Self-contained — no imports from ``src.rl``. Delegates per-tick state machine
to ``src.sim.runner.Simulation`` via ``build_world`` + ``Simulation.tick()``.
"""

from __future__ import annotations

from src.sim.episode_sampler import EpisodeSpec
from src.sim.metrics import RunSlice, aggregate_episode
from src.sim.policy import Policy
from src.sim.runner import build_world


def _record_active_subset(
    store,
    action: dict,
    run_slice: RunSlice,
    catalog_lookup: dict,
    order_fee: float,
) -> None:
    """Project one tick of store state onto the active-subset ``RunSlice``.

    Reads unit cost from ``catalog_lookup`` (keyed by product_id → ``Ware``);
    this is equivalent to reading ``store.costs[pid]`` because ``store.costs``
    is populated from ``Ware.unit_cost`` at construction.
    """
    pids = run_slice.active_pids
    orders = action.get("order", {})

    run_slice.sales.append([store.sales.get(pid, 0) for pid in pids])
    run_slice.demand.append([store.demand.get(pid, 0) for pid in pids])
    run_slice.inventory.append([store.inventory.get(pid, 0) for pid in pids])
    run_slice.price.append([store.prices.get(pid, 0.0) for pid in pids])
    run_slice.msrp.append([store.base_prices.get(pid, 1.0) for pid in pids])
    run_slice.revenue.append([store.revenue.get(pid, 0.0) for pid in pids])
    run_slice.holding_cost.append([store.holding_cost.get(pid, 0.0) for pid in pids])
    run_slice.order_cost.append([
        orders.get(pid, 0) * catalog_lookup[pid].unit_cost for pid in pids
    ])
    run_slice.order_fee.append([
        float(order_fee) if orders.get(pid, 0) > 0 else 0.0 for pid in pids
    ])


def run_policy_episode(policy: Policy, spec: EpisodeSpec) -> dict[str, float]:
    """Roll out ``policy`` on ``spec`` for one full episode and return aggregate KPIs.

    Tick order: world tick → observe → decide → dispatch → demand.
    Delegates the per-tick state machine to ``Simulation.tick()``.
    """
    sim = build_world(spec.scenario, policy_overrides=[policy])
    store = sim.stores[0]

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    catalog_lookup = {w.product_id: w for w in spec.scenario.catalog}
    order_fee = float(spec.scenario.stores[0].template.order_fee)

    for _ in range(spec.scenario.n_steps):
        result = sim.tick()
        _record_active_subset(
            store,
            result.actions[0],
            run_slice,
            catalog_lookup,
            order_fee,
        )

    return aggregate_episode(run_slice)


__all__ = ["run_policy_episode"]
