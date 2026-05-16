"""Single-episode rollout primitives for tuning studies.

Self-contained — no imports from ``src.rl``. Mirrors the tick order used by
the simulator's ``Runner`` so per-episode metrics are comparable to other
arms in the project.
"""

from __future__ import annotations

from random import Random

from src.sim.event_engine import EventEngine
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.policy import Policy
from src.sim.scenario import StoreInstance
from src.sim.store import Store
from src.tuning.episode import TuningEpisodeSpec
from src.sim.metrics import RunSlice, aggregate_episode


def _build_world(spec: TuningEpisodeSpec):
    """Instantiate simulator subsystems from a ``TuningEpisodeSpec``.

    Returns ``(world_rng, item_registry, market, event_engine, store)``
    with ``Store`` having ``policy=None`` — callers wire up the policy.
    """
    scenario = spec.scenario
    world_rng = Random(scenario.world_seed)

    item_registry = ItemRegistry(
        scenario.item_lifecycle,
        scenario.catalog,
        world_rng,
    )
    market = Market(
        scenario.market,
        world_rng,
        scenario.start_date,
        registry=item_registry,
    )
    event_engine = EventEngine(scenario.disruption, world_rng)

    si: StoreInstance = scenario.stores[0]
    store = Store(
        si.template,
        si.init_seed,
        None,
        scenario.catalog,
        freshness_alpha=item_registry.default_freshness_alpha,
        freshness_decay=item_registry.default_freshness_decay,
        item_registry=item_registry,
    )
    return world_rng, item_registry, market, event_engine, store


def _make_delivery_callback(store: Store, pid: str, qty: int):
    def _cb() -> None:
        store.deliver(pid, qty)

    return _cb


def _dispatch_orders(store: Store, market: Market, event_engine: EventEngine, action: dict) -> None:
    orders = action.get("order", {})
    if not orders:
        return
    current_step = market.current_step()
    supply = market.market_state[store.region]["market_supply"]
    supply_factor = max(market.params.supply_factor_min, supply)
    for pid, qty in orders.items():
        if qty <= 0:
            continue
        base_lead = store.delivery_lags[pid]
        adjusted_lead = int(base_lead / supply_factor)
        arrival_time = current_step + adjusted_lead
        event_engine.schedule(
            event_type="order_arrival",
            delay=arrival_time,
            callback=_make_delivery_callback(store, pid, qty),
        )


def _process_demand(store: Store, market: Market, action: dict) -> dict[str, float]:
    prices = action.get("price", {})
    orders = action.get("order", {})
    current_step = market.current_step()
    traces: dict[str, float] = {}
    for pid in list(store.inventory.keys()):
        price = prices.get(pid, store.prices[pid])
        order_qty = orders.get(pid, 0)
        demand = market.sample_demand(pid, store, price, current_step=current_step)
        store.settle(pid, demand=demand, price=price, order_qty=order_qty)
        store.prices[pid] = price
        traces[pid] = demand
    return traces


def run_policy_episode(policy: Policy, spec: TuningEpisodeSpec) -> dict[str, float]:
    """Roll out ``policy`` on ``spec`` for one full episode and return aggregate KPIs.

    Tick order: world tick → observe → decide → dispatch → demand.
    """
    _, item_registry, market, event_engine, store = _build_world(spec)
    store.policy = policy

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    episode_length = spec.scenario.n_steps

    for _tick in range(episode_length):
        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        obs_dict = store.observe(market.current_step(), item_registry)
        action = store.decide(obs_dict)

        _dispatch_orders(store, market, event_engine, action)
        _process_demand(store, market, action)

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


__all__ = ["run_policy_episode"]
