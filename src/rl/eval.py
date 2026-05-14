"""CRN-paired evaluation harness for the RL training framework.

``build_eval_seeds`` generates a fixed, deterministic list of ``EpisodeSpec``
objects from the configured eval-seed range (disjoint from the training range
via ``config.eval_seed_offset``).

``evaluate`` runs the RL policy and ``OrderUpToPolicy`` on bit-identical
``EpisodeSpec`` tuples (Common Random Numbers), computes paired metrics, and
returns a flat dict suitable for logging to TensorBoard under ``eval/*`` keys.

CRN guarantee
-------------
Both the RL and baseline runs for a given ``EpisodeSpec`` use the *same*
``world_seed``, capacity, balance, active subset, and slot permutation.  The
world trajectory is therefore bit-identical between the two runs, so any
difference in the returned metrics is attributable to the policy alone.

Implementation notes
--------------------
- The RL policy is invoked as a callable ``(obs_tensor: np.ndarray) →
  action_vec: np.ndarray`` so the same eval can serve PPO, SAC, or any
  future agent without rewriting plumbing.
- The baseline policy is obtained by calling ``baseline_policy_factory()``
  once per seed so per-policy RNG state is reset between paired runs.
- The eval runs the RL policy through ``RLEnv`` (step-by-step) and the
  baseline through a lightweight inline loop that replicates the env's tick
  order.  Both share the same ``EpisodeSpec`` inputs so the CRN guarantee
  holds.
"""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Any, Callable

import numpy as np

from src.rl.configs.default import RLConfig
from src.rl.encoders import decode_action, encode_observation
from src.rl.episode_sampler import EpisodeSpec, sample_episode
from src.rl.metrics import RunSlice, aggregate_episode
from src.sim.event_engine import EventEngine
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.policy import OrderUpToPolicy, Policy
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    StoreInstance,
    Ware,
)
from src.sim.store import Store


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyFn = Callable[[np.ndarray], np.ndarray]
"""RL policy callable: takes an obs tensor, returns an action vector."""

BaselineFactory = Callable[[], Policy]
"""Factory that returns a fresh Policy (e.g. OrderUpToPolicy) per seed."""


# ---------------------------------------------------------------------------
# Public: build_eval_seeds
# ---------------------------------------------------------------------------


def build_eval_seeds(
    catalog: list[Ware],
    base_template: Any,
    config: RLConfig,
    n_seeds: int | None = None,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
) -> list[EpisodeSpec]:
    """Generate a fixed list of held-out ``EpisodeSpec`` objects for evaluation.

    Seeds are taken from ``[config.eval_seed_offset, config.eval_seed_offset
    + n_seeds)``.  This range is intentionally far above the training seed
    space (default offset is 10_000_000) so eval seeds are never seen during
    training.

    Parameters
    ----------
    catalog:
        Full product universe; must have at least ``config.K_active`` items.
    base_template:
        ``StoreTemplate`` carrying non-episodic knobs (region, delivery lag,
        …).  Episodic fields are overridden by ``sample_episode``.
    config:
        ``RLConfig`` instance.  Consumes ``eval_seed_offset``,
        ``n_eval_seeds``, and all fields forwarded to ``sample_episode``.
    n_seeds:
        Number of eval seeds.  Defaults to ``config.n_eval_seeds``.
    market_params, disruption_params, lifecycle_params:
        Optional simulator parameter overrides forwarded to ``sample_episode``.

    Returns
    -------
    list[EpisodeSpec]
        Deterministic, ordered list of ``n_seeds`` ``EpisodeSpec`` objects.
        Calling this function twice with the same arguments returns identical
        specs.
    """
    if n_seeds is None:
        n_seeds = config.n_eval_seeds

    specs: list[EpisodeSpec] = []
    for i in range(n_seeds):
        seed = config.eval_seed_offset + i
        spec = sample_episode(
            catalog=catalog,
            base_template=base_template,
            config=config,
            episode_seed=seed,
            market_params=market_params,
            disruption_params=disruption_params,
            lifecycle_params=lifecycle_params,
        )
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------------


def evaluate(
    rl_policy_fn: PolicyFn,
    baseline_policy_factory: BaselineFactory,
    eval_specs: list[EpisodeSpec],
    *,
    config: RLConfig | None = None,
) -> dict[str, float]:
    """Run paired CRN evaluation and return a flat ``eval/*`` metrics dict.

    For each ``EpisodeSpec``:

    1. Run the RL policy through a step-by-step loop identical to ``RLEnv``
       (same tick order; same world seed → bit-identical world trajectory).
    2. Run a fresh baseline policy through the same ``EpisodeSpec`` using
       the same tick loop.
    3. Compute ``aggregate_episode`` metrics for each run.
    4. Compute paired uplift (RL − baseline) per seed, then average.

    Parameters
    ----------
    rl_policy_fn:
        Callable ``(obs_tensor: np.ndarray) → action_vec: np.ndarray``.
        Receives a flat float32 observation and must return a float32 array
        of shape ``(2 * K_active,)`` in ``[-1, 1]``.  The same function is
        called for every eval seed so it must be stateless or carry its own
        internal state across seeds.
    baseline_policy_factory:
        Zero-argument callable returning a fresh ``Policy`` instance.  Called
        once per seed so per-policy RNG state is reset between paired runs.
    eval_specs:
        List of ``EpisodeSpec`` objects, typically produced by
        ``build_eval_seeds``.
    config:
        ``RLConfig`` instance.  When ``None``, ``RLConfig()`` (PRD defaults)
        is used.  Only ``K_active`` and ``episode_length`` are consumed here.

    Returns
    -------
    dict[str, float]
        Flat dict with keys prefixed ``eval/``.  Keys:

        - ``eval/rl_return``         — mean episode return across eval seeds
        - ``eval/baseline_return``   — mean baseline episode return
        - ``eval/paired_uplift``     — mean (RL return − baseline return) per seed
        - ``eval/win_rate``          — fraction of seeds where RL return ≥ baseline
        - ``eval/rl_service_level``
        - ``eval/rl_stockout_rate``
        - ``eval/rl_inventory_turnover``
        - ``eval/rl_mean_price_pct_of_msrp``
        - ``eval/rl_revenue``
        - ``eval/rl_net_profit``
        - ``eval/baseline_service_level``
        - ``eval/baseline_stockout_rate``
        - ``eval/baseline_inventory_turnover``
        - ``eval/baseline_mean_price_pct_of_msrp``
        - ``eval/baseline_revenue``
        - ``eval/baseline_net_profit``
    """
    if config is None:
        config = RLConfig()

    if not eval_specs:
        return {}

    rl_metrics_list: list[dict[str, float]] = []
    baseline_metrics_list: list[dict[str, float]] = []

    for spec in eval_specs:
        rl_metrics = _run_rl(rl_policy_fn, spec, config=config)
        baseline_policy = baseline_policy_factory()
        baseline_metrics = _run_baseline(baseline_policy, spec)
        rl_metrics_list.append(rl_metrics)
        baseline_metrics_list.append(baseline_metrics)

    return _aggregate_paired(rl_metrics_list, baseline_metrics_list)


# ---------------------------------------------------------------------------
# Private: single-episode runners
# ---------------------------------------------------------------------------


def _build_world(spec: EpisodeSpec):
    """Instantiate the simulator subsystems from an ``EpisodeSpec``.

    Returns ``(world_rng, item_registry, market, event_engine, store)``
    with the ``Store`` having ``policy=None`` (caller must wire up the
    policy separately).
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
    # Store requires a policy; we pass None and expect the caller to attach
    # or set pending actions before Store.decide() is called.
    store = Store(
        si.template,
        si.init_seed,
        None,  # policy wired up by callers
        scenario.catalog,
        freshness_alpha=item_registry.default_freshness_alpha,
        freshness_decay=item_registry.default_freshness_decay,
        item_registry=item_registry,
    )
    return world_rng, item_registry, market, event_engine, store


def _make_delivery_callback(store: Store, pid: str, qty: int):
    """Bind ``(store, pid, qty)`` into a zero-arg delivery callback."""

    def _cb() -> None:
        store.deliver(pid, qty)

    return _cb


def _dispatch_orders(store: Store, market: Market, event_engine: EventEngine, action: dict) -> None:
    """Schedule delivery callbacks for positive-qty orders (same as Runner)."""
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
    """Sample demand and settle accounting — mirrors Runner._process_demand."""
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


def _run_rl(
    rl_policy_fn: PolicyFn,
    spec: EpisodeSpec,
    *,
    config: RLConfig,
) -> dict[str, float]:
    """Run the RL policy on ``spec`` for one full episode.

    Uses the same tick order as ``RLEnv.step()``.  Collects per-tick traces
    into a ``RunSlice`` and returns ``aggregate_episode`` metrics.
    """
    from src.sim.policy import RLPolicy

    _, item_registry, market, event_engine, store = _build_world(spec)

    rl_policy = RLPolicy()
    store.policy = rl_policy

    K = config.K_active
    initial_cash = float(store.balance)
    sales_history: dict[str, deque] = {
        pid: deque(maxlen=100) for pid in [w.product_id for w in spec.scenario.catalog]
    }
    slot_perm = spec.slot_permutation

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    episode_length = spec.scenario.n_steps

    for tick in range(episode_length):
        # 1–3: world ticks
        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        # 4: encode obs, call rl_policy_fn, decode action
        obs = encode_observation(
            store,
            market,
            item_registry,
            step=tick,
            slot_perm=slot_perm,
            K_active=K,
            initial_cash=initial_cash,
            sales_history=sales_history,
        )
        action_vec = rl_policy_fn(obs)
        action_dict = decode_action(
            np.asarray(action_vec, dtype=np.float32),
            slot_perm,
            store,
            K,
            store.base_prices,
        )
        action_dict["activate"] = []
        action_dict["deactivate"] = []
        action_dict["promotions"] = {}

        # 5: set pending action and call decide
        rl_policy.set_pending_action(action_dict)
        obs_dict = store.observe(market.current_step(), item_registry)
        action_used = store.decide(obs_dict)

        # 6–7: dispatch orders, process demand
        _dispatch_orders(store, market, event_engine, action_used)
        _process_demand(store, market, action_used)

        # Update sales history
        for pid, qty in store.sales.items():
            if pid in sales_history:
                sales_history[pid].append(qty)

        # Collect per-tick traces for active SKUs
        active_set = set(store.active_items)
        tick_sales = [store.sales.get(pid, 0) for pid in spec.active_subset]
        tick_demand = [store.demand.get(pid, 0) for pid in spec.active_subset]
        tick_inv = [store.inventory.get(pid, 0) for pid in spec.active_subset]
        tick_price = [store.prices.get(pid, 0.0) for pid in spec.active_subset]
        tick_msrp = [store.base_prices.get(pid, 1.0) for pid in spec.active_subset]
        tick_rev = [store.revenue.get(pid, 0.0) for pid in spec.active_subset]
        tick_hold = [store.holding_cost.get(pid, 0.0) for pid in spec.active_subset]
        tick_order_cost = [
            action_used.get("order", {}).get(pid, 0) * store.costs.get(pid, 0.0)
            for pid in spec.active_subset
        ]
        tick_fee: list[float] = []
        for pid in spec.active_subset:
            ordered = action_used.get("order", {}).get(pid, 0)
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


def _run_baseline(
    policy: Policy,
    spec: EpisodeSpec,
) -> dict[str, float]:
    """Run a baseline policy on ``spec`` for one full episode.

    Uses the same tick order as ``Runner.run()``.  Attaches ``policy``
    directly to the ``Store`` and uses ``Store.decide()``.  Collects
    per-tick traces into a ``RunSlice`` and returns ``aggregate_episode``
    metrics.
    """
    _, item_registry, market, event_engine, store = _build_world(spec)
    store.policy = policy

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    episode_length = spec.scenario.n_steps

    for _tick in range(episode_length):
        # 1–3: world ticks
        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        # 4–5: observe and decide
        obs_dict = store.observe(market.current_step(), item_registry)
        action = store.decide(obs_dict)

        # 6–7: dispatch orders, process demand
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


# ---------------------------------------------------------------------------
# Private: aggregate paired metrics
# ---------------------------------------------------------------------------


def _aggregate_paired(
    rl_list: list[dict[str, float]],
    baseline_list: list[dict[str, float]],
) -> dict[str, float]:
    """Compute paired summary statistics from per-seed metric dicts.

    Returns a flat dict with ``eval/`` prefix.
    """
    n = len(rl_list)
    assert n == len(baseline_list) and n > 0

    # Paired uplift and win-rate are computed on net_profit which is the
    # episode return proxy (maps to the reward accumulated by the RL env).
    rl_returns = [m.get("net_profit", 0.0) for m in rl_list]
    bl_returns = [m.get("net_profit", 0.0) for m in baseline_list]

    paired_uplift = [r - b for r, b in zip(rl_returns, bl_returns)]
    win_rate = float(sum(1 for u in paired_uplift if u >= 0)) / n

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    result: dict[str, float] = {
        "eval/rl_return": _mean(rl_returns),
        "eval/baseline_return": _mean(bl_returns),
        "eval/paired_uplift": _mean(paired_uplift),
        "eval/win_rate": win_rate,
        # RL business KPIs
        "eval/rl_service_level": _mean([m.get("service_level", 0.0) for m in rl_list]),
        "eval/rl_stockout_rate": _mean([m.get("stockout_rate", 0.0) for m in rl_list]),
        "eval/rl_inventory_turnover": _mean([m.get("inventory_turnover", 0.0) for m in rl_list]),
        "eval/rl_mean_price_pct_of_msrp": _mean([m.get("mean_price_pct_of_msrp", 1.0) for m in rl_list]),
        "eval/rl_revenue": _mean([m.get("revenue", 0.0) for m in rl_list]),
        "eval/rl_net_profit": _mean([m.get("net_profit", 0.0) for m in rl_list]),
        # Baseline business KPIs
        "eval/baseline_service_level": _mean([m.get("service_level", 0.0) for m in baseline_list]),
        "eval/baseline_stockout_rate": _mean([m.get("stockout_rate", 0.0) for m in baseline_list]),
        "eval/baseline_inventory_turnover": _mean([m.get("inventory_turnover", 0.0) for m in baseline_list]),
        "eval/baseline_mean_price_pct_of_msrp": _mean([m.get("mean_price_pct_of_msrp", 1.0) for m in baseline_list]),
        "eval/baseline_revenue": _mean([m.get("revenue", 0.0) for m in baseline_list]),
        "eval/baseline_net_profit": _mean([m.get("net_profit", 0.0) for m in baseline_list]),
    }

    return result


__all__ = ["build_eval_seeds", "evaluate", "PolicyFn", "BaselineFactory"]
