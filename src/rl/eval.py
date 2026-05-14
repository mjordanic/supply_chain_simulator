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

import dataclasses
import json
import os
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Callable

import numpy as np

from src.rl.configs.default import RLConfig
from src.rl.encoders import compute_effective_rate, decode_action, encode_observation
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

    # Derive market-demand prior for cold-start ordering (mirrors RLEnv.reset).
    from random import Random as _Random
    from src.sim.distributions import Distribution as _Distribution

    market_base_demand = getattr(spec.scenario.market, "base_demand", None)
    if isinstance(market_base_demand, _Distribution):
        prior_rng = _Random(spec.scenario.world_seed + 1)
        base_demand_prior = float(market_base_demand.sample(prior_rng))
    elif market_base_demand is not None:
        base_demand_prior = float(market_base_demand)
    else:
        base_demand_prior = 1.0

    run_slice = RunSlice(active_pids=list(spec.active_subset))
    episode_length = spec.scenario.n_steps

    for tick in range(episode_length):
        # 1–3: world ticks
        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        # Compute effective rate once per tick.
        effective_rate = compute_effective_rate(sales_history, base_demand_prior)

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
            effective_rate=effective_rate,
            target_centre_lead_times=config.target_centre_lead_times,
            target_half_span_lead_times=config.target_half_span_lead_times,
            target_max_lead_times=config.target_max_lead_times,
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


# ---------------------------------------------------------------------------
# Two-scale eval dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairedSeedResult:
    """Per-seed metrics for a paired CRN evaluation run.

    Fields
    ------
    seed:
        The episode seed used for both RL and baseline runs.
    rl_net_profit:
        Net profit for the RL policy run.
    baseline_net_profit:
        Net profit for the baseline (OrderUpToPolicy) run.
    uplift:
        Paired uplift ``rl_net_profit - baseline_net_profit``.
    cold_start_qty_per_sku:
        Mean order quantity per active SKU on tick 0 for the RL policy.
    """

    seed: int
    rl_net_profit: float
    baseline_net_profit: float
    uplift: float
    cold_start_qty_per_sku: float


@dataclass
class ScaleResult:
    """Aggregate metrics for one scale (small or flagship) across n_seeds.

    Fields
    ------
    mean_uplift:
        Mean paired uplift across all seeds.
    uplift_ci_low:
        2.5th percentile of the bootstrap distribution of mean uplift.
    uplift_ci_high:
        97.5th percentile of the bootstrap distribution of mean uplift.
    mean_cold_start_qty:
        Mean tick-0 order qty per active SKU across all seeds.
    cold_start_qty_p05:
        5th percentile of the per-seed cold-start qty distribution.
    cold_start_qty_p95:
        95th percentile of the per-seed cold-start qty distribution.
    per_seed:
        Per-seed breakdown; length == n_seeds.  Retained for downstream
        auditing and recomputation.
    """

    mean_uplift: float
    uplift_ci_low: float
    uplift_ci_high: float
    mean_cold_start_qty: float
    cold_start_qty_p05: float
    cold_start_qty_p95: float
    per_seed: list[PairedSeedResult]


@dataclass
class TwoScaleEvalResult:
    """Combined result of a two-scale paired-CRN evaluation.

    Fields
    ------
    small:
        Results for the ``small`` scale (``capacity_dist = Uniform(150, 400)``).
    flagship:
        Results for the ``flagship`` scale (``capacity_dist = Constant(10_000)``).
    """

    small: ScaleResult
    flagship: ScaleResult


# ---------------------------------------------------------------------------
# Public: evaluate_two_scale
# ---------------------------------------------------------------------------


def evaluate_two_scale(
    checkpoint_path: str,
    *,
    catalog: list[Ware],
    base_template: Any,
    config: RLConfig,
    n_seeds: int = 32,
    seed_offset: int | None = None,
) -> TwoScaleEvalResult:
    """Run paired-CRN evaluation against OrderUpToPolicy at two scales.

    Two held-out eval sets, disjoint from the training seed range:
      - small: capacity_dist = Uniform(150, 400); balance proportional.
      - flagship: capacity_dist = Constant(10_000); balance proportional.

    Each set runs n_seeds CRN-paired seeds (same world_seed / init_seed
    for the RL policy and OrderUpToPolicy). Per-scale results carry
    uplift mean, 95% bootstrap CI, and tick-0 cold-start qty distribution.

    Parameters
    ----------
    checkpoint_path:
        Path to a PyTorch checkpoint file (``*.pt``).  The checkpoint is
        loaded and the actor is used as the RL policy callable.  Pass an
        empty string or ``""`` when the rl_policy_fn override is used
        (e.g. in smoke tests via ``_rl_policy_fn_override``).
    catalog:
        Full product universe for episode sampling.
    base_template:
        ``StoreTemplate`` carrying non-episodic knobs.
    config:
        Base ``RLConfig``.  Per-scale configs are derived via
        ``dataclasses.replace`` — the input config is never mutated.
    n_seeds:
        Number of CRN seeds per scale.
    seed_offset:
        Base seed offset.  Defaults to
        ``config.eval_seed_offset + 1_000_000``.

    Returns
    -------
    TwoScaleEvalResult
        ``small`` and ``flagship`` ``ScaleResult`` objects with uplift
        means, 95 % bootstrap CIs, and cold-start qty statistics.
    """
    from src.sim.distributions import Constant, Uniform

    if seed_offset is None:
        seed_offset = config.eval_seed_offset + 1_000_000

    # Per-scale seed offsets — disjoint from each other and from training.
    _SMALL_SCALE_OFFSET = 0
    _FLAGSHIP_SCALE_OFFSET = 100_000

    # Build per-scale configs (never mutate the input config).
    small_config = dataclasses.replace(
        config,
        capacity_dist=Uniform(150, 400),
        balance_dist=Uniform(15_000, 40_000),
    )
    flagship_config = dataclasses.replace(
        config,
        capacity_dist=Constant(10_000),
        balance_dist=Uniform(500_000, 2_000_000),
    )

    # Load RL policy callable from checkpoint (skip for override mode).
    rl_policy_fn: PolicyFn
    if hasattr(evaluate_two_scale, "_rl_policy_fn_override"):
        rl_policy_fn = evaluate_two_scale._rl_policy_fn_override  # type: ignore[attr-defined]
    elif checkpoint_path:
        rl_policy_fn = _load_policy_fn(checkpoint_path, config)
    else:
        raise ValueError(
            "evaluate_two_scale: checkpoint_path is empty and no "
            "_rl_policy_fn_override is set.  Pass a valid checkpoint path."
        )

    baseline_factory: BaselineFactory = lambda: OrderUpToPolicy(policy_seed=0)

    small_result = _run_scale_eval(
        rl_policy_fn=rl_policy_fn,
        baseline_factory=baseline_factory,
        catalog=catalog,
        base_template=base_template,
        config=small_config,
        n_seeds=n_seeds,
        base_seed=seed_offset + _SMALL_SCALE_OFFSET,
    )

    flagship_result = _run_scale_eval(
        rl_policy_fn=rl_policy_fn,
        baseline_factory=baseline_factory,
        catalog=catalog,
        base_template=base_template,
        config=flagship_config,
        n_seeds=n_seeds,
        base_seed=seed_offset + _FLAGSHIP_SCALE_OFFSET,
    )

    return TwoScaleEvalResult(small=small_result, flagship=flagship_result)


# ---------------------------------------------------------------------------
# Private: per-scale runner
# ---------------------------------------------------------------------------


def _run_scale_eval(
    *,
    rl_policy_fn: PolicyFn,
    baseline_factory: BaselineFactory,
    catalog: list[Ware],
    base_template: Any,
    config: RLConfig,
    n_seeds: int,
    base_seed: int,
) -> ScaleResult:
    """Run paired CRN evaluation for a single scale.

    Parameters
    ----------
    rl_policy_fn:
        RL policy callable.
    baseline_factory:
        Factory producing a fresh baseline policy per seed.
    catalog, base_template, config:
        Episode sampling inputs.
    n_seeds:
        Number of seeds.
    base_seed:
        First seed integer; seeds are ``[base_seed, base_seed + n_seeds)``.

    Returns
    -------
    ScaleResult
        Aggregate stats including bootstrap CI and cold-start qty.
    """
    per_seed_results: list[PairedSeedResult] = []

    for i in range(n_seeds):
        seed = base_seed + i
        spec = sample_episode(
            catalog=catalog,
            base_template=base_template,
            config=config,
            episode_seed=seed,
        )

        rl_metrics = _run_rl(rl_policy_fn, spec, config=config)
        baseline_policy = baseline_factory()
        baseline_metrics = _run_baseline(baseline_policy, spec)

        cold_start_qty = _cold_start_qty_per_sku(rl_policy_fn, spec, config)

        per_seed_results.append(
            PairedSeedResult(
                seed=seed,
                rl_net_profit=rl_metrics.get("net_profit", 0.0),
                baseline_net_profit=baseline_metrics.get("net_profit", 0.0),
                uplift=rl_metrics.get("net_profit", 0.0) - baseline_metrics.get("net_profit", 0.0),
                cold_start_qty_per_sku=cold_start_qty,
            )
        )

    uplifts = [r.uplift for r in per_seed_results]
    cold_starts = [r.cold_start_qty_per_sku for r in per_seed_results]

    mean_uplift = float(np.mean(uplifts))
    uplift_ci_low, uplift_ci_high = _bootstrap_ci(uplifts, seed=seed_from_list(uplifts))
    mean_cold_start = float(np.mean(cold_starts))
    p05 = float(np.percentile(cold_starts, 5))
    p95 = float(np.percentile(cold_starts, 95))

    return ScaleResult(
        mean_uplift=mean_uplift,
        uplift_ci_low=uplift_ci_low,
        uplift_ci_high=uplift_ci_high,
        mean_cold_start_qty=mean_cold_start,
        cold_start_qty_p05=p05,
        cold_start_qty_p95=p95,
        per_seed=per_seed_results,
    )


def seed_from_list(vals: list[float]) -> int:
    """Derive a deterministic integer seed from a list of floats.

    Used to seed the bootstrap RNG in a reproducible, input-dependent way
    so the CI is consistent across repeated calls with the same uplift data.
    """
    h = hash(tuple(round(v, 6) for v in vals))
    return abs(h) % (2**31)


def _bootstrap_ci(
    samples: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of ``samples``.

    Parameters
    ----------
    samples:
        Observed data (per-seed paired uplifts).
    n_resamples:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.
    ci_low, ci_high:
        Percentile bounds (default: 2.5 / 97.5 → 95 % CI).

    Returns
    -------
    tuple[float, float]
        ``(lower_bound, upper_bound)`` of the CI.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    return float(np.percentile(boot_means, ci_low)), float(np.percentile(boot_means, ci_high))


def _cold_start_qty_per_sku(
    rl_policy_fn: PolicyFn,
    spec: EpisodeSpec,
    config: RLConfig,
) -> float:
    """Return the mean per-active-SKU order qty emitted by the RL policy on tick 0.

    Runs a fresh episode rollout and captures the action dict produced on
    the first tick, then averages the positive order quantities across
    the K_active active SKUs.
    """
    from src.sim.distributions import Distribution as _Distribution
    from random import Random as _Random

    _, item_registry, market, event_engine, store = _build_world(spec)
    from src.sim.policy import RLPolicy

    rl_policy = RLPolicy()
    store.policy = rl_policy

    K = config.K_active
    initial_cash = float(store.balance)
    sales_history: dict[str, deque] = {
        pid: deque(maxlen=100) for pid in [w.product_id for w in spec.scenario.catalog]
    }
    slot_perm = spec.slot_permutation

    market_base_demand = getattr(spec.scenario.market, "base_demand", None)
    if isinstance(market_base_demand, _Distribution):
        prior_rng = _Random(spec.scenario.world_seed + 1)
        base_demand_prior = float(market_base_demand.sample(prior_rng))
    elif market_base_demand is not None:
        base_demand_prior = float(market_base_demand)
    else:
        base_demand_prior = 1.0

    # Only need tick 0.
    market.tick()
    event_engine.tick(market)
    item_registry.tick()

    effective_rate = compute_effective_rate(sales_history, base_demand_prior)

    obs = encode_observation(
        store,
        market,
        item_registry,
        step=0,
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
        effective_rate=effective_rate,
        target_centre_lead_times=config.target_centre_lead_times,
        target_half_span_lead_times=config.target_half_span_lead_times,
        target_max_lead_times=config.target_max_lead_times,
    )

    orders = action_dict.get("order", {})
    active_pids = list(spec.active_subset)
    total_qty = sum(orders.get(pid, 0) for pid in active_pids)
    n_active = max(len(active_pids), 1)
    return float(total_qty) / n_active


def _load_policy_fn(checkpoint_path: str, config: RLConfig) -> PolicyFn:
    """Load an RL actor from a PyTorch checkpoint and return a policy callable.

    The actor is wrapped into a stateless ``(obs: np.ndarray) → action: np.ndarray``
    callable that maps a flat float32 observation to a float32 action vector
    in ``[-1, 1]^(2 * K_active)``.

    Raises ``FileNotFoundError`` if ``checkpoint_path`` does not exist.
    ``RuntimeError`` propagates if the checkpoint is incompatible with the
    current observation shape (e.g. a stale ADR-0005-era checkpoint).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"evaluate_two_scale: checkpoint not found: {checkpoint_path!r}"
        )

    import torch
    from src.rl.agents.actor_critic import Actor

    from src.rl.encoders import observation_dim, action_dim

    obs_dim = observation_dim(config.K_active)
    act_dim = action_dim(config.K_active)
    actor = Actor(obs_dim=obs_dim, action_dim=act_dim)

    state = torch.load(checkpoint_path, map_location="cpu")
    actor_state = state.get("actor", state)
    actor.load_state_dict(actor_state)
    actor.eval()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            action, _, _ = actor.get_action(t)
            return action.squeeze(0).numpy()

    return policy_fn


# ---------------------------------------------------------------------------
# Serialisation helpers for TwoScaleEvalResult
# ---------------------------------------------------------------------------


def _two_scale_result_to_dict(result: TwoScaleEvalResult) -> dict:
    """Recursively convert a ``TwoScaleEvalResult`` to a JSON-serialisable dict."""

    def _seed_result_to_dict(r: PairedSeedResult) -> dict:
        return dataclasses.asdict(r)

    def _scale_result_to_dict(s: ScaleResult) -> dict:
        d = dataclasses.asdict(s)
        return d

    return {
        "small": _scale_result_to_dict(result.small),
        "flagship": _scale_result_to_dict(result.flagship),
    }


def _print_two_scale_table(result: TwoScaleEvalResult) -> None:
    """Print a human-readable console summary of a ``TwoScaleEvalResult``."""
    print("\n=== Two-Scale Paired-CRN Evaluation Results ===\n")
    for name, scale in [("small", result.small), ("flagship", result.flagship)]:
        n = len(scale.per_seed)
        print(f"  Scale: {name}  (n={n} seeds)")
        print(f"    Mean uplift:       {scale.mean_uplift:+.2f}")
        print(f"    95% CI:            [{scale.uplift_ci_low:+.2f}, {scale.uplift_ci_high:+.2f}]")
        print(f"    Cold-start qty/SKU (mean): {scale.mean_cold_start_qty:.2f}")
        print(f"    Cold-start qty/SKU (p05/p95): {scale.cold_start_qty_p05:.2f} / {scale.cold_start_qty_p95:.2f}")
        print()


__all__ = [
    "build_eval_seeds",
    "evaluate",
    "evaluate_two_scale",
    "TwoScaleEvalResult",
    "ScaleResult",
    "PairedSeedResult",
    "PolicyFn",
    "BaselineFactory",
]


# ---------------------------------------------------------------------------
# __main__ — CLI entry point
# ---------------------------------------------------------------------------
# Invoked as:
#   uv run python -m src.rl.eval --checkpoint runs/<name>/<step>.pt \
#       --world rl_train --n-seeds 32 --seed-offset 11000000
#
# Output: JSON report written to <checkpoint_dir>/<stem>__two_scale_eval.json
# and a console table printed to stdout.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Run two-scale paired-CRN evaluation for an RL checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a PyTorch checkpoint file (*.pt).",
    )
    parser.add_argument(
        "--world",
        default="rl_train",
        help="World archetype label (default: rl_train).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=32,
        help="Number of CRN seeds per scale (default: 32).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help="Base seed offset (default: config.eval_seed_offset + 1_000_000).",
    )

    args = parser.parse_args()

    # Build a minimal catalog and template for the CLI run.
    from src.sim.scenario import StoreTemplate, load_catalog

    _cli_catalog = load_catalog(
        [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0 + i * 0.3,
                "seasonality": "all_season",
            }
            for i in range(20)
        ]
    )
    _cli_template = StoreTemplate(
        id="cli_eval",
        region="US",
        capacity=200,
        init_balance=20_000.0,
        init_stock_pct=0.0,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )

    _cli_config = RLConfig(world_archetype=args.world)

    print(f"Loading checkpoint: {args.checkpoint}")
    _result = evaluate_two_scale(
        checkpoint_path=args.checkpoint,
        catalog=_cli_catalog,
        base_template=_cli_template,
        config=_cli_config,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
    )

    _print_two_scale_table(_result)

    # Write JSON report next to the checkpoint.
    _ckpt = pathlib.Path(args.checkpoint)
    _report_path = _ckpt.parent / f"{_ckpt.stem}__two_scale_eval.json"
    _report_path.write_text(
        json.dumps(_two_scale_result_to_dict(_result), indent=2)
    )
    print(f"JSON report written to: {_report_path}")
