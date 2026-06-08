"""CRN-paired evaluation harness for the RL training framework.

``build_eval_seeds`` generates a fixed, deterministic list of ``RLEpisodeSpec``
objects from the configured eval-seed range (disjoint from the training range
via ``config.eval_seed_offset``).

``evaluate`` runs the RL policy and ``OrderUpToPolicy`` (via
``SingleSupplierAdapter``) on bit-identical ``RLEpisodeSpec`` tuples (Common
Random Numbers), computes paired metrics, and returns a flat dict suitable
for logging to TensorBoard under ``eval/*`` keys.

CRN guarantee
-------------
Both the RL and baseline runs for a given ``RLEpisodeSpec`` use the *same*
``world_seed``, capacity, balance, active subset, and slot permutation.  The
world trajectory is therefore bit-identical between the two runs, so any
difference in the returned metrics is attributable to the policy alone.

The CRN tuple is now expanded to include the ``allocation`` sub-seed (ADR 0016):
    (world_seed, capacity, balance, active_subset, slot_permutation, allocation_sub_seed)

Implementation notes
--------------------
- The RL policy is invoked as a callable ``(obs_tensor: np.ndarray) →
  action_vec: np.ndarray`` so the same eval can serve PPO, SAC, or any
  future agent without rewriting plumbing.
- The baseline policy is ``OrderUpToPolicy`` wrapped in ``SingleSupplierAdapter``
  so it runs on the graph engine's ``IntermediateNode``.
- The RL run uses a bespoke two-phase stepping loop (``tick_world()`` /
  ``tick_decide_and_settle()``) because the RL observation encoder must run
  between phases. The per-tick engine state (``_tick_purchases``,
  ``_tick_decision_state``, ``_tick_sink_demand``, closing inventory) is
  accumulated into a minimal run-log dict that is fed to the shared
  ``inspect`` + ``metrics`` DataFrame path (ADR 0019). The baseline run uses
  ``Runner.run()`` directly.

RL eval path finding (issue 07)
-------------------------------
``_run_rl`` does NOT go through ``Runner.run()``. It drives the engine with
the two-phase ``tick_world()`` / ``tick_decide_and_settle()`` API so the RL
observation encoder can snapshot state between phases. After each
``tick_decide_and_settle()`` the per-tick engine state is harvested and
appended to a run-log compatible list; this list is passed to the shared
``inspect.*`` builders at the end of the episode.

``_run_baseline`` DOES use ``Runner.run()`` via the ``Runner`` class — it
attaches the baseline policy via ``policy_overrides`` and calls
``Runner.run()`` exactly once.

Both paths then call ``inspect.flow_frame`` / ``inspect.purchase_frame`` /
``inspect.closing_inventory_frame`` / ``metrics.business_metrics`` /
``metrics.profit_decomposition`` for KPI extraction (ADR 0019).
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.rl.configs.default import RLConfig
from src.rl.encoders import compute_effective_rate, decode_action, encode_observation
from src.rl.episode_sampler import RLEpisodeSpec, sample_episode
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    RLIntermediatePolicy,
)
from src.sim.runner import build_world, Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Ware,
)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyFn = Callable[[np.ndarray], np.ndarray]
"""RL policy callable: takes an obs tensor, returns an action vector."""

BaselineFactory = Callable[[], IntermediatePolicy]
"""Factory that returns a fresh IntermediatePolicy (e.g. SingleSupplierAdapter
wrapping OrderUpToPolicy) per seed."""


# ---------------------------------------------------------------------------
# Public: build_eval_seeds
# ---------------------------------------------------------------------------


def build_eval_seeds(
    catalog: list[Ware],
    config: RLConfig,
    n_seeds: int | None = None,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
) -> list[RLEpisodeSpec]:
    """Generate a fixed list of held-out ``RLEpisodeSpec`` objects for evaluation.

    Seeds are taken from ``[config.eval_seed_offset, config.eval_seed_offset
    + n_seeds)``.  This range is intentionally far above the training seed
    space (default offset is 10_000_000) so eval seeds are never seen during
    training.

    Parameters
    ----------
    catalog:
        Full product universe; must have at least ``config.K_active`` items.
    config:
        ``RLConfig`` instance.
    n_seeds:
        Number of eval seeds.  Defaults to ``config.n_eval_seeds``.
    market_params, disruption_params, lifecycle_params:
        Optional simulator parameter overrides forwarded to ``sample_episode``.

    Returns
    -------
    list[RLEpisodeSpec]
        Deterministic, ordered list of ``n_seeds`` ``RLEpisodeSpec`` objects.
    """
    if n_seeds is None:
        n_seeds = config.n_eval_seeds

    specs: list[RLEpisodeSpec] = []
    for i in range(n_seeds):
        seed = config.eval_seed_offset + i
        spec = sample_episode(
            catalog=catalog,
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
    eval_specs: list[RLEpisodeSpec],
    *,
    config: RLConfig | None = None,
) -> dict[str, float]:
    """Run paired CRN evaluation and return a flat ``eval/*`` metrics dict.

    For each ``RLEpisodeSpec``:

    1. Run the RL policy through a step-by-step loop using the graph engine.
    2. Run a fresh baseline policy through the same spec.
    3. Compute ``aggregate_episode`` metrics for each run.
    4. Compute paired uplift (RL − baseline) per seed, then average.

    Parameters
    ----------
    rl_policy_fn:
        Callable ``(obs_tensor: np.ndarray) → action_vec: np.ndarray``.
    baseline_policy_factory:
        Zero-argument callable returning a fresh ``IntermediatePolicy`` instance.
    eval_specs:
        List of ``RLEpisodeSpec`` objects from ``build_eval_seeds``.
    config:
        ``RLConfig`` instance.  When ``None``, ``RLConfig()`` is used.

    Returns
    -------
    dict[str, float]
        Flat dict with keys prefixed ``eval/``.
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
        baseline_metrics = _run_baseline(baseline_policy, spec, config=config)
        rl_metrics_list.append(rl_metrics)
        baseline_metrics_list.append(baseline_metrics)

    return _aggregate_paired(rl_metrics_list, baseline_metrics_list)


# ---------------------------------------------------------------------------
# Private: single-episode runners
# ---------------------------------------------------------------------------


def _run_rl(
    rl_policy_fn: PolicyFn,
    spec: RLEpisodeSpec,
    *,
    config: RLConfig,
) -> dict[str, float]:
    """Run the RL policy on ``spec`` for one full episode.

    Uses the graph engine's two-phase tick API:
      sim.tick_world() → encode obs → decode action →
      RLIntermediatePolicy.set_pending_action() →
      sim.tick_decide_and_settle().

    After each tick the engine's internal per-tick state (``_tick_purchases``,
    ``_tick_decision_state``, ``_tick_sink_demand``, closing inventory) is
    harvested into a minimal run-log dict.  At the end of the episode this
    run-log is passed to the shared ``inspect`` + ``metrics`` DataFrame builders
    (ADR 0019) to produce KPIs on the same code path as the baseline.
    """
    from src.sim.inspect import (
        flow_frame,
        purchase_frame,
        closing_inventory_frame,
        node_timeseries_df,
    )
    from src.sim.metrics import business_metrics, profit_decomposition
    from src.sim.node import IntermediateNode

    rl_policy = RLIntermediatePolicy()
    sim = build_world(spec.scenario, policy_overrides={"S": rl_policy})
    node_s = sim.nodes["S"]

    K = config.K_active
    active_subset = spec.active_subset
    initial_cash = float(node_s.cash)
    sales_history: dict[str, deque] = {
        pid: deque(maxlen=100) for pid in active_subset
    }
    slot_perm = spec.slot_permutation
    supplier_ids_for = {pid: [f"F_{pid}"] for pid in active_subset}

    # Derive market-demand prior for cold-start ordering.
    from random import Random as _Random
    from src.sim.distributions import Distribution as _Distribution

    market_base_demand = getattr(spec.scenario.market, "base_demand", None)
    if isinstance(market_base_demand, _Distribution):
        prior_rng = _Random(spec.world_seed + 1)
        base_demand_prior = float(market_base_demand.sample(prior_rng))
    elif market_base_demand is not None:
        base_demand_prior = float(market_base_demand)
    else:
        base_demand_prior = 1.0

    episode_length = spec.scenario.n_steps
    base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_subset}

    # Accumulate per-tick log entries for the shared inspect builders.
    tick_entries: list[dict[str, Any]] = []

    for tick in range(episode_length):
        # Phase 1: advance world, publish offers.
        current_tick = sim.tick_world()
        central_table = getattr(sim, "_current_table", None)

        # Compute effective rate once per tick.
        effective_rate = compute_effective_rate(sales_history, base_demand_prior)

        # Encode obs.
        obs = encode_observation(
            node=node_s,
            market=sim.market,
            registry=sim.item_registry,
            step=tick,
            slot_perm=slot_perm,
            K_active=K,
            central_table=central_table,
            active_subset=active_subset,
            supplier_ids_for=supplier_ids_for,
            initial_cash=initial_cash,
            sales_history=sales_history,
            effective_rate=effective_rate,
        )

        # Call RL policy function.
        action_vec = rl_policy_fn(obs)

        # Decode action.
        all_supplier_ids = [f"F_{pid}" for pid in active_subset]
        action_dict = decode_action(
            np.asarray(action_vec, dtype=np.float32),
            slot_perm,
            node_s,
            K,
            base_prices,
            supplier_ids=all_supplier_ids,
            active_subset=active_subset,
            effective_rate=effective_rate,
            target_centre_lead_times=config.target_centre_lead_times,
            target_half_span_lead_times=config.target_half_span_lead_times,
            target_max_lead_times=config.target_max_lead_times,
        )

        # Inject action into RLIntermediatePolicy shim.
        rl_policy.set_pending_action(action_dict)

        # Phase 2: demand-pull walk (S.policy.decide() returns pending action).
        sim.tick_decide_and_settle(current_tick)

        # Update rolling sales history.
        last_sales = sim._tick_sales.get("S", {})
        for pid in active_subset:
            qty = last_sales.get(pid, 0)
            if pid in sales_history:
                sales_history[pid].append(qty)

        # Harvest per-tick engine state into a run-log compatible entry.
        # This mirrors what Runner._snapshot_tick() does for the standard path.
        node_cash: dict[str, float] = {}
        node_inventory: dict[str, Any] = {}
        node_pending: dict[str, Any] = {}

        for node_id, node in sim.nodes.items():
            from src.sim.node import DemandSinkNode, FactoryNode
            node_cash[node_id] = float(node.cash)
            if isinstance(node, FactoryNode):
                node_inventory[node_id] = {"_total": node.inventory}
                node_pending[node_id] = {}
            elif isinstance(node, IntermediateNode):
                node_inventory[node_id] = dict(node.inventory)
                pending_totals: dict[str, int] = {}
                for sup_pending in node.pending.values():
                    for pid, qty in sup_pending.items():
                        pending_totals[pid] = pending_totals.get(pid, 0) + qty
                node_pending[node_id] = pending_totals
            else:
                node_inventory[node_id] = {}
                node_pending[node_id] = {}

        # Build node_flows from decision_state + sink_demand + sales.
        purchases = list(sim._tick_purchases)
        decision_state = sim._tick_decision_state
        sink_demand = sim._tick_sink_demand

        sales_by: dict[tuple[str, str], int] = {}
        demand_by: dict[tuple[str, str], int] = {}
        for p in purchases:
            key = (p["supplier_id"], p["pid"])
            sales_by[key] = sales_by.get(key, 0) + p["qty_filled"]
            demand_by[key] = demand_by.get(key, 0) + p["qty_requested"]

        node_flows: list[dict[str, Any]] = []
        for node_id, pidmap in decision_state.items():
            for pid, st in pidmap.items():
                key = (node_id, pid)
                node_flows.append({
                    "node_id": node_id,
                    "pid": pid,
                    "sales": sales_by.get(key, 0),
                    "demand": demand_by.get(key, 0),
                    "price": st["price"],
                    "stockout": st["on_hand"] == 0,
                })
        for sink_id, pidmap in sink_demand.items():
            for pid, dt in pidmap.items():
                node_flows.append({
                    "node_id": sink_id,
                    "pid": pid,
                    "sales": 0,
                    "demand": int(dt),
                    "price": None,
                    "stockout": False,
                })

        purchases_log = [
            {
                "buyer_id": p["buyer_id"],
                "supplier_id": p["supplier_id"],
                "pid": p["pid"],
                "qty_filled": p["qty_filled"],
                "cash_paid": p["cash_paid"],
            }
            for p in purchases
        ]

        # Derive node_orders for completeness (inspect builders don't need it).
        node_orders: dict[str, dict[str, int]] = {
            nid: {}
            for nid, n in sim.nodes.items()
            if isinstance(n, IntermediateNode)
        }
        for p in purchases:
            buyer = sim.nodes.get(p["buyer_id"])
            if isinstance(buyer, IntermediateNode):
                d = node_orders[p["buyer_id"]]
                d[p["pid"]] = d.get(p["pid"], 0) + p["qty_requested"]

        tick_entries.append({
            "tick": current_tick,
            "node_cash": node_cash,
            "node_inventory": node_inventory,
            "node_pending": node_pending,
            "node_orders": node_orders,
            "node_flows": node_flows,
            "purchases": purchases_log,
        })

    # Build the run_log dict compatible with inspect builders.
    run_log: dict[str, Any] = {
        "n_steps": episode_length,
        "ticks": tick_entries,
        "global": {
            "time": {
                "simulation_step": list(range(episode_length + 1)),
                "simulation_date": [None] * (episode_length + 1),
            },
            "market_supply": {},
            "market_demand": {},
            "events": {"occurrences": [None] * episode_length},
        },
    }

    ff = flow_frame(run_log)
    pf = purchase_frame(run_log)
    cif = closing_inventory_frame(run_log)
    ts_df = node_timeseries_df(run_log, spec.scenario)

    bm = business_metrics(ff, ts_df, spec.scenario)
    pd_df = profit_decomposition(ff, pf, cif, spec.scenario)

    return _extract_node_s_metrics(bm, pd_df)


def _run_baseline(
    policy: IntermediatePolicy,
    spec: RLEpisodeSpec,
    *,
    config: RLConfig | None = None,
) -> dict[str, float]:
    """Run a baseline policy on ``spec`` for one full episode via ``Runner.run()``.

    Attaches ``policy`` via ``policy_overrides={"S": policy}`` and runs
    through ``Runner.run()`` (single-phase, standard run log).
    Returns metrics via the shared ``inspect`` + ``metrics`` DataFrame path.
    """
    from src.sim.inspect import (
        flow_frame,
        purchase_frame,
        closing_inventory_frame,
        node_timeseries_df,
    )
    from src.sim.metrics import business_metrics, profit_decomposition

    runner = Runner(spec.scenario, policy_overrides={"S": policy})
    run_log = runner.run()

    ff = flow_frame(run_log)
    pf = purchase_frame(run_log)
    cif = closing_inventory_frame(run_log)
    ts_df = node_timeseries_df(run_log, spec.scenario)

    bm = business_metrics(ff, ts_df, spec.scenario)
    pd_df = profit_decomposition(ff, pf, cif, spec.scenario)

    return _extract_node_s_metrics(bm, pd_df)


def _extract_node_s_metrics(
    bm: "Any",
    pd_df: "Any",
) -> dict[str, float]:
    """Extract KPIs for node "S" from business_metrics and profit_decomposition frames."""
    s_bm = bm[bm["node_id"] == "S"]
    if s_bm.empty:
        s_bm = bm[bm["node_id"] == "_system"]

    service_level = float(s_bm["service_level"].iloc[0]) if not s_bm.empty else 0.0
    stockout_rate = float(s_bm["stockout_rate"].iloc[0]) if not s_bm.empty else 0.0
    inventory_turnover = float(s_bm["inventory_turnover"].iloc[0]) if not s_bm.empty else 0.0
    mean_price_pct_of_msrp = float(s_bm["mean_price_pct_of_msrp"].iloc[0]) if not s_bm.empty else 1.0

    s_pd = pd_df[pd_df["node_id"] == "S"]
    net_profit = float(s_pd["net_profit"].iloc[0]) if not s_pd.empty else 0.0

    return {
        "net_profit": net_profit,
        "service_level": service_level,
        "stockout_rate": stockout_rate,
        "inventory_turnover": inventory_turnover,
        "mean_price_pct_of_msrp": mean_price_pct_of_msrp,
    }


# ---------------------------------------------------------------------------
# Private: aggregate paired metrics
# ---------------------------------------------------------------------------


def _aggregate_paired(
    rl_list: list[dict[str, float]],
    baseline_list: list[dict[str, float]],
) -> dict[str, float]:
    """Compute paired summary statistics from per-seed metric dicts."""
    n = len(rl_list)
    assert n == len(baseline_list) and n > 0

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
        "eval/rl_stockout_rate": _mean([m.get("stockout_rate", 0.0) for m in rl_list]),
        "eval/rl_inventory_turnover": _mean([m.get("inventory_turnover", 0.0) for m in rl_list]),
        "eval/rl_mean_price_pct_of_msrp": _mean([m.get("mean_price_pct_of_msrp", 1.0) for m in rl_list]),
        "eval/rl_net_profit": _mean([m.get("net_profit", 0.0) for m in rl_list]),
        "eval/baseline_stockout_rate": _mean([m.get("stockout_rate", 0.0) for m in baseline_list]),
        "eval/baseline_inventory_turnover": _mean([m.get("inventory_turnover", 0.0) for m in baseline_list]),
        "eval/baseline_mean_price_pct_of_msrp": _mean([m.get("mean_price_pct_of_msrp", 1.0) for m in baseline_list]),
        "eval/baseline_net_profit": _mean([m.get("net_profit", 0.0) for m in baseline_list]),
    }

    return result


# ---------------------------------------------------------------------------
# Two-scale eval dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairedSeedResult:
    """Per-seed metrics for a paired CRN evaluation run."""

    seed: int
    rl_net_profit: float
    baseline_net_profit: float
    uplift: float
    cold_start_qty_per_sku: float


@dataclass
class ScaleResult:
    """Aggregate metrics for one scale (small or flagship) across n_seeds."""

    mean_uplift: float
    uplift_ci_low: float
    uplift_ci_high: float
    mean_cold_start_qty: float
    cold_start_qty_p05: float
    cold_start_qty_p95: float
    per_seed: list[PairedSeedResult]


@dataclass
class TwoScaleEvalResult:
    """Combined result of a two-scale paired-CRN evaluation."""

    small: ScaleResult
    flagship: ScaleResult


# ---------------------------------------------------------------------------
# Public: evaluate_two_scale
# ---------------------------------------------------------------------------


def evaluate_two_scale(
    checkpoint_path: str,
    *,
    catalog: list[Ware],
    config: RLConfig,
    n_seeds: int = 32,
    seed_offset: int | None = None,
) -> TwoScaleEvalResult:
    """Run paired-CRN evaluation against OrderUpToPolicy at two scales.

    Two held-out eval sets, disjoint from the training seed range:
      - small: capacity_dist = Uniform(150, 400); balance proportional.
      - flagship: capacity_dist = Constant(10_000); balance proportional.

    Each set runs n_seeds CRN-paired seeds.  Per-scale results carry
    uplift mean, 95% bootstrap CI, and tick-0 cold-start qty distribution.

    Parameters
    ----------
    checkpoint_path:
        Path to a PyTorch checkpoint file (``*.pt``).  Pass ``""`` when
        the ``_rl_policy_fn_override`` attribute is set (e.g. in smoke tests).
    catalog, config, n_seeds, seed_offset:
        See class docstring.
    """
    from src.sim.distributions import Constant, Uniform

    if seed_offset is None:
        seed_offset = config.eval_seed_offset + 1_000_000

    _SMALL_SCALE_OFFSET = 0
    _FLAGSHIP_SCALE_OFFSET = 100_000

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

    # Baseline factory: OrderUpToPolicy via SingleSupplierAdapter
    # (wraps the store-based policy to run on the graph IntermediateNode).
    baseline_factory: BaselineFactory = _make_baseline_factory()

    small_result = _run_scale_eval(
        rl_policy_fn=rl_policy_fn,
        baseline_factory=baseline_factory,
        catalog=catalog,
        config=small_config,
        n_seeds=n_seeds,
        base_seed=seed_offset + _SMALL_SCALE_OFFSET,
    )

    flagship_result = _run_scale_eval(
        rl_policy_fn=rl_policy_fn,
        baseline_factory=baseline_factory,
        catalog=catalog,
        config=flagship_config,
        n_seeds=n_seeds,
        base_seed=seed_offset + _FLAGSHIP_SCALE_OFFSET,
    )

    return TwoScaleEvalResult(small=small_result, flagship=flagship_result)


def _make_baseline_factory() -> BaselineFactory:
    """Return a factory that creates a fresh baseline policy each call.

    The baseline is ``OrderUpToPolicy`` (via ``MultiSupplierTextbookPolicy``)
    which is already an ``IntermediatePolicy`` and runs natively on the graph engine.
    """
    from src.sim.policy import OrderUpToPolicy

    def factory() -> IntermediatePolicy:
        return OrderUpToPolicy(policy_seed=0)

    return factory


# ---------------------------------------------------------------------------
# Private: per-scale runner
# ---------------------------------------------------------------------------


def _run_scale_eval(
    *,
    rl_policy_fn: PolicyFn,
    baseline_factory: BaselineFactory,
    catalog: list[Ware],
    config: RLConfig,
    n_seeds: int,
    base_seed: int,
) -> ScaleResult:
    """Run paired CRN evaluation for a single scale."""
    per_seed_results: list[PairedSeedResult] = []

    for i in range(n_seeds):
        seed = base_seed + i
        spec = sample_episode(
            catalog=catalog,
            config=config,
            episode_seed=seed,
        )

        rl_metrics = _run_rl(rl_policy_fn, spec, config=config)
        baseline_policy = baseline_factory()
        baseline_metrics = _run_baseline(baseline_policy, spec, config=config)

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
    """Derive a deterministic integer seed from a list of floats."""
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
    """Compute a bootstrap confidence interval for the mean of ``samples``."""
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
    spec: RLEpisodeSpec,
    config: RLConfig,
) -> float:
    """Return the mean per-active-SKU order qty emitted by the RL policy on tick 0."""
    from src.sim.distributions import Distribution as _Distribution
    from random import Random as _Random

    rl_policy = RLIntermediatePolicy()
    sim = build_world(spec.scenario, policy_overrides={"S": rl_policy})
    node_s = sim.nodes["S"]

    K = config.K_active
    active_subset = spec.active_subset
    initial_cash = float(node_s.cash)
    sales_history: dict[str, deque] = {
        pid: deque(maxlen=100) for pid in active_subset
    }
    slot_perm = spec.slot_permutation
    supplier_ids_for = {pid: [f"F_{pid}"] for pid in active_subset}

    market_base_demand = getattr(spec.scenario.market, "base_demand", None)
    if isinstance(market_base_demand, _Distribution):
        prior_rng = _Random(spec.world_seed + 1)
        base_demand_prior = float(market_base_demand.sample(prior_rng))
    elif market_base_demand is not None:
        base_demand_prior = float(market_base_demand)
    else:
        base_demand_prior = 1.0

    # Only need tick 0 to measure cold-start.
    current_tick = sim.tick_world()
    central_table = getattr(sim, "_current_table", None)

    effective_rate = compute_effective_rate(sales_history, base_demand_prior)

    obs = encode_observation(
        node=node_s,
        market=sim.market,
        registry=sim.item_registry,
        step=0,
        slot_perm=slot_perm,
        K_active=K,
        central_table=central_table,
        active_subset=active_subset,
        supplier_ids_for=supplier_ids_for,
        initial_cash=initial_cash,
        sales_history=sales_history,
        effective_rate=effective_rate,
    )

    action_vec = rl_policy_fn(obs)
    base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_subset}
    all_supplier_ids = [f"F_{pid}" for pid in active_subset]

    action_dict = decode_action(
        np.asarray(action_vec, dtype=np.float32),
        slot_perm,
        node_s,
        K,
        base_prices,
        supplier_ids=all_supplier_ids,
        active_subset=active_subset,
        effective_rate=effective_rate,
        target_centre_lead_times=config.target_centre_lead_times,
        target_half_span_lead_times=config.target_half_span_lead_times,
        target_max_lead_times=config.target_max_lead_times,
    )

    orders = action_dict.get("order", {})
    # Count total qty across all order lines.
    total_qty = 0
    for pid, order_lines in orders.items():
        for _sup, qty in order_lines:
            total_qty += qty

    n_active = max(len(active_subset), 1)
    return float(total_qty) / n_active


def _load_policy_fn(checkpoint_path: str, config: RLConfig) -> PolicyFn:
    """Load an RL actor from a PyTorch checkpoint and return a policy callable."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"evaluate_two_scale: checkpoint not found: {checkpoint_path!r}"
        )

    import torch
    from src.rl.agents.ppo import Actor
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
            action, _, _ = actor.get_action_and_log_prob(t)
            return action.squeeze(0).numpy()

    return policy_fn


# ---------------------------------------------------------------------------
# Serialisation helpers for TwoScaleEvalResult
# ---------------------------------------------------------------------------


def _two_scale_result_to_dict(result: TwoScaleEvalResult) -> dict:
    """Recursively convert a ``TwoScaleEvalResult`` to a JSON-serialisable dict."""

    def _scale_result_to_dict(s: ScaleResult) -> dict:
        return dataclasses.asdict(s)

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

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Run two-scale paired-CRN evaluation for an RL checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world", default="rl_train")
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=None)

    args = parser.parse_args()

    from src.rl.episode_sampler import make_synthetic_catalog

    _cli_catalog = make_synthetic_catalog(20)

    _cli_config = RLConfig()

    print(f"Loading checkpoint: {args.checkpoint}")
    _result = evaluate_two_scale(
        checkpoint_path=args.checkpoint,
        catalog=_cli_catalog,
        config=_cli_config,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
    )

    _print_two_scale_table(_result)

    _ckpt = pathlib.Path(args.checkpoint)
    _report_path = _ckpt.parent / f"{_ckpt.stem}__two_scale_eval.json"
    _report_path.write_text(json.dumps(_two_scale_result_to_dict(_result), indent=2))
    print(f"JSON report written to: {_report_path}")
