"""Synthetic per-SKU transition log on the tuner episode shape.

The buyer is a mixture: all four textbook families, with hyperparameters
drawn from the tuning search spaces; ``ExploratoryPolicy``; never-order;
and over-order. Over-order contributes more episodes than each other arm,
because a competent cover orders too rarely to carry the action
distribution. Planning depth varies by episode. Factories stay on
``StaticFactoryPolicy``. Seeds start at ``14_000_000``, disjoint from the
tuner search and holdout offsets. Every episode uses one holding rate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.jepa.dynamics import (
    ARRIVAL_AGE_SLOTS,
    SALES_WINDOW,
    arrival_age_profile,
    padded_sales_window,
)
from src.jepa.policy import ExploratoryPolicy, NeverOrderPolicy, OverOrderPolicy
from src.sim.distributions import Constant
from src.sim.node import FactoryNode
from src.sim.policy import IntermediatePolicy, StaticFactoryPolicy
from src.sim.runner import Runner
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode
from src.tuning.search_spaces import (
    COVER_HORIZON_TICKS,
    sample_cover_horizon_ticks,
    sample_textbook_policy,
)

LOG_SEED_OFFSET = 14_000_000
EPISODES_EACH = 64
# Competent covers place a handful of orders and then wait. An equal share
# of those arms leaves the realised non-zero fill near the old ~13%.
# Over-order keeps asking, so it supplies most of the purchases.
OVER_ORDER_WEIGHT = 8

# Textbook families, then the exploratory arm, then the two poor arms.
# Over-order is last: its seed block follows the others.
BEHAVIORS: tuple[str, ...] = (
    "order_up_to",
    "reorder_point",
    "periodic_order_up_to",
    "periodic_reorder",
    "exploratory",
    "never_order",
    "over_order",
)
TEXTBOOK_BEHAVIORS: frozenset[str] = frozenset(
    {
        "order_up_to",
        "reorder_point",
        "periodic_order_up_to",
        "periodic_reorder",
    }
)


def catalog_identity(catalog) -> str:
    """Stable hash of the catalog records that produced the log."""
    rows = [ware._asdict() for ware in catalog]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


COLUMNS = (
    "episode_seed",
    "behavior",
    "tick",
    "product_id",
    "on_hand",
    "in_transit",
    "sales_window",
    "cash",
    "capacity",
    "requested_qty",
    "filled_qty",
    "delivery_lag",
    "next_on_hand",
    "next_in_transit",
    "next_sales",
)


def collection_config(
    *,
    episode_length: int = 365,
    k_active: int | None = None,
    delivery_lag: int | None = None,
    holding_rate: float = 0.01,
    order_fee: float | None = None,
) -> TuningConfig:
    """Tuner episode shape used for the log.

    Unspecified knobs (capacity, opening cash, initial stock, assortment
    size, nominal delivery lag) stay at the ``TuningConfig`` defaults, so
    K and the delivery lag are sampled unless the caller pins them.
    ``order_fee`` stays at that default unless the caller passes one.
    """
    kwargs: dict[str, Any] = {
        "episode_length": episode_length,
        "delivery_lag": delivery_lag,
        "holding_rate": holding_rate,
    }
    if k_active is not None:
        kwargs["K_active"] = k_active
    if order_fee is not None:
        kwargs["order_fee"] = order_fee
    return TuningConfig(**kwargs)


def build_log_scenario(config: TuningConfig, episode_seed: int, catalog: list | None = None):
    """One tuner-graph episode at ``episode_seed``."""
    if catalog is None:
        catalog = make_synthetic_catalog(config.active_catalog_size())
    return sample_episode(catalog, config, episode_seed)


def episodes_for(behavior: str, n_episodes_each: int) -> int:
    """How many episodes ``behavior`` contributes at this base count."""
    weight = OVER_ORDER_WEIGHT if behavior == "over_order" else 1
    return int(n_episodes_each) * weight


def behavior_schedule(
    n_episodes_each: int,
    seed_offset: int,
) -> list[tuple[str, int]]:
    """``(behavior, episode_seed)`` in manifest order, one block per arm."""
    schedule: list[tuple[str, int]] = []
    cursor = int(seed_offset)
    for behavior in BEHAVIORS:
        count = episodes_for(behavior, n_episodes_each)
        for offset in range(count):
            schedule.append((behavior, cursor + offset))
        cursor += count
    return schedule


def log_buyer(
    behavior: str,
    episode_seed: int,
    delivery_lag: int,
) -> tuple[IntermediatePolicy, dict[str, Any]]:
    """Buyer policy and the hyperparameters the manifest records.

    Textbook arms are built by the tuning space factories. Planning depth
    is that draw's ``cover_horizon_ticks``. Exploratory depth is its own
    draw of the same range and replaces ``delivery_lag`` in the target.
    Never-order is depth 0. Over-order's depth sits strictly above the
    textbook cover range.
    """
    if behavior in TEXTBOOK_BEHAVIORS:
        policy, params = sample_textbook_policy(behavior, episode_seed)
        _reseed(policy, episode_seed)
        depth = int(params["cover_horizon_ticks"])
        hyper = {"episode_seed": int(episode_seed), "planning_depth": depth, **params}
        return policy, hyper
    if behavior == "exploratory":
        depth = sample_cover_horizon_ticks(episode_seed)
        policy = ExploratoryPolicy(
            policy_seed=episode_seed,
            delivery_lag=delivery_lag,
            planning_depth=depth,
        )
        return policy, {"episode_seed": int(episode_seed), "planning_depth": depth}
    if behavior == "never_order":
        return NeverOrderPolicy(policy_seed=episode_seed), {
            "episode_seed": int(episode_seed),
            "planning_depth": 0,
        }
    if behavior == "over_order":
        depth = COVER_HORIZON_TICKS[1] + sample_cover_horizon_ticks(episode_seed)
        return OverOrderPolicy(policy_seed=episode_seed, planning_depth=depth), {
            "episode_seed": int(episode_seed),
            "planning_depth": depth,
        }
    raise ValueError(f"unknown log behavior {behavior!r}")


def _reseed(policy: IntermediatePolicy, seed: int) -> None:
    """Pin ``policy_rng`` so a space factory's unseeded constructor is stable."""
    policy.policy_seed = int(seed)
    policy.policy_rng.seed(int(seed))
    inner = getattr(policy, "_inner", None)
    if inner is not None and hasattr(inner, "policy_rng"):
        inner.policy_seed = int(seed)
        inner.policy_rng.seed(int(seed))


def _factory_overrides(scenario) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for instance in scenario.nodes:
        node = instance.node
        if isinstance(node, FactoryNode):
            overrides[node.id] = StaticFactoryPolicy(
                capacity_per_tick=_factory_capacity(node),
                unit_cost=float(node.unit_cost),
                policy_seed=node.init_seed,
            )
    return overrides


def build_log_overrides(
    scenario,
    *,
    behavior: str,
    policy_seed: int,
    delivery_lag: int,
) -> dict[str, Any]:
    """``StaticFactoryPolicy`` on every factory, and one buyer policy on ``S``.

    The buyer is the mixture arm for ``behavior`` at ``policy_seed``.
    Textbook hyperparameters come from the tuning search spaces.
    """
    overrides = _factory_overrides(scenario)
    policy, _hyper = log_buyer(behavior, policy_seed, delivery_lag)
    overrides["S"] = policy
    return overrides


def collect_episode_transitions(
    scenario,
    buyer_policy: IntermediatePolicy,
    *,
    behavior: str,
    episode_seed: int,
    delivery_lag: int,
) -> pd.DataFrame:
    """Per-SKU transitions for one episode on shop ``S``.

    ``requested_qty`` is the quantity ``buyer_policy`` emitted. ``filled_qty``
    is the quantity Allocation filled.
    """
    rows, _capacity_ticks, _cash_ticks = _record_episode(
        scenario,
        buyer_policy,
        behavior=behavior,
        episode_seed=episode_seed,
        delivery_lag=delivery_lag,
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def collect_transition_log(
    *,
    n_episodes_each: int = EPISODES_EACH,
    episode_length: int | None = None,
    k_active: int | None = None,
    delivery_lag: int | None = None,
    seed_offset: int = LOG_SEED_OFFSET,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Collect the per-SKU transition log.

    Defaults are the published shape from ``collection_config``: 365 ticks,
    K and the nominal delivery lag sampled from the tuner ranges, one
    ``holding_rate`` (0.01), the tuner order fee, and ``EPISODES_EACH``
    episodes of each arm except over-order, which runs
    ``OVER_ORDER_WEIGHT`` times that many, from ``LOG_SEED_OFFSET``. Pass a
    shorter episode or a smaller count for a smoke run. Pass
    ``delivery_lag`` to pin the nominal lag.

    Arms occupy contiguous seed blocks in ``BEHAVIORS`` order. The last
    tick of each episode is omitted: it has no next observation.

    When ``path`` is set, the parquet is written there and a
    ``<stem>.manifest.json`` sidecar is written beside it.
    """
    shape: dict[str, Any] = {}
    if episode_length is not None:
        shape["episode_length"] = episode_length
    if k_active is not None:
        shape["k_active"] = k_active
    if delivery_lag is not None:
        shape["delivery_lag"] = delivery_lag
    config = collection_config(**shape)
    catalog = make_synthetic_catalog(config.active_catalog_size())

    rows: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    capacity_bound: set[tuple[int, int]] = set()
    cash_bound: set[tuple[int, int]] = set()
    for behavior, episode_seed in behavior_schedule(n_episodes_each, seed_offset):
        spec = build_log_scenario(config, episode_seed, catalog)
        policy, hyper = log_buyer(behavior, episode_seed, spec.delivery_lag)
        episode_rows, cap_ticks, cash_ticks = _run_episode(
            spec.scenario,
            policy,
            behavior=behavior,
            episode_seed=episode_seed,
            delivery_lag=spec.delivery_lag,
        )
        rows.extend(episode_rows)
        sampled.append({"behavior": behavior, "policy": type(policy).__name__, **hyper})
        capacity_bound.update((episode_seed, tick) for tick in cap_ticks)
        cash_bound.update((episode_seed, tick) for tick in cash_ticks)

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        _write_manifest(
            destination,
            frame,
            config,
            catalog,
            n_episodes_each=n_episodes_each,
            seed_offset=seed_offset,
            sampled=sampled,
            capacity_bound=capacity_bound,
            cash_bound=cash_bound,
        )
    return frame


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def state_coverage(
    frame: pd.DataFrame,
    *,
    capacity_bound: set[tuple[int, int]],
    cash_bound: set[tuple[int, int]],
) -> dict[str, float]:
    """Share of transition rows in each reported state.

    ``low_inventory`` is on-hand of zero. ``high_inventory`` is a tick whose
    on-hand sums to at least half the shared shelf. ``capacity_bound`` and
    ``cash_bound`` are ticks where the request was short and the shelf, or
    the cash left after payment, is what stopped another unit.
    """
    keys = (
        "low_inventory",
        "high_inventory",
        "capacity_bound",
        "cash_bound",
    )
    if len(frame) == 0:
        return {key: 0.0 for key in keys}
    low = float((frame["on_hand"] == 0).mean())
    shelf = frame.groupby(["episode_seed", "tick"])["on_hand"].transform("sum")
    high = float(((frame["capacity"] > 0) & (shelf >= 0.5 * frame["capacity"])).mean())
    index = pd.MultiIndex.from_frame(frame[["episode_seed", "tick"]].astype(int))
    cap = float(index.isin(pd.MultiIndex.from_tuples(capacity_bound)).mean()) if capacity_bound else 0.0
    cash = float(index.isin(pd.MultiIndex.from_tuples(cash_bound)).mean()) if cash_bound else 0.0
    return {
        "low_inventory": low,
        "high_inventory": high,
        "capacity_bound": cap,
        "cash_bound": cash,
    }


def _mixture_manifest(sampled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {behavior: [] for behavior in BEHAVIORS}
    policies: dict[str, str] = {}
    for row in sampled:
        behavior = str(row["behavior"])
        policies[behavior] = str(row["policy"])
        hyper = {key: value for key, value in row.items() if key not in {"behavior", "policy"}}
        grouped[behavior].append(hyper)
    return [
        {
            "behavior": behavior,
            "episodes": len(grouped[behavior]),
            "policy": policies[behavior],
            "hyperparameters": grouped[behavior],
        }
        for behavior in BEHAVIORS
        if grouped[behavior]
    ]


def _write_manifest(
    path: Path,
    frame: pd.DataFrame,
    config: TuningConfig,
    catalog,
    *,
    n_episodes_each: int,
    seed_offset: int,
    sampled: list[dict[str, Any]],
    capacity_bound: set[tuple[int, int]],
    cash_bound: set[tuple[int, int]],
) -> None:
    """Sidecar of every parameter that produced ``frame``.

    ``nonzero_order_fraction`` is the share of rows whose filled quantity
    is non-zero: the order Allocation realised, not the request.
    ``holding_rate`` is one number. The log is not a holding-rate sweep.
    """
    schedule = behavior_schedule(n_episodes_each, seed_offset)
    if len(frame) == 0:
        fraction = 0.0
    else:
        fraction = float((frame["filled_qty"] > 0).mean())
    seeds = [seed for _behavior, seed in schedule]
    manifest = {
        "episode_count": len(schedule),
        "seed_min": int(min(seeds)) if seeds else int(seed_offset),
        "seed_max": int(max(seeds)) if seeds else int(seed_offset),
        "episode_length": int(config.episode_length),
        "K_active": None if config.K_active is None else int(config.K_active),
        "K_min": int(config.K_min),
        "K_max_episode": int(config.K_max_episode),
        "delivery_lag": (
            None if config.delivery_lag is None else int(config.delivery_lag)
        ),
        "holding_rate": float(config.holding_rate),
        "order_fee": float(config.order_fee),
        "catalog_identity": catalog_identity(catalog),
        "behavior_mixture": _mixture_manifest(sampled),
        "state_coverage": state_coverage(
            frame, capacity_bound=capacity_bound, cash_bound=cash_bound
        ),
        "git_sha": _git_sha(),
        "nonzero_order_fraction": fraction,
    }
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _TransitionRecorder(IntermediatePolicy):
    """Record the observation window and the requested quantity at each decide."""

    def __init__(self, inner: IntermediatePolicy) -> None:
        super().__init__(policy_seed=getattr(inner, "policy_seed", None))
        self._inner = inner
        self._sales: dict[str, list[int]] = {}
        self.steps: list[dict[str, dict[str, Any]]] = []

    def decide(self, obs_intermediate: Mapping[str, Any], central_table: Any) -> dict[str, Any]:
        action = self._inner.decide(obs_intermediate, central_table)
        self.steps.append(_snapshot(obs_intermediate, action, self._sales))
        return action


def _run_episode(
    scenario,
    buyer: IntermediatePolicy,
    *,
    behavior: str,
    episode_seed: int,
    delivery_lag: int,
) -> tuple[list[dict], set[int], set[int]]:
    overrides = _factory_overrides(scenario)
    return _record_episode(
        scenario,
        buyer,
        behavior=behavior,
        episode_seed=episode_seed,
        delivery_lag=delivery_lag,
        policy_overrides=overrides,
    )


def _record_episode(
    scenario,
    buyer_policy: IntermediatePolicy,
    *,
    behavior: str,
    episode_seed: int,
    delivery_lag: int,
    policy_overrides: Mapping[str, Any] | None = None,
) -> tuple[list[dict], set[int], set[int]]:
    overrides = dict(policy_overrides or {})
    recorder = _TransitionRecorder(buyer_policy)
    overrides["S"] = recorder
    run_log = Runner(scenario, policy_overrides=overrides).run()
    _annotate_steps(recorder.steps, run_log, scenario, "S", delivery_lag)
    capacity_ticks, cash_ticks = _constraint_ticks(recorder.steps, run_log)
    return (
        _transitions(recorder.steps, episode_seed, behavior, delivery_lag),
        capacity_ticks,
        cash_ticks,
    )


def _constraint_ticks(
    steps: list[dict[str, dict[str, Any]]],
    run_log: Mapping[str, Any],
) -> tuple[set[int], set[int]]:
    """Ticks where the shelf or the till stopped a short order.

    The last step has no transition row, so it is not counted. Cash left is
    the observation cash minus what Allocation paid. A unit costs whatever
    this tick paid per filled unit, or the last such price in the episode.
    """
    spent_by_tick: dict[int, float] = {}
    for tick_log in run_log["ticks"]:
        spent = 0.0
        for purchase in tick_log.get("purchases") or []:
            if purchase.get("buyer_id") != "S":
                continue
            spent += float(purchase.get("cash_paid") or 0.0)
        spent_by_tick[int(tick_log["tick"])] = spent

    capacity_ticks: set[int] = set()
    cash_ticks: set[int] = set()
    last_unit: float | None = None
    for step in steps[:-1]:
        if not step:
            continue
        first = next(iter(step.values()))
        tick = int(first["tick"])
        on_hand = sum(int(snap["on_hand"]) for snap in step.values())
        filled = sum(int(snap["filled_qty"]) for snap in step.values())
        requested = sum(int(snap["requested_qty"]) for snap in step.values())
        capacity = int(first["capacity"])
        cash = float(first["cash"])
        spent = spent_by_tick.get(tick, 0.0)
        if capacity > 0 and requested > filled and on_hand + filled >= capacity:
            capacity_ticks.add(tick)
        unit = (spent / filled) if filled > 0 and spent > 0 else None
        if unit is not None:
            last_unit = unit
        price = unit if unit is not None else last_unit
        if requested > filled and price is not None and cash - spent < price - 1e-6:
            cash_ticks.add(tick)
    return capacity_ticks, cash_ticks


def _annotate_steps(
    steps: list[dict[str, dict[str, Any]]],
    run_log: Mapping[str, Any],
    scenario,
    buyer_id: str,
    delivery_lag: int,
) -> None:
    """Write the filled quantity and the in-transit arrival-age profile.

    ``filled_qty`` sums Allocation's fills across suppliers. Profile index
    ``age`` is units filled ``age`` ticks ago that have not arrived. Age 0
    stays 0 because ``decide`` runs before this tick's fill.
    """
    fills_by_tick: dict[int, list[tuple[str, str, int]]] = {}
    for tick_log in run_log["ticks"]:
        rows: list[tuple[str, str, int]] = []
        for purchase in tick_log.get("purchases") or []:
            if purchase["buyer_id"] != buyer_id:
                continue
            rows.append(
                (str(purchase["pid"]), str(purchase["supplier_id"]), int(purchase["qty_filled"]))
            )
        fills_by_tick[int(tick_log["tick"])] = rows

    history: dict[str, list[tuple[int, int, int]]] = {}
    for step in steps:
        if not step:
            continue
        tick = int(next(iter(step.values()))["tick"])
        filled: dict[str, int] = {}
        for pid, _supplier, qty in fills_by_tick.get(tick, []):
            filled[pid] = filled.get(pid, 0) + qty
        for pid, snapshot in step.items():
            snapshot["filled_qty"] = int(filled.get(pid, 0))
            snapshot["in_transit"] = arrival_age_profile(
                history.get(pid, []),
                tick,
                ARRIVAL_AGE_SLOTS,
            )
        for pid, supplier_id, qty in fills_by_tick.get(tick, []):
            if qty <= 0:
                continue
            lead = _edge_lead(scenario, supplier_id, buyer_id, pid, int(delivery_lag))
            history.setdefault(pid, []).append((tick, qty, lead))


def _edge_lead(scenario, supplier_id: str, buyer_id: str, pid: str, default: int) -> int:
    for edge in scenario.edges:
        if edge.supplier_id == supplier_id and edge.buyer_id == buyer_id:
            return int(edge.lead_time_for(pid))
    return int(default)


def _snapshot(
    obs: Mapping[str, Any],
    action: Mapping[str, Any],
    sales_hist: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    observed = dict(obs.get("observed_sales") or {})
    inventory = {pid: int(qty) for pid, qty in dict(obs.get("inventory") or {}).items()}
    cash = float(obs.get("cash") or 0.0)
    capacity = int(obs.get("capacity") or 0)
    tick = int(obs.get("tick") or 0)
    orders = action.get("order") or {}
    step: dict[str, dict[str, Any]] = {}
    for pid, on_hand in inventory.items():
        sold = int(observed.get(pid, 0))
        history = sales_hist.setdefault(pid, [])
        history.append(sold)
        window = [int(value) for value in padded_sales_window(history, SALES_WINDOW)]
        requested = sum(int(qty) for _supplier, qty in orders.get(pid, []))
        step[pid] = {
            "tick": tick,
            "on_hand": on_hand,
            "in_transit": [0] * ARRIVAL_AGE_SLOTS,
            "sales_window": window,
            "cash": cash,
            "capacity": capacity,
            "requested_qty": requested,
            "filled_qty": 0,
            "sales": sold,
        }
    return step


def _transitions(
    steps: list[dict[str, dict[str, Any]]],
    episode_seed: int,
    behavior: str,
    delivery_lag: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(steps) - 1):
        current = steps[index]
        nxt = steps[index + 1]
        for pid, cur in current.items():
            if pid not in nxt:
                continue
            rows.append(
                {
                    "episode_seed": episode_seed,
                    "behavior": behavior,
                    "tick": cur["tick"],
                    "product_id": pid,
                    "on_hand": cur["on_hand"],
                    "in_transit": list(cur["in_transit"]),
                    "sales_window": list(cur["sales_window"]),
                    "cash": cur["cash"],
                    "capacity": cur["capacity"],
                    "requested_qty": cur["requested_qty"],
                    "filled_qty": cur["filled_qty"],
                    "delivery_lag": int(delivery_lag),
                    "next_on_hand": nxt[pid]["on_hand"],
                    "next_in_transit": list(nxt[pid]["in_transit"]),
                    "next_sales": nxt[pid]["sales"],
                }
            )
    return rows


def _factory_capacity(node: FactoryNode) -> int:
    cap = node.capacity_per_tick
    if isinstance(cap, Constant):
        return int(cap.value)
    return int(cap)
