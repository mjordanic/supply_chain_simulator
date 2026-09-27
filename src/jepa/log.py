"""Synthetic per-SKU transition log on the tuner episode shape.

Half the episodes run published-default ``OrderUpToPolicy``. Half run
``ExploratoryPolicy``. Factories stay on ``StaticFactoryPolicy``. Seeds
start at ``14_000_000``, disjoint from the tuner search and holdout offsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.jepa.policy import ExploratoryPolicy, _flatten_pending
from src.sim.distributions import Constant
from src.sim.node import FactoryNode
from src.sim.policy import IntermediatePolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.sim.runner import Runner
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode

LOG_SEED_OFFSET = 14_000_000
EPISODES_EACH = 64

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
    "granted_qty",
    "next_on_hand",
    "next_in_transit",
    "next_sales",
)


def collection_config(
    *,
    episode_length: int = 365,
    k_active: int = 5,
    delivery_lag: int = 3,
    holding_rate: float = 0.01,
    order_fee: float | None = None,
) -> TuningConfig:
    """Tuner episode shape used for the log.

    Unspecified knobs (capacity, opening cash, initial stock) stay at the
    ``TuningConfig`` defaults. ``order_fee`` stays at that default unless
    the caller passes one.
    """
    kwargs: dict[str, Any] = {
        "episode_length": episode_length,
        "K_active": k_active,
        "delivery_lag": delivery_lag,
        "holding_rate": holding_rate,
    }
    if order_fee is not None:
        kwargs["order_fee"] = order_fee
    return TuningConfig(**kwargs)


def build_log_scenario(config: TuningConfig, episode_seed: int, catalog: list | None = None):
    """One tuner-graph episode at ``episode_seed``."""
    if catalog is None:
        catalog = make_synthetic_catalog(max(config.K_catalog, config.K_active))
    return sample_episode(catalog, config, episode_seed)


def build_log_overrides(
    scenario,
    *,
    behavior: str,
    policy_seed: int,
    delivery_lag: int,
) -> dict[str, Any]:
    """``StaticFactoryPolicy`` on every factory, and one buyer policy on ``S``.

    ``order_up_to`` is ``OrderUpToPolicy()`` with no arguments — the published
    default, fair-share included. ``exploratory`` draws covers from
    ``policy_seed`` (not from the world seed).
    """
    overrides: dict[str, Any] = {}
    for instance in scenario.nodes:
        node = instance.node
        if isinstance(node, FactoryNode):
            overrides[node.id] = StaticFactoryPolicy(
                capacity_per_tick=_factory_capacity(node),
                unit_cost=float(node.unit_cost),
                policy_seed=node.init_seed,
            )
    if behavior == "order_up_to":
        overrides["S"] = OrderUpToPolicy()
    elif behavior == "exploratory":
        overrides["S"] = ExploratoryPolicy(
            policy_seed=policy_seed,
            delivery_lag=delivery_lag,
        )
    else:
        raise ValueError(f"unknown log behavior {behavior!r}")
    return overrides


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
    five SKUs, lead time 3, ``holding_rate`` 0.01, the tuner order fee, and
    64 episodes of each behavior from ``LOG_SEED_OFFSET``. Pass a shorter
    episode or a smaller count for a smoke run.

    Order-up-to episodes use seeds ``seed_offset .. seed_offset + n - 1``.
    Exploratory episodes use the next ``n`` seeds. The last tick of each
    episode is omitted: it has no next observation.
    """
    shape: dict[str, Any] = {}
    if episode_length is not None:
        shape["episode_length"] = episode_length
    if k_active is not None:
        shape["k_active"] = k_active
    if delivery_lag is not None:
        shape["delivery_lag"] = delivery_lag
    config = collection_config(**shape)
    catalog = make_synthetic_catalog(max(config.K_catalog, config.K_active))

    rows: list[dict[str, Any]] = []
    for behavior, start in (
        ("order_up_to", seed_offset),
        ("exploratory", seed_offset + n_episodes_each),
    ):
        for offset in range(n_episodes_each):
            episode_seed = start + offset
            spec = build_log_scenario(config, episode_seed, catalog)
            rows.extend(
                _run_episode(
                    spec.scenario,
                    behavior=behavior,
                    episode_seed=episode_seed,
                    delivery_lag=config.delivery_lag,
                )
            )

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
    return frame


class _TransitionRecorder(IntermediatePolicy):
    """Record the observation window and the granted quantity at each decide."""

    def __init__(self, inner: IntermediatePolicy, delivery_lag: int) -> None:
        super().__init__(policy_seed=getattr(inner, "policy_seed", None))
        self._inner = inner
        self._delivery_lag = int(delivery_lag)
        self._sales: dict[str, list[int]] = {}
        self.steps: list[dict[str, dict[str, Any]]] = []

    def decide(self, obs_intermediate: Mapping[str, Any], central_table: Any) -> dict[str, Any]:
        action = self._inner.decide(obs_intermediate, central_table)
        self.steps.append(_snapshot(obs_intermediate, action, self._sales, self._delivery_lag))
        return action


def _run_episode(scenario, *, behavior: str, episode_seed: int, delivery_lag: int) -> list[dict]:
    overrides = build_log_overrides(
        scenario,
        behavior=behavior,
        policy_seed=episode_seed,
        delivery_lag=delivery_lag,
    )
    recorder = _TransitionRecorder(overrides["S"], delivery_lag)
    overrides["S"] = recorder
    Runner(scenario, policy_overrides=overrides).run()
    return _transitions(recorder.steps, episode_seed, behavior)


def _snapshot(
    obs: Mapping[str, Any],
    action: Mapping[str, Any],
    sales_hist: dict[str, list[int]],
    delivery_lag: int,
) -> dict[str, dict[str, Any]]:
    inventory = {pid: int(qty) for pid, qty in dict(obs.get("inventory") or {}).items()}
    in_transit = _flatten_pending(obs.get("pending") or {})
    observed = dict(obs.get("observed_sales") or {})
    cash = float(obs.get("cash") or 0.0)
    capacity = int(obs.get("capacity") or 0)
    tick = int(obs.get("tick") or 0)
    orders = action.get("order") or {}
    step: dict[str, dict[str, Any]] = {}
    for pid, on_hand in inventory.items():
        sold = int(observed.get(pid, 0))
        history = sales_hist.setdefault(pid, [])
        history.append(sold)
        window = history[-delivery_lag:] if delivery_lag > 0 else []
        if delivery_lag > 0 and len(window) < delivery_lag:
            window = [0] * (delivery_lag - len(window)) + window
        granted = sum(int(qty) for _supplier, qty in orders.get(pid, []))
        step[pid] = {
            "tick": tick,
            "on_hand": on_hand,
            "in_transit": int(in_transit.get(pid, 0)),
            "sales_window": window,
            "cash": cash,
            "capacity": capacity,
            "granted_qty": granted,
            "sales": sold,
        }
    return step


def _transitions(
    steps: list[dict[str, dict[str, Any]]],
    episode_seed: int,
    behavior: str,
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
                    "in_transit": cur["in_transit"],
                    "sales_window": list(cur["sales_window"]),
                    "cash": cur["cash"],
                    "capacity": cur["capacity"],
                    "granted_qty": cur["granted_qty"],
                    "next_on_hand": nxt[pid]["on_hand"],
                    "next_in_transit": nxt[pid]["in_transit"],
                    "next_sales": nxt[pid]["sales"],
                }
            )
    return rows


def _factory_capacity(node: FactoryNode) -> int:
    cap = node.capacity_per_tick
    if isinstance(cap, Constant):
        return int(cap.value)
    return int(cap)
