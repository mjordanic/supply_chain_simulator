"""Synthetic per-SKU transition log on the tuner episode shape.

Half the episodes run published-default ``OrderUpToPolicy``. Half run
``ExploratoryPolicy``. Factories stay on ``StaticFactoryPolicy``. Seeds
start at ``14_000_000``, disjoint from the tuner search and holdout offsets.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.jepa.policy import ExploratoryPolicy
from src.sim.distributions import Constant
from src.sim.node import FactoryNode
from src.sim.policy import IntermediatePolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.sim.runner import Runner
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode

LOG_SEED_OFFSET = 14_000_000
EPISODES_EACH = 64


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
    "next_on_hand",
    "next_in_transit",
    "next_sales",
)


def collection_config(
    *,
    episode_length: int = 365,
    k_active: int | None = None,
    delivery_lag: int = 3,
    holding_rate: float = 0.01,
    order_fee: float | None = None,
) -> TuningConfig:
    """Tuner episode shape used for the log.

    Unspecified knobs (capacity, opening cash, initial stock, assortment
    size) stay at the ``TuningConfig`` defaults, so K is sampled unless
    ``k_active`` is passed. ``order_fee`` stays at that default unless
    the caller passes one.
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
    rows = _record_episode(
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
    K sampled from the tuner range, lead time 3, ``holding_rate`` 0.01, the
    tuner order fee, and 64 episodes of each behavior from
    ``LOG_SEED_OFFSET``. Pass a shorter episode or a smaller count for a
    smoke run.

    Order-up-to episodes use seeds ``seed_offset .. seed_offset + n - 1``.
    Exploratory episodes use the next ``n`` seeds. The last tick of each
    episode is omitted: it has no next observation.

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
        _write_manifest(
            destination,
            frame,
            config,
            catalog,
            n_episodes_each=n_episodes_each,
            seed_offset=seed_offset,
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


def _write_manifest(
    path: Path,
    frame: pd.DataFrame,
    config: TuningConfig,
    catalog,
    *,
    n_episodes_each: int,
    seed_offset: int,
) -> None:
    """Sidecar of every parameter that produced ``frame``.

    ``nonzero_order_fraction`` is the share of rows whose filled quantity
    is non-zero: the order Allocation realised, not the request.
    """
    n = int(n_episodes_each)
    if len(frame) == 0:
        fraction = 0.0
    else:
        fraction = float((frame["filled_qty"] > 0).mean())
    manifest = {
        "episode_count": 2 * n,
        "seed_min": int(seed_offset),
        "seed_max": int(seed_offset) + 2 * n - 1,
        "episode_length": int(config.episode_length),
        "K_active": None if config.K_active is None else int(config.K_active),
        "K_min": int(config.K_min),
        "K_max_episode": int(config.K_max_episode),
        "delivery_lag": int(config.delivery_lag),
        "holding_rate": float(config.holding_rate),
        "order_fee": float(config.order_fee),
        "catalog_identity": catalog_identity(catalog),
        "behavior_mixture": [
            {"behavior": "order_up_to", "episodes": n, "policy": "OrderUpToPolicy"},
            {"behavior": "exploratory", "episodes": n, "policy": "ExploratoryPolicy"},
        ],
        "git_sha": _git_sha(),
        "nonzero_order_fraction": fraction,
    }
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _TransitionRecorder(IntermediatePolicy):
    """Record the observation window and the requested quantity at each decide."""

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
    buyer = overrides.pop("S")
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
) -> list[dict]:
    overrides = dict(policy_overrides or {})
    recorder = _TransitionRecorder(buyer_policy, delivery_lag)
    overrides["S"] = recorder
    run_log = Runner(scenario, policy_overrides=overrides).run()
    _annotate_steps(recorder.steps, run_log, scenario, "S", delivery_lag)
    return _transitions(recorder.steps, episode_seed, behavior)


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
    length = max(0, int(delivery_lag))
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
            profile = [0] * length
            for order_tick, qty, lead in history.get(pid, []):
                age = tick - order_tick
                if 0 < age < lead and age < length:
                    profile[age] += qty
            snapshot["filled_qty"] = int(filled.get(pid, 0))
            snapshot["in_transit"] = profile
        for pid, supplier_id, qty in fills_by_tick.get(tick, []):
            if qty <= 0:
                continue
            lead = _edge_lead(scenario, supplier_id, buyer_id, pid, length)
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
    delivery_lag: int,
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
        window = history[-delivery_lag:] if delivery_lag > 0 else []
        if delivery_lag > 0 and len(window) < delivery_lag:
            window = [0] * (delivery_lag - len(window)) + window
        requested = sum(int(qty) for _supplier, qty in orders.get(pid, []))
        step[pid] = {
            "tick": tick,
            "on_hand": on_hand,
            "in_transit": [0] * max(0, int(delivery_lag)),
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
