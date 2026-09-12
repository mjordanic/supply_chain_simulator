"""CRN policy overlay helpers for the Streamlit demo.

Sibling consumer of ``src.sim`` (ADR 0010 / ADR 0023): load a setup, overlay
demo knobs in memory, run textbook families on ``shop-1`` via
``Runner(..., policy_overrides=...)``.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pandas as pd

from src.sim.distributions import Constant, Normal
from src.sim.node import DemandSinkNode, IntermediateNode
from src.sim.policy_registry import build_policy
from src.sim.runner import Runner
from src.sim.setup_io import _derive_node_seed

FOCAL_NODE_ID = "shop-1"
MAX_N_STEPS = 60
DEMO_CYCLE_LEN = 20
DISRUPTION_ON_PROB = 0.2
DISRUPTION_SEVERITY = 0.4
DISRUPTION_DURATION = 5


def apply_knobs(
    scenario: Any,
    *,
    world_seed: int,
    n_steps: int,
    demand_mean: float,
    cycle_amp: float,
    disruption: bool,
) -> Any:
    """Return a deep-copied Scenario with demo knobs applied.

    Does not mutate ``scenario`` or any setup files on disk.
    """
    overlaid = copy.deepcopy(scenario)
    overlaid.world_seed = int(world_seed)
    overlaid.n_steps = min(MAX_N_STEPS, max(1, int(n_steps)))
    overlaid.market = replace(
        overlaid.market,
        cycle_amp=float(cycle_amp),
        cycle_len=DEMO_CYCLE_LEN,
    )
    if disruption:
        overlaid.disruption = replace(
            overlaid.disruption,
            event_prob=DISRUPTION_ON_PROB,
            severity=Constant(DISRUPTION_SEVERITY),
            duration=Constant(DISRUPTION_DURATION),
        )
    else:
        overlaid.disruption = replace(overlaid.disruption, event_prob=0.0)

    for ni in overlaid.nodes:
        node = ni.node
        if not isinstance(node, DemandSinkNode) or node.demand_dist is None:
            continue
        dist = node.demand_dist
        if isinstance(dist, Normal):
            node.demand_dist = Normal(mean=float(demand_mean), std=dist.std, clip=dist.clip)
        else:
            node.demand_dist = Constant(float(demand_mean))

    return overlaid


TEXTBOOK_FAMILIES = (
    "order_up_to",
    "reorder_point",
    "periodic_order_up_to",
    "periodic_reorder",
)

FAMILY_LABELS = {
    "order_up_to": "(s, S) order-up-to",
    "reorder_point": "(s, Q) reorder-point",
    "periodic_order_up_to": "(R, S) periodic order-up-to",
    "periodic_reorder": "(R, s, S) periodic reorder",
}


def _policy_params_for_family(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Hyperparameters the registry will splat; facts are injected separately."""
    out: dict[str, Any] = {"cover_horizon_ticks": int(params["cover_horizon_ticks"])}
    if "safety_lead_pct_of_lag" in params:
        out["safety_lead_pct_of_lag"] = float(params["safety_lead_pct_of_lag"])
    if name == "reorder_point":
        out["Q"] = int(params["Q"])
    if name in ("periodic_order_up_to", "periodic_reorder"):
        out["review_interval"] = int(params["review_interval"])
    return out


def _focal_node(scenario: Any) -> Any:
    for ni in scenario.nodes:
        if ni.node.id == FOCAL_NODE_ID and isinstance(ni.node, IntermediateNode):
            return ni.node
    raise ValueError(f"scenario has no IntermediateNode {FOCAL_NODE_ID!r}")


def run_crn_overlay(
    scenario: Any,
    families: list[str],
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run each textbook family on shop-1; all other nodes keep setup policies.

    Same ``scenario`` (hence same ``world_seed``) for every family — CRN.
    """
    if not families:
        raise ValueError("families must be non-empty")

    shop = _focal_node(scenario)
    policy_seed = _derive_node_seed(scenario.world_seed, FOCAL_NODE_ID, "policy")
    logs: dict[str, dict[str, Any]] = {}
    for name in families:
        pol = build_policy(
            name,
            _policy_params_for_family(name, params),
            node=shop,
            edges=scenario.edges,
            policy_seed=policy_seed,
        )
        logs[name] = Runner(scenario, policy_overrides={FOCAL_NODE_ID: pol}).run()
    return logs


def _rejection_qty(run_log: dict[str, Any]) -> int:
    total = 0
    for tick_log in run_log.get("ticks", []):
        for row in tick_log.get("rejections", []):
            total += int(row.get("qty_rejected", 0))
    return total


def scorecard(
    run_logs: dict[str, dict[str, Any]],
    scenario: Any,
) -> pd.DataFrame:
    """Shop-1 operational KPIs + profit, with CRN delta vs ``order_up_to``."""
    from src.sim.inspect import (
        closing_inventory_frame,
        flow_frame,
        node_timeseries_df,
        purchase_frame,
    )
    from src.sim.metrics import business_metrics, profit_decomposition

    rows: list[dict[str, Any]] = []
    for family, run_log in run_logs.items():
        ff = flow_frame(run_log)
        ts = node_timeseries_df(run_log, scenario)
        kpis = business_metrics(ff, ts, scenario)
        shop_kpis = kpis[kpis["node_id"] == FOCAL_NODE_ID]
        profit = profit_decomposition(
            ff, purchase_frame(run_log), closing_inventory_frame(run_log), scenario
        )
        shop_profit = profit[profit["node_id"] == FOCAL_NODE_ID]
        row: dict[str, Any] = {
            "family": family,
            "service_level": float("nan"),
            "stockout_rate": float("nan"),
            "inventory_turnover": float("nan"),
            "net_profit": float("nan"),
            "rejections": _rejection_qty(run_log),
        }
        if not shop_kpis.empty:
            row["service_level"] = float(shop_kpis.iloc[0]["service_level"])
            row["stockout_rate"] = float(shop_kpis.iloc[0]["stockout_rate"])
            row["inventory_turnover"] = float(shop_kpis.iloc[0]["inventory_turnover"])
        if not shop_profit.empty:
            row["net_profit"] = float(shop_profit.iloc[0]["net_profit"])
        rows.append(row)

    card = pd.DataFrame(rows)
    if "order_up_to" in run_logs and not card.empty:
        anchor = float(card.loc[card["family"] == "order_up_to", "net_profit"].iloc[0])
        card["delta_net_profit"] = card["net_profit"] - anchor
    else:
        card["delta_net_profit"] = float("nan")
    return card


def focal_family(families: list[str]) -> str:
    """Playback family: ``order_up_to`` when selected, else the first checked."""
    if "order_up_to" in families:
        return "order_up_to"
    if not families:
        raise ValueError("families must be non-empty")
    return families[0]
