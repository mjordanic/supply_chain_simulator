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

# One fixed colour per family, used by every chart and the network title.
FAMILY_COLORS = {
    "order_up_to": "#4C9BE8",
    "reorder_point": "#F28E2B",
    "periodic_order_up_to": "#59A14F",
    "periodic_reorder": "#E15759",
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
    Each run gets its own deep copy: ``build_world`` copies nodes but
    re-attaches the setup policy objects, so a stateful setup policy (shop-2
    on the two-shop graph) would otherwise carry history between runs.
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
        world = copy.deepcopy(scenario)
        logs[name] = Runner(world, policy_overrides={FOCAL_NODE_ID: pol}).run()
    return logs


def _rejection_qty(run_log: dict[str, Any]) -> int:
    total = 0
    for tick_log in run_log.get("ticks", []):
        for row in tick_log.get("rejections", []):
            total += int(row.get("qty_rejected", 0))
    return total


def _fill_rate(flow: pd.DataFrame, scenario: Any, node_id: str) -> float:
    """Shop units sold ÷ demand target of the sinks it supplies.

    The engine's ``service_level`` divides by orders the shop actually saw,
    which equals sales here, so it reads 1.0 even for a starved shop.
    """
    sink_ids = {
        ni.node.id for ni in scenario.nodes if isinstance(ni.node, DemandSinkNode)
    }
    fed = {e.buyer_id for e in scenario.edges if e.supplier_id == node_id}
    wanted = flow.loc[flow["node_id"].isin(sink_ids & fed), "demand"].sum()
    sold = flow.loc[flow["node_id"] == node_id, "sales"].sum()
    return float(sold / wanted) if wanted > 0 else float("nan")


def scorecard(
    run_logs: dict[str, dict[str, Any]],
    scenario: Any,
    node_id: str = FOCAL_NODE_ID,
) -> pd.DataFrame:
    """One shop's operational KPIs + profit, with CRN delta vs ``order_up_to``."""
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
        shop_kpis = kpis[kpis["node_id"] == node_id]
        profit = profit_decomposition(
            ff, purchase_frame(run_log), closing_inventory_frame(run_log), scenario
        )
        shop_profit = profit[profit["node_id"] == node_id]
        on_hand = ts.loc[ts["node_id"] == node_id, "inventory_total"]
        row: dict[str, Any] = {
            "family": family,
            "fill_rate": _fill_rate(ff, scenario, node_id),
            "days_empty": int((on_hand == 0).sum()),
            "inventory_turnover": float("nan"),
            "net_profit": float("nan"),
            "rejections": _rejection_qty(run_log),
        }
        if not shop_kpis.empty:
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


def identical_groups(card: pd.DataFrame) -> list[list[str]]:
    """Families whose scorecard rows came out identical (two or more each)."""
    cols = ["net_profit", "fill_rate", "days_empty", "rejections"]
    groups = card.groupby(cols, sort=False, dropna=False)["family"].agg(list)
    return [g for g in groups if len(g) > 1]


ANCHOR_FAMILY = "order_up_to"
LUCK_SEEDS = tuple(range(30))


def _net_profit(run_log: dict[str, Any], scenario: Any, node_id: str) -> float:
    from src.sim.inspect import closing_inventory_frame, flow_frame, purchase_frame
    from src.sim.metrics import profit_decomposition

    profit = profit_decomposition(
        flow_frame(run_log),
        purchase_frame(run_log),
        closing_inventory_frame(run_log),
        scenario,
    )
    row = profit[profit["node_id"] == node_id]
    return float(row.iloc[0]["net_profit"]) if not row.empty else float("nan")


def luck_sweep(
    scenario: Any,
    families: list[str],
    params: dict[str, Any],
    seeds: tuple[int, ...] = LUCK_SEEDS,
) -> pd.DataFrame:
    """Run each family plus the (s, S) anchor on every seed's world.

    Only ``world_seed`` changes between worlds; every other knob stays as in
    ``scenario``. Each world is shared by all families (CRN), so the delta
    vs (s, S) is a paired difference. Columns: ``world_seed, family,
    net_profit, delta_net_profit``.
    """
    run_families = [ANCHOR_FAMILY] + [f for f in families if f != ANCHOR_FAMILY]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        world = copy.deepcopy(scenario)
        world.world_seed = int(seed)
        logs = run_crn_overlay(world, run_families, params)
        profits = {f: _net_profit(log, world, FOCAL_NODE_ID) for f, log in logs.items()}
        for family, profit in profits.items():
            rows.append(
                {
                    "world_seed": int(seed),
                    "family": family,
                    "net_profit": profit,
                    "delta_net_profit": profit - profits[ANCHOR_FAMILY],
                }
            )
    return pd.DataFrame(rows)


def luck_summary(sweep: pd.DataFrame, n_boot: int = 2000, rng_seed: int = 0) -> pd.DataFrame:
    """Per family: mean paired delta vs (s, S), bootstrap 95% CI, wins and ties.

    Percentile bootstrap of the mean over worlds, with a fixed RNG seed per
    family so the same deltas always show the same interval, whichever other
    families ran. A win is a world where the family made strictly more than
    (s, S); a tie is exactly the same profit.
    """
    import numpy as np

    rows: list[dict[str, Any]] = []
    for family, group in sweep.groupby("family", sort=False):
        if family == ANCHOR_FAMILY:
            continue
        rng = np.random.default_rng(rng_seed)
        deltas = group["delta_net_profit"].to_numpy(dtype=float)
        n = len(deltas)
        means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        rows.append(
            {
                "family": family,
                "n_worlds": n,
                "mean_delta": float(deltas.mean()),
                "ci_low": float(np.percentile(means, 2.5)),
                "ci_high": float(np.percentile(means, 97.5)),
                "wins": int((deltas > 0).sum()),
                "ties": int((deltas == 0).sum()),
            }
        )
    return pd.DataFrame(
        rows, columns=["family", "n_worlds", "mean_delta", "ci_low", "ci_high", "wins", "ties"]
    )
