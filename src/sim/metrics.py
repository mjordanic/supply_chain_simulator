"""Pure-function business-metrics module — DataFrame-native (ADR 0019).

The module is schema-free: it never reads the run-log structure directly.
All run-log parsing lives in ``src.sim.inspect`` (the flow/purchase builders).
``business_metrics`` receives a ``flow_frame`` (from ``inspect.flow_frame``) and
a ``node_timeseries_df`` (from ``inspect.node_timeseries_df``) so the math here
stays pure.

Public surface
--------------
business_metrics(flow_frame, node_timeseries_df, scenario) → pd.DataFrame
    Per-node and system-wide operational KPI rows.

    Columns: node_id, service_level, stockout_rate,
             inventory_turnover, mean_price_pct_of_msrp.

    One row per selling node (IntermediateNode / FactoryNode) plus one
    ``"_system"`` row that aggregates all selling nodes together.

    Selling nodes are those with at least one non-null ``price`` entry in
    the flow frame (demand sinks never sell).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# business_metrics — the new DataFrame-native entry point
# ---------------------------------------------------------------------------


def business_metrics(
    flow_frame: "pd.DataFrame",
    node_timeseries_df: "pd.DataFrame",
    scenario: Any,
) -> "pd.DataFrame":
    """Return per-node + system-wide operational KPI rows.

    Parameters
    ----------
    flow_frame:
        Output of ``inspect.flow_frame(run_log)`` — tidy per-(node, pid, tick)
        with columns ``tick, node_id, pid, sales, demand, price, stockout``.
    node_timeseries_df:
        Output of ``inspect.node_timeseries_df(run_log, scenario)`` — per
        ``(tick, node_id)`` with ``inventory_total``.
    scenario:
        The ``Scenario`` used for the run.  Provides per-product ``base_price``
        (MSRP) via ``scenario.catalog``.

    Returns
    -------
    pd.DataFrame
        Columns: ``node_id, service_level, stockout_rate,
        inventory_turnover, mean_price_pct_of_msrp``.
        One row per selling node + one ``"_system"`` roll-up row.

    Notes
    -----
    - Selling nodes are identified by having at least one non-null ``price``
      entry in the flow frame (demand sinks have ``price == None``).
    - ``stockout_rate`` uses the decision-time ``stockout`` boolean captured
      in the flow frame — not a post-hoc zero-inventory check.
    - ``inventory_turnover`` = sum(sales) / max(1, mean(inventory_total))
      where ``inventory_total`` comes from ``node_timeseries_df``.
    - ``mean_price_pct_of_msrp`` compares ``price`` (from flow frame) with
      ``base_price`` (MSRP) from the catalog.
    """
    # Build MSRP lookup: {pid: base_price}.
    msrp: dict[str, float] = {
        w.product_id: float(w.base_price) for w in scenario.catalog
    }

    # Identify selling nodes: those with non-null price in the flow frame.
    if flow_frame.empty:
        return pd.DataFrame(
            columns=[
                "node_id", "service_level", "stockout_rate",
                "inventory_turnover", "mean_price_pct_of_msrp",
            ]
        )

    selling_mask = flow_frame["price"].notna()
    seller_ids: list[str] = list(flow_frame.loc[selling_mask, "node_id"].unique())

    rows: list[dict] = []
    for nid in seller_ids:
        ndf = flow_frame[flow_frame["node_id"] == nid]
        rows.append(_node_kpis(nid, ndf, node_timeseries_df, msrp))

    # System-wide roll-up across all selling nodes.
    sellers_df = flow_frame[flow_frame["node_id"].isin(seller_ids)]
    rows.append(_node_kpis("_system", sellers_df, node_timeseries_df, msrp, system=True))

    return pd.DataFrame(rows)


def _node_kpis(
    node_id: str,
    ndf: "pd.DataFrame",
    ts_df: "pd.DataFrame",
    msrp: dict[str, float],
    *,
    system: bool = False,
) -> dict:
    """Compute the four operational KPIs for one node (or system roll-up)."""
    total_sales = float(ndf["sales"].sum())
    total_demand = float(ndf["demand"].sum())
    sl = total_sales / max(1.0, total_demand)

    # stockout_rate: fraction of (node, pid, tick) rows where stockout==True.
    n_rows = len(ndf)
    n_stockout = int(ndf["stockout"].sum()) if n_rows > 0 else 0
    sor = float(n_stockout) / float(n_rows) if n_rows > 0 else 0.0

    # inventory_turnover: sum(sales) / max(1, mean(inventory_total)).
    if system:
        # Roll-up over all selling nodes in ts_df.
        node_ids = ndf["node_id"].unique()
        inv_df = ts_df[ts_df["node_id"].isin(node_ids)]
    else:
        inv_df = ts_df[ts_df["node_id"] == node_id]
    mean_inv = float(inv_df["inventory_total"].mean()) if len(inv_df) > 0 else 0.0
    it = total_sales / max(1.0, mean_inv)

    # mean_price_pct_of_msrp: mean of price / base_price across non-null rows.
    priced = ndf[ndf["price"].notna()]
    if priced.empty:
        mpp = 1.0
    else:
        ratios = priced.apply(
            lambda r: float(r["price"]) / max(1e-9, msrp.get(r["pid"], float(r["price"]))),
            axis=1,
        )
        mpp = float(ratios.mean())

    return {
        "node_id": node_id,
        "service_level": sl,
        "stockout_rate": sor,
        "inventory_turnover": it,
        "mean_price_pct_of_msrp": mpp,
    }


__all__ = [
    "business_metrics",
]
