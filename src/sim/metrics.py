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

profit_decomposition(flow_frame, purchase_frame, closing_inventory_frame,
                     scenario) → pd.DataFrame
    Per-node profit decomposition for ``IntermediateNode``s (ADR 0019 Rule 3).

    Columns: node_id, revenue, order_cost, holding_cost, order_fees,
             net_profit.

    - ``revenue``      = Σ price × sales for this node (selling rows only).
    - ``order_cost``   = Σ cash_paid for all purchases where buyer == node
                         (real cash transferred, not catalog unit_cost).
    - ``holding_cost`` = Σ_tick Σ_pid closing_qty × holding_rate × unit_cost
                         (derived from closing inventory frame + node rates).
    - ``order_fees``   = Σ_tick n_distinct_suppliers_ordered_from × order_fee
                         (derived from purchase frame + node order_fee).
    - ``net_profit``   = revenue − order_cost − holding_cost − order_fees.

    ``net_profit`` reconciles to the node's Δcash over the episode by
    construction (ADR 0019 Rule 3).  The full equity reconciliation identity
    is: net_profit + Δ(inventory_value + outstanding_value) = Δequity.
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


def profit_decomposition(
    flow_frame: "pd.DataFrame",
    purchase_frame: "pd.DataFrame",
    closing_inventory_frame: "pd.DataFrame",
    scenario: Any,
) -> "pd.DataFrame":
    """Return per-node profit decomposition for IntermediateNodes.

    Parameters
    ----------
    flow_frame:
        Output of ``inspect.flow_frame(run_log)`` — used to compute revenue
        (``price × sales`` for selling rows).
    purchase_frame:
        Output of ``inspect.purchase_frame(run_log)`` — used to compute order
        cost (``sum(cash_paid)`` where ``buyer_id == node``) and order fees
        (count of distinct suppliers per tick × ``order_fee``).
    closing_inventory_frame:
        Output of ``inspect.closing_inventory_frame(run_log)`` — per
        ``(tick, node_id, pid)`` closing on-hand qty, used to derive holding
        cost.
    scenario:
        The ``Scenario`` used for the run.  Provides ``unit_cost`` per product
        (from ``scenario.catalog``) and ``holding_rate`` / ``order_fee`` per
        ``IntermediateNode`` (from ``scenario.nodes``).

    Returns
    -------
    pd.DataFrame
        Columns: ``node_id, revenue, order_cost, holding_cost, order_fees,
        net_profit``.
        One row per ``IntermediateNode`` in the scenario.

    Notes
    -----
    - ``net_profit`` reconciles to Δcash (the node's change in cash balance
      over the episode) by construction (ADR 0019 Rule 3).
    - Factories and demand sinks are excluded: factories are zero-margin
      bookkeeping nodes; sinks hold no inventory and never pay order fees.
    """
    from src.sim.node import IntermediateNode

    # Build unit-cost lookup from catalog.
    unit_cost: dict[str, float] = {
        w.product_id: float(w.unit_cost) for w in scenario.catalog
    }

    # Build IntermediateNode rate lookup: {node_id: (holding_rate, order_fee)}.
    node_rates: dict[str, tuple[float, float]] = {}
    for ni in scenario.nodes:
        if isinstance(ni.node, IntermediateNode):
            node_rates[ni.node.id] = (
                float(ni.node.holding_rate),
                float(ni.node.order_fee),
            )

    if not node_rates:
        return pd.DataFrame(
            columns=[
                "node_id", "revenue", "order_cost",
                "holding_cost", "order_fees", "net_profit",
            ]
        )

    rows: list[dict] = []
    for node_id, (holding_rate, order_fee) in node_rates.items():
        # --- Revenue: sum(price * sales) for this selling node ---
        node_ff = flow_frame[
            (flow_frame["node_id"] == node_id) & flow_frame["price"].notna()
        ]
        revenue = float((node_ff["price"] * node_ff["sales"]).sum())

        # --- Order cost: real cash paid by this node to its suppliers ---
        node_pf = purchase_frame[purchase_frame["buyer_id"] == node_id]
        order_cost = float(node_pf["cash_paid"].sum())

        # --- Holding cost: derived from closing inventory + rates ---
        node_inv = closing_inventory_frame[
            closing_inventory_frame["node_id"] == node_id
        ]
        holding_cost = float(
            (
                node_inv["qty"]
                * node_inv["pid"].map(unit_cost).fillna(0.0)
                * holding_rate
            ).sum()
        )

        # --- Order fees: distinct (buyer, supplier) pairs per tick × fee ---
        if node_pf.empty or order_fee == 0.0:
            order_fees = 0.0
        else:
            # Count distinct supplier_id values per tick for this buyer.
            suppliers_per_tick = (
                node_pf.groupby("tick")["supplier_id"].nunique()
            )
            order_fees = float(suppliers_per_tick.sum() * order_fee)

        net_profit = revenue - order_cost - holding_cost - order_fees
        rows.append(
            {
                "node_id": node_id,
                "revenue": revenue,
                "order_cost": order_cost,
                "holding_cost": holding_cost,
                "order_fees": order_fees,
                "net_profit": net_profit,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "node_id", "revenue", "order_cost",
            "holding_cost", "order_fees", "net_profit",
        ],
    )


__all__ = [
    "business_metrics",
    "profit_decomposition",
]
