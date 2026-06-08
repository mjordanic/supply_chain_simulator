"""Run-log inspection helpers.

Pure functions over the run-log dict produced by ``Runner.run()``.  No
plotting, no file I/O — just tidy DataFrames for notebooks and tests.

Public API
----------
- ``node_timeseries_df(run_log, scenario)`` — long-form per-(tick, node).
- ``global_timeseries_df(run_log)``         — per-tick world-state series.
- ``per_product_df(run_log, scenario, node_id)`` — per-(tick, pid) for one node.
- ``node_equity(run_log, scenario, node_id)`` — cash + inventory + outstanding
  at cost, per tick.
- ``flow_frame(run_log)`` — tidy per-(node, pid, tick) flow log (sales, demand,
  price, stockout).
- ``purchase_frame(run_log)`` — per-(buyer, supplier, pid, tick) purchase rows
  (qty_filled, cash_paid).
- ``closing_inventory_frame(run_log)`` — per-(tick, node_id, pid) closing
  on-hand inventory for non-factory nodes.
- ``conservation_identity_terms(run_log, scenario)`` — the four ADR 0013
  Rule 5 terms for the cash-conservation identity.

The flow / purchase builders own *all* run-log flow-schema knowledge so that
``metrics.py`` can stay schema-free and DataFrame-native (ADR 0019).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def node_timeseries_df(run_log: dict[str, Any], scenario: Any) -> pd.DataFrame:
    """Long-form DataFrame: one row per ``(tick, node_id)``.

    Columns
    -------
    tick              int
    node_id           str
    node_type         str   — ``"FactoryNode"``, ``"IntermediateNode"``, or
                              ``"DemandSinkNode"``
    region            str
    level             int | None
    cash              float
    inventory_total   int
    pending_total     int
    orders_total      int
    """
    # Build node metadata lookup from scenario.
    meta: dict[str, dict[str, Any]] = {}
    for ni in scenario.nodes:
        node = ni.node
        meta[node.id] = {
            "node_type": type(node).__name__,
            "region": node.region,
            "level": node.level,
        }

    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        node_cash = tick_log.get("node_cash", {})
        node_inventory = tick_log.get("node_inventory", {})
        node_pending = tick_log.get("node_pending", {})
        node_orders = tick_log.get("node_orders", {})

        for node_id, m in meta.items():
            inv_dict = node_inventory.get(node_id, {})
            if "_total" in inv_dict:
                inventory_total = inv_dict["_total"]
            else:
                inventory_total = sum(
                    v for k, v in inv_dict.items() if k != "_total"
                )

            pending_dict = node_pending.get(node_id, {})
            pending_total = sum(pending_dict.values())

            orders_dict = node_orders.get(node_id, {})
            orders_total = sum(orders_dict.values())

            rows.append(
                {
                    "tick": tick,
                    "node_id": node_id,
                    "node_type": m["node_type"],
                    "region": m["region"],
                    "level": m["level"],
                    "cash": node_cash.get(node_id, 0.0),
                    "inventory_total": inventory_total,
                    "pending_total": pending_total,
                    "orders_total": orders_total,
                }
            )

    return pd.DataFrame(rows)


def global_timeseries_df(run_log: dict[str, Any]) -> pd.DataFrame:
    """Per-tick world-state DataFrame.

    Columns
    -------
    tick              int
    market_supply_<region>   float  — one column per region
    market_demand_<region>   float  — one column per region

    The ``global.time`` series has ``n_steps + 1`` entries (step 0 is the
    pre-run snapshot).  We align by ``simulation_step`` so the resulting
    DataFrame has ``n_steps`` rows matching the ``ticks`` list.
    """
    g = run_log["global"]
    time_steps: list[int] = g["time"]["simulation_step"]
    market_supply: dict[str, list[float]] = g.get("market_supply", {})
    market_demand: dict[str, list[float]] = g.get("market_demand", {})

    rows: list[dict[str, Any]] = []
    for i, step in enumerate(time_steps):
        row: dict[str, Any] = {"tick": step}
        for region, series in market_supply.items():
            row[f"market_supply_{region}"] = series[i]
        for region, series in market_demand.items():
            row[f"market_demand_{region}"] = series[i]
        rows.append(row)

    return pd.DataFrame(rows)


def per_product_df(
    run_log: dict[str, Any], scenario: Any, node_id: str
) -> pd.DataFrame:
    """Per-(tick, pid) inventory and orders for one node.

    Excludes the synthetic ``_total`` key present in factory inventory
    snapshots — only real product rows are emitted.

    Columns
    -------
    tick          int
    node_id       str
    pid           str
    inventory     int
    pending       int
    orders        int
    """
    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        inv_dict = tick_log.get("node_inventory", {}).get(node_id, {})
        pending_dict = tick_log.get("node_pending", {}).get(node_id, {})
        orders_dict = tick_log.get("node_orders", {}).get(node_id, {})

        for pid, qty in inv_dict.items():
            if pid == "_total":
                continue
            rows.append(
                {
                    "tick": tick,
                    "node_id": node_id,
                    "pid": pid,
                    "inventory": qty,
                    "pending": pending_dict.get(pid, 0),
                    "orders": orders_dict.get(pid, 0),
                }
            )

    return pd.DataFrame(rows)


def node_equity(
    run_log: dict[str, Any], scenario: Any, node_id: str
) -> pd.DataFrame:
    """Per-tick equity estimate for one node.

    equity = cash
           + inventory-at-cost  (on-hand units × unit_cost from catalog)
           + outstanding-at-cost (in-transit units × unit_cost)

    ``unit_cost`` is looked up from ``scenario.catalog`` by product id.
    For factory nodes (single product, ``_total`` key) the factory's own
    ``unit_cost`` is used.  For intermediate and sink nodes, per-product
    ``unit_cost`` from the catalog is used.

    Columns
    -------
    tick                int
    node_id             str
    cash                float
    inventory_value     float
    outstanding_value   float
    equity              float
    """
    # Build unit-cost lookup from catalog.
    unit_cost: dict[str, float] = {
        w.product_id: float(w.unit_cost) for w in scenario.catalog
    }

    # Find which node this is to handle factory vs. intermediate.
    node_obj = None
    for ni in scenario.nodes:
        if ni.node.id == node_id:
            node_obj = ni.node
            break

    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        cash = tick_log.get("node_cash", {}).get(node_id, 0.0)
        inv_dict = tick_log.get("node_inventory", {}).get(node_id, {})
        pending_dict = tick_log.get("node_pending", {}).get(node_id, {})

        # Inventory-at-cost
        inv_value = 0.0
        if "_total" in inv_dict:
            # Factory: use the node's own unit_cost if available.
            uc = 0.0
            if node_obj is not None:
                uc = float(getattr(node_obj, "unit_cost", 0.0))
            inv_value = inv_dict["_total"] * uc
        else:
            for pid, qty in inv_dict.items():
                inv_value += qty * unit_cost.get(pid, 0.0)

        # Outstanding-at-cost (pending in-transit)
        outstanding_value = 0.0
        for pid, qty in pending_dict.items():
            outstanding_value += qty * unit_cost.get(pid, 0.0)

        equity = cash + inv_value + outstanding_value
        rows.append(
            {
                "tick": tick,
                "node_id": node_id,
                "cash": cash,
                "inventory_value": inv_value,
                "outstanding_value": outstanding_value,
                "equity": equity,
            }
        )

    return pd.DataFrame(rows)


def flow_frame(run_log: dict[str, Any]) -> pd.DataFrame:
    """Tidy long-form per-product flow log: one row per ``(node, pid, tick)``.

    This is the deep builder every downstream consumer of the per-tick flow
    log shares (ADR 0019).  It reads the ``node_flows`` records the engine
    logs each tick and emits a tidy frame so ``metrics.py`` never has to know
    the run-log schema.

    Columns
    -------
    tick        int
    node_id     str
    pid         str
    sales       int     — units the node sold as a supplier this tick.
    demand      int     — units requested of it; for a ``DemandSinkNode`` the
                          exogenous ``demand_target``.
    price       float   — the node's ``list_price`` at decision time (``NaN``
                          for demand sinks, which do not sell).
    stockout    bool    — decision-time on-hand == 0 (captured during the walk,
                          not inferred from the closing snapshot).
    """
    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        for f in tick_log.get("node_flows", []):
            rows.append(
                {
                    "tick": tick,
                    "node_id": f["node_id"],
                    "pid": f["pid"],
                    "sales": f["sales"],
                    "demand": f["demand"],
                    "price": f["price"],
                    "stockout": f["stockout"],
                }
            )

    return pd.DataFrame(
        rows,
        columns=["tick", "node_id", "pid", "sales", "demand", "price", "stockout"],
    )


def purchase_frame(run_log: dict[str, Any]) -> pd.DataFrame:
    """Per-(buyer, supplier, pid, tick) purchase rows from the flow log.

    These are the realised allocations behind every ``execute_buy`` call —
    the rows the old ``node_orders`` aggregate is derived from (ADR 0019).

    Columns
    -------
    tick          int
    buyer_id      str
    supplier_id   str
    pid           str
    qty_filled    int     — units actually allocated.
    cash_paid     float   — ``qty_filled × supplier list_price``.
    """
    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        for p in tick_log.get("purchases", []):
            rows.append(
                {
                    "tick": tick,
                    "buyer_id": p["buyer_id"],
                    "supplier_id": p["supplier_id"],
                    "pid": p["pid"],
                    "qty_filled": p["qty_filled"],
                    "cash_paid": p["cash_paid"],
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "tick",
            "buyer_id",
            "supplier_id",
            "pid",
            "qty_filled",
            "cash_paid",
        ],
    )


def closing_inventory_frame(run_log: dict[str, Any]) -> pd.DataFrame:
    """Per-(tick, node_id, pid) closing on-hand inventory.

    Extracts the post-settle closing inventory for every non-factory node
    (i.e. every entry in ``node_inventory`` that does NOT use the
    ``_total`` synthetic key).  Factory nodes expose only ``_total`` and
    never pay holding cost, so they are excluded; the caller must pass
    ``IntermediateNode``-only node_ids when using this frame to derive
    holding cost.

    Columns
    -------
    tick        int
    node_id     str
    pid         str
    qty         int   — closing on-hand units for this (node, pid, tick).
    """
    rows: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        tick = tick_log["tick"]
        for node_id, inv_dict in tick_log.get("node_inventory", {}).items():
            for pid, qty in inv_dict.items():
                if pid == "_total":
                    continue
                rows.append(
                    {
                        "tick": tick,
                        "node_id": node_id,
                        "pid": pid,
                        "qty": qty,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["tick", "node_id", "pid", "qty"],
    )


def conservation_identity_terms(
    run_log: dict[str, Any],
    scenario: Any,
) -> dict[str, float]:
    """Compute the ADR 0013 Rule 5 cash-conservation identity terms.

    Rule 5 (amended by ADR 0019) states that, for a system with no active
    factory production:

        initial_cash + sink_cash_created == node_cash_total + cumulative_void

    where

    - ``initial_cash``      is the total cash across all nodes *before* any
                            tick (not returned here — callers supply it from
                            a pre-run ``build_world`` snapshot; see examples
                            in the integration tests).
    - ``sink_cash_created`` = ``n_steps × income_rate`` for each
                            ``DemandSinkNode`` (the only source of new cash
                            per ADR 0013 Rule 1).
    - ``node_cash_total``   = Σ node cash balances at end of run.
    - ``cumulative_void``   = Σ holding-cost + Σ order-fee charged to
                            ``IntermediateNode``s over all ticks (derived
                            from the logged flows + the node rates per
                            ADR 0019 Rule 5 — not logged per tick).
    - ``in_transit_value``  = Σ pending units × unit_cost at end of run
                            (paid-for but not-yet-delivered inventory,
                            returned for diagnostic purposes but NOT part
                            of the identity — payment is at allocation time
                            per ADR 0012, so the cash is already in the
                            seller's balance).

    Parameters
    ----------
    run_log:
        Output of ``Runner.run()`` — the dict with ``"n_steps"`` and
        ``"ticks"`` keys.
    scenario:
        The ``Scenario`` used for the run.  Provides ``DemandSinkNode``
        ``income_rate``s, ``IntermediateNode`` ``holding_rate``/
        ``order_fee``s, and the catalog ``unit_cost`` per product.

    Returns
    -------
    dict[str, float]
        Keys: ``"sink_cash_created"``, ``"node_cash_total"``,
        ``"cumulative_void"``, ``"in_transit_value"``.

    Notes
    -----
    The identity holds when factory nodes have zero net production over the
    episode (``capacity_per_tick == 0`` or all produced units are sold).
    With active production the factory injects inventory value into the
    system outside the cash-conservation scope; the graph-runner integration
    test uses a zero-production factory scenario so the identity holds
    exactly.

    Payment is at allocation time (ADR 0012 — ``execute_buy`` transfers
    cash immediately).  In-transit goods are therefore already paid for and
    the seller's cash already reflects the payment.  The ``in_transit_value``
    term is provided for diagnostic use but is NOT included in the identity
    check.
    """
    from src.sim.node import DemandSinkNode, IntermediateNode

    unit_cost: dict[str, float] = {
        w.product_id: float(w.unit_cost) for w in scenario.catalog
    }

    # Collect node-rate lookup for IntermediateNodes.
    intermediate_rates: dict[str, tuple[float, float]] = {}
    sink_income_rates: dict[str, float] = {}
    for ni in scenario.nodes:
        if isinstance(ni.node, IntermediateNode):
            intermediate_rates[ni.node.id] = (
                float(ni.node.holding_rate),
                float(ni.node.order_fee),
            )
        elif isinstance(ni.node, DemandSinkNode):
            sink_income_rates[ni.node.id] = float(ni.node.income_rate)

    n_steps: int = int(run_log["n_steps"])

    # --- sink_cash_created: income_rate × n_steps per sink ---
    sink_cash_created = sum(
        rate * n_steps for rate in sink_income_rates.values()
    )

    # --- Cumulate void over all ticks ---
    cumulative_void = 0.0
    for tick_log in run_log["ticks"]:
        purchases_this_tick = tick_log.get("purchases", [])

        for node_id, (holding_rate, order_fee) in intermediate_rates.items():
            # Holding cost: closing_qty × holding_rate × unit_cost per pid.
            inv_dict = tick_log.get("node_inventory", {}).get(node_id, {})
            for pid, qty in inv_dict.items():
                if pid == "_total":
                    continue
                uc = unit_cost.get(pid, 0.0)
                cumulative_void += qty * holding_rate * uc

            # Order fee: one per distinct (buyer, supplier) pair for this buyer.
            if order_fee > 0.0:
                suppliers_this_tick = {
                    p["supplier_id"]
                    for p in purchases_this_tick
                    if p["buyer_id"] == node_id
                }
                cumulative_void += len(suppliers_this_tick) * order_fee

    # --- Final tick: node_cash_total and in_transit_value ---
    last_tick = run_log["ticks"][-1]

    node_cash_total = sum(last_tick.get("node_cash", {}).values())

    in_transit_value = sum(
        qty * unit_cost.get(pid, 0.0)
        for node_id, pend in last_tick.get("node_pending", {}).items()
        for pid, qty in pend.items()
    )

    return {
        "sink_cash_created": sink_cash_created,
        "node_cash_total": node_cash_total,
        "cumulative_void": cumulative_void,
        "in_transit_value": in_transit_value,
    }


__all__ = [
    "node_timeseries_df",
    "global_timeseries_df",
    "per_product_df",
    "node_equity",
    "flow_frame",
    "purchase_frame",
    "closing_inventory_frame",
    "conservation_identity_terms",
]
