"""Per-node-type observation builder stubs.

These functions will be fleshed out in later phases when the graph
simulation engine (``GraphSimulation``, ``GraphRunner``) is implemented.
For now they are stubs that define the intended interface so that policy
``decide`` methods can be written against a stable signature.

Each builder receives the relevant node instance plus any shared context
(central table, current tick) and returns a typed dict that the policy's
``decide`` method can inspect.

All functions here are pure (no side effects) and import-safe — they do
not depend on runtime simulation state beyond their explicit arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode


def build_factory_obs(node: "FactoryNode", *, tick: int) -> dict[str, Any]:
    """Build a ``FactoryNode`` observation dict for one tick.

    Parameters
    ----------
    node:
        The factory node being observed.
    tick:
        Current simulation tick.

    Returns
    -------
    dict with at minimum:
        ``node_id``     — node identifier
        ``tick``        — current tick
        ``inventory``   — current on-hand stock
        ``capacity``    — resolved capacity for this tick (int, post-sampling)
        ``list_price``  — current list price
    """
    # Stub — real implementation in Phase 1 (GraphSimulation).
    return {
        "node_id": node.id,
        "tick": tick,
        "inventory": node.inventory,
        "list_price": node.list_price,
    }


def build_intermediate_obs(
    node: "IntermediateNode",
    *,
    tick: int,
) -> dict[str, Any]:
    """Build an ``IntermediateNode`` observation dict for one tick.

    Parameters
    ----------
    node:
        The intermediate node being observed.
    tick:
        Current simulation tick.

    Returns
    -------
    dict with at minimum:
        ``node_id``            — node identifier
        ``tick``               — current tick
        ``inventory``          — ``{pid: qty}`` current on-hand stock
        ``pending``            — ``{supplier_id: {pid: qty}}`` in-transit
        ``list_prices``        — ``{pid: price}`` current selling prices
        ``min_order_imposed``  — ``{pid: min_qty}`` server-side minimums
    """
    # Stub — real implementation in Phase 1 (GraphSimulation).
    return {
        "node_id": node.id,
        "tick": tick,
        "inventory": dict(node.inventory),
        "pending": {k: dict(v) for k, v in node.pending.items()},
        "list_prices": dict(node.list_prices),
        "min_order_imposed": dict(node.min_order_imposed),
    }


def build_sink_obs(
    node: "DemandSinkNode",
    *,
    tick: int,
    demand_target: float = 0.0,
) -> dict[str, Any]:
    """Build a ``DemandSinkNode`` observation dict for one tick.

    Parameters
    ----------
    node:
        The demand-sink node being observed.
    tick:
        Current simulation tick.
    demand_target:
        The resolved demand target for this tick (post lifecycle/freshness
        composition). Defaults to 0.0 until the graph engine provides it.

    Returns
    -------
    dict with at minimum:
        ``node_id``        — node identifier
        ``tick``           — current tick
        ``product_id``     — bound product
        ``cash``           — current cash balance
        ``demand_target``  — resolved demand for this tick
    """
    # Stub — real implementation in Phase 1 (GraphSimulation).
    return {
        "node_id": node.id,
        "tick": tick,
        "product_id": node.product_id,
        "cash": node.cash,
        "demand_target": demand_target,
    }
