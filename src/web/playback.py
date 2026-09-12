"""Turn a Runner Run Log into per-tick DAG playback frames."""

from __future__ import annotations

from typing import Any


def _inventory_total(inv_dict: dict[str, Any]) -> int:
    if not inv_dict:
        return 0
    if "_total" in inv_dict:
        return int(inv_dict["_total"])
    return int(sum(v for k, v in inv_dict.items() if k != "_total"))


def _node_types(scenario: Any) -> dict[str, str]:
    return {ni.node.id: ni.node._node_type for ni in scenario.nodes}


def playback_frames(run_log: dict[str, Any], scenario: Any) -> list[dict[str, Any]]:
    """One frame per tick: node cash/inventory, purchases, rejection qty."""
    types = _node_types(scenario)
    frames: list[dict[str, Any]] = []
    for tick_log in run_log["ticks"]:
        nodes: dict[str, dict[str, Any]] = {}
        cash = tick_log.get("node_cash", {})
        inv = tick_log.get("node_inventory", {})
        for node_id, ntype in types.items():
            nodes[node_id] = {
                "cash": float(cash.get(node_id, 0.0)),
                "inventory": _inventory_total(inv.get(node_id, {})),
                "node_type": ntype,
            }
        purchases = [
            {
                "supplier_id": p["supplier_id"],
                "buyer_id": p["buyer_id"],
                "qty_filled": int(p["qty_filled"]),
            }
            for p in tick_log.get("purchases", [])
        ]
        rejections = sum(int(r.get("qty_rejected", 0)) for r in tick_log.get("rejections", []))
        frames.append(
            {
                "tick": tick_log["tick"],
                "nodes": nodes,
                "purchases": purchases,
                "rejections": rejections,
            }
        )
    return frames
