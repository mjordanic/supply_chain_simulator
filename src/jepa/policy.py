"""Exploratory replenishment policy.

Each tick draws a cover in lead times of recent demand and a priority over
SKUs, both from ``policy_rng``. The quantity before the ration is base-stock
against the textbook censored-sales rate. Pricing stays at ``base_price``.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.sim.policy import IntermediatePolicy, _estimate_rate, cheapest_direct_supplier

COVERS: tuple[int, ...] = (0, 1, 2, 4)


def _flatten_pending(pending: Mapping[str, Any]) -> dict[str, int]:
    """Sum in-transit units per product across suppliers."""
    totals: dict[str, int] = {}
    for supplier_pending in pending.values():
        for pid, qty in supplier_pending.items():
            totals[pid] = totals.get(pid, 0) + int(qty)
    return totals


class ExploratoryPolicy(IntermediatePolicy):
    """Covers in ``{0, 1, 2, 4}`` lead times of recent demand, rationed by priority.

    ``proposals`` records one entry per ``decide`` call: the tick, the cover
    per SKU, and the priority order (first is highest). Those draws use only
    ``policy_rng`` and the sorted product ids, so a different world seed does
    not change the sequence.
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        delivery_lag: int = 3,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.delivery_lag = int(delivery_lag)
        self.min_order = 0
        self.sales_log: dict[str, list[int]] = {}
        self.proposals: list[dict[str, Any]] = []

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        inventory = {
            pid: int(qty)
            for pid, qty in dict(obs_intermediate.get("inventory") or {}).items()
        }
        pids = sorted(inventory)
        pending = _flatten_pending(obs_intermediate.get("pending") or {})
        sales = dict(obs_intermediate.get("observed_sales") or {})
        for pid in pids:
            self.sales_log.setdefault(pid, []).append(int(sales.get(pid, 0)))

        covers = {pid: int(self.policy_rng.choice(COVERS)) for pid in pids}
        priority = list(self.policy_rng.sample(pids, len(pids))) if pids else []
        self.proposals.append(
            {
                "tick": int(obs_intermediate.get("tick", len(self.proposals))),
                "covers": dict(covers),
                "priority": priority,
            }
        )

        window = max(1, self.delivery_lag)
        rates = _estimate_rate(
            {pid: self.sales_log[pid] for pid in pids},
            window,
            {pid: [] for pid in pids},
        )
        desired: dict[str, int] = {}
        for pid in pids:
            position = inventory[pid] + pending.get(pid, 0)
            rate = float(rates.get(pid, 0.0))
            target = round(covers[pid] * rate * self.delivery_lag)
            desired[pid] = max(0, int(target) - position)

        free_space = max(
            0,
            int(obs_intermediate.get("capacity") or 0)
            - sum(inventory.values())
            - sum(pending.values()),
        )
        suppliers, unit_costs = _suppliers_and_costs(pids, obs_intermediate, central_table)
        for pid, supplier_id in suppliers.items():
            if supplier_id is None:
                desired[pid] = 0
        granted = _priority_ration(
            desired,
            priority,
            free_space,
            float(obs_intermediate.get("cash") or 0.0),
            unit_costs,
        )

        order: dict[str, list[tuple[str, int]]] = {}
        for pid in priority:
            qty = granted.get(pid, 0)
            supplier_id = suppliers.get(pid)
            if qty > 0 and supplier_id is not None:
                order[pid] = [(supplier_id, qty)]

        base = obs_intermediate.get("base_prices") or obs_intermediate.get("list_prices") or {}
        list_price = {pid: float(base.get(pid, 0.0)) for pid in pids}
        return {
            "order": order,
            "list_price": list_price,
            "min_order_imposed": {pid: 0 for pid in pids},
            "promotions": {},
        }


def _priority_ration(
    desired: Mapping[str, int],
    priority: list[str],
    free_space: int,
    cash: float,
    unit_costs: Mapping[str, float],
) -> dict[str, int]:
    """Grant units in priority order until free space or cash is exhausted."""
    remaining_space = max(0, int(free_space))
    remaining_cash = max(0.0, float(cash))
    granted = {pid: 0 for pid in desired}
    for pid in priority:
        want = int(desired.get(pid, 0))
        if want <= 0 or remaining_space <= 0:
            continue
        cost = float(unit_costs.get(pid, 0.0))
        affordable = want if cost <= 0 else int(remaining_cash / cost)
        qty = min(want, remaining_space, max(0, affordable))
        if qty <= 0:
            continue
        granted[pid] = qty
        remaining_space -= qty
        remaining_cash -= qty * cost
    return granted


def _suppliers_and_costs(
    pids: list[str],
    obs: Mapping[str, Any],
    central_table: Any,
) -> tuple[dict[str, str | None], dict[str, float]]:
    """Cheapest direct supplier per SKU, and the price cash is spent at."""
    direct = obs.get("direct_supplier_ids")
    suppliers: dict[str, str | None] = {}
    unit_costs: dict[str, float] = {}
    for pid in pids:
        supplier_id, price = cheapest_direct_supplier(
            pid,
            direct,
            central_table,
            fallback_cost=0.0,
        )
        suppliers[pid] = supplier_id
        unit_costs[pid] = price
    return suppliers, unit_costs
