"""Behaviour policies for the transition log.

``ExploratoryPolicy`` draws a cover in lead times of recent demand and a
priority over SKUs, both from ``policy_rng``. The quantity before the ration
is base-stock against the textbook censored-sales rate. ``NeverOrderPolicy``
and ``OverOrderPolicy`` are the deliberately poor arms. Pricing stays at
``base_price``.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.sim.policy import (
    IntermediatePolicy,
    _estimate_rate,
    cheapest_direct_supplier,
    direct_offers,
)

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

    ``planning_depth``, when set, replaces ``delivery_lag`` as the tick
    horizon in the target ``cover × rate × horizon``. The rate window stays
    ``delivery_lag``.

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
        planning_depth: int | None = None,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.delivery_lag = int(delivery_lag)
        # Ticks of demand in the order-up-to target. ``None`` keeps the
        # target in lead times of ``delivery_lag``. The transition log sets
        # a depth per episode so the horizon is not the same on every run.
        self.planning_depth = None if planning_depth is None else int(planning_depth)
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
        horizon = self.delivery_lag if self.planning_depth is None else self.planning_depth
        desired: dict[str, int] = {}
        for pid in pids:
            position = inventory[pid] + pending.get(pid, 0)
            rate = float(rates.get(pid, 0.0))
            target = round(covers[pid] * rate * horizon)
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


class NeverOrderPolicy(IntermediatePolicy):
    """Order nothing.

    The transition log includes this arm so the shelf can sit empty.
    ``planning_depth`` is 0: there is no horizon to cover.
    """

    def __init__(self, *, policy_seed: int | None = None) -> None:
        super().__init__(policy_seed=policy_seed)
        self.planning_depth = 0

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        del central_table
        inventory = dict(obs_intermediate.get("inventory") or {})
        return _flat_order(obs_intermediate, sorted(inventory), {})


class OverOrderPolicy(IntermediatePolicy):
    """Keep buying, and overshoot the shelf when cash can pay for it.

    A competent cover places one large order and then waits. This arm
    asks for at least one unit every tick, so a purchase lands whenever
    supply and cash allow. When cash covers the open shelf it asks for
    ``free space + planning_depth`` units, past what the shelf can hold,
    so the capacity clamp is reachable. ``planning_depth`` is above every
    textbook cover horizon.
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        planning_depth: int,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.planning_depth = int(planning_depth)

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
        capacity = int(obs_intermediate.get("capacity") or 0)
        free = max(0, capacity - sum(inventory.values()))
        cash = float(obs_intermediate.get("cash") or 0.0)
        chosen: dict[str, str] = {}
        prices: list[float] = []
        for pid in pids:
            offers = list(
                direct_offers(pid, obs_intermediate.get("direct_supplier_ids"), central_table)
            )
            if not offers:
                continue
            stocked = [item for item in offers if _available(item) > 0] or offers
            supplier_id, offer = min(
                stocked, key=lambda item: (float(item[1].list_price), item[0])
            )
            chosen[pid] = supplier_id
            prices.append(float(offer.list_price))
        price = min(prices) if prices else 0.0
        if price > 0 and free > 0 and cash >= free * price:
            qty = free + self.planning_depth
        else:
            qty = 1
        order = {pid: [(supplier_id, max(int(qty), 1))] for pid, supplier_id in chosen.items()}
        return _flat_order(obs_intermediate, pids, order)


def _available(item: tuple[str, Any]) -> int:
    return int(getattr(item[1], "available_qty", 0) or 0)


def _flat_order(
    obs: Mapping[str, Any],
    pids: list[str],
    order: dict[str, list[tuple[str, int]]],
) -> dict[str, Any]:
    base = obs.get("base_prices") or obs.get("list_prices") or {}
    return {
        "order": order,
        "list_price": {pid: float(base.get(pid, 0.0)) for pid in pids},
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
