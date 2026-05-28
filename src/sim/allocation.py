"""Allocation — single-call FCFS allocation primitive.

``execute_buy`` is the atomic allocation primitive for the multi-echelon
graph engine.  Issue 07 completes the full FCFS allocation contract per
ADR 0012: capacity clamp, two-layer min-order rejection, cash_ledger
parameter, and all unit-testable clamp paths.

Public API
----------
- ``AllocationResult``  — frozen result dataclass
- ``execute_buy``       — single buyer→supplier allocation
- ``shuffle_buyers``    — deterministic per-phase buyer shuffle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.sim.central_table import CentralTable


@dataclass(frozen=True)
class AllocationResult:
    """Result of one ``execute_buy`` call.

    Fields
    ------
    qty_filled:
        Units actually allocated (≤ qty_requested).
    qty_rejected:
        Units that could not be filled (due to inventory, cash, capacity,
        or min-order rejection).
    cash_paid:
        Total cash transferred from buyer to seller.
    """

    qty_filled: int
    qty_rejected: int
    cash_paid: float


def execute_buy(
    buyer: Any,
    supplier: Any,
    pid: str,
    qty_requested: int,
    table: "CentralTable",
    cash_ledger: Any = None,
    event_engine: Any = None,
    *,
    current_tick: int,
    lead_time: int,
) -> AllocationResult:
    """Execute a single buyer→supplier allocation (full FCFS contract).

    Implements the complete ADR 0012 allocation primitive with all clamp
    paths, two-layer min-order rejection, and cash_ledger support.

    Steps
    -----
    1. Look up the live offer from ``table``.
    2. Two-layer min-order check: reject when
       ``qty_requested < min(supplier-imposed min_order, buyer-side policy
       min_order)``.  Supplier-side comes from ``Offer.min_order``; buyer-side
       comes from ``buyer.policy.min_order`` (defaulting to 0 if absent).
    3. Clamp ``qty_to_fill`` by:
       - ``Offer.available_qty`` — never over-allocate from inventory.
       - ``buyer.cash / list_price`` — buyer pays at allocation time
         (ADR 0013); can't spend more than available.
       - Remaining buyer capacity — for ``IntermediateNode`` buyers with a
         finite ``capacity``, the fill is further clamped to
         ``capacity - sum(inventory.values())``.
    4. ``table.commit(supplier.id, pid, qty_to_fill)`` — decrement live
       available_qty and update fill-rate EMA.
    4b. Decrement supplier node inventory at allocation time.
    4c. Record in-transit order in ``buyer.pending[supplier_id][pid]``.
    5. Debit buyer cash and credit supplier cash by
       ``qty_to_fill × list_price``.
    6. Schedule a delivery callback on ``event_engine`` at
       ``current_tick + lead_time``.

    Parameters
    ----------
    buyer:
        The buying node.  Must have ``.cash: float`` and
        ``.id: str`` attributes.  Delivery increments
        ``buyer.inventory[pid]`` via a callback.
    supplier:
        The selling node.  Must have ``.id: str`` and ``.cash: float``.
    pid:
        Product ID being purchased.
    qty_requested:
        Units the buyer wants to acquire.
    table:
        Live ``CentralTable`` instance.  ``publish`` must have been called
        for ``(supplier.id, pid)`` this tick before ``execute_buy``.
    cash_ledger:
        Optional external cash ledger for audit tracking.  Currently
        reserved for future use; pass ``None`` (the default) if not needed.
    event_engine:
        ``EventEngine`` used to schedule delivery callbacks.  May be
        ``None`` in unit tests that don't need delivery scheduling.
    current_tick:
        The current simulation tick.
    lead_time:
        Number of ticks until physical delivery arrives.

    Returns
    -------
    AllocationResult
        Populated with ``qty_filled``, ``qty_rejected``, and ``cash_paid``.
    """
    # 1. Look up the live offer from the central table.
    offer_rows = table.snapshot_for_buyer(pid)
    offer = None
    for sid, o in offer_rows:
        if sid == supplier.id:
            offer = o
            break

    if offer is None:
        # Supplier has not published an offer for this pid this tick.
        return AllocationResult(
            qty_filled=0,
            qty_rejected=qty_requested,
            cash_paid=0.0,
        )

    # 2. Two-layer min-order check (ADR 0012).
    # The effective minimum is min(supplier_min, buyer_policy_min) when both
    # layers have an explicit constraint.  When the buyer has no policy
    # min-order constraint, only the supplier-side threshold applies.
    supplier_min = offer.min_order
    buyer_policy = getattr(buyer, "policy", None)
    buyer_policy_min = getattr(buyer_policy, "min_order", None)
    if buyer_policy_min is not None and buyer_policy_min > 0:
        effective_min = min(supplier_min, buyer_policy_min)
    else:
        effective_min = supplier_min
    if qty_requested < effective_min:
        return AllocationResult(
            qty_filled=0,
            qty_rejected=qty_requested,
            cash_paid=0.0,
        )

    # 3. Clamp by available inventory.
    qty_to_fill = min(qty_requested, offer.available_qty)

    # 3b. Clamp by buyer's cash (payment at allocation time per ADR 0013).
    list_price = offer.list_price
    if list_price > 0:
        cash_affordable = int(buyer.cash / list_price)
        qty_to_fill = min(qty_to_fill, cash_affordable)

    # 3c. Clamp by remaining buyer capacity (for IntermediateNode with finite capacity).
    buyer_capacity = getattr(buyer, "capacity", 0) or 0
    if buyer_capacity > 0:
        buyer_inventory = getattr(buyer, "inventory", None)
        if isinstance(buyer_inventory, dict):
            current_stock = sum(buyer_inventory.values())
        elif isinstance(buyer_inventory, (int, float)):
            current_stock = buyer_inventory
        else:
            current_stock = 0
        remaining_capacity = max(0, buyer_capacity - current_stock)
        qty_to_fill = min(qty_to_fill, remaining_capacity)

    qty_to_fill = max(0, qty_to_fill)
    qty_rejected = qty_requested - qty_to_fill
    cash_paid = qty_to_fill * list_price

    if qty_to_fill == 0:
        return AllocationResult(
            qty_filled=0,
            qty_rejected=qty_rejected,
            cash_paid=0.0,
        )

    # 4. Commit to the central table — decrements available_qty and updates EMA.
    table.commit(supplier.id, pid, qty_to_fill)

    # 4b. Decrement supplier inventory — goods leave the supplier at allocation
    #     time.  Physical delivery to the buyer is deferred by lead_time ticks
    #     (scheduled below), but the goods must leave the seller's shelf now so
    #     that subsequent buyers in the same tick see reduced available stock
    #     (the central table already reflects this via commit, but the supplier's
    #     node.inventory must match so the next tick's publish_offers step
    #     starts from the correct physical stock level).
    if isinstance(getattr(supplier, "inventory", None), dict):
        # IntermediateNode: dict-based inventory.
        current = supplier.inventory.get(pid, 0)
        supplier.inventory[pid] = max(0, current - qty_to_fill)
    elif isinstance(getattr(supplier, "inventory", None), (int, float)):
        # FactoryNode: scalar inventory.
        supplier.inventory = max(0, int(supplier.inventory) - qty_to_fill)

    # 4c. For IntermediateNode buyers, record the in-transit order in
    #     ``buyer.pending[supplier_id][pid]``.  This keeps ``pending`` in
    #     sync with what is actually in transit so the policy's position
    #     estimate (inventory + pending) is accurate.
    buyer_pending = getattr(buyer, "pending", None)
    if isinstance(buyer_pending, dict):
        sup_map = buyer_pending.setdefault(supplier.id, {})
        sup_map[pid] = sup_map.get(pid, 0) + qty_to_fill

    # 5. Transfer cash: buyer pays, supplier receives.
    buyer.cash -= cash_paid
    supplier.cash += cash_paid

    # 6. Schedule delivery callback on the event engine (if provided).
    if event_engine is not None:
        delivery_tick = current_tick + lead_time
        _schedule_delivery(
            event_engine, buyer, supplier.id, pid, qty_to_fill, delivery_tick
        )

    return AllocationResult(
        qty_filled=qty_to_fill,
        qty_rejected=qty_rejected,
        cash_paid=cash_paid,
    )


def shuffle_buyers(buyers: list[Any], allocation_rng: Any) -> list[Any]:
    """Return a deterministically shuffled copy of *buyers*.

    Uses Fisher-Yates shuffle driven by ``allocation_rng`` so the order is
    reproducible from the same ``world_seed`` (ADR 0016) and orthogonal to
    ``world_rng``.

    Parameters
    ----------
    buyers:
        List of buyer nodes to shuffle.
    allocation_rng:
        A ``random.Random`` instance seeded from the ``"allocation"``
        sub-seed.

    Returns
    -------
    list
        A new list containing the same elements in shuffled order.
    """
    shuffled = list(buyers)
    allocation_rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _schedule_delivery(
    event_engine: Any,
    buyer: Any,
    supplier_id: str,
    pid: str,
    qty: int,
    arrival_tick: int,
) -> None:
    """Schedule a delivery callback on ``event_engine``.

    The callback fires at ``arrival_tick`` and delivers inventory to the
    buyer node.

    - ``IntermediateNode``: increments ``buyer.inventory[pid]`` by ``qty``
      and decrements ``buyer.pending[supplier_id][pid]`` by ``qty``.
    - ``FactoryNode``: increments scalar ``buyer.inventory`` (the factory
      receives returned/transferred goods, which is unusual but handled).
    - ``DemandSinkNode``: no inventory field — delivered goods are
      considered consumed at allocation time (payment already debited).
      The delivery callback is a no-op for sinks.
    """
    def _callback() -> None:
        if not hasattr(buyer, "inventory"):
            # DemandSinkNode and other sink-like nodes — no inventory field.
            return
        if isinstance(buyer.inventory, dict):
            # IntermediateNode: dict-based inventory.
            buyer.inventory[pid] = buyer.inventory.get(pid, 0) + qty
            # Clear the in-transit counter now that goods have arrived.
            buyer_pending = getattr(buyer, "pending", None)
            if isinstance(buyer_pending, dict) and supplier_id in buyer_pending:
                sup_map = buyer_pending[supplier_id]
                remaining = max(0, sup_map.get(pid, 0) - qty)
                if remaining == 0:
                    sup_map.pop(pid, None)
                else:
                    sup_map[pid] = remaining
                if not sup_map:
                    buyer_pending.pop(supplier_id, None)
        else:
            # FactoryNode: scalar inventory.
            buyer.inventory += qty

    event_engine.schedule(
        event_type="order_arrival",
        delay=arrival_tick,
        callback=_callback,
    )


__all__ = ["AllocationResult", "execute_buy", "shuffle_buyers"]
