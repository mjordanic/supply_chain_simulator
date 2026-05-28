"""Allocation — single-call FCFS allocation primitive.

``execute_buy`` is the atomic allocation primitive for the multi-echelon
graph engine.  Issue 05 implements the **minimum single-supplier path**:
payment debit/credit, central-table commit, and delivery scheduling.
The multi-supplier clamping logic and min-order rules are present but
the full contention path (buyer-capacity clamp, multi-supplier routing)
lands in issue 07.

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
    event_engine: Any,
    *,
    current_tick: int,
    lead_time: int,
) -> AllocationResult:
    """Execute a single buyer→supplier allocation (minimum single-supplier path).

    This implementation handles the single-supplier-no-contention case that
    the Phase 1 chain scenario exercises.  Multi-supplier clamping logic and
    strict min-order enforcement across both layers land in issue 07.

    Steps
    -----
    1. Look up the live offer from ``table``.
    2. Check supplier-imposed minimum order (``Offer.min_order``).  If
       ``qty_requested < min_order``, reject entirely.
    3. Clamp ``qty_to_fill`` by:
       - ``Offer.available_qty`` — can't over-allocate from inventory.
       - ``buyer.cash / list_price`` — buyer pays at allocation time
         (ADR 0013); can't spend more than available.
    4. ``table.commit(supplier.id, pid, qty_to_fill)`` — decrement live
       available_qty and update fill-rate EMA.
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
    event_engine:
        ``EventEngine`` used to schedule delivery callbacks.
    current_tick:
        The current simulation tick.
    lead_time:
        Number of ticks until physical delivery arrives.

    Returns
    -------
    AllocationResult
        Populated with ``qty_filled``, ``qty_rejected``, and ``cash_paid``.
    """
    # Look up the live offer from the central table.
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

    # 2. Supplier-imposed minimum order check.
    if qty_requested < offer.min_order:
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

    # 5. Transfer cash: buyer pays, supplier receives.
    buyer.cash -= cash_paid
    supplier.cash += cash_paid

    # 6. Schedule delivery callback on the event engine.
    delivery_tick = current_tick + lead_time
    _schedule_delivery(event_engine, buyer, pid, qty_to_fill, delivery_tick)

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
    pid: str,
    qty: int,
    arrival_tick: int,
) -> None:
    """Schedule a delivery callback on ``event_engine``.

    The callback fires at ``arrival_tick`` and delivers inventory to the
    buyer node.

    - ``IntermediateNode``: increments ``buyer.inventory[pid]`` by ``qty``.
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
        else:
            # FactoryNode: scalar inventory.
            buyer.inventory += qty

    event_engine.schedule(
        event_type="order_arrival",
        delay=arrival_tick,
        callback=_callback,
    )


__all__ = ["AllocationResult", "execute_buy", "shuffle_buyers"]
