"""CentralTable deep module — live supplier-offer table for the allocation phase.

This module is the live-state contract of ADR 0012. It is a small interface
around rich semantics: mutation happens during a phase (per-tick), not between ticks.

Public API
----------
- ``Offer``        — immutable snapshot of one seller's offer for one product
- ``CentralTable`` — mutable table keyed by ``(seller_id, pid)``

Semantics
---------
At the start of each tick all sellers call ``publish(seller_id, pid, offer)``
to reset their offer. During a phase buyers sequentially call
``commit(seller_id, pid, qty)`` which decrements ``available_qty`` live and
updates the rolling qty-weighted EMA on ``fill_rate_recent``.

``snapshot_for_buyer(pid)`` returns the current post-commit state as a list of
``(seller_id, Offer)`` pairs, reflecting every prior allocation in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_EMA_WINDOW = 10


@dataclass
class Offer:
    """One seller's offer for one product at a point in time.

    Attributes
    ----------
    available_qty:
        Units currently available (decremented live by ``commit``).
    list_price:
        Per-unit price charged at allocation time.
    min_order:
        Server-imposed minimum order quantity. Orders below this threshold
        are rejected by the allocator.
    fill_rate_recent:
        Exponential moving average of qty-weighted partial-fill ratios over
        the last ``_EMA_WINDOW`` ticks. 1.0 = always fully filled; 0.0 = never
        filled. Updated by ``commit``.
    """

    available_qty: int
    list_price: float
    min_order: int
    fill_rate_recent: float = 1.0


class CentralTable:
    """Globally visible table of live supplier offers keyed by ``(seller_id, pid)``.

    Thread-safety: not thread-safe. Within one tick, allocation is sequential.
    """

    def __init__(self) -> None:
        # (seller_id, pid) -> Offer
        self._rows: dict[tuple[str, str], Offer] = {}
        # EMA state: per key, store the running EMA value
        # (same as fill_rate_recent; kept separately for clarity)
        self._ema: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def publish(self, seller_id: str, pid: str, offer: Offer) -> None:
        """Overwrite the row for ``(seller_id, pid)`` with *offer*.

        Called by sellers at the start of each tick to reset their offer.
        Preserves the existing ``fill_rate_recent`` EMA — the published
        ``Offer.fill_rate_recent`` value is used as the initial EMA if no
        prior history exists.
        """
        key = (seller_id, pid)
        existing_ema = self._ema.get(key, offer.fill_rate_recent)
        # Overwrite, but keep the accumulated EMA unless this is a fresh key
        new_offer = Offer(
            available_qty=offer.available_qty,
            list_price=offer.list_price,
            min_order=offer.min_order,
            fill_rate_recent=existing_ema,
        )
        self._rows[key] = new_offer
        self._ema[key] = existing_ema

    def commit(self, seller_id: str, pid: str, qty: int) -> None:
        """Decrement ``available_qty`` by *qty* and update the fill-rate EMA.

        Parameters
        ----------
        seller_id, pid:
            Identify the row to update.
        qty:
            Quantity being allocated. Must be ≤ ``available_qty``.

        Raises
        ------
        ValueError
            If *qty* exceeds ``available_qty`` (would push stock below zero).
        """
        key = (seller_id, pid)
        offer = self._rows[key]

        if qty > offer.available_qty:
            raise ValueError(
                f"Cannot commit {qty} units for ({seller_id!r}, {pid!r}): "
                f"only {offer.available_qty} available."
            )

        # Qty-weighted fill rate: ratio of what was committed vs what was available
        # (available_qty before this commit is the "requested" from the seller's side)
        fill_fraction = qty / offer.available_qty if offer.available_qty > 0 else 1.0

        # EMA update: alpha = 2 / (window + 1) for window-size EMA
        alpha = 2.0 / (_EMA_WINDOW + 1)
        new_ema = alpha * fill_fraction + (1.0 - alpha) * self._ema.get(key, 1.0)

        self._rows[key] = Offer(
            available_qty=offer.available_qty - qty,
            list_price=offer.list_price,
            min_order=offer.min_order,
            fill_rate_recent=new_ema,
        )
        self._ema[key] = new_ema

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def snapshot_for_buyer(self, pid: str) -> list[tuple[str, Offer]]:
        """Return all ``(seller_id, Offer)`` pairs for *pid* in current state.

        Reflects all ``commit`` calls made so far this phase. Returns an empty
        list if no seller has published an offer for *pid*.
        """
        return [
            (seller_id, offer)
            for (seller_id, p), offer in self._rows.items()
            if p == pid
        ]
