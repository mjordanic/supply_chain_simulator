"""Allocation stub — full implementation in issue 07.

``execute_buy`` is the single-call FCFS allocation primitive for the
multi-echelon graph engine. Its full implementation (clamping by inventory,
cash, capacity; payment + delivery scheduling; ``CentralTable.commit``) is
the subject of issue 07-allocation-execute-buy-full.

This stub exists so that:
1. ``NodePolicy`` subclasses can import ``AllocationResult`` for type hints.
2. The ``"allocation"`` sub-seed in ``episode_sampler._SUB_SEED_PARAMS`` has
   a module to land in.
3. Later phases can ``from src.sim.allocation import execute_buy`` without
   changing the import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    table: Any,
    cash_ledger: Any,
    *,
    current_tick: int,
    lead_time: int,
) -> AllocationResult:
    """Execute a single buyer→supplier allocation (stub).

    Full implementation in issue 07-allocation-execute-buy-full.

    Raises
    ------
    NotImplementedError
        Always — this is a stub.
    """
    raise NotImplementedError(
        "allocation.execute_buy is not yet implemented; see issue 07."
    )


def shuffle_buyers(buyers: list[Any], allocation_rng: Any) -> list[Any]:
    """Deterministically shuffle a list of buyers using ``allocation_rng`` (stub).

    Full implementation in issue 07-allocation-execute-buy-full.
    """
    raise NotImplementedError(
        "allocation.shuffle_buyers is not yet implemented; see issue 07."
    )
