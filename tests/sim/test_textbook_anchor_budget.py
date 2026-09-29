"""The textbook anchor's cash check uses the supplier's price.

``OrderUpToPolicy`` is constructed with the default ``unit_cost`` of 1.0.
The offer on the table is the price cash actually leaves at. ``decide``
is the whole path under test: Allocation is not called, so a request that
costs more than cash at the offer price would be the clamp the anchor
used to defer.
"""

from __future__ import annotations

from src.sim.central_table import CentralTable, Offer
from src.sim.policy import OrderUpToPolicy

PRICE = 11.0
CASH = 55.0  # five units at PRICE; fifty-five units at the old default of 1.0


def _table() -> CentralTable:
    table = CentralTable()
    table.publish(
        "F",
        "A",
        Offer(available_qty=10_000, list_price=PRICE, min_order=0),
    )
    return table


def _obs(*, on_hand: int, cash: float, sales: int) -> dict:
    return {
        "tick": 0,
        "inventory": {"A": on_hand},
        "pending": {},
        "observed_sales": {"A": sales},
        "capacity": 100_000,
        "cash": cash,
        "list_prices": {"A": 20.0},
        "base_prices": {"A": 20.0},
        "direct_supplier_ids": {"F"},
        "min_order_imposed": {"A": 0},
    }


def _qty(action: dict) -> int:
    return sum(qty for _supplier, qty in action["order"].get("A", []))


def test_internal_budget_binds_at_the_supplier_price_when_cash_is_short():
    """A short till cuts the request inside decide, at the offer price.

    Steady-state demand wants far more than five units. Cash of 55 buys
    five units at 11 and fifty-five units at a unit cost of 1. The
    emitted request is the five-unit order. Nothing downstream has to
    clamp it.
    """
    policy = OrderUpToPolicy(
        unit_cost=1.0,
        opening_budget_pct=0.0,
        safety_lead_pct_of_lag=0.0,
        cover_horizon_ticks=10,
        delivery_lag=2,
    )
    table = _table()
    policy.decide(_obs(on_hand=100, cash=1_000_000.0, sales=10), table)
    action = policy.decide(_obs(on_hand=0, cash=CASH, sales=10), table)

    qty = _qty(action)
    assert qty * PRICE <= CASH
    assert qty == int(CASH // PRICE)
    assert qty < int(CASH // 1.0)
