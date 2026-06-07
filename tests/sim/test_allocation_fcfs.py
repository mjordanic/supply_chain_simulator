"""Unit tests for allocation.execute_buy — full FCFS contract (issue 07).

Tests cover all clamp paths and the two-layer min-order rejection rule.
All tests use pure in-memory fixtures; no live GraphSimulation is required.
"""

from __future__ import annotations

from random import Random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.sim.allocation import AllocationResult, execute_buy
from src.sim.central_table import CentralTable, Offer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_table(
    supplier_id: str,
    pid: str,
    available: int = 100,
    price: float = 5.0,
    min_order: int = 0,
) -> CentralTable:
    """Return a CentralTable with a single published offer."""
    table = CentralTable()
    table.publish(supplier_id, pid, Offer(
        available_qty=available,
        list_price=price,
        min_order=min_order,
    ))
    return table


def _make_buyer(
    cash: float = 1000.0,
    inventory: dict | None = None,
    capacity: int = 0,
    policy=None,
) -> SimpleNamespace:
    """Return a minimal buyer namespace."""
    b = SimpleNamespace(
        id="buyer-1",
        cash=cash,
        inventory=inventory if inventory is not None else {},
        capacity=capacity,
        policy=policy,
    )
    return b


def _make_supplier(cash: float = 0.0) -> SimpleNamespace:
    """Return a minimal supplier namespace."""
    return SimpleNamespace(id="supplier-1", cash=cash)


def _make_event_engine() -> MagicMock:
    """Return a minimal event-engine mock."""
    ee = MagicMock()
    # Track scheduled callbacks for assertion.
    ee.scheduled_callbacks = []

    def _schedule(event_type, delay, callback):
        ee.scheduled_callbacks.append((event_type, delay, callback))

    ee.schedule.side_effect = _schedule
    return ee


# ---------------------------------------------------------------------------
# AllocationResult dataclass
# ---------------------------------------------------------------------------

class TestAllocationResult:
    def test_fields(self):
        r = AllocationResult(qty_filled=5, qty_rejected=2, cash_paid=25.0)
        assert r.qty_filled == 5
        assert r.qty_rejected == 2
        assert r.cash_paid == 25.0

    def test_frozen(self):
        r = AllocationResult(qty_filled=5, qty_rejected=2, cash_paid=25.0)
        with pytest.raises(Exception):
            r.qty_filled = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Happy path: basic purchase
# ---------------------------------------------------------------------------

class TestExecuteBuyBasicPurchase:
    def test_debits_buyer_credits_supplier(self):
        buyer = _make_buyer(cash=500.0)
        supplier = _make_supplier(cash=0.0)
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=2,
        )

        assert result.qty_filled == 10
        assert result.qty_rejected == 0
        assert result.cash_paid == 50.0
        assert buyer.cash == 450.0
        assert supplier.cash == 50.0

    def test_qty_filled_plus_rejected_equals_requested(self):
        buyer = _make_buyer(cash=25.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled + result.qty_rejected == 10

    def test_cash_paid_equals_qty_filled_times_price(self):
        buyer = _make_buyer(cash=1000.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=7.5)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=8,
            table=table,
            event_engine=ee,
            current_tick=3,
            lead_time=2,
        )

        assert result.cash_paid == result.qty_filled * 7.5


# ---------------------------------------------------------------------------
# Clamp path: inventory-limited
# ---------------------------------------------------------------------------

class TestExecuteBuyInventoryLimited:
    def test_clamp_to_available_qty(self):
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=3, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 3
        assert result.qty_rejected == 7

    def test_zero_available_returns_all_rejected(self):
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=0, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5
        assert result.cash_paid == 0.0
        # No cash transferred.
        assert buyer.cash == 9999.0
        assert supplier.cash == 0.0


# ---------------------------------------------------------------------------
# Clamp path: cash-limited
# ---------------------------------------------------------------------------

class TestExecuteBuyCashLimited:
    def test_clamp_to_affordable_qty(self):
        # Can afford 5 at price=5 with cash=25.
        buyer = _make_buyer(cash=25.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 5
        assert result.cash_paid == 25.0
        assert buyer.cash == 0.0

    def test_zero_cash_returns_all_rejected(self):
        buyer = _make_buyer(cash=0.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5
        assert result.cash_paid == 0.0

    def test_partial_unit_cash_floors_to_integer(self):
        # cash=12.0, price=5.0 → affordable = int(12/5) = 2 units.
        buyer = _make_buyer(cash=12.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 2
        assert result.cash_paid == 10.0


# ---------------------------------------------------------------------------
# Clamp path: capacity-limited (buyer's remaining inventory capacity)
# ---------------------------------------------------------------------------

class TestExecuteBuyCapacityLimited:
    def test_clamp_to_remaining_capacity(self):
        # capacity=10, inventory currently has 7 → remaining=3.
        buyer = _make_buyer(
            cash=9999.0,
            inventory={"P0001": 7},
            capacity=10,
        )
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 3
        assert result.qty_rejected == 7

    def test_full_capacity_returns_all_rejected(self):
        # capacity=10, inventory already at 10 → no room.
        buyer = _make_buyer(
            cash=9999.0,
            inventory={"P0001": 10},
            capacity=10,
        )
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5
        assert result.cash_paid == 0.0

    def test_zero_capacity_means_no_clamp(self):
        # capacity=0 (or absent) means no capacity clamp applied.
        buyer = _make_buyer(cash=9999.0, inventory={"P0001": 50}, capacity=0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=10, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        # Clamp is only by inventory (10 available).
        assert result.qty_filled == 10

    def test_multi_product_capacity_uses_sum(self):
        # capacity=10, two products: P0001=4, P0002=4 → total=8, remaining=2.
        buyer = _make_buyer(
            cash=9999.0,
            inventory={"P0001": 4, "P0002": 4},
            capacity=10,
        )
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 2


# ---------------------------------------------------------------------------
# Two-layer min-order rejection
# ---------------------------------------------------------------------------

class TestExecuteBuyMinOrderRejection:
    def test_supplier_min_order_rejects(self):
        # Supplier requires min 5; buyer requests 3.
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=5)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=3,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 3
        assert result.cash_paid == 0.0

    def test_buyer_policy_min_order_rejects(self):
        # Buyer policy requires min 10; supplier min is 0; buyer requests 5.
        # effective_min = min(0, 10) = 0 → NOT rejected (since qty=5 >= 0).
        # Wait — that's wrong. Let me re-read the spec.
        # "Reject when qty_requested < min(supplier_min, buyer_min)"
        # min(0, 10) = 0, so 5 < 0 is False → NOT rejected.
        # For buyer policy min to reject, supplier_min must also be >= the threshold.
        # This test: supplier_min=10, buyer_policy_min=8.
        # effective_min = min(10, 8) = 8; qty=5 < 8 → rejected.
        policy = SimpleNamespace(min_order=8)
        buyer = _make_buyer(cash=9999.0, policy=policy)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=10)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5
        assert result.cash_paid == 0.0

    def test_two_layer_min_uses_minimum_of_both(self):
        # supplier_min=10, buyer_min=5; effective_min=5.
        # qty_requested=7 >= 5 → NOT rejected.
        policy = SimpleNamespace(min_order=5)
        buyer = _make_buyer(cash=9999.0, policy=policy)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=10)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=7,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        # Not rejected (7 >= min(10, 5) = 5).
        assert result.qty_filled == 7

    def test_no_policy_min_order_uses_supplier_only(self):
        # No buyer policy → supplier_min is the only active constraint.
        # supplier_min=5, qty=3 < 5 → rejected.
        # This tests that policy=None doesn't cause AttributeError.
        buyer = _make_buyer(cash=9999.0, policy=None)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=5)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=3,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        # No buyer policy min → effective_min = supplier_min = 5; qty=3 < 5 → rejected.
        assert result.qty_filled == 0
        assert result.qty_rejected == 3

    def test_supplier_min_only_both_layers_same(self):
        # Both layers agree: supplier_min=5, buyer_min=5.
        # effective_min=5; qty=4 < 5 → rejected.
        policy = SimpleNamespace(min_order=5)
        buyer = _make_buyer(cash=9999.0, policy=policy)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=5)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=4,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 4


# ---------------------------------------------------------------------------
# Delivery scheduling
# ---------------------------------------------------------------------------

class TestExecuteBuyDeliveryScheduling:
    def test_delivery_scheduled_at_correct_tick(self):
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=7,
            lead_time=3,
        )

        # Delivery should be at tick 7 + 3 = 10.
        assert len(ee.scheduled_callbacks) == 1
        event_type, delay, _ = ee.scheduled_callbacks[0]
        assert event_type == "order_arrival"
        assert delay == 10

    def test_no_delivery_scheduled_when_nothing_filled(self):
        buyer = _make_buyer(cash=0.0)  # cash-limited → zero fill
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert len(ee.scheduled_callbacks) == 0

    def test_none_event_engine_skips_delivery_scheduling(self):
        """execute_buy with event_engine=None does not raise."""
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=None,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 5

    def test_delivery_callback_increments_dict_inventory(self):
        buyer = _make_buyer(cash=9999.0, inventory={"P0001": 0})
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=2,
        )

        # Fire the callback manually.
        _, _, callback = ee.scheduled_callbacks[0]
        callback()
        assert buyer.inventory["P0001"] == 10


# ---------------------------------------------------------------------------
# Missing offer path
# ---------------------------------------------------------------------------

class TestExecuteBuyMissingOffer:
    def test_no_offer_returns_all_rejected(self):
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = CentralTable()  # Empty table — no offer published.
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5
        assert result.cash_paid == 0.0

    def test_wrong_supplier_returns_all_rejected(self):
        """Offer exists but for a different supplier."""
        buyer = _make_buyer(cash=9999.0)
        supplier = SimpleNamespace(id="other-supplier", cash=0.0)
        # Table has an offer for "supplier-1", not "other-supplier".
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 0
        assert result.qty_rejected == 5


# ---------------------------------------------------------------------------
# Cash conservation
# ---------------------------------------------------------------------------

class TestExecuteBuyCashConservation:
    def test_cash_is_conserved(self):
        """Total cash before == total cash after."""
        buyer = _make_buyer(cash=200.0)
        supplier = _make_supplier(cash=50.0)
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        total_before = buyer.cash + supplier.cash

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=15,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        total_after = buyer.cash + supplier.cash
        assert abs(total_after - total_before) < 1e-9

    def test_cash_ledger_none_does_not_raise(self):
        """Passing cash_ledger=None explicitly is a no-op."""
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)
        ee = _make_event_engine()

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid="P0001",
            qty_requested=5,
            table=table,
            cash_ledger=None,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 5


# ---------------------------------------------------------------------------
# AllocationResult.reason — binding-constraint annotation (issue 05)
# ---------------------------------------------------------------------------

class TestAllocationResultReason:
    """``AllocationResult.reason`` reports the binding constraint."""

    def test_reason_none_on_full_fill(self):
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=10,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason is None

    def test_reason_no_offer(self):
        """Supplier published no offer for this pid."""
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        # Publish a different pid so "P0001" has no offer.
        table = CentralTable()
        table.publish("supplier-1", "OTHER", Offer(available_qty=100, list_price=5.0, min_order=0))

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=10,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "no_offer"
        assert result.qty_filled == 0
        assert result.qty_rejected == 10

    def test_reason_below_min_order(self):
        """Order quantity below the supplier's min_order."""
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0, min_order=20)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=5,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "below_min_order"
        assert result.qty_filled == 0
        assert result.qty_rejected == 5

    def test_reason_insufficient_stock(self):
        """Supplier has fewer units than requested."""
        buyer = _make_buyer(cash=9999.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=3, price=5.0)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=10,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "insufficient_stock"
        assert result.qty_filled == 3
        assert result.qty_rejected == 7

    def test_reason_insufficient_cash(self):
        """Buyer's cash limits the fill."""
        # cash=10, price=5 → affordable=2; requested=10
        buyer = _make_buyer(cash=10.0)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=10,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "insufficient_cash"
        assert result.qty_filled == 2
        assert result.qty_rejected == 8

    def test_reason_insufficient_capacity(self):
        """Buyer capacity limits the fill."""
        # capacity=5, inventory already has 3 → remaining=2; requested=10
        buyer = _make_buyer(cash=9999.0, inventory={"P0001": 3}, capacity=5)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=10,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "insufficient_capacity"
        assert result.qty_filled == 2
        assert result.qty_rejected == 8

    def test_reason_on_zero_fill_due_to_capacity(self):
        """capacity fully exhausted → reason is insufficient_capacity."""
        buyer = _make_buyer(cash=9999.0, inventory={"P0001": 10}, capacity=10)
        supplier = _make_supplier()
        table = _make_table("supplier-1", "P0001", available=100, price=5.0)

        result = execute_buy(
            buyer=buyer, supplier=supplier, pid="P0001", qty_requested=5,
            table=table, event_engine=None, current_tick=0, lead_time=1,
        )

        assert result.reason == "insufficient_capacity"
        assert result.qty_filled == 0
