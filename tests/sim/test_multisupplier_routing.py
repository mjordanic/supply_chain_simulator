"""Unit tests for MultiSupplierTextbookPolicy._split_across_suppliers.

Pure unit tests on the routing function — no simulation engine required.
Tests verify the four routing contracts described in the issue:

1. Cheapest-first supplier ordering (default strategy).
2. Both min-order layers reject sub-minimum lines.
3. Partial-fill arithmetic when no single supplier covers the requested qty.
4. Pluggable routing_strategy invoked when supplied.
"""

from __future__ import annotations

from src.sim.central_table import CentralTable, Offer
from src.sim.policy import MultiSupplierTextbookPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(offers: dict[tuple[str, str], dict]) -> CentralTable:
    """Build a CentralTable with the given offers.

    offers: {(seller_id, pid): {"available_qty", "list_price", "min_order"}}
    """
    table = CentralTable()
    for (seller_id, pid), o in offers.items():
        table.publish(
            seller_id,
            pid,
            Offer(
                available_qty=o["available_qty"],
                list_price=o["list_price"],
                min_order=o.get("min_order", 0),
            ),
        )
    return table


# ---------------------------------------------------------------------------
# 1. Cheapest-first ordering (default routing strategy)
# ---------------------------------------------------------------------------


class TestCheapestFirst:
    def test_routes_all_to_cheapest_when_it_has_enough_stock(self):
        """When the cheapest supplier has enough stock, all goes there."""
        pid = "P0000"
        table = _make_table({
            ("cheap", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 0},
            ("expensive", pid): {"available_qty": 100, "list_price": 5.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=30,
            supplier_ids=["cheap", "expensive"],
            central_table=table,
        )
        # All 30 from cheap, nothing from expensive.
        assert result == [("cheap", 30)]

    def test_routes_to_expensive_only_when_cheap_is_empty(self):
        """When the cheapest supplier has no stock, fall back to the next."""
        pid = "P0000"
        table = _make_table({
            ("cheap", pid): {"available_qty": 0, "list_price": 1.0, "min_order": 0},
            ("expensive", pid): {"available_qty": 100, "list_price": 5.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=20,
            supplier_ids=["cheap", "expensive"],
            central_table=table,
        )
        assert result == [("expensive", 20)]

    def test_cheapest_first_order_respected_regardless_of_input_order(self):
        """Input list order does not override cheapest-first sorting."""
        pid = "P0000"
        table = _make_table({
            ("supplier_hi", pid): {"available_qty": 100, "list_price": 9.0, "min_order": 0},
            ("supplier_lo", pid): {"available_qty": 100, "list_price": 2.0, "min_order": 0},
        })
        # supplier_hi is listed first, but should not receive any units
        # because supplier_lo is cheaper and has enough stock.
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=50,
            supplier_ids=["supplier_hi", "supplier_lo"],  # expensive listed first
            central_table=table,
        )
        assert result == [("supplier_lo", 50)]

    def test_empty_result_when_no_stock_anywhere(self):
        """Return empty list when all suppliers are out of stock."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 0, "list_price": 1.0, "min_order": 0},
            ("s2", pid): {"available_qty": 0, "list_price": 2.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=10,
            supplier_ids=["s1", "s2"],
            central_table=table,
        )
        assert result == []

    def test_empty_result_when_no_suppliers(self):
        """Return empty list when supplier_ids is empty."""
        pid = "P0000"
        table = CentralTable()
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=10,
            supplier_ids=[],
            central_table=table,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 2. Min-order discipline — both layers reject sub-min lines
# ---------------------------------------------------------------------------


class TestMinOrderDiscipline:
    def test_supplier_min_order_rejects_small_request(self):
        """A request below the supplier-imposed min_order is rejected for that supplier."""
        pid = "P0000"
        table = _make_table({
            ("strict", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 50},
            ("flexible", pid): {"available_qty": 100, "list_price": 2.0, "min_order": 0},
        })
        # Request only 10 units — below strict supplier's min_order of 50.
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=10,
            supplier_ids=["strict", "flexible"],
            central_table=table,
        )
        # "strict" min_order=50 > qty_total=10, so it's skipped.
        # "flexible" min_order=0, so it can fill.
        assert result == [("flexible", 10)]

    def test_supplier_min_order_honoured_when_qty_above_threshold(self):
        """When qty_total meets the supplier min_order, the supplier IS used."""
        pid = "P0000"
        table = _make_table({
            ("strict", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 10},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=10,  # exactly at the min_order threshold
            supplier_ids=["strict"],
            central_table=table,
        )
        assert result == [("strict", 10)]

    def test_all_suppliers_below_min_order_returns_empty(self):
        """When all suppliers have min_order > qty_total, return empty."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 100},
            ("s2", pid): {"available_qty": 100, "list_price": 2.0, "min_order": 200},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=50,  # below both min_orders
            supplier_ids=["s1", "s2"],
            central_table=table,
        )
        assert result == []

    def test_buyer_side_min_order_rejects_small_lines(self):
        """Buyer-side min_order_floor on MultiSupplierTextbookPolicy rejects sub-min lines."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 5, "list_price": 1.0, "min_order": 0},
            ("s2", pid): {"available_qty": 100, "list_price": 2.0, "min_order": 0},
        })
        # s1 can only fill 5 units; if buyer-side min_order_floor=10,
        # the 5-unit partial should be rejected (below buyer floor).
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=40,
            supplier_ids=["s1", "s2"],
            central_table=table,
            buyer_min_order_floor=10,
        )
        # s1 would contribute only 5 (below floor of 10), so it's skipped.
        # s2 can fill the full 40.
        assert result == [("s2", 40)]


# ---------------------------------------------------------------------------
# 3. Partial-fill arithmetic when no single supplier covers the full qty
# ---------------------------------------------------------------------------


class TestPartialFillArithmetic:
    def test_splits_across_two_suppliers_when_first_short(self):
        """When the cheapest supplier is partially stocked, split across two."""
        pid = "P0000"
        table = _make_table({
            ("cheap", pid): {"available_qty": 20, "list_price": 1.0, "min_order": 0},
            ("expensive", pid): {"available_qty": 100, "list_price": 5.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=50,
            supplier_ids=["cheap", "expensive"],
            central_table=table,
        )
        # cheap fills 20, expensive fills remaining 30.
        assert ("cheap", 20) in result
        assert ("expensive", 30) in result
        assert sum(qty for _, qty in result) == 50

    def test_total_fill_capped_at_available_when_all_short(self):
        """When total available < qty_total, fill as much as possible."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 10, "list_price": 1.0, "min_order": 0},
            ("s2", pid): {"available_qty": 15, "list_price": 2.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=100,  # more than total available (25)
            supplier_ids=["s1", "s2"],
            central_table=table,
        )
        total = sum(qty for _, qty in result)
        assert total == 25  # only 25 available across both suppliers

    def test_splits_across_three_suppliers(self):
        """Splits correctly across three suppliers, cheapest-first."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 10, "list_price": 1.0, "min_order": 0},
            ("s2", pid): {"available_qty": 15, "list_price": 2.0, "min_order": 0},
            ("s3", pid): {"available_qty": 20, "list_price": 3.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=30,
            supplier_ids=["s1", "s2", "s3"],
            central_table=table,
        )
        total = sum(qty for _, qty in result)
        assert total == 30
        # Order should be cheapest-first.
        supplier_order = [sid for sid, _ in result]
        assert supplier_order == sorted(supplier_order, key=lambda s: {"s1": 1, "s2": 2, "s3": 3}[s])

    def test_zero_qty_total_returns_empty(self):
        """A zero request returns an empty list."""
        pid = "P0000"
        table = _make_table({
            ("s1", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=0,
            supplier_ids=["s1"],
            central_table=table,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 4. Pluggable routing_strategy
# ---------------------------------------------------------------------------


class TestPluggableRoutingStrategy:
    def test_custom_strategy_invoked_when_supplied(self):
        """When routing_strategy is provided, it is called instead of default cheapest-first."""
        pid = "P0000"
        calls: list[dict] = []

        def recording_strategy(
            pid: str,
            qty_total: int,
            supplier_ids: list[str],
            central_table: CentralTable,
            **kwargs,
        ) -> list[tuple[str, int]]:
            calls.append({"pid": pid, "qty_total": qty_total, "supplier_ids": supplier_ids})
            # Always route to the last supplier in the list.
            last = supplier_ids[-1]
            offers = {sid: o for sid, o in central_table.snapshot_for_buyer(pid)}
            offer = offers.get(last)
            if offer is None or offer.available_qty == 0:
                return []
            qty = min(qty_total, offer.available_qty)
            return [(last, qty)]

        table = _make_table({
            ("s1", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 0},
            ("s2", pid): {"available_qty": 100, "list_price": 2.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=25,
            supplier_ids=["s1", "s2"],
            central_table=table,
            routing_strategy=recording_strategy,
        )
        # Strategy routes to last supplier (s2).
        assert result == [("s2", 25)]
        # Strategy was actually called.
        assert len(calls) == 1
        assert calls[0]["pid"] == pid
        assert calls[0]["qty_total"] == 25

    def test_fill_rate_weighted_strategy_routes_to_most_reliable(self):
        """A fill-rate-weighted strategy can be used as an alternative."""
        pid = "P0000"

        def fill_rate_weighted(
            pid: str,
            qty_total: int,
            supplier_ids: list[str],
            central_table: CentralTable,
            **kwargs,
        ) -> list[tuple[str, int]]:
            """Route to the supplier with the highest fill_rate_recent."""
            offers = [
                (sid, o)
                for sid, o in central_table.snapshot_for_buyer(pid)
                if sid in supplier_ids and o.available_qty > 0
            ]
            if not offers:
                return []
            # Sort by fill_rate descending (most reliable first).
            offers.sort(key=lambda t: t[1].fill_rate_recent, reverse=True)
            result: list[tuple[str, int]] = []
            remaining = qty_total
            for sid, offer in offers:
                if remaining <= 0:
                    break
                qty = min(remaining, offer.available_qty)
                result.append((sid, qty))
                remaining -= qty
            return result

        # Build a table where s2 has higher fill_rate_recent.
        table = CentralTable()
        # Publish initial offer for s1 (cheap but unreliable — low fill rate).
        table.publish("s1", pid, Offer(available_qty=100, list_price=1.0, min_order=0))
        # Simulate partial fill on s1 to drive down its fill_rate.
        for _ in range(5):
            table.publish("s1", pid, Offer(available_qty=100, list_price=1.0, min_order=0))
            table.commit("s1", pid, 10)  # only fills 10 of 100 → 10% fill rate
        # Reset s1 for current tick.
        table.publish("s1", pid, Offer(available_qty=100, list_price=1.0, min_order=0))

        # s2: always fully filled → fill_rate stays high.
        table.publish("s2", pid, Offer(available_qty=100, list_price=2.0, min_order=0))

        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=30,
            supplier_ids=["s1", "s2"],
            central_table=table,
            routing_strategy=fill_rate_weighted,
        )
        # Strategy should route to s2 (higher fill_rate) first.
        assert result[0][0] == "s2"

    def test_default_strategy_used_when_routing_strategy_is_none(self):
        """When routing_strategy=None, default cheapest-first is used."""
        pid = "P0000"
        table = _make_table({
            ("cheap", pid): {"available_qty": 100, "list_price": 1.0, "min_order": 0},
            ("pricey", pid): {"available_qty": 100, "list_price": 10.0, "min_order": 0},
        })
        result = MultiSupplierTextbookPolicy._split_across_suppliers(
            pid=pid,
            qty_total=20,
            supplier_ids=["cheap", "pricey"],
            central_table=table,
            routing_strategy=None,
        )
        # Cheapest first by default.
        assert result == [("cheap", 20)]
