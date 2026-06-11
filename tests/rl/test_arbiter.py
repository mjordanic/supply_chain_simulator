"""Tests for src/rl/arbiter.py.

Covers the deterministic Arbiter module's two modes:
  - proportional fair-share (default)
  - priority greedy

All assertions are on externally observable behaviour (feasibility invariants,
ratio preservation, priority ordering, integer outputs, edge cases) — not
internal implementation details.

K-free: the same code path must work for K = 1 and K = 32.
"""

from __future__ import annotations

import pytest

from src.rl.arbiter import allocate, arbitrate_orders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_prices(pids: list[str], price: float = 1.0) -> dict[str, float]:
    return {pid: price for pid in pids}


def _uniform_headroom(pids: list[str], headroom: int = 1_000) -> dict[str, int]:
    return {pid: headroom for pid in pids}


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_proposed_returns_empty():
    result = allocate({}, {}, 100, 1000.0, {})
    assert result == {}


def test_zero_budget_returns_all_zero():
    pids = ["A", "B"]
    result = allocate(
        proposed={"A": 10.0, "B": 5.0},
        per_sku_headroom=_uniform_headroom(pids),
        global_free_space=100,
        cash_budget=0.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode="proportional",
    )
    assert result == {"A": 0, "B": 0}


def test_zero_global_space_returns_all_zero_proportional():
    pids = ["A", "B"]
    result = allocate(
        proposed={"A": 10.0, "B": 5.0},
        per_sku_headroom=_uniform_headroom(pids),
        global_free_space=0,
        cash_budget=10_000.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode="proportional",
    )
    assert result == {"A": 0, "B": 0}


def test_zero_global_space_returns_all_zero_greedy():
    pids = ["A", "B"]
    result = allocate(
        proposed={"A": 10.0, "B": 5.0},
        per_sku_headroom=_uniform_headroom(pids),
        global_free_space=0,
        cash_budget=10_000.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode="greedy",
    )
    assert result == {"A": 0, "B": 0}


def test_zero_proposals_return_zero_proportional():
    pids = ["A", "B"]
    result = allocate(
        proposed={"A": 0.0, "B": 0.0},
        per_sku_headroom=_uniform_headroom(pids),
        global_free_space=100,
        cash_budget=1000.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode="proportional",
    )
    assert result == {"A": 0, "B": 0}


def test_zero_proposals_return_zero_greedy():
    pids = ["A", "B"]
    result = allocate(
        proposed={"A": 0.0, "B": 0.0},
        per_sku_headroom=_uniform_headroom(pids),
        global_free_space=100,
        cash_budget=1000.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode="greedy",
    )
    assert result == {"A": 0, "B": 0}


# ---------------------------------------------------------------------------
# Feasibility invariants (both modes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["proportional", "greedy"])
def test_allocations_are_integer_outputs(mode):
    pids = ["A", "B", "C"]
    result = allocate(
        proposed={"A": 7.5, "B": 3.3, "C": 11.1},
        per_sku_headroom=_uniform_headroom(pids, 20),
        global_free_space=15,
        cash_budget=200.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode=mode,
    )
    for pid in pids:
        assert isinstance(result[pid], int), f"{pid}: expected int, got {type(result[pid])}"


@pytest.mark.parametrize("mode", ["proportional", "greedy"])
def test_allocations_never_exceed_per_sku_headroom(mode):
    """Allocations must not exceed each product's per-SKU headroom."""
    per_sku = {"A": 5, "B": 10, "C": 3}
    result = allocate(
        proposed={"A": 100.0, "B": 100.0, "C": 100.0},
        per_sku_headroom=per_sku,
        global_free_space=1000,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["A", "B", "C"], 1.0),
        mode=mode,
    )
    for pid, cap in per_sku.items():
        assert result[pid] <= cap, f"{pid}: allocated {result[pid]} > headroom {cap}"


@pytest.mark.parametrize("mode", ["proportional", "greedy"])
def test_allocations_never_exceed_global_free_space(mode):
    """Sum of all allocations must not exceed global_free_space."""
    pids = ["A", "B", "C"]
    free = 20
    result = allocate(
        proposed={"A": 50.0, "B": 50.0, "C": 50.0},
        per_sku_headroom=_uniform_headroom(pids, 1000),
        global_free_space=free,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(pids, 1.0),
        mode=mode,
    )
    total = sum(result.values())
    assert total <= free, f"total allocated {total} > global_free_space {free}"


@pytest.mark.parametrize("mode", ["proportional", "greedy"])
def test_allocations_never_exceed_cash_budget(mode):
    """Total cash cost of allocations must not exceed cash_budget."""
    prices = {"A": 3.0, "B": 5.0, "C": 2.0}
    budget = 30.0
    result = allocate(
        proposed={"A": 100.0, "B": 100.0, "C": 100.0},
        per_sku_headroom=_uniform_headroom(["A", "B", "C"], 1000),
        global_free_space=10_000,
        cash_budget=budget,
        unit_prices=prices,
        mode=mode,
    )
    total_cost = sum(result[pid] * prices[pid] for pid in ["A", "B", "C"])
    assert total_cost <= budget + 1e-9, (
        f"total cost {total_cost:.4f} exceeds budget {budget}"
    )


# ---------------------------------------------------------------------------
# Proportional: ratio preservation under binding constraint
# ---------------------------------------------------------------------------


def test_proportional_preserves_ratios_under_global_space_constraint():
    """Under a binding global-space constraint, proportional mode scales all
    proposals by the same factor, so proposal ratios are preserved."""
    # Proposals: A=10, B=20, C=30 — but only 30 units of space available.
    # After per-SKU cap (headroom = 1000, no binding): capped = {A:10, B:20, C:30}
    # sum = 60, free = 30 → scale = 0.5
    # Expected: A≈5, B≈10, C≈15 (integer truncation applies)
    result = allocate(
        proposed={"A": 10.0, "B": 20.0, "C": 30.0},
        per_sku_headroom=_uniform_headroom(["A", "B", "C"], 1000),
        global_free_space=30,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["A", "B", "C"], 0.0),  # no cash constraint
        mode="proportional",
    )
    # Ratios should be approximately 1:2:3.
    # Allow rounding slack of ±1.
    assert result["A"] >= 4 and result["A"] <= 6
    assert result["B"] >= 9 and result["B"] <= 11
    assert result["C"] >= 14 and result["C"] <= 16
    # Check ratio B/A ≈ 2, C/A ≈ 3 approximately.
    if result["A"] > 0:
        assert abs(result["B"] / result["A"] - 2.0) < 0.5
        assert abs(result["C"] / result["A"] - 3.0) < 0.5


def test_proportional_preserves_ratios_under_cash_constraint():
    """Under a binding cash constraint, proportional mode scales all proposals
    proportionally, preserving ratios."""
    # Proposals: A=10, B=10 at price 5.0 each → cost 100. Budget = 50.
    # Space is non-binding. Scale by 0.5 → A=5, B=5.
    result = allocate(
        proposed={"A": 10.0, "B": 10.0},
        per_sku_headroom=_uniform_headroom(["A", "B"], 1000),
        global_free_space=10_000,
        cash_budget=50.0,
        unit_prices={"A": 5.0, "B": 5.0},
        mode="proportional",
    )
    assert result["A"] == result["B"]  # equal proposals → equal allocations
    assert result["A"] <= 5
    total_cost = result["A"] * 5.0 + result["B"] * 5.0
    assert total_cost <= 50.0 + 1e-9


def test_proportional_no_constraint_gets_full_proposal():
    """When no resource is binding, proportional mode allocates the full
    proposed quantity (truncated to int)."""
    result = allocate(
        proposed={"A": 7.0, "B": 3.0},
        per_sku_headroom={"A": 100, "B": 100},
        global_free_space=1000,
        cash_budget=10_000.0,
        unit_prices={"A": 1.0, "B": 1.0},
        mode="proportional",
    )
    assert result["A"] == 7
    assert result["B"] == 3


# ---------------------------------------------------------------------------
# Greedy: priority ordering
# ---------------------------------------------------------------------------


def test_greedy_fills_highest_priority_first():
    """Greedy mode must exhaust resources starting with the highest-priority
    product. The lowest-priority product is starved first."""
    # Global space = 10, proposals = {A:10, B:10}, prices=0 (no cash constraint).
    # A has priority 2.0, B has priority 1.0 → A should be filled first.
    result = allocate(
        proposed={"A": 10.0, "B": 10.0},
        per_sku_headroom=_uniform_headroom(["A", "B"], 1000),
        global_free_space=10,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["A", "B"], 0.0),
        priorities={"A": 2.0, "B": 1.0},
        mode="greedy",
    )
    assert result["A"] == 10  # fully filled
    assert result["B"] == 0   # starved — no space left


def test_greedy_starves_lowest_priority_on_space():
    """When space runs out, greedy mode starves the lowest-priority product."""
    # Space = 15. Three products each proposing 10.
    # Priorities: X=3, Y=2, Z=1. Fill X(10), then Y(5 remaining), Z(0).
    result = allocate(
        proposed={"X": 10.0, "Y": 10.0, "Z": 10.0},
        per_sku_headroom=_uniform_headroom(["X", "Y", "Z"], 1000),
        global_free_space=15,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["X", "Y", "Z"], 0.0),
        priorities={"X": 3.0, "Y": 2.0, "Z": 1.0},
        mode="greedy",
    )
    assert result["X"] == 10
    assert result["Y"] == 5
    assert result["Z"] == 0


def test_greedy_starves_lowest_priority_on_cash():
    """When cash budget runs out, greedy mode starves the lowest-priority product."""
    # Each unit costs 3.0. Budget = 15.0 → 5 units total.
    # Priorities: A=2, B=1. Fill A(5), B starved.
    result = allocate(
        proposed={"A": 10.0, "B": 10.0},
        per_sku_headroom=_uniform_headroom(["A", "B"], 1000),
        global_free_space=10_000,
        cash_budget=15.0,
        unit_prices={"A": 3.0, "B": 3.0},
        priorities={"A": 2.0, "B": 1.0},
        mode="greedy",
    )
    assert result["A"] == 5
    assert result["B"] == 0


def test_greedy_per_sku_headroom_limits_individual_products():
    """Greedy mode respects per-SKU headroom for each product individually,
    even when the high-priority product has a low headroom."""
    # A has headroom 3, B has headroom 20. Space = 100. Priority A > B.
    # A should get 3 (headroom cap), then B gets its full proposal.
    result = allocate(
        proposed={"A": 50.0, "B": 10.0},
        per_sku_headroom={"A": 3, "B": 20},
        global_free_space=100,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["A", "B"], 0.0),
        priorities={"A": 5.0, "B": 1.0},
        mode="greedy",
    )
    assert result["A"] == 3   # capped by headroom
    assert result["B"] == 10  # gets its full proposal


# ---------------------------------------------------------------------------
# K-free: identical code path for K=1 and K=32
# ---------------------------------------------------------------------------


def test_k_equals_1_proportional():
    result = allocate(
        proposed={"only": 20.0},
        per_sku_headroom={"only": 100},
        global_free_space=50,
        cash_budget=500.0,
        unit_prices={"only": 2.0},
        mode="proportional",
    )
    assert isinstance(result["only"], int)
    assert result["only"] <= 50  # global space
    assert result["only"] * 2.0 <= 500.0 + 1e-9  # cash


def test_k_equals_1_greedy():
    result = allocate(
        proposed={"only": 20.0},
        per_sku_headroom={"only": 100},
        global_free_space=50,
        cash_budget=500.0,
        unit_prices={"only": 2.0},
        priorities={"only": 1.0},
        mode="greedy",
    )
    assert isinstance(result["only"], int)
    assert result["only"] <= 50  # global space


def test_k_equals_32_proportional():
    """32-product proportional allocation satisfies all feasibility constraints."""
    pids = [f"P{i:02d}" for i in range(32)]
    proposed = {pid: float(10 + i) for i, pid in enumerate(pids)}
    per_sku = {pid: 20 for pid in pids}
    prices = {pid: 1.5 for pid in pids}

    result = allocate(
        proposed=proposed,
        per_sku_headroom=per_sku,
        global_free_space=100,
        cash_budget=200.0,
        unit_prices=prices,
        mode="proportional",
    )
    assert len(result) == 32
    assert all(isinstance(v, int) for v in result.values())
    assert sum(result.values()) <= 100
    total_cost = sum(result[pid] * prices[pid] for pid in pids)
    assert total_cost <= 200.0 + 1e-6
    for pid in pids:
        assert result[pid] <= per_sku[pid]


def test_k_equals_32_greedy():
    """32-product greedy allocation satisfies all feasibility constraints."""
    pids = [f"P{i:02d}" for i in range(32)]
    proposed = {pid: 10.0 for pid in pids}
    per_sku = {pid: 15 for pid in pids}
    prices = {pid: 2.0 for pid in pids}
    priorities = {pid: float(31 - i) for i, pid in enumerate(pids)}  # P00 highest

    result = allocate(
        proposed=proposed,
        per_sku_headroom=per_sku,
        global_free_space=80,
        cash_budget=200.0,
        unit_prices=prices,
        priorities=priorities,
        mode="greedy",
    )
    assert len(result) == 32
    assert all(isinstance(v, int) for v in result.values())
    assert sum(result.values()) <= 80
    total_cost = sum(result[pid] * prices[pid] for pid in pids)
    assert total_cost <= 200.0 + 1e-6
    for pid in pids:
        assert result[pid] <= per_sku[pid]


# ---------------------------------------------------------------------------
# Zero-price products (no cash constraint for that product)
# ---------------------------------------------------------------------------


def test_zero_price_products_not_cash_constrained_proportional():
    """Products with unit_price=0 impose no cash cost and should be fully
    served up to capacity/space constraints."""
    result = allocate(
        proposed={"free": 50.0},
        per_sku_headroom={"free": 100},
        global_free_space=50,
        cash_budget=0.01,  # tiny budget, but price=0 so no constraint
        unit_prices={"free": 0.0},
        mode="proportional",
    )
    assert result["free"] == 50


def test_zero_price_products_not_cash_constrained_greedy():
    result = allocate(
        proposed={"free": 50.0},
        per_sku_headroom={"free": 100},
        global_free_space=50,
        cash_budget=0.01,
        unit_prices={"free": 0.0},
        priorities={"free": 1.0},
        mode="greedy",
    )
    assert result["free"] == 50


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        allocate(
            proposed={"A": 10.0},
            per_sku_headroom={"A": 100},
            global_free_space=100,
            cash_budget=100.0,
            unit_prices={"A": 1.0},
            mode="invalid",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Greedy: determinism (tie-breaking by pid)
# ---------------------------------------------------------------------------


def test_greedy_deterministic_on_equal_priority():
    """When two products have the same priority, the result is deterministic
    (tie broken alphabetically by pid)."""
    result1 = allocate(
        proposed={"A": 10.0, "B": 10.0},
        per_sku_headroom=_uniform_headroom(["A", "B"], 1000),
        global_free_space=10,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["A", "B"], 0.0),
        priorities={"A": 1.0, "B": 1.0},  # tie
        mode="greedy",
    )
    result2 = allocate(
        proposed={"B": 10.0, "A": 10.0},  # reversed dict insertion order
        per_sku_headroom=_uniform_headroom(["B", "A"], 1000),
        global_free_space=10,
        cash_budget=100_000.0,
        unit_prices=_uniform_prices(["B", "A"], 0.0),
        priorities={"B": 1.0, "A": 1.0},  # tie
        mode="greedy",
    )
    assert result1 == result2  # deterministic


# ---------------------------------------------------------------------------
# arbitrate_orders — shared helper
# ---------------------------------------------------------------------------

class _FakeNode:
    """Minimal node stand-in for arbitrate_orders tests."""

    def __init__(
        self,
        inventory: dict,
        pending: dict,
        capacity: float,
        cash: float,
        costs: dict | None = None,
        list_prices: dict | None = None,
    ) -> None:
        self.inventory = inventory
        self.pending = pending
        self.capacity = capacity
        self.cash = cash
        self.costs = costs or {}
        self.list_prices = list_prices or {}


class _FakeConfig:
    arbiter_mode: str = "proportional"
    cash_budget_fraction: float = 1.0


def _make_raw_orders(pids: list[str], qty: float = 10.0) -> dict:
    """Build a raw_orders dict as decode_set_action would produce it."""
    return {pid: [(f"F_{pid}", qty)] for pid in pids}


def test_arbitrate_orders_returns_order_dict_structure():
    """arbitrate_orders returns a dict mapping pid → [(supplier, qty)] list."""
    pids = ["A", "B"]
    node = _FakeNode(
        inventory={"A": 5, "B": 3},
        pending={},
        capacity=1000.0,
        cash=500.0,
    )
    config = _FakeConfig()
    unit_prices = {"A": 1.0, "B": 1.0}
    priorities = {"A": 1.0, "B": 1.0}
    raw_orders = _make_raw_orders(pids, qty=10.0)

    result = arbitrate_orders(
        raw_orders=raw_orders,
        active_subset=tuple(pids),
        node=node,
        unit_prices=unit_prices,
        priorities=priorities,
        config=config,
    )
    # Must return a dict with the same keys as active_subset.
    assert set(result.keys()) == set(pids)
    # Each value is a list (possibly empty if qty==0).
    for pid in pids:
        assert isinstance(result[pid], list)


def test_arbitrate_orders_respects_cash_budget():
    """When cash is tiny, arbitrate_orders returns zero-qty (empty list) for all."""
    pids = ["A", "B"]
    # cash=0 forces cash_budget=0 → no allocations
    node = _FakeNode(
        inventory={},
        pending={},
        capacity=1000.0,
        cash=0.0,
    )
    config = _FakeConfig()
    unit_prices = {"A": 1.0, "B": 1.0}
    priorities = {"A": 1.0, "B": 1.0}
    raw_orders = _make_raw_orders(pids, qty=10.0)

    result = arbitrate_orders(
        raw_orders=raw_orders,
        active_subset=tuple(pids),
        node=node,
        unit_prices=unit_prices,
        priorities=priorities,
        config=config,
    )
    for pid in pids:
        assert result[pid] == [], f"Expected empty list for {pid} when cash=0"


def test_arbitrate_orders_supplier_prefix_matches():
    """Supplier in the returned order line uses F_<pid> convention."""
    pids = ["X"]
    node = _FakeNode(inventory={}, pending={}, capacity=1000.0, cash=10_000.0)
    config = _FakeConfig()
    raw_orders = _make_raw_orders(pids, qty=5.0)
    result = arbitrate_orders(
        raw_orders=raw_orders,
        active_subset=tuple(pids),
        node=node,
        unit_prices={"X": 1.0},
        priorities={"X": 1.0},
        config=config,
    )
    # Should have one line with supplier "F_X" and qty > 0.
    assert len(result["X"]) == 1
    supplier, qty = result["X"][0]
    assert supplier == "F_X"
    assert qty > 0


def test_arbitrate_orders_consistent_with_direct_allocate():
    """arbitrate_orders result must equal calling allocate() + building order_dict by hand."""
    pids = ["A", "B", "C"]
    inventory = {"A": 10, "B": 5, "C": 0}
    capacity = 200.0
    cash = 300.0
    unit_prices = {"A": 2.0, "B": 3.0, "C": 1.0}
    priorities = {"A": 1.0, "B": 2.0, "C": 3.0}
    qty = 20.0
    raw_orders = _make_raw_orders(pids, qty=qty)

    node = _FakeNode(inventory=inventory, pending={}, capacity=capacity, cash=cash)
    config = _FakeConfig()

    result = arbitrate_orders(
        raw_orders=raw_orders,
        active_subset=tuple(pids),
        node=node,
        unit_prices=unit_prices,
        priorities=priorities,
        config=config,
    )

    # Compute expected via allocate() directly.
    proposed = {pid: qty for pid in pids}
    total_inv = sum(float(v) for v in inventory.values())
    global_free_space = max(0, int(capacity - total_inv))
    per_sku_headroom = {pid: max(0, int(capacity - float(inventory.get(pid, 0)))) for pid in pids}
    cash_budget = cash * config.cash_budget_fraction
    expected_alloc = allocate(
        proposed=proposed,
        per_sku_headroom=per_sku_headroom,
        global_free_space=global_free_space,
        cash_budget=cash_budget,
        unit_prices=unit_prices,
        priorities=priorities,
        mode=config.arbiter_mode,
    )
    expected = {
        pid: [(f"F_{pid}", expected_alloc[pid])] if expected_alloc[pid] > 0 else []
        for pid in pids
    }
    assert result == expected
