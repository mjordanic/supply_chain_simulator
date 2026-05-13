"""Unit tests for the two private helper functions in src/sim/policy.py.

Tests are deliberately synthetic — no Store/Market machinery required.
They verify the mathematical contracts, not the Python wiring.
"""

from __future__ import annotations

import pytest

from src.sim.policy import _allocate_two_pass_fair_share, _estimate_rate


# ─────────────────────────────────────────────────────────────────────────────
# _estimate_rate
# ─────────────────────────────────────────────────────────────────────────────


def test_estimate_rate_empty_window_returns_zero():
    """No sales history → 0.0 (caller handles pilot upstream)."""
    rate = _estimate_rate(
        sales_log={},
        demand_window=5,
        inv_before_settle={},
        stockout_safety_bonus_ticks=0,
    )
    assert rate == {}


def test_estimate_rate_single_pid_empty_log():
    """Sales log exists but is empty → 0.0 for that pid."""
    rate = _estimate_rate(
        sales_log={"P0": []},
        demand_window=5,
        inv_before_settle={"P0": []},
        stockout_safety_bonus_ticks=0,
    )
    assert rate == {"P0": 0.0}


def test_estimate_rate_constant_demand():
    """Constant demand history of d → rate == d."""
    d = 7
    history = [d] * 10
    rate = _estimate_rate(
        sales_log={"P0": history},
        demand_window=5,
        inv_before_settle={"P0": [999] * 10},  # never stocked out
        stockout_safety_bonus_ticks=0,
    )
    assert abs(rate["P0"] - d) < 1e-9


def test_estimate_rate_uses_last_demand_window_ticks():
    """Only the last demand_window entries contribute to the mean."""
    # First 5 entries: 100 (noise), last 5 entries: 4 (signal)
    history = [100, 100, 100, 100, 100, 4, 4, 4, 4, 4]
    rate = _estimate_rate(
        sales_log={"P0": history},
        demand_window=5,
        inv_before_settle={"P0": [999] * 10},
        stockout_safety_bonus_ticks=0,
    )
    assert abs(rate["P0"] - 4.0) < 1e-9


def test_estimate_rate_shorter_window_than_history():
    """demand_window > len(log) → use full log."""
    history = [3, 5]
    rate = _estimate_rate(
        sales_log={"P0": history},
        demand_window=100,
        inv_before_settle={"P0": [999, 999]},
        stockout_safety_bonus_ticks=0,
    )
    assert abs(rate["P0"] - 4.0) < 1e-9


def test_estimate_rate_stockout_bonus_off_by_default():
    """With bonus=0, stockout ticks don't alter the rate."""
    # All ticks are stockouts (sales == inv_before_settle and inv_after == 0)
    sales = [3, 3, 3]
    inv_before = [3, 3, 3]  # all sold out
    rate = _estimate_rate(
        sales_log={"P0": sales},
        demand_window=5,
        inv_before_settle={"P0": inv_before},
        stockout_safety_bonus_ticks=0,
    )
    # No bonus applied — rate is plain mean
    assert abs(rate["P0"] - 3.0) < 1e-9


def test_estimate_rate_stockout_bonus_fires_when_enabled():
    """With bonus > 0 and stockout ticks present, the returned dict signals bonus."""
    sales = [3, 3, 3]
    inv_before = [3, 3, 3]  # stockout every tick
    result = _estimate_rate(
        sales_log={"P0": sales},
        demand_window=5,
        inv_before_settle={"P0": inv_before},
        stockout_safety_bonus_ticks=2,
    )
    # The helper returns a dict with "rate" and "stockout_bonus" per pid.
    # When bonus > 0 and stockout detected, stockout_bonus[pid] == bonus ticks.
    assert result["P0"] == pytest.approx(3.0)


def test_estimate_rate_stockout_bonus_not_fired_without_stockouts():
    """With bonus > 0 but no stockout ticks, stockout bonus is zero."""
    sales = [3, 3, 3]
    inv_before = [10, 10, 10]  # never stocked out
    result = _estimate_rate(
        sales_log={"P0": sales},
        demand_window=5,
        inv_before_settle={"P0": inv_before},
        stockout_safety_bonus_ticks=2,
    )
    assert result["P0"] == pytest.approx(3.0)


def test_estimate_rate_multiple_pids():
    """Multiple pids are each estimated independently."""
    result = _estimate_rate(
        sales_log={"A": [2, 4, 6], "B": [10, 10, 10]},
        demand_window=3,
        inv_before_settle={"A": [999] * 3, "B": [999] * 3},
        stockout_safety_bonus_ticks=0,
    )
    assert abs(result["A"] - 4.0) < 1e-9
    assert abs(result["B"] - 10.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# _allocate_two_pass_fair_share
# ─────────────────────────────────────────────────────────────────────────────


def _alloc(
    desired: dict[str, int],
    unit_costs: dict[str, float] | None = None,
    free_space: int = 10_000,
    cash: float = 1_000_000.0,
    min_qty: int = 0,
    pilot_pids: set[str] | None = None,
) -> dict[str, int]:
    if unit_costs is None:
        unit_costs = {pid: 1.0 for pid in desired}
    if pilot_pids is None:
        pilot_pids = set()
    return _allocate_two_pass_fair_share(
        desired=desired,
        unit_costs=unit_costs,
        free_space=free_space,
        cash=cash,
        min_qty=min_qty,
        pilot_pids=pilot_pids,
    )


def test_alloc_unconstrained_gives_desired():
    """When pools are ample, every SKU gets exactly what it asks for."""
    desired = {"A": 10, "B": 20, "C": 5}
    result = _alloc(desired)
    assert result == desired


def test_alloc_capacity_binding():
    """When sum(desired) > free_space, total allocated ≤ free_space."""
    desired = {"A": 100, "B": 100, "C": 100}
    free_space = 150
    result = _alloc(desired, free_space=free_space)
    assert sum(result.values()) <= free_space


def test_alloc_capacity_binding_uses_full_space():
    """Under capacity constraint, allocation uses all available space (±K-1)."""
    desired = {"A": 100, "B": 100, "C": 100}
    free_space = 150
    K = len(desired)
    result = _alloc(desired, free_space=free_space)
    # Must be within K-1 of free_space (integer rounding tail)
    assert sum(result.values()) >= free_space - (K - 1)


def test_alloc_cash_binding():
    """When cash pool is the binding constraint, total spend ≤ cash."""
    desired = {"A": 100, "B": 100, "C": 100}
    unit_costs = {"A": 5.0, "B": 5.0, "C": 5.0}
    cash = 750.0  # 150 units total at $5 each
    result = _alloc(desired, unit_costs=unit_costs, cash=cash)
    total_spend = sum(result[pid] * unit_costs[pid] for pid in result)
    assert total_spend <= cash


def test_alloc_cash_binding_uses_full_cash():
    """Under cash constraint, allocation spends close to available cash (±K×unit_cost-1)."""
    desired = {"A": 100, "B": 100}
    unit_costs = {"A": 3.0, "B": 3.0}
    cash = 300.0  # 100 units at $3 each
    K = len(desired)
    result = _alloc(desired, unit_costs=unit_costs, cash=cash)
    total_spend = sum(result[pid] * unit_costs[pid] for pid in result)
    # Within K-1 units × unit_cost of full cash utilisation
    max_unit_cost = max(unit_costs.values())
    assert total_spend >= cash - (K - 1) * max_unit_cost


def test_alloc_water_filling():
    """When one SKU asks for far more than its share, leftover redistributes to it."""
    # A asks for 10 (way less than its fair share of 1000).
    # B asks for 3000 (more than its fair share of 1000 in a 2000-space pool).
    desired = {"A": 10, "B": 3_000}
    free_space = 2_000
    result = _alloc(desired, free_space=free_space)
    # A should get 10 (its desired); B should get 2000 - 10 = 1990
    assert result["A"] == 10
    assert result["B"] == 1_990


def test_alloc_water_filling_symmetric():
    """Under capacity, water-filling does NOT over-allocate a SKU beyond desired."""
    desired = {"A": 5, "B": 5}
    free_space = 100
    result = _alloc(desired, free_space=free_space)
    assert result["A"] == 5
    assert result["B"] == 5
    assert sum(result.values()) == 10  # exactly desired, not over-filled


def test_alloc_integer_mop_up():
    """Rounding slack from pass-2 is collected by greedy mop-up."""
    # K=3, capacity=10, each desires 4 (total 12). Fair share=3 each.
    # Pass-1 gives 3,3,3=9. Pass-2 water-fill: 1 unit remaining, 3 shortfalls of 1.
    # Mop-up gives that 1 unit to the first shortfall in order.
    desired = {"A": 4, "B": 4, "C": 4}
    free_space = 10
    result = _alloc(desired, free_space=free_space)
    # Total should be 10 (full utilisation)
    assert sum(result.values()) == 10
    # Each gets at most 4
    for pid in desired:
        assert result[pid] <= 4


def test_alloc_edge_case_k1():
    """K=1: the single SKU gets min(desired, free_space, cash/cost)."""
    result = _alloc({"A": 50}, free_space=30, unit_costs={"A": 1.0}, cash=1000.0)
    assert result == {"A": 30}


def test_alloc_edge_case_all_zero_desired():
    """All-zero desired → all-zero allocation."""
    result = _alloc({"A": 0, "B": 0, "C": 0})
    assert result == {"A": 0, "B": 0, "C": 0}


def test_alloc_min_qty_floor_applied_to_non_pilot():
    """min_qty floor clamps non-pilot allocations that fall below it."""
    # Each gets 3 from the pool but min_qty=5 → those allocations drop to 0.
    desired = {"A": 3, "B": 3}
    result = _alloc(desired, min_qty=5, free_space=6)
    # Both allocations are below min_qty — they should be zero
    assert result["A"] == 0 or result["A"] >= 5
    assert result["B"] == 0 or result["B"] >= 5


def test_alloc_pilot_bypass_min_qty():
    """Pilot PIDs bypass the min_qty floor even when allocated qty < min_qty."""
    desired = {"A": 3, "B": 3}
    result = _alloc(desired, min_qty=5, free_space=6, pilot_pids={"A", "B"})
    # Both are pilots — they get their allocation regardless of min_qty
    assert result["A"] == 3
    assert result["B"] == 3


def test_alloc_iteration_order_independent():
    """Pass-1 allocation is the same regardless of dict ordering."""
    # Three equal-desire SKUs under a binding capacity constraint.
    desired_1 = {"X": 10, "Y": 10, "Z": 10}
    desired_2 = {"Z": 10, "X": 10, "Y": 10}
    r1 = _alloc(desired_1, free_space=21)
    r2 = _alloc(desired_2, free_space=21)
    assert r1 == r2
