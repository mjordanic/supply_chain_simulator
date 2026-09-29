"""Tests for src/tuning/search_spaces.py.

Each bundled trial-callback factory is exercised with:
  - A FixedTrial at representative mid-range values → construction succeeds
    and the sampled kwargs are propagated correctly.
  - Boundary sweeps → construction succeeds at every range boundary so that
    silent range-narrowing in the factory is caught immediately.

Routing strategy and ``per_supplier_min_order_floor`` are pinned to the
same values the replenishment planner uses. A trial dict that still
carries the old keys does not change the constructed policy.

We deliberately avoid running a real Optuna study here — factories are
pure constructors and the Optuna machinery is tested by Optuna itself.
"""

from __future__ import annotations

import optuna
import pytest

from src.sim.policy import (
    MIN_ORDER_FLOOR,
    OPENING_BUDGET_PCT,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
)
from src.tuning.search_spaces import (
    order_up_to_space,
    periodic_order_up_to_space,
    periodic_reorder_space,
    reorder_point_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed(params: dict) -> optuna.trial.FixedTrial:
    """Convenience wrapper: build a FixedTrial from a plain dict."""
    return optuna.trial.FixedTrial(params)


def _base_shared(
    cover_horizon_ticks: int = 10,
    safety_lead_pct_of_lag: float = 0.5,
    stockout_safety_bonus_pct_of_lag: float = 0.0,
    per_supplier_min_order_floor: int = 0,
    routing_strategy: str = "cheapest_first",
) -> dict:
    """Return a dict with all shared tunables at the given values."""
    return {
        "cover_horizon_ticks": cover_horizon_ticks,
        "safety_lead_pct_of_lag": safety_lead_pct_of_lag,
        "stockout_safety_bonus_pct_of_lag": stockout_safety_bonus_pct_of_lag,
        "per_supplier_min_order_floor": per_supplier_min_order_floor,
        "routing_strategy": routing_strategy,
    }


# ---------------------------------------------------------------------------
# order_up_to_space
# ---------------------------------------------------------------------------


def test_order_up_to_space_constructs():
    """Factory returns an OrderUpToPolicy with the sampled kwargs."""
    trial = _fixed(_base_shared(cover_horizon_ticks=10, safety_lead_pct_of_lag=0.5))
    policy = order_up_to_space(trial)

    assert isinstance(policy, OrderUpToPolicy)
    assert policy.cover_horizon_ticks == 10
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.5)
    assert policy.opening_budget_pct == pytest.approx(OPENING_BUDGET_PCT)
    assert policy.stockout_safety_bonus_pct_of_lag == pytest.approx(0.0)
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


def test_order_up_to_space_ignores_a_trial_floor_and_routing_choice():
    """The old routing and floor keys are not applied."""
    trial = _fixed(
        _base_shared(per_supplier_min_order_floor=5, routing_strategy="fill_rate_weighted")
    )
    policy = order_up_to_space(trial)
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


def test_order_up_to_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory should not suggest it."""
    trial = _fixed(_base_shared())
    policy = order_up_to_space(trial)
    assert policy.min_qty == 0


# ---------------------------------------------------------------------------
# reorder_point_space
# ---------------------------------------------------------------------------


def test_reorder_point_space_constructs():
    """Factory returns a ReorderPointPolicy with all sampled kwargs."""
    params = {**_base_shared(cover_horizon_ticks=7, safety_lead_pct_of_lag=1.0,
                              stockout_safety_bonus_pct_of_lag=0.5), "Q": 15}
    trial = _fixed(params)
    policy = reorder_point_space(trial)

    assert isinstance(policy, ReorderPointPolicy)
    assert policy.cover_horizon_ticks == 7
    assert policy.safety_lead_pct_of_lag == pytest.approx(1.0)
    assert policy.opening_budget_pct == pytest.approx(OPENING_BUDGET_PCT)
    assert policy.Q == 15


def test_reorder_point_space_ignores_a_trial_floor_and_routing_choice():
    trial = _fixed(
        {**_base_shared(per_supplier_min_order_floor=3, routing_strategy="fill_rate_weighted"), "Q": 10}
    )
    policy = reorder_point_space(trial)
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


def test_reorder_point_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory constructs without suggesting it."""
    trial = _fixed({**_base_shared(), "Q": 10})
    policy = reorder_point_space(trial)
    # ReorderPointPolicy delegates min_qty to its inner core; the public API
    # does not expose it.  We simply verify construction succeeds and the
    # factory did not attempt to tune min_qty (which would fail on FixedTrial).
    assert isinstance(policy, ReorderPointPolicy)


# ---------------------------------------------------------------------------
# periodic_order_up_to_space
# ---------------------------------------------------------------------------


def test_periodic_order_up_to_space_constructs():
    """Factory returns a PeriodicOrderUpToPolicy with all sampled kwargs."""
    params = {**_base_shared(cover_horizon_ticks=14, safety_lead_pct_of_lag=0.667),
              "review_interval": 7}
    trial = _fixed(params)
    policy = periodic_order_up_to_space(trial)

    assert isinstance(policy, PeriodicOrderUpToPolicy)
    assert policy.cover_horizon_ticks == 14
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.667)
    assert policy.opening_budget_pct == pytest.approx(OPENING_BUDGET_PCT)
    assert policy.review_interval == 7


def test_periodic_order_up_to_space_ignores_a_trial_floor_and_routing_choice():
    trial = _fixed(
        {
            **_base_shared(per_supplier_min_order_floor=2, routing_strategy="fill_rate_weighted"),
            "review_interval": 7,
        }
    )
    policy = periodic_order_up_to_space(trial)
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


def test_periodic_order_up_to_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory constructs without suggesting it."""
    trial = _fixed({**_base_shared(), "review_interval": 3})
    policy = periodic_order_up_to_space(trial)
    assert isinstance(policy, PeriodicOrderUpToPolicy)


# ---------------------------------------------------------------------------
# periodic_reorder_space
# ---------------------------------------------------------------------------


def test_periodic_reorder_space_constructs():
    """Factory returns a PeriodicReorderPolicy with all sampled kwargs."""
    params = {**_base_shared(cover_horizon_ticks=10, safety_lead_pct_of_lag=0.5,
                              stockout_safety_bonus_pct_of_lag=1.0), "review_interval": 5}
    trial = _fixed(params)
    policy = periodic_reorder_space(trial)

    assert isinstance(policy, PeriodicReorderPolicy)
    assert policy.cover_horizon_ticks == 10
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.5)
    assert policy.opening_budget_pct == pytest.approx(OPENING_BUDGET_PCT)
    assert policy.review_interval == 5


def test_periodic_reorder_space_ignores_a_trial_floor_and_routing_choice():
    trial = _fixed(
        {
            **_base_shared(per_supplier_min_order_floor=7, routing_strategy="fill_rate_weighted"),
            "review_interval": 5,
        }
    )
    policy = periodic_reorder_space(trial)
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


def test_periodic_reorder_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory constructs without suggesting it."""
    trial = _fixed({**_base_shared(), "review_interval": 7})
    policy = periodic_reorder_space(trial)
    assert isinstance(policy, PeriodicReorderPolicy)


# ---------------------------------------------------------------------------
# Boundary sweeps — all four factories
# ---------------------------------------------------------------------------

_SHARED_LOW = {
    "cover_horizon_ticks": 1,
    "safety_lead_pct_of_lag": 0.0,
    "stockout_safety_bonus_pct_of_lag": 0.0,
    "per_supplier_min_order_floor": 0,
    "routing_strategy": "cheapest_first",
}

_SHARED_HIGH = {
    "cover_horizon_ticks": 30,
    "safety_lead_pct_of_lag": 3.0,
    "stockout_safety_bonus_pct_of_lag": 2.0,
    "per_supplier_min_order_floor": 10,
    "routing_strategy": "fill_rate_weighted",
}


@pytest.mark.parametrize("params", [_SHARED_LOW, _SHARED_HIGH])
def test_order_up_to_space_boundary(params):
    """Construction succeeds at every range boundary."""
    policy = order_up_to_space(_fixed(params))
    assert isinstance(policy, OrderUpToPolicy)
    assert policy.cover_horizon_ticks == params["cover_horizon_ticks"]
    assert policy.safety_lead_pct_of_lag == pytest.approx(params["safety_lead_pct_of_lag"])
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


@pytest.mark.parametrize("q_val,params", [(1, _SHARED_LOW), (30, _SHARED_HIGH)])
def test_reorder_point_space_boundary(q_val, params):
    """Construction succeeds at every range boundary."""
    full = {**params, "Q": q_val}
    policy = reorder_point_space(_fixed(full))
    assert isinstance(policy, ReorderPointPolicy)
    assert policy.Q == q_val
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


@pytest.mark.parametrize("ri_val,params", [(1, _SHARED_LOW), (14, _SHARED_HIGH)])
def test_periodic_order_up_to_space_boundary(ri_val, params):
    """Construction succeeds at every range boundary."""
    full = {**params, "review_interval": ri_val}
    policy = periodic_order_up_to_space(_fixed(full))
    assert isinstance(policy, PeriodicOrderUpToPolicy)
    assert policy.review_interval == ri_val
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


@pytest.mark.parametrize("ri_val,params", [(1, _SHARED_LOW), (14, _SHARED_HIGH)])
def test_periodic_reorder_space_boundary(ri_val, params):
    """Construction succeeds at every range boundary."""
    full = {**params, "review_interval": ri_val}
    policy = periodic_reorder_space(_fixed(full))
    assert isinstance(policy, PeriodicReorderPolicy)
    assert policy.review_interval == ri_val
    assert policy.per_supplier_min_order_floor == MIN_ORDER_FLOOR
    assert policy.routing_strategy is None


# ---------------------------------------------------------------------------
# Documented ranges guard — all four factories at both low and high boundaries
# ---------------------------------------------------------------------------


def test_all_factories_use_documented_ranges():
    """All four factories construct without error at every range boundary.

    Routing and the min-order floor stay pinned at every boundary.
    """
    for params in (_SHARED_LOW, _SHARED_HIGH):
        p = order_up_to_space(_fixed(params))
        assert isinstance(p, OrderUpToPolicy)

    for q_val, params in [(1, _SHARED_LOW), (30, _SHARED_HIGH)]:
        full = {**params, "Q": q_val}
        p = reorder_point_space(_fixed(full))
        assert isinstance(p, ReorderPointPolicy)
        assert p.Q == q_val

    for ri_val, params in [(1, _SHARED_LOW), (14, _SHARED_HIGH)]:
        full = {**params, "review_interval": ri_val}
        p = periodic_order_up_to_space(_fixed(full))
        assert isinstance(p, PeriodicOrderUpToPolicy)
        assert p.review_interval == ri_val

    for ri_val, params in [(1, _SHARED_LOW), (14, _SHARED_HIGH)]:
        full = {**params, "review_interval": ri_val}
        p = periodic_reorder_space(_fixed(full))
        assert isinstance(p, PeriodicReorderPolicy)
        assert p.review_interval == ri_val


# ---------------------------------------------------------------------------
# Public API re-export test
# ---------------------------------------------------------------------------


def test_factories_importable_from_tuning_package():
    """All four factories are importable from the top-level tuning package."""
    from src.tuning import (
        order_up_to_space as f1,
        periodic_order_up_to_space as f3,
        periodic_reorder_space as f4,
        reorder_point_space as f2,
    )

    assert f1 is order_up_to_space
    assert f2 is reorder_point_space
    assert f3 is periodic_order_up_to_space
    assert f4 is periodic_reorder_space


class _Bounds:
    """Records the inclusive range each ``suggest_*`` call was given."""

    def __init__(self) -> None:
        self.seen: dict[str, tuple[float, float]] = {}

    def suggest_int(self, name: str, low: int, high: int, *args, **kwargs) -> int:
        del args, kwargs
        self.seen[name] = (int(low), int(high))
        return int(low)

    def suggest_float(self, name: str, low: float, high: float, *args, **kwargs) -> float:
        del args, kwargs
        self.seen[name] = (float(low), float(high))
        return float(low)


def test_factories_suggest_the_documented_ranges():
    """Narrowing a constant must fail here. The numbers are the old literals."""
    shared = _Bounds()
    order_up_to_space(shared)
    assert shared.seen["cover_horizon_ticks"] == (1, 30)
    assert shared.seen["safety_lead_pct_of_lag"] == (0.0, 3.0)
    assert shared.seen["stockout_safety_bonus_pct_of_lag"] == (0.0, 2.0)

    reorder = _Bounds()
    reorder_point_space(reorder)
    assert reorder.seen["Q"] == (1, 30)

    periodic = _Bounds()
    periodic_order_up_to_space(periodic)
    assert periodic.seen["review_interval"] == (1, 14)

    both = _Bounds()
    periodic_reorder_space(both)
    assert both.seen["review_interval"] == (1, 14)
    assert "Q" not in both.seen
