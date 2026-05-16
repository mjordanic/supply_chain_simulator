"""Tests for src/tuning/search_spaces.py.

Each bundled trial-callback factory is exercised with:
  - A FixedTrial at representative mid-range values → construction succeeds
    and the sampled kwargs are propagated correctly.
  - Boundary sweeps → construction succeeds at every range boundary so that
    silent range-narrowing in the factory is caught immediately.

We deliberately avoid running a real Optuna study here — factories are
pure constructors and the Optuna machinery is tested by Optuna itself.
"""

from __future__ import annotations

import optuna
import pytest

from src.sim.policy import (
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


# ---------------------------------------------------------------------------
# order_up_to_space
# ---------------------------------------------------------------------------


def test_order_up_to_space_constructs():
    """Factory returns an OrderUpToPolicy with the sampled kwargs."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.5,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        }
    )
    policy = order_up_to_space(trial)

    assert isinstance(policy, OrderUpToPolicy)
    assert policy.cover_horizon_ticks == 10
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.5)
    assert policy.opening_budget_pct == pytest.approx(0.5)
    assert policy.stockout_safety_bonus_pct_of_lag == pytest.approx(0.0)


def test_order_up_to_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory should not suggest it."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 5,
            "safety_lead_pct_of_lag": 1.0,
            "opening_budget_pct": 0.3,
            "stockout_safety_bonus_pct_of_lag": 0.5,
        }
    )
    policy = order_up_to_space(trial)
    # min_qty stays at the class default (0)
    assert policy.min_qty == 0


# ---------------------------------------------------------------------------
# reorder_point_space
# ---------------------------------------------------------------------------


def test_reorder_point_space_constructs():
    """Factory returns a ReorderPointPolicy with all sampled kwargs."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 7,
            "safety_lead_pct_of_lag": 1.0,
            "opening_budget_pct": 0.4,
            "stockout_safety_bonus_pct_of_lag": 0.5,
            "Q": 15,
        }
    )
    policy = reorder_point_space(trial)

    assert isinstance(policy, ReorderPointPolicy)
    assert policy.cover_horizon_ticks == 7
    assert policy.safety_lead_pct_of_lag == pytest.approx(1.0)
    assert policy.opening_budget_pct == pytest.approx(0.4)
    assert policy.stockout_safety_bonus_pct_of_lag == pytest.approx(0.5)
    assert policy.Q == 15


def test_reorder_point_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory should not suggest it."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 5,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "Q": 10,
        }
    )
    policy = reorder_point_space(trial)
    assert policy.min_qty == 0


# ---------------------------------------------------------------------------
# periodic_order_up_to_space
# ---------------------------------------------------------------------------


def test_periodic_order_up_to_space_constructs():
    """Factory returns a PeriodicOrderUpToPolicy with all sampled kwargs."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 14,
            "safety_lead_pct_of_lag": 0.667,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "review_interval": 7,
        }
    )
    policy = periodic_order_up_to_space(trial)

    assert isinstance(policy, PeriodicOrderUpToPolicy)
    assert policy.cover_horizon_ticks == 14
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.667)
    assert policy.opening_budget_pct == pytest.approx(0.5)
    assert policy.stockout_safety_bonus_pct_of_lag == pytest.approx(0.0)
    assert policy.review_interval == 7


def test_periodic_order_up_to_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory should not suggest it."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.5,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "review_interval": 3,
        }
    )
    policy = periodic_order_up_to_space(trial)
    assert policy.min_qty == 0


# ---------------------------------------------------------------------------
# periodic_reorder_space
# ---------------------------------------------------------------------------


def test_periodic_reorder_space_constructs():
    """Factory returns a PeriodicReorderPolicy with all sampled kwargs."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.5,
            "opening_budget_pct": 0.6,
            "stockout_safety_bonus_pct_of_lag": 1.0,
            "review_interval": 5,
        }
    )
    policy = periodic_reorder_space(trial)

    assert isinstance(policy, PeriodicReorderPolicy)
    assert policy.cover_horizon_ticks == 10
    assert policy.safety_lead_pct_of_lag == pytest.approx(0.5)
    assert policy.opening_budget_pct == pytest.approx(0.6)
    assert policy.stockout_safety_bonus_pct_of_lag == pytest.approx(1.0)
    assert policy.review_interval == 5


def test_periodic_reorder_space_no_min_qty_tuned():
    """min_qty is not a tunable — factory should not suggest it."""
    trial = _fixed(
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "review_interval": 7,
        }
    )
    policy = periodic_reorder_space(trial)
    assert policy.min_qty == 0


# ---------------------------------------------------------------------------
# test_all_factories_use_documented_ranges
# ---------------------------------------------------------------------------
#
# Guard against silent range-narrowing: for every tunable in every factory,
# both the low boundary and the high boundary must construct successfully and
# the resulting attribute must equal the boundary value.


_ORDER_UP_TO_BOUNDARY_CASES = [
    # (param being pushed to boundary, params dict)
    (
        "cover_horizon_ticks_low",
        {
            "cover_horizon_ticks": 1,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "cover_horizon_ticks_high",
        {
            "cover_horizon_ticks": 30,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "safety_lead_pct_of_lag_low",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "safety_lead_pct_of_lag_high",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 3.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "opening_budget_pct_low",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.05,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "opening_budget_pct_high",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.95,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "stockout_safety_bonus_pct_of_lag_low",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
        },
    ),
    (
        "stockout_safety_bonus_pct_of_lag_high",
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.0,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 2.0,
        },
    ),
]


@pytest.mark.parametrize("label,params", _ORDER_UP_TO_BOUNDARY_CASES)
def test_order_up_to_space_boundary(label: str, params: dict):
    """Construction succeeds and kwarg is propagated at every range boundary."""
    policy = order_up_to_space(_fixed(params))
    assert isinstance(policy, OrderUpToPolicy)
    # Verify the specific param that was pushed to its boundary
    param_name = label.rsplit("_", 1)[0]  # strip "_low" / "_high" suffix
    boundary_value = params[param_name]
    actual = getattr(policy, param_name)
    assert actual == pytest.approx(boundary_value), (
        f"{label}: expected {param_name}={boundary_value!r}, got {actual!r}"
    )


_REORDER_POINT_EXTRA_CASES = [
    ("Q_low", {"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.0,
               "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0,
               "Q": 1}),
    ("Q_high", {"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.0,
                "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0,
                "Q": 30}),
]


@pytest.mark.parametrize("label,params", _REORDER_POINT_EXTRA_CASES)
def test_reorder_point_space_boundary(label: str, params: dict):
    """Construction succeeds and Q kwarg is propagated at every range boundary."""
    policy = reorder_point_space(_fixed(params))
    assert isinstance(policy, ReorderPointPolicy)
    param_name = label.rsplit("_", 1)[0]
    boundary_value = params[param_name]
    actual = getattr(policy, param_name)
    assert actual == pytest.approx(boundary_value), (
        f"{label}: expected {param_name}={boundary_value!r}, got {actual!r}"
    )


_PERIODIC_REVIEW_INTERVAL_CASES = [
    ("review_interval_low",
     {"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.0,
      "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0,
      "review_interval": 1}),
    ("review_interval_high",
     {"cover_horizon_ticks": 10, "safety_lead_pct_of_lag": 0.0,
      "opening_budget_pct": 0.5, "stockout_safety_bonus_pct_of_lag": 0.0,
      "review_interval": 14}),
]


@pytest.mark.parametrize("label,params", _PERIODIC_REVIEW_INTERVAL_CASES)
def test_periodic_order_up_to_space_boundary(label: str, params: dict):
    """Construction succeeds and review_interval is propagated at every boundary."""
    policy = periodic_order_up_to_space(_fixed(params))
    assert isinstance(policy, PeriodicOrderUpToPolicy)
    param_name = label.rsplit("_", 1)[0]
    boundary_value = params[param_name]
    actual = getattr(policy, param_name)
    assert actual == pytest.approx(boundary_value), (
        f"{label}: expected {param_name}={boundary_value!r}, got {actual!r}"
    )


@pytest.mark.parametrize("label,params", _PERIODIC_REVIEW_INTERVAL_CASES)
def test_periodic_reorder_space_boundary(label: str, params: dict):
    """Construction succeeds and review_interval is propagated at every boundary."""
    policy = periodic_reorder_space(_fixed(params))
    assert isinstance(policy, PeriodicReorderPolicy)
    param_name = label.rsplit("_", 1)[0]
    boundary_value = params[param_name]
    actual = getattr(policy, param_name)
    assert actual == pytest.approx(boundary_value), (
        f"{label}: expected {param_name}={boundary_value!r}, got {actual!r}"
    )


# ---------------------------------------------------------------------------
# test_all_factories_use_documented_ranges  (issue-spec required test)
# ---------------------------------------------------------------------------


def test_all_factories_use_documented_ranges():
    """All four factories construct without error at every range boundary.

    Iterates each factory with a FixedTrial set to the low and high boundary
    for each tunable; asserts construction succeeds and the kwarg is the
    boundary value.  Guards against silent range-narrowing in any factory.
    """
    shared_low = {
        "cover_horizon_ticks": 1,
        "safety_lead_pct_of_lag": 0.0,
        "opening_budget_pct": 0.05,
        "stockout_safety_bonus_pct_of_lag": 0.0,
    }
    shared_high = {
        "cover_horizon_ticks": 30,
        "safety_lead_pct_of_lag": 3.0,
        "opening_budget_pct": 0.95,
        "stockout_safety_bonus_pct_of_lag": 2.0,
    }

    # order_up_to_space — 4 tunables
    for params in (shared_low, shared_high):
        p = order_up_to_space(_fixed(params))
        assert isinstance(p, OrderUpToPolicy)
        for k, v in params.items():
            assert getattr(p, k) == pytest.approx(v), f"order_up_to_space: {k}={v}"

    # reorder_point_space — 5 tunables (shared + Q)
    for q_val, params in [(1, shared_low), (30, shared_high)]:
        full = {**params, "Q": q_val}
        p = reorder_point_space(_fixed(full))
        assert isinstance(p, ReorderPointPolicy)
        for k, v in full.items():
            assert getattr(p, k) == pytest.approx(v), f"reorder_point_space: {k}={v}"

    # periodic_order_up_to_space — 5 tunables (shared + review_interval)
    for ri_val, params in [(1, shared_low), (14, shared_high)]:
        full = {**params, "review_interval": ri_val}
        p = periodic_order_up_to_space(_fixed(full))
        assert isinstance(p, PeriodicOrderUpToPolicy)
        for k, v in full.items():
            assert getattr(p, k) == pytest.approx(v), (
                f"periodic_order_up_to_space: {k}={v}"
            )

    # periodic_reorder_space — 5 tunables (shared + review_interval)
    for ri_val, params in [(1, shared_low), (14, shared_high)]:
        full = {**params, "review_interval": ri_val}
        p = periodic_reorder_space(_fixed(full))
        assert isinstance(p, PeriodicReorderPolicy)
        for k, v in full.items():
            assert getattr(p, k) == pytest.approx(v), (
                f"periodic_reorder_space: {k}={v}"
            )


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

    # They must be the same objects as the direct module imports
    assert f1 is order_up_to_space
    assert f2 is reorder_point_space
    assert f3 is periodic_order_up_to_space
    assert f4 is periodic_reorder_space
