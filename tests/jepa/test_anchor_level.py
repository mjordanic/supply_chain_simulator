"""Anchor and planner share routing, the min-order floor, and cold start.

The search space used to sample ``routing_strategy`` and
``per_supplier_min_order_floor`` while the planner was pinned. A trial
that asks for a different route or a non-zero floor is ignored. Both
arms then fill the cheapest supplier first and spill the remainder.
"""

from __future__ import annotations

import optuna

from src.jepa.dynamics import ARRIVAL_AGE_SLOTS
from src.jepa.plan import ReplenishmentPlanner
from src.sim.central_table import CentralTable, Offer
from src.sim.policy import MIN_ORDER_FLOOR, OPENING_BUDGET_PCT, OrderUpToPolicy
from src.tuning.search_spaces import order_up_to_space

CHEAP = "cheap"
DEAR = "dear"
CHEAP_QTY = 5


class _OracleDynamics:
    """Granted units arrive ``delivery_lag`` steps later and sell at a fixed rate."""

    def __init__(self) -> None:
        self.delivery_lag = 3
        self.pipeline = [0.0] * self.delivery_lag

    def open_rollout(self, features: list[list[float]]) -> _OracleRollout:
        return _OracleRollout(self, features)


class _OracleRollout:
    def __init__(self, dynamics: _OracleDynamics, features: list[list[float]]) -> None:
        self.rows: list[list] = []
        pipeline = list(dynamics.pipeline)
        for feat in features:
            on_hand = float(feat[0])
            in_transit = float(sum(feat[1 : 1 + ARRIVAL_AGE_SLOTS]))
            profile = list(pipeline)
            if int(round(sum(profile))) != int(round(in_transit)):
                profile = [0.0] * (len(profile) - 1) + [in_transit]
            self.rows.append([on_hand, profile])

    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        predicted: list[tuple[float, float, float]] = []
        for row, qty in zip(self.rows, granted, strict=True):
            on_hand, profile = row
            arriving = float(profile.pop(0))
            profile.append(float(qty))
            on_hand += arriving
            sales = min(on_hand, 10.0)
            row[0] = on_hand - sales
            predicted.append((row[0], float(sum(profile)), sales))
        return predicted


def _two_supplier_table() -> CentralTable:
    table = CentralTable()
    table.publish(CHEAP, "A", Offer(available_qty=CHEAP_QTY, list_price=4.0, min_order=0))
    table.publish(DEAR, "A", Offer(available_qty=10_000, list_price=9.0, min_order=0))
    return table


def _lines(action: dict) -> list[tuple[str, int]]:
    return [(str(sid), int(qty)) for sid, qty in action["order"].get("A", [])]


def test_search_space_cannot_tune_routing_or_the_min_order_floor():
    """A trial's routing choice and floor do not reach the anchor or the planner."""
    trial = optuna.trial.FixedTrial(
        {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "per_supplier_min_order_floor": 7,
            "routing_strategy": "fill_rate_weighted",
        }
    )
    anchor = order_up_to_space(trial)
    planner = ReplenishmentPlanner(
        _OracleDynamics(),
        policy_seed=0,
        delivery_lag=3,
        budget=4,
        order_fee=0.0,
    )

    assert anchor.per_supplier_min_order_floor == MIN_ORDER_FLOOR == 0
    assert planner.per_supplier_min_order_floor == anchor.per_supplier_min_order_floor
    assert anchor.routing_strategy is None
    assert planner.routing_strategy is None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    study.optimize(lambda t: order_up_to_space(t) and 0.0, n_trials=1)
    sampled = set(study.best_trial.params)
    assert "routing_strategy" not in sampled
    assert "per_supplier_min_order_floor" not in sampled


def test_both_arms_fill_the_cheapest_supplier_then_spill():
    """The cheap offer is short. Both arms take it, then the remainder."""
    table = _two_supplier_table()
    planner = ReplenishmentPlanner(
        _OracleDynamics(),
        policy_seed=0,
        delivery_lag=3,
        budget=64,
        holding_rate=0.01,
        order_fee=0.0,
    )
    planner_action = planner.decide(
        {
            "tick": 0,
            "inventory": {"A": 0},
            "pending": {},
            "observed_sales": {"A": 10},
            "capacity": 100_000,
            "cash": 1_000_000.0,
            "base_prices": {"A": 20.0},
            "list_prices": {"A": 20.0},
            "direct_supplier_ids": {CHEAP, DEAR},
            "min_order_imposed": {"A": 0},
        },
        table,
    )
    planner_lines = _lines(planner_action)
    assert planner_lines[0] == (CHEAP, CHEAP_QTY)
    assert sum(qty for _sid, qty in planner_lines) > CHEAP_QTY
    assert any(sid == DEAR and qty > 0 for sid, qty in planner_lines)

    anchor = OrderUpToPolicy(
        unit_cost=1.0,
        opening_budget_pct=0.0,
        safety_lead_pct_of_lag=0.0,
        cover_horizon_ticks=10,
        delivery_lag=3,
        per_supplier_min_order_floor=MIN_ORDER_FLOOR,
    )
    anchor.decide(
        {
            "tick": 0,
            "inventory": {"A": 50},
            "pending": {},
            "observed_sales": {"A": 10},
            "capacity": 100_000,
            "cash": 1_000_000.0,
            "base_prices": {"A": 20.0},
            "list_prices": {"A": 20.0},
            "direct_supplier_ids": {CHEAP, DEAR},
            "min_order_imposed": {"A": 0},
        },
        table,
    )
    anchor_action = anchor.decide(
        {
            "tick": 1,
            "inventory": {"A": 0},
            "pending": {},
            "observed_sales": {"A": 10},
            "capacity": 100_000,
            "cash": 1_000_000.0,
            "base_prices": {"A": 20.0},
            "list_prices": {"A": 20.0},
            "direct_supplier_ids": {CHEAP, DEAR},
            "min_order_imposed": {"A": 0},
        },
        table,
    )
    anchor_lines = _lines(anchor_action)
    assert anchor_lines[0] == (CHEAP, CHEAP_QTY)
    assert any(sid == DEAR and qty > 0 for sid, qty in anchor_lines)


def test_cold_start_pilot_matches_at_the_supplier_price():
    """Empty shelf and no sales: both arms spend the same fraction of cash.

    The fraction is ``OPENING_BUDGET_PCT`` (half of cash), split across the
    active SKUs and priced at the offer. The constructor default unit cost
    of 1.0 does not size the anchor's pilot. The search space pins the same
    fraction, so a tuned anchor cannot pilot harder than the planner.
    """
    price = 11.0
    cash = 1100.0
    expected = int(OPENING_BUDGET_PCT * cash / price)
    table = CentralTable()
    table.publish("F", "A", Offer(available_qty=10_000, list_price=price, min_order=0))
    obs = {
        "tick": 0,
        "inventory": {"A": 0},
        "pending": {},
        "observed_sales": {"A": 0},
        "capacity": 100_000,
        "cash": cash,
        "base_prices": {"A": 20.0},
        "list_prices": {"A": 20.0},
        "direct_supplier_ids": {"F"},
        "min_order_imposed": {"A": 0},
    }
    anchor = OrderUpToPolicy(unit_cost=1.0)
    planner = ReplenishmentPlanner(
        _OracleDynamics(),
        policy_seed=0,
        delivery_lag=3,
        budget=4,
        order_fee=0.0,
    )
    tuned = order_up_to_space(
        optuna.trial.FixedTrial(
            {
                "cover_horizon_ticks": 10,
                "safety_lead_pct_of_lag": 0.5,
                "stockout_safety_bonus_pct_of_lag": 0.0,
            }
        )
    )

    assert sum(qty for _sid, qty in anchor.decide(obs, table)["order"].get("A", [])) == expected
    assert sum(qty for _sid, qty in planner.decide(obs, table)["order"].get("A", [])) == expected
    assert tuned.opening_budget_pct == planner.opening_budget_pct == OPENING_BUDGET_PCT
