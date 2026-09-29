"""Planner scoring horizon and terminal value.

The pre-fix planner scores ``delivery_lag`` steps. An order placed on the
first imagined step arrives on step ``delivery_lag``, which is the first
step outside that window, so every positive cover loses to ordering
nothing. These tests use an oracle dynamics with that arrival timing.
"""

from __future__ import annotations

from src.jepa.plan import ReplenishmentPlanner
from src.sim.central_table import CentralTable, Offer

LAG = 3
DEMAND = 10
PRICE = 10.0
COST = 4.0


class _OracleDynamics:
    """Exact on-hand recursion. Granted units are sellable ``delivery_lag`` steps later."""

    def __init__(self, delivery_lag: int, demand: float) -> None:
        self.delivery_lag = int(delivery_lag)
        self.demand = float(demand)
        # Index 0 arrives at the next imagined step, before that step's sales.
        self.pipeline = [0.0] * self.delivery_lag

    def open_rollout(self, features: list[list[float]]) -> _OracleRollout:
        return _OracleRollout(self, features)


class _OracleRollout:
    def __init__(self, dynamics: _OracleDynamics, features: list[list[float]]) -> None:
        self.demand = dynamics.demand
        self.rows: list[list] = []
        pipeline = list(dynamics.pipeline)
        for feat in features:
            on_hand = float(feat[0])
            in_transit = float(feat[1])
            profile = list(pipeline)
            if int(round(sum(profile))) != int(round(in_transit)):
                raise AssertionError(
                    f"in-transit sum {in_transit} does not match the arrival profile {sum(profile)}"
                )
            self.rows.append([on_hand, profile])

    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        predicted: list[tuple[float, float, float]] = []
        for row, qty in zip(self.rows, granted, strict=True):
            on_hand, profile = row
            on_hand, sales = _arrive_and_sell(on_hand, profile, float(qty), self.demand)
            row[0] = on_hand
            predicted.append((on_hand, float(sum(profile)), sales))
        return predicted


def _table() -> CentralTable:
    table = CentralTable()
    table.publish("F_A", "A", Offer(available_qty=10_000, list_price=COST, min_order=0))
    return table


def _obs(*, on_hand: int, in_transit: int, observed_sales: int) -> dict:
    pending = {"F_A": {"A": in_transit}} if in_transit else {}
    return {
        "tick": 0,
        "inventory": {"A": on_hand},
        "pending": pending,
        "observed_sales": {"A": observed_sales},
        "base_prices": {"A": PRICE},
        "list_prices": {"A": 1.0},
        "capacity": 100_000,
        "cash": 1_000_000.0,
        "direct_supplier_ids": {"F_A"},
    }


def _granted(action: dict) -> int:
    return sum(qty for _supplier, qty in action["order"].get("A", []))


def _arrive_and_sell(
    on_hand: float,
    profile: list[float],
    granted: float,
    demand: float,
) -> tuple[float, float]:
    """The oldest lot arrives, then sales. `granted` is the new lot."""
    arriving = float(profile.pop(0))
    profile.append(granted)
    on_hand += arriving
    sales = min(on_hand, demand)
    return on_hand - sales, sales


def _planner(**kwargs) -> ReplenishmentPlanner:
    dynamics = kwargs.pop("dynamics", None) or _OracleDynamics(LAG, DEMAND)
    defaults = dict(
        policy_seed=0,
        delivery_lag=LAG,
        budget=64,
        holding_rate=0.01,
        order_fee=0.0,
    )
    defaults.update(kwargs)
    return ReplenishmentPlanner(dynamics, **defaults)


def test_planner_orders_a_lead_time_of_demand_when_margin_is_positive():
    """Empty shelf, steady demand, price above cost, room in cash and capacity.

    A lead time of demand is the smallest positive cover. The cover grid
    cannot ask for more than four. Ordering nothing is the pre-fix winner.
    """
    planner = _planner()
    action = planner.decide(_obs(on_hand=0, in_transit=0, observed_sales=DEMAND), _table())

    qty = _granted(action)
    assert qty >= DEMAND * LAG
    assert qty <= 4 * DEMAND * LAG
    assert action["list_price"] == {"A": PRICE}
    assert action["promotions"] == {}
    assert action["min_order_imposed"] == {"A": 0}


def _realized_profit(horizon_multiple: int, *, ticks: int = 36) -> float:
    """Cash profit of one episode. The oracle the planner rolls out is the world."""
    dynamics = _OracleDynamics(LAG, DEMAND)
    planner = _planner(dynamics=dynamics, horizon_multiple=horizon_multiple)
    on_hand = 0.0
    profit = 0.0
    observed = DEMAND
    for _tick in range(ticks):
        action = planner.decide(
            _obs(
                on_hand=int(round(on_hand)),
                in_transit=int(round(sum(dynamics.pipeline))),
                observed_sales=observed,
            ),
            _table(),
        )
        granted = float(_granted(action))
        holding = planner.holding_rate * on_hand * COST
        on_hand, sales = _arrive_and_sell(on_hand, dynamics.pipeline, granted, float(DEMAND))
        observed = int(round(sales))
        fee = planner.order_fee if granted > 0 else 0.0
        profit += sales * PRICE - granted * COST - holding - fee
    return profit


def test_realized_profit_is_insensitive_to_horizon_multiple():
    """Once ending stock keeps its value, 2×, 4× and 8× lead times score alike.

    Without a terminal value the short window still confiscates pipeline, and
    realized profit moves with the multiple.
    """
    profits = [_realized_profit(multiple) for multiple in (2, 4, 8)]
    spread = max(profits) - min(profits)
    scale = max(abs(profit) for profit in profits)
    assert scale > 0
    assert spread / scale < 0.05
