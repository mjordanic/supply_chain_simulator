"""The four textbook rules of ``src/sim/policy.py`` on the lean simulator.

Levels, triggers, rounding, the stockout bonus, and the fair-share ration
follow ``TextbookReorderPolicy`` and its cores line for line; the parity test
runs both on one demand path. The one addition is ``demand_window``: the
engine always averages the last ``delivery_lag`` days of sales, which is
noisy. ``None`` keeps that rule; tuning may pick a longer window. The
stockout lookback uses the same window. The fair-share ration spends cash
net of the order fee (the engine's parity test runs with fee 0).

The engine's review counter starts at tick 1, so day ``t`` here is step ``t + 1``.
Every product has sales history, so the engine's cold-start pilot never fires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.perfect_supplier.sim import Decision, Obs
from src.sim.policy import _allocate_two_pass_fair_share

FAMILIES = ("order_up_to", "reorder_point", "periodic_order_up_to", "periodic_reorder")


@dataclass(frozen=True)
class TextbookParams:
    family: str = "order_up_to"
    cover_horizon_ticks: int = 14
    safety_lead_pct_of_lag: float = 1 / 3
    stockout_safety_bonus_pct_of_lag: float = 0.0
    review_interval: int | None = None  # periodic families; None = delivery lag
    demand_window: int | None = None  # None = delivery lag, as in the engine


class TextbookPolicy:
    def __init__(self, params: TextbookParams, name: str | None = None):
        if params.family not in FAMILIES:
            raise ValueError(f"unknown family {params.family!r}")
        self.params = params
        self.name = name or params.family

    def levels(self, obs: Obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-product ``(rate, s, S)`` on this observation."""
        p, lag = self.params, obs.delivery_lag
        window = max(1, int(p.demand_window if p.demand_window is not None else lag))
        recent = obs.sales[-window:]
        rate = recent.mean(axis=0)
        safety = np.full(obs.n_products, round(p.safety_lead_pct_of_lag * lag), dtype=float)
        if p.stockout_safety_bonus_pct_of_lag > 0:
            # Engine stockout: positive sales that emptied the shelf.
            stocked_out = ((recent > 0) & obs.censored[-window:]).any(axis=0)
            safety += np.where(stocked_out, round(p.stockout_safety_bonus_pct_of_lag * lag), 0)
        s = (lag + safety) * rate
        S = s + p.cover_horizon_ticks * rate
        return rate, s, S

    def decide(self, obs: Obs) -> Decision:
        p, lag = self.params, obs.delivery_lag
        rate, s, S = self.levels(obs)
        position = obs.position
        step = obs.day + 1
        review = p.review_interval if p.review_interval is not None else max(1, int(lag))
        due = step % review == 0
        if p.family == "order_up_to":
            trigger, qty = position < s, np.rint(S - position)
        elif p.family == "reorder_point":
            trigger, qty = position < s, np.rint(S - s)
        elif p.family == "periodic_order_up_to":
            trigger, qty = np.full(obs.n_products, due), np.rint(S - position)
        else:
            trigger, qty = due & (position < s), np.rint(S - position)
        desired = {str(i): int(q) for i, (t, q) in enumerate(zip(trigger, qty)) if t and q > 0}
        free = max(0, obs.capacity - int(position.sum()))
        alloc = _allocate_two_pass_fair_share(
            desired=desired,
            unit_costs={str(i): obs.unit_cost for i in range(obs.n_products)},
            free_space=free,
            # Reserve the fee, as the sim does, so the sim never re-rations this split.
            cash=max(obs.cash - obs.order_fee, 0.0),
            min_qty=0,
            pilot_pids=set(),
        )
        out = np.array([alloc.get(str(i), 0) for i in range(obs.n_products)], dtype=np.int64)
        return Decision(out, {"rate": rate, "s": s, "S": S})

    def describe(self) -> dict:
        return asdict(self.params)


@dataclass(frozen=True)
class EoqParams:
    service_z: float = 1.65
    demand_window: int = 28
    q_scale: float = 1.0
    joint: bool = False
    family: str = "eoq"


class EoqPolicy:
    """Continuous-review (s, Q) with the economic order quantity.

    Unlike the four engine rules above, this one reads the shop's economics,
    as the planner does. Per product, from the last ``demand_window`` days of
    sales: rate ``d`` and daily standard deviation ``sd``.

    - reorder point ``s = d * lag + z * sd * sqrt(lag)``
    - order quantity ``Q = q_scale * sqrt(2 * fee * d / (holding_rate * unit_cost))``

    When position falls below ``s`` the shop orders up to ``s + Q``. With no
    fee, ``Q`` is 0 and this is a daily base-stock rule.

    ``joint=False``: each product triggers on its own and pays its own fee in
    the formula. ``joint=True`` is the textbook can-order idea at its simplest:
    when any product is below ``s``, every product orders up to ``s + Q``, and
    the fee in ``Q`` is shared by all products. That is the planner's ordering
    pattern without the forecast.
    """

    def __init__(self, params: EoqParams, name: str | None = None):
        self.params = params
        self.name = name or "eoq"

    def decide(self, obs: Obs) -> Decision:
        p, lag = self.params, obs.delivery_lag
        recent = obs.sales[-p.demand_window :]
        rate = recent.mean(axis=0)
        sd = recent.std(axis=0, ddof=1)
        s = rate * lag + p.service_z * sd * np.sqrt(lag)
        fee = obs.order_fee / obs.n_products if p.joint else obs.order_fee
        Q = p.q_scale * np.sqrt(2 * fee * rate / (obs.holding_rate * obs.unit_cost))
        S = s + Q
        position = obs.position
        below = position < s
        trigger = np.full(obs.n_products, below.any()) if p.joint else below
        qty = np.where(trigger, np.maximum(np.rint(S - position), 0), 0).astype(np.int64)
        return Decision(qty, {"rate": rate, "s": s, "S": S})

    def describe(self) -> dict:
        return asdict(self.params)
