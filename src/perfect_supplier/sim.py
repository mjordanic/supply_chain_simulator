"""Lean single-shop simulator with one perfect supplier.

Day ``t`` runs in the engine's tick order (``src/sim/runner.py``):

1. Units ordered on day ``t - delivery_lag`` arrive and go on the shelf.
2. Demand arrives; sales are ``min(demand, on_hand)``; the rest is lost.
3. The policy reads an :class:`Obs` (post-sale stock, today's sales) and
   requests a quantity per product.
4. The request is clamped by free shelf (capacity minus on-hand minus
   in-transit) and by cash (purchase cost plus the order fee). Cost and fee
   leave cash now. The clamped quantity is sellable on ``t + delivery_lag``.
5. Holding cost ``closing on-hand * holding_rate * unit_cost`` leaves cash.

Equity is cash plus on-hand and in-transit at unit cost. Policies see sales
and a censoring flag, never demand.

``simulate`` drives the shop with one policy. ``shadow`` policies read the
same observation every day; their requests are logged and never executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.perfect_supplier.world import HISTORY_DAYS, World, demand_mean_path, demand_path


@dataclass
class Obs:
    """What a policy may read on day ``day`` after arrivals."""

    day: int
    weekday: int
    delivery_lag: int
    on_hand: np.ndarray  # (P,)
    pipeline: np.ndarray  # (delivery_lag - 1, P); row k arrives on day + 1 + k
    sales: np.ndarray  # (HISTORY_DAYS + day, P), oldest first
    censored: np.ndarray  # same shape, True when the day closed with an empty shelf
    cash: float
    capacity: int
    unit_cost: float
    price: float
    order_fee: float
    holding_rate: float

    @property
    def in_transit(self) -> np.ndarray:
        return self.pipeline.sum(axis=0)

    @property
    def position(self) -> np.ndarray:
        return self.on_hand + self.in_transit

    @property
    def n_products(self) -> int:
        return len(self.on_hand)


@dataclass
class Decision:
    qty: np.ndarray  # (P,) requested units
    info: dict[str, Any] = field(default_factory=dict)  # scalar or (P,) values for the log


class Policy(Protocol):
    name: str

    def decide(self, obs: Obs) -> Decision: ...


def ration(requested: np.ndarray, free_shelf: int, budget: float, unit_cost: float) -> np.ndarray:
    """Scale ``requested`` down to fit shelf and cash; largest remainder rounding."""
    requested = np.maximum(np.asarray(requested, dtype=np.int64), 0)
    total = int(requested.sum())
    limit = min(max(int(free_shelf), 0), max(int(np.floor(budget / unit_cost + 1e-9)), 0))
    if total <= limit:
        return requested
    if limit == 0:
        return np.zeros_like(requested)
    share = requested * (limit / total)
    out = np.floor(share).astype(np.int64)
    rest = limit - int(out.sum())
    order = np.argsort(-(share - out), kind="stable")
    out[order[:rest]] += 1
    return out


def _to_column(value: Any, n_products: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(n_products, arr.item())
    return arr


@dataclass
class EpisodeResult:
    frame: pd.DataFrame
    summary: dict[str, Any]


def simulate(
    world: World,
    policy: Policy,
    *,
    shadow: Mapping[str, Policy] | None = None,
    demand: np.ndarray | None = None,
) -> EpisodeResult:
    """Run one episode; return the per product-day frame and the episode summary."""
    shadow = dict(shadow or {})
    full_demand = demand_path(world) if demand is None else demand
    full_mean = demand_mean_path(world)
    P, T, L = world.n_products, world.n_days, world.delivery_lag

    sales = np.zeros((HISTORY_DAYS + T, P), dtype=np.int64)
    censored = np.zeros((HISTORY_DAYS + T, P), dtype=bool)
    sales[:HISTORY_DAYS] = full_demand[:HISTORY_DAYS]
    arrivals = np.zeros((T + L + 1, P), dtype=np.int64)
    on_hand = np.asarray(world.initial_on_hand, dtype=np.int64).copy()
    cash = float(world.initial_cash)
    initial_equity = cash + world.unit_cost * on_hand.sum()

    cols: dict[str, list] = {
        k: []
        for k in (
            "on_hand_open arrivals in_transit position requested filled demand demand_mean "
            "sales lost censored on_hand_close"
        ).split()
    }
    shop: dict[str, list] = {
        k: [] for k in "cash_open cash_close purchase_cost fee_paid revenue holding_cost equity_close".split()
    }
    info_cols: dict[str, list] = {}
    shadow_cols: dict[str, list] = {f"shadow_{n}_requested": [] for n in shadow}

    for t in range(T):
        on_hand = on_hand + arrivals[t]
        on_hand_open = on_hand.copy()
        cash_open = cash
        d = full_demand[HISTORY_DAYS + t]
        sold = np.minimum(d, on_hand)
        on_hand = on_hand - sold
        revenue = world.price * float(sold.sum())
        cash += revenue
        sales[HISTORY_DAYS + t] = sold
        censored[HISTORY_DAYS + t] = on_hand == 0

        pipeline = arrivals[t + 1 : t + L] if L > 1 else np.zeros((0, P), dtype=np.int64)
        obs = Obs(
            day=t,
            weekday=(HISTORY_DAYS + t) % 7,
            delivery_lag=L,
            on_hand=on_hand.copy(),
            pipeline=pipeline.copy(),
            sales=sales[: HISTORY_DAYS + t + 1],
            censored=censored[: HISTORY_DAYS + t + 1],
            cash=cash,
            capacity=world.capacity,
            unit_cost=world.unit_cost,
            price=world.price,
            order_fee=world.order_fee,
            holding_rate=world.holding_rate,
        )
        in_transit = obs.in_transit
        decision = policy.decide(obs)
        for name, other in shadow.items():
            shadow_cols[f"shadow_{name}_requested"].append(np.asarray(other.decide(obs).qty, dtype=np.int64))
        requested = np.maximum(np.asarray(decision.qty, dtype=np.int64), 0)
        free_shelf = world.capacity - int(on_hand.sum()) - int(in_transit.sum())
        budget = cash - (world.order_fee if requested.sum() > 0 else 0.0)
        filled = ration(requested, free_shelf, budget, world.unit_cost)
        fee = world.order_fee if filled.sum() > 0 else 0.0
        purchase = world.unit_cost * float(filled.sum())
        arrivals[t + L] += filled
        holding = world.holding_rate * world.unit_cost * float(on_hand.sum())
        cash -= purchase + fee + holding
        pipeline_after = arrivals[t + 1 : t + L + 1].sum(axis=0)
        equity = cash + world.unit_cost * float(on_hand.sum() + pipeline_after.sum())

        for k, v in (
            ("on_hand_open", on_hand_open),
            ("arrivals", arrivals[t]),
            ("in_transit", in_transit),
            ("position", on_hand + in_transit),
            ("requested", requested),
            ("filled", filled),
            ("demand", d),
            ("demand_mean", full_mean[HISTORY_DAYS + t]),
            ("sales", sold),
            ("lost", d - sold),
            ("censored", censored[HISTORY_DAYS + t]),
            ("on_hand_close", on_hand),
        ):
            cols[k].append(np.asarray(v).copy())
        for k, v in (
            ("cash_open", cash_open),
            ("cash_close", cash),
            ("purchase_cost", purchase),
            ("fee_paid", fee),
            ("revenue", revenue),
            ("holding_cost", holding),
            ("equity_close", equity),
        ):
            shop[k].append(v)
        for k, v in decision.info.items():
            info_cols.setdefault(f"dec_{k}", [None] * t).append(_to_column(v, P))

    days = np.repeat(np.arange(T), P)
    frame = pd.DataFrame(
        {
            "world_seed": world.world_seed,
            "split": world.split,
            "method": policy.name,
            "day": days,
            "weekday": (HISTORY_DAYS + days) % 7,
            "product": np.tile(np.arange(P), T),
        }
    )
    for k, v in cols.items():
        frame[k] = np.concatenate(v)
    for k, v in shop.items():
        frame[k] = np.repeat(np.asarray(v, dtype=float), P)
    for k, v in info_cols.items():
        v = [np.full(P, np.nan) if x is None else x for x in v]
        frame[k] = np.concatenate(v)
    for k, v in shadow_cols.items():
        frame[k] = np.concatenate(v)

    final_equity = shop["equity_close"][-1]
    summary = episode_summary(world, policy.name, frame, initial_equity, final_equity)
    return EpisodeResult(frame=frame, summary=summary)


def episode_summary(
    world: World, method: str, frame: pd.DataFrame, initial_equity: float, final_equity: float
) -> dict[str, Any]:
    """One summary row: the world, per-product demand and ordering, and the score."""
    row: dict[str, Any] = {"method": method, **world.params()}
    row["initial_equity"] = initial_equity
    row["final_equity"] = final_equity
    row["delta_equity"] = final_equity - initial_equity
    # Share of the margin on every demanded unit that ended up as equity.
    # 1.0 would sell all demand with no fee and no holding cost.
    row["total_demand"] = float(frame["demand"].sum())
    row["score"] = row["delta_equity"] / ((world.price - world.unit_cost) * row["total_demand"])
    row["delta_equity_over_initial"] = row["delta_equity"] / initial_equity
    per_day = frame.groupby("day").first()
    row["net_profit"] = float(per_day["cash_close"].iloc[-1] - world.initial_cash)
    row["revenue"] = float(per_day["revenue"].sum())
    row["purchase_cost"] = float(per_day["purchase_cost"].sum())
    row["fees"] = float(per_day["fee_paid"].sum())
    row["holding_cost"] = float(per_day["holding_cost"].sum())
    # Days with any filled order; not fee_paid, which is 0 every day when the fee is 0.
    row["order_days"] = int((frame.groupby("day")["filled"].sum() > 0).sum())
    row["fill_rate_demand"] = float(frame["sales"].sum() / max(frame["demand"].sum(), 1))
    row["lost_units"] = int(frame["lost"].sum())
    row["stockout_product_days"] = int((frame["lost"] > 0).sum())
    row["mean_on_hand"] = float(frame["on_hand_close"].mean())
    for p, g in frame.groupby("product"):
        ordered = g.loc[g["filled"] > 0, "filled"]
        row[f"mean_demand_p{p}"] = float(g["demand"].mean())
        row[f"zero_demand_days_p{p}"] = int((g["demand"] == 0).sum())
        row[f"mean_order_qty_p{p}"] = float(ordered.mean()) if len(ordered) else 0.0
        row[f"order_count_p{p}"] = int(len(ordered))
        row[f"fill_rate_p{p}"] = float(g["sales"].sum() / max(g["demand"].sum(), 1))
        row[f"mean_on_hand_p{p}"] = float(g["on_hand_close"].mean())
    return row
