"""Pure-function business-metrics module for tuning episodes.

Mirrors the layout used by other parts of the project but lives here so the
tuning package is fully self-contained — no cross-package imports from RL.

Public surface
--------------
RunSlice
    Dataclass collecting per-active-SKU per-tick traces for one episode.

aggregate_episode(run_slice) → dict[str, float]
    Bundled flat dict (service_level, stockout_rate, mean_price_pct_of_msrp,
    inventory_turnover, revenue, holding_cost, order_cost, order_fees,
    net_profit) used as the per-seed KPI payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RunSlice:
    """Per-active-SKU per-tick traces for one tuning episode."""

    sales: List[List[float]] = field(default_factory=list)
    demand: List[List[float]] = field(default_factory=list)
    inventory: List[List[float]] = field(default_factory=list)
    price: List[List[float]] = field(default_factory=list)
    msrp: List[List[float]] = field(default_factory=list)
    revenue: List[List[float]] = field(default_factory=list)
    holding_cost: List[List[float]] = field(default_factory=list)
    order_cost: List[List[float]] = field(default_factory=list)
    order_fee: List[List[float]] = field(default_factory=list)

    active_pids: List[str] = field(default_factory=list)


def _flat(matrix: List[List[float]]) -> List[float]:
    out: list[float] = []
    for row in matrix:
        out.extend(row)
    return out


def _safe_sum(values: List[float]) -> float:
    return float(sum(values)) if values else 0.0


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def service_level(run_slice: RunSlice) -> float:
    total_sales = _safe_sum(_flat(run_slice.sales))
    total_demand = _safe_sum(_flat(run_slice.demand))
    return total_sales / max(1.0, total_demand)


def stockout_rate(run_slice: RunSlice) -> float:
    flat_inv = _flat(run_slice.inventory)
    if not flat_inv:
        return 0.0
    n_zero = sum(1 for v in flat_inv if v == 0.0)
    return float(n_zero) / float(len(flat_inv))


def mean_price_pct_of_msrp(run_slice: RunSlice) -> float:
    flat_price = _flat(run_slice.price)
    flat_msrp = _flat(run_slice.msrp)
    if not flat_price or not flat_msrp:
        return 1.0
    ratios = [p / max(1e-9, m) for p, m in zip(flat_price, flat_msrp)]
    return _safe_mean(ratios)


def inventory_turnover(run_slice: RunSlice) -> float:
    total_sales = _safe_sum(_flat(run_slice.sales))
    flat_inv = _flat(run_slice.inventory)
    mean_inv = _safe_mean(flat_inv)
    return total_sales / max(1.0, mean_inv)


def profit_decomposition(run_slice: RunSlice) -> Dict[str, float]:
    revenue = _safe_sum(_flat(run_slice.revenue))
    holding = _safe_sum(_flat(run_slice.holding_cost))
    order_cost = _safe_sum(_flat(run_slice.order_cost))
    fees = _safe_sum(_flat(run_slice.order_fee))
    net = revenue - (holding + order_cost + fees)
    return {
        "revenue": revenue,
        "holding_cost": holding,
        "order_cost": order_cost,
        "order_fees": fees,
        "net_profit": net,
    }


def aggregate_episode(run_slice: RunSlice) -> Dict[str, float]:
    decomp = profit_decomposition(run_slice)
    return {
        "service_level": service_level(run_slice),
        "stockout_rate": stockout_rate(run_slice),
        "mean_price_pct_of_msrp": mean_price_pct_of_msrp(run_slice),
        "inventory_turnover": inventory_turnover(run_slice),
        **decomp,
    }


__all__ = ["RunSlice", "aggregate_episode"]
