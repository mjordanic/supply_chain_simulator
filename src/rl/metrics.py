"""Pure-function business-metrics module for RL episodes.

All functions are deterministic and stateless — no I/O, no global state,
no DataFrame coupling.  ``RunSlice`` is a plain dataclass populated from
the env's per-tick ``info`` dicts; aggregation happens entirely in memory.

Public surface
--------------
RunSlice
    Dataclass collecting per-active-SKU per-tick traces for one episode.

service_level(run_slice) → float
    sum(sales) / max(1, sum(demand)) — fraction of demand fulfilled.

stockout_rate(run_slice) → float
    Fraction of (active_SKU, tick) pairs where inventory was 0 at
    decision time.

mean_price_pct_of_msrp(run_slice) → float
    Mean of price / MSRP across all active-SKU ticks.

inventory_turnover(run_slice) → float
    sum(sales) / max(1, mean(inventory)) — how often stock is replenished.

profit_decomposition(run_slice) → dict[str, float]
    Totals for revenue, holding_cost, order_cost, order_fees, net_profit.
    net_profit = revenue - (holding_cost + order_cost + order_fees).

aggregate_episode(run_slice) → dict[str, float]
    All of the above bundled into one flat dict for one episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunSlice:
    """Per-active-SKU per-tick traces for one RL episode.

    Each field is a list-of-lists: outer index is tick (0 … T-1), inner
    index is active-SKU position (0 … K-1).  Alternatively, callers may
    append per-tick dicts and use the helper constructors, but the plain
    list-of-lists representation keeps the aggregation math simple and
    avoids any pandas dependency.

    Fields
    ------
    sales:
        Units sold per (tick, sku) — realised sales after stockout clamping.
    demand:
        Realised demand per (tick, sku) before stockout clamping.
    inventory:
        On-hand inventory per (tick, sku) *at decision time* (before
        settling demand for that tick).
    price:
        Selling price per (tick, sku).
    msrp:
        MSRP (base price) per (tick, sku).  Assumed constant per SKU
        across an episode but stored per-tick for generality.
    revenue:
        Revenue per (tick, sku) = sales × price.
    holding_cost:
        Holding cost per (tick, sku).
    order_cost:
        Order cost per (tick, sku) = order_qty × unit_cost.
    order_fee:
        Fixed order fee per (tick, sku) (non-zero iff an order was placed).
    active_pids:
        Optional list of product ids in the same order as the inner axis.
        Used for reference only; metrics are computed purely from arrays.
    """

    # Per-tick, per-sku traces — list of lists (tick × sku).
    sales: List[List[float]] = field(default_factory=list)
    demand: List[List[float]] = field(default_factory=list)
    inventory: List[List[float]] = field(default_factory=list)
    price: List[List[float]] = field(default_factory=list)
    msrp: List[List[float]] = field(default_factory=list)
    revenue: List[List[float]] = field(default_factory=list)
    holding_cost: List[List[float]] = field(default_factory=list)
    order_cost: List[List[float]] = field(default_factory=list)
    order_fee: List[List[float]] = field(default_factory=list)

    # Optional metadata — not used in metric computations.
    active_pids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Flat helpers
# ---------------------------------------------------------------------------


def _flat(matrix: List[List[float]]) -> List[float]:
    """Flatten a list-of-lists into a single list."""
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


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def service_level(run_slice: RunSlice) -> float:
    """Fraction of demand fulfilled: sum(sales) / max(1, sum(demand)).

    Handles empty slices and zero-demand episodes without divide-by-zero.
    Returns 0.0 when there is no demand and no sales.
    """
    total_sales = _safe_sum(_flat(run_slice.sales))
    total_demand = _safe_sum(_flat(run_slice.demand))
    return total_sales / max(1.0, total_demand)


def stockout_rate(run_slice: RunSlice) -> float:
    """Fraction of (active_SKU, tick) pairs where inventory was 0 at decision time.

    Returns 0.0 on an empty slice.
    """
    flat_inv = _flat(run_slice.inventory)
    if not flat_inv:
        return 0.0
    n_zero = sum(1 for v in flat_inv if v == 0.0)
    return float(n_zero) / float(len(flat_inv))


def mean_price_pct_of_msrp(run_slice: RunSlice) -> float:
    """Mean of price / MSRP across all (active_SKU, tick) pairs.

    Returns 1.0 when MSRP data is unavailable (graceful fallback).
    Handles zero MSRP entries by treating them as 1.0.
    """
    flat_price = _flat(run_slice.price)
    flat_msrp = _flat(run_slice.msrp)
    if not flat_price or not flat_msrp:
        return 1.0
    ratios = [p / max(1e-9, m) for p, m in zip(flat_price, flat_msrp)]
    return _safe_mean(ratios)


def inventory_turnover(run_slice: RunSlice) -> float:
    """sum(sales) / max(1, mean(inventory)) across active-SKU ticks.

    A higher value indicates stock is selling quickly relative to average
    on-hand levels.  Returns 0.0 on an empty slice.
    """
    total_sales = _safe_sum(_flat(run_slice.sales))
    flat_inv = _flat(run_slice.inventory)
    mean_inv = _safe_mean(flat_inv)
    return total_sales / max(1.0, mean_inv)


def profit_decomposition(run_slice: RunSlice) -> Dict[str, float]:
    """Total revenue, cost components, and net profit for the episode.

    Returns
    -------
    dict with keys:
        revenue, holding_cost, order_cost, order_fees, net_profit
    where net_profit = revenue - (holding_cost + order_cost + order_fees).
    """
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
    """Bundle all business metrics for one episode into a flat dict.

    Keys
    ----
    service_level, stockout_rate, mean_price_pct_of_msrp,
    inventory_turnover, revenue, holding_cost, order_cost,
    order_fees, net_profit.
    """
    decomp = profit_decomposition(run_slice)
    return {
        "service_level": service_level(run_slice),
        "stockout_rate": stockout_rate(run_slice),
        "mean_price_pct_of_msrp": mean_price_pct_of_msrp(run_slice),
        "inventory_turnover": inventory_turnover(run_slice),
        **decomp,
    }


__all__ = [
    "RunSlice",
    "service_level",
    "stockout_rate",
    "mean_price_pct_of_msrp",
    "inventory_turnover",
    "profit_decomposition",
    "aggregate_episode",
]
