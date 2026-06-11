"""Set encoder/decoder for variable-K RL.

Produces a ``(K_max, F)`` observation tensor and decodes a ``(K_max, 3)`` action
tensor for the shared-weight per-product policy introduced in ADR 0021.

Public API
----------
encode_set_observation(...) → np.ndarray shape (K_MAX, F) float32
decode_set_action(...) → action dict for RLIntermediatePolicy.set_pending_action()
set_obs_shape() → (K_MAX, F)
set_action_shape() → (K_MAX, 3)

Layout constants (row-feature indices)
---------------------------------------
Per-product row (F features):

  ROW_INVENTORY        0  inventory / per-SKU capacity           [0, 1]
  ROW_SALES            1  rolling-5-tick mean sales / capacity   [0, 1]
  ROW_PENDING          2  pending orders / capacity              [0, 1]
  ROW_PRICE_RATIO      3  price / MSRP
  ROW_MSRP_RATIO       4  MSRP / mean MSRP
  ROW_COST_RATIO       5  unit_cost / MSRP                       [0, 1]
  ROW_SUPPLIER_COUNT   6  supplier_count / MAX_SUPPLIER_COUNT    [0, 1]
  ROW_MIN_PRICE        7  min_price / MSRP
  ROW_FILL_RATE        8  mean_fill_rate                         [0, 1]
  --- global features, broadcast onto every active row ---
  ROW_CASH             9  cash / initial_cash
  ROW_TOTAL_INV       10  total_inventory / capacity             [0, 1]
  ROW_SIN             11  sin(2π · step / 360)                   [-1, 1]
  ROW_COS             12  cos(2π · step / 360)                   [-1, 1]
  --- contention aggregates, broadcast onto every active row ---
  ROW_CONTENTION_QTY  13  Σ_proposed_qty / free_space            [0, 1]
  ROW_CONTENTION_COST 14  Σ_estimated_order_cost / cash_budget   [0, 1]
  --- mask channel ---
  ROW_MASK            15  1.0 = active row, 0.0 = padding

F = 16  (total features per row)
K_MAX = 32  (maximum products per episode)

Action layout (3 values per row)
----------------------------------
  col 0: price multiplier raw   [-1, 1] → linear → multiplier in [0.5, 1.5]
  col 1: order-up-to target raw [-1, 1] → order-up-to in lead-time units
  col 2: priority scalar        [-1, 1] → used by the greedy Arbiter variant

Padded rows (mask == 0) produce no orders or prices in the decode output.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Layout version — single source of truth for checkpoint validation
# ---------------------------------------------------------------------------

OBS_LAYOUT_VERSION: int = 1
"""Observation layout version.

Checkpoints validate against this constant (issue 04). Any change to the
feature list (adding, removing, or reordering features) MUST bump this value.
Current layout: F=16 per-product features, K_MAX=32, action shape (K_MAX, 3).
"""

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

K_MAX: int = 32
"""Maximum number of product rows (including padding). K is sampled per episode
from [1, 20]; rows K..K_MAX-1 are zero-padded with mask=0."""

F: int = 16
"""Number of features per product row.

9 per-SKU live features + 4 global broadcast + 2 contention aggregates + 1 mask.
"""

_MAX_SUPPLIER_COUNT: float = 8.0
_MAX_LEAD_TIME: float = 30.0  # unused in current obs row; kept for future

# Action head indices (within each row of the (K_MAX, 3) action tensor)
_ACT_PRICE: int = 0
_ACT_ORDER: int = 1
_ACT_PRIORITY: int = 2

# Order-up-to decoder parameters (same semantics as legacy encoders.py)
_TARGET_CENTRE_LEAD_TIMES: int = 15
_TARGET_HALF_SPAN_LEAD_TIMES: int = 15
_TARGET_MAX_LEAD_TIMES: int = 30

# ---------------------------------------------------------------------------
# Named row-feature index constants
# ---------------------------------------------------------------------------

ROW_INVENTORY: int = 0
ROW_SALES: int = 1
ROW_PENDING: int = 2
ROW_PRICE_RATIO: int = 3
ROW_MSRP_RATIO: int = 4
ROW_COST_RATIO: int = 5
ROW_SUPPLIER_COUNT: int = 6
ROW_MIN_PRICE: int = 7
ROW_FILL_RATE: int = 8
ROW_CASH: int = 9
ROW_TOTAL_INV: int = 10
ROW_SIN: int = 11
ROW_COS: int = 12
ROW_CONTENTION_QTY: int = 13
ROW_CONTENTION_COST: int = 14
ROW_MASK: int = 15  # must equal F - 1


# ---------------------------------------------------------------------------
# Public shape helpers
# ---------------------------------------------------------------------------


def set_obs_shape() -> tuple[int, int]:
    """Return the observation tensor shape ``(K_MAX, F)``."""
    return (K_MAX, F)


def set_action_shape() -> tuple[int, int]:
    """Return the action tensor shape ``(K_MAX, 3)``."""
    return (K_MAX, 3)


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------


def encode_set_observation(
    node: Any,
    market: Any,
    *,
    active_subset: Sequence[str],
    initial_cash: float,
    sales_history: dict[str, deque] | None = None,
    central_table: Any = None,
    supplier_ids_for: dict[str, list[str]] | None = None,
    proposed_quantities: dict[str, float] | None = None,
    unit_prices: dict[str, float] | None = None,
) -> np.ndarray:
    """Encode node/market state into a ``(K_MAX, F)`` float32 observation tensor.

    Parameters
    ----------
    node:
        An ``IntermediateNode`` instance.  Fields consulted: ``inventory``,
        ``list_prices``, ``base_prices``, ``pending`` (nested {sup: {pid: qty}}),
        ``capacity``, ``cash``, ``costs``.
    market:
        A ``Market`` instance.  Fields consulted: ``step``.
    active_subset:
        Ordered list of product ids for active rows (length K ≤ K_MAX).
        Row ``i`` carries the product at ``active_subset[i]``.
    initial_cash:
        Opening episode balance for normalising the cash feature.
    sales_history:
        Optional dict mapping pid → deque of per-tick sales values.
        Used for the rolling-5-tick mean sales feature.
    central_table:
        Optional ``CentralTable`` snapshot.  Populates supplier_count,
        min_price, and mean_fill_rate features.
    supplier_ids_for:
        Optional ``{pid: [supplier_id, ...]}`` mapping to filter central-table
        offers to direct suppliers only.
    proposed_quantities:
        Optional ``{pid: float}`` proposed order quantities from the current
        action decoding pass, used to compute contention aggregates.
        When ``None``, contention features are 0.
    unit_prices:
        Optional ``{pid: float}`` purchase prices for each product, used to
        compute the cash-contention aggregate.  When ``None``, cash contention
        is 0.

    Returns
    -------
    numpy.ndarray
        Shape ``(K_MAX, F)``, dtype ``float32``.
        Active rows 0..K-1 carry live features; padding rows K..K_MAX-1 are
        all-zero (including mask=0).
    """
    active_items: list[str] = list(active_subset)
    K = len(active_items)

    # --- Node field extraction ---
    if hasattr(node, "inventory") and isinstance(node.inventory, dict):
        inventory: dict[str, int] = node.inventory
    else:
        inventory = {}

    if hasattr(node, "list_prices"):
        prices: dict[str, float] = node.list_prices
    else:
        prices = getattr(node, "prices", {})

    base_prices_map: dict[str, float] = getattr(node, "base_prices", {})
    costs_map: dict[str, float] = getattr(node, "costs", {})

    capacity: float = float(getattr(node, "capacity", max(1, K * 100)))
    per_sku_capacity: float = max(1.0, capacity / max(1, K))
    cash_val: float = float(getattr(node, "cash", getattr(node, "balance", 0.0)))

    # Flatten nested pending dict ({supplier: {pid: qty}} → {pid: qty})
    raw_pending = getattr(node, "pending", {})
    if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
        pending: dict[str, int] = {}
        for _sup_pend in raw_pending.values():
            for pid, qty in _sup_pend.items():
                pending[pid] = pending.get(pid, 0) + qty
    else:
        pending = dict(raw_pending) if raw_pending else {}

    # --- Global features ---
    total_inv: float = float(sum(inventory.values()))
    total_pending_all: float = float(sum(pending.values()))
    cash_norm: float = max(1.0, float(initial_cash))
    step: int = int(getattr(market, "step", 0))
    angle: float = 2.0 * math.pi * step / 360.0
    sin_val: float = math.sin(angle)
    cos_val: float = math.cos(angle)

    # --- Contention aggregates ---
    free_space: float = max(0.0, capacity - total_inv - total_pending_all)
    total_proposed: float = 0.0
    total_estimated_cost: float = 0.0
    if proposed_quantities is not None:
        total_proposed = sum(max(0.0, float(v)) for v in proposed_quantities.values())
        if unit_prices is not None:
            total_estimated_cost = sum(
                max(0.0, float(proposed_quantities.get(pid, 0.0))) * float(unit_prices.get(pid, 0.0))
                for pid in proposed_quantities
            )

    contention_qty: float = float(np.clip(
        total_proposed / max(1.0, free_space), 0.0, 1.0
    )) if free_space > 0.0 else (1.0 if total_proposed > 0.0 else 0.0)
    contention_cost: float = float(np.clip(
        total_estimated_cost / max(1.0, cash_val), 0.0, 1.0
    )) if cash_val > 0.0 else (1.0 if total_estimated_cost > 0.0 else 0.0)

    # Mean MSRP across active SKUs (for ROW_MSRP_RATIO normalisation)
    active_msrps: list[float] = [
        float(base_prices_map.get(pid, prices.get(pid, 1.0)))
        for pid in active_items
    ]
    mean_msrp: float = float(np.mean(active_msrps)) if active_msrps else 1.0

    # Build the tensor
    obs = np.zeros((K_MAX, F), dtype=np.float32)

    for i, pid in enumerate(active_items):
        row = obs[i]

        inv = float(inventory.get(pid, 0))
        price = float(prices.get(pid, 1.0))
        msrp = float(base_prices_map.get(pid, prices.get(pid, 1.0)))
        cost = float(costs_map.get(pid, 0.0))
        pend = float(pending.get(pid, 0))

        # 0: inventory / per-SKU capacity
        row[ROW_INVENTORY] = float(np.clip(inv / per_sku_capacity, 0.0, 1.0))

        # 1: rolling-5-tick mean sales / per-SKU capacity
        if sales_history is not None and pid in sales_history:
            hist = sales_history[pid]
            recent = list(hist)[-5:]
            mean_sales = float(np.mean(recent)) if recent else 0.0
        else:
            mean_sales = 0.0
        row[ROW_SALES] = float(np.clip(mean_sales / per_sku_capacity, 0.0, 1.0))

        # 2: pending orders / per-SKU capacity
        row[ROW_PENDING] = float(np.clip(pend / per_sku_capacity, 0.0, 1.0))

        # 3: price / MSRP
        row[ROW_PRICE_RATIO] = float(price / max(1e-9, msrp))

        # 4: MSRP / mean MSRP
        row[ROW_MSRP_RATIO] = float(msrp / max(1e-9, mean_msrp))

        # 5: unit_cost / MSRP
        row[ROW_COST_RATIO] = float(np.clip(cost / max(1e-9, msrp), 0.0, 1.0))

        # 6-8: central-table snapshot (supplier_count, min_price, fill_rate)
        if central_table is not None:
            _offers = central_table.snapshot_for_buyer(pid)
            if supplier_ids_for is not None:
                _allowed = set(supplier_ids_for.get(pid, []))
                _offers = [(sid, o) for sid, o in _offers if sid in _allowed]
            if _offers:
                n_sup = len(_offers)
                _offer_prices = [o.list_price for _, o in _offers]
                _fill_rates = [o.fill_rate_recent for _, o in _offers]
                row[ROW_SUPPLIER_COUNT] = float(
                    np.clip(n_sup / _MAX_SUPPLIER_COUNT, 0.0, 1.0)
                )
                row[ROW_MIN_PRICE] = float(min(_offer_prices) / max(1e-9, msrp))
                row[ROW_FILL_RATE] = float(
                    np.clip(float(np.mean(_fill_rates)), 0.0, 1.0)
                )

        # 9-12: global features (broadcast)
        row[ROW_CASH] = float(cash_val / cash_norm)
        row[ROW_TOTAL_INV] = float(np.clip(total_inv / max(1.0, capacity), 0.0, 1.0))
        row[ROW_SIN] = float(sin_val)
        row[ROW_COS] = float(cos_val)

        # 13-14: contention aggregates (broadcast)
        row[ROW_CONTENTION_QTY] = float(contention_qty)
        row[ROW_CONTENTION_COST] = float(contention_cost)

        # 15: mask = 1.0 for active rows
        row[ROW_MASK] = 1.0

    # Rows K..K_MAX-1 remain zero (mask=0, all features=0) — already initialised.
    return obs


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------


def decode_set_action(
    action: np.ndarray,
    node: Any,
    *,
    active_subset: Sequence[str],
    base_prices: dict[str, float],
    effective_rate: dict[str, float] | None = None,
    supplier_ids: list[str] | None = None,
    target_centre_lead_times: int = _TARGET_CENTRE_LEAD_TIMES,
    target_half_span_lead_times: int = _TARGET_HALF_SPAN_LEAD_TIMES,
    target_max_lead_times: int = _TARGET_MAX_LEAD_TIMES,
) -> dict[str, Any]:
    """Decode a ``(K_MAX, 3)`` action tensor into the graph engine's action dict.

    Only active rows (indices 0..K-1, where K = len(active_subset)) are
    decoded.  Padded rows are ignored; no orders or prices are emitted for them.

    Action column semantics (same as legacy encoders.py):
      col 0  price multiplier: [-1, 1] → linear → [0.5, 1.5] × MSRP
      col 1  order-up-to:      [-1, 1] → target lead-time → order qty via demand rate
      col 2  priority scalar:  passed through to the Arbiter; not decoded here

    Parameters
    ----------
    action:
        Numpy array of shape ``(K_MAX, 3)``.
    node:
        An ``IntermediateNode`` instance.
    active_subset:
        Ordered list of active product ids (length K ≤ K_MAX).
        Row ``i`` corresponds to ``active_subset[i]``.
    base_prices:
        Dict mapping pid → MSRP.
    effective_rate:
        Per-pid demand rate for the order-up-to decoder.  When ``None``,
        all order quantities are 0.
    supplier_ids:
        List of upstream supplier node ids.  When provided, each active product's
        order is attributed to the first matching supplier.  When ``None``,
        a synthetic ``"F_<pid>"`` id is used.
    target_centre_lead_times, target_half_span_lead_times, target_max_lead_times:
        Order-up-to decoder knobs (same defaults as legacy encoders.py).

    Returns
    -------
    dict
        ``{"order": {pid: [(supplier_id, qty), ...]},
           "list_price": {pid: float},
           "min_order_imposed": {pid: int}}``

        Only active product ids appear as keys.
    """
    active_items: list[str] = list(active_subset)

    # Node field extraction (mirrors encode_set_observation)
    if hasattr(node, "inventory") and isinstance(node.inventory, dict):
        inventory: dict[str, int] = node.inventory
    else:
        inventory = {}

    raw_pending = getattr(node, "pending", {})
    if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
        pending: dict[str, int] = {}
        for _sup_pend in raw_pending.values():
            for pid, qty in _sup_pend.items():
                pending[pid] = pending.get(pid, 0) + qty
    else:
        pending = dict(raw_pending) if raw_pending else {}

    capacity: float = float(getattr(node, "capacity", max(1, len(active_items) * 100)))
    total_inv: float = sum(float(v) for v in inventory.values())
    total_pending: float = sum(float(v) for v in pending.values())
    global_free_space: int = max(0, int(capacity - total_inv - total_pending))

    order_dict: dict[str, list[tuple[str, int]]] = {}
    price_dict: dict[str, float] = {}
    min_order_imposed: dict[str, int] = {}
    requested: dict[str, float] = {}
    per_sku_headroom: dict[str, int] = {}

    for i, pid in enumerate(active_items):
        msrp = float(base_prices.get(pid, 1.0))

        # Price head: col 0, [-1, 1] → [0.5, 1.5] * MSRP
        price_raw = float(np.clip(float(action[i, _ACT_PRICE]), -1.0, 1.0))
        price_mult = 0.5 + (price_raw + 1.0) * 0.5
        price_dict[pid] = msrp * price_mult
        min_order_imposed[pid] = 0

        inv = float(inventory.get(pid, 0))
        pend = float(pending.get(pid, 0))
        headroom = max(0, int(capacity - inv - pend))
        per_sku_headroom[pid] = headroom

        if effective_rate is None:
            requested[pid] = 0.0
        else:
            order_raw = float(np.clip(float(action[i, _ACT_ORDER]), -1.0, 1.0))
            target_lt = float(
                np.clip(
                    target_centre_lead_times + order_raw * target_half_span_lead_times,
                    0.0,
                    float(target_max_lead_times),
                )
            )
            rate = float(effective_rate.get(pid, 0.0))
            position = inv + pend
            req = max(0.0, target_lt * rate - position)
            requested[pid] = req

    # Return raw requested quantities; capacity/cash contention is resolved
    # by the Arbiter in the env step path, not here.
    for pid in active_items:
        qty = int(round(requested.get(pid, 0.0)))
        qty = max(0, min(qty, per_sku_headroom.get(pid, 0)))
        if supplier_ids is not None:
            _sup = next(
                (s for s in supplier_ids if s == f"F_{pid}"),
                f"F_{pid}",
            )
            order_dict[pid] = [(_sup, qty)] if qty > 0 else []
        else:
            order_dict[pid] = [(f"F_{pid}", qty)] if qty > 0 else []

    return {
        "order": order_dict,
        "list_price": price_dict,
        "min_order_imposed": min_order_imposed,
    }


__all__ = [
    "K_MAX",
    "F",
    "OBS_LAYOUT_VERSION",
    "ROW_INVENTORY",
    "ROW_SALES",
    "ROW_PENDING",
    "ROW_PRICE_RATIO",
    "ROW_MSRP_RATIO",
    "ROW_COST_RATIO",
    "ROW_SUPPLIER_COUNT",
    "ROW_MIN_PRICE",
    "ROW_FILL_RATE",
    "ROW_CASH",
    "ROW_TOTAL_INV",
    "ROW_SIN",
    "ROW_COS",
    "ROW_CONTENTION_QTY",
    "ROW_CONTENTION_COST",
    "ROW_MASK",
    "encode_set_observation",
    "decode_set_action",
    "set_obs_shape",
    "set_action_shape",
]
