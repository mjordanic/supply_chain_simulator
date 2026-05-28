"""Pure-function observation/action encoders for the RL env.

All functions are deterministic and stateless — no simulator state is
held, no I/O is performed.  The module is the single source of truth for
the observation layout and the action decoding contract.

Public surface
--------------
encode_observation(node, market, registry, central_table, step, slot_perm, K_active)
    → numpy float32 1-D array

decode_action(action_vec, slot_perm, node, K_active, base_prices, supplier_ids)
    → action dict suitable for RLIntermediatePolicy.set_pending_action()

observation_dim(K_active) → int
action_dim(K_active)      → int

Observation layout
------------------
Per-SKU block (repeated K_active times, in slot_perm order):

  0  inventory / per-SKU capacity slice              [0, 1]
  1  recent sales rolling-5-tick mean / capacity     [0, 1]  (clamped)
  2  pending orders / capacity                       [0, 1]  (clamped)
  3  price / MSRP                                    [0, ∞)  (in practice ≈[0.5,2])
  4  MSRP / mean MSRP                                [0, ∞)
  5  unit_cost / MSRP                                [0, 1]
  6-10  lifecycle stage one-hot (5 stages)           {0,1}
  11 ticks_since_activation: log1p(tau)/log1p(360)   [0, 1]  (clamped)
  12 in_season flag                                  {0,1}
  13 clip(inv/rate, 0, max_lt) / max_lt              [0, 1]  demand-units inventory
  14 supplier_count / max_supplier_count             [0, 1]  central-table snapshot
  15 min_price / MSRP                                [0, ∞)  central-table snapshot
  16 mean_lead_time / max_lead_time                  [0, 1]  central-table snapshot
  17 mean_fill_rate                                  [0, 1]  central-table snapshot

N_PER_SKU = 18

Global block (4 values, appended after all SKU blocks):

  0  cash / initial_cash                             [0, ∞)  (in practice ≈[0,∞))
  1  total_inventory / capacity                      [0, 1]
  2  sin(2π · step / 360)                            [-1, 1]
  3  cos(2π · step / 360)                            [-1, 1]

N_GLOBAL = 4

Total: K_active * N_PER_SKU + N_GLOBAL

Action layout
-------------
action_vec ∈ [-1, 1]^(2*K_active)
  first  K_active → price multipliers in [0.5, 1.5]  (linear: 0.5 + (a+1)*0.5)
  second K_active → order fractions in [0, 1]         (linear: (a+1)/2)

The action decoder emits per-supplier splits so the encoder/decoder pair
generalises when an episode runs against a multi-supplier graph.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

import numpy as np

from src.sim.lifecycle_clock import CANONICAL_STAGES

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

N_PER_SKU: int = 18
"""Number of features per active-SKU slot in the observation.

Increased from 14 to 18 to include the 4-feature central-table snapshot
block (slots 14-17): supplier_count, min_price, mean_lead_time, mean_fill_rate.
"""

N_GLOBAL: int = 4
"""Number of global features appended after all per-SKU blocks."""

# Central-table snapshot block constants.
_CT_SLOTS: int = 4  # 4 features per SKU for the central-table snapshot
_MAX_SUPPLIER_COUNT: float = 8.0  # normalisation cap for supplier_count
_MAX_LEAD_TIME: float = 30.0       # normalisation cap for mean_lead_time

# Pre-build a stage → one-hot index mapping from the canonical order.
_STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(CANONICAL_STAGES)}
_N_STAGES: int = len(CANONICAL_STAGES)  # 5

# Log scale normaliser for ticks_since_activation: clamp tau to [0, 360] then
# scale by log1p(360) so the feature lives in [0, 1].
_LOG_TAU_MAX: float = math.log1p(360.0)

# Seasonality labels that map to "in-season" during a tick.
_ALL_SEASON: str = "all_season"


# ---------------------------------------------------------------------------
# Public shape helpers
# ---------------------------------------------------------------------------


def observation_dim(K_active: int) -> int:
    """Return the flat observation vector length for ``K_active`` active SKUs."""
    return K_active * N_PER_SKU + N_GLOBAL


def action_dim(K_active: int) -> int:
    """Return the action vector length for ``K_active`` active SKUs."""
    return 2 * K_active


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------


def encode_observation(
    node: Any,
    market: Any,
    registry: Any,
    step: int,
    slot_perm: Sequence[int],
    K_active: int,
    *,
    central_table: Any = None,
    active_subset: Sequence[str] | None = None,
    supplier_ids_for: dict[str, list[str]] | None = None,
    initial_cash: float | None = None,
    sales_history: dict[str, deque] | None = None,
    effective_rate: dict[str, float] | None = None,
    max_inventory_lt: float = 30.0,
    # Legacy store-compat kwargs (ignored on graph engine but kept so callers
    # that pass them don't break with TypeError).
    store: Any = None,
) -> np.ndarray:
    """Encode the current graph-node/market/registry state into a flat float32 observation.

    Parameters
    ----------
    node:
        An ``IntermediateNode`` instance.  Fields consulted: ``inventory``,
        ``list_prices``, ``pending`` (dict of {supplier_id: {pid: qty}}),
        ``capacity``, ``cash``, ``carried_products``.
    market:
        A ``Market`` instance.  Fields consulted: ``date`` (for month),
        ``params.season_months``, ``step``.
    registry:
        An ``ItemRegistry`` instance.  ``registry.stage(pid)`` is used for
        lifecycle stage; ``registry.seasonality(pid)`` for the in-season flag.
    step:
        Current simulation tick (0-indexed).
    slot_perm:
        A sequence of length K_active mapping slot index → position in
        ``active_subset``.  Slot i carries the SKU at
        ``active_subset[slot_perm[i]]``.
    K_active:
        Number of active SKU slots.  Must match len(slot_perm).
    central_table:
        Optional ``CentralTable`` snapshot.  When supplied, slots 14-17 of
        each SKU block are populated with supplier-side signal.
    active_subset:
        Ordered list of K_active product ids.  When ``None`` and ``node`` is
        an ``IntermediateNode``, falls back to ``sorted(node.carried_products)``.
    supplier_ids_for:
        Optional ``{pid: [supplier_id, ...]}`` mapping giving the set of
        upstream suppliers for each product.  Used to filter central-table
        snapshots to direct suppliers only.  When ``None``, all offers in the
        table are used.
    initial_cash:
        The opening balance for this episode, used to normalise the cash
        feature.  When None, falls back to max(1, node.cash).
    sales_history:
        Optional dict mapping pid → deque of per-tick sales values.
    effective_rate:
        Per-pid demand rate dict, as returned by ``compute_effective_rate``.
    max_inventory_lt:
        Saturation point for the demand-units inventory feature (slot 13).
    store:
        Ignored — kept for backward-compatibility with callers that pass
        the old ``Store`` object.

    Returns
    -------
    numpy.ndarray
        Shape ``(observation_dim(K_active),)``, dtype float32.
    """
    # Support both IntermediateNode (graph) and Store (legacy) objects
    # by normalising field access.
    _node = node if node is not None else store

    # Determine the active items list.
    if active_subset is not None:
        active_items: list[str] = list(active_subset)
    elif hasattr(_node, "active_items"):
        # Legacy Store
        active_items = list(_node.active_items)
    elif hasattr(_node, "carried_products"):
        # IntermediateNode — use a stable sorted order.
        active_items = sorted(_node.carried_products)
    else:
        active_items = []

    n_active = len(active_items)

    # Capacity: IntermediateNode uses .capacity (scalar), Store uses .capacity.
    if hasattr(_node, "capacity") and not callable(getattr(_node, "capacity", None)):
        capacity: float = float(_node.capacity)
    else:
        capacity = float(max(1, n_active * 100))
    per_sku_capacity: float = max(1.0, capacity / max(1, K_active))

    # Prices: IntermediateNode uses list_prices; Store uses prices / base_prices.
    if hasattr(_node, "list_prices"):
        prices: dict[str, float] = _node.list_prices
        base_prices: dict[str, float] = {
            pid: float(getattr(_node, "base_prices", {}).get(pid, prices.get(pid, 1.0)))
            for pid in active_items
        }
    else:
        prices = getattr(_node, "prices", {})
        base_prices = getattr(_node, "base_prices", prices)

    # Unit costs: IntermediateNode doesn't have per-SKU costs; use MSRP as proxy.
    if hasattr(_node, "costs"):
        costs: dict[str, float] = _node.costs
    else:
        costs = {}

    # Inventory: dict for both.
    if hasattr(_node, "inventory") and isinstance(_node.inventory, dict):
        inventory: dict[str, int] = _node.inventory
    else:
        inventory = {}

    # Pending orders: IntermediateNode.pending is {supplier_id: {pid: qty}},
    # Store.pending is {pid: qty}.
    raw_pending = getattr(_node, "pending", {})
    if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
        # IntermediateNode: sum across all suppliers.
        pending: dict[str, int] = {}
        for _sup_pend in raw_pending.values():
            for pid, qty in _sup_pend.items():
                pending[pid] = pending.get(pid, 0) + qty
    else:
        pending = dict(raw_pending) if raw_pending else {}

    # Cash: both have .cash / .balance.
    cash_val = float(getattr(_node, "cash", getattr(_node, "balance", 0.0)))

    # activation_tick: used for freshness; may be on the node or absent.
    activation_tick: dict[str, int] = getattr(_node, "activation_tick", {})

    # Mean MSRP across active SKUs.
    active_base_prices = [base_prices.get(pid, prices.get(pid, 1.0)) for pid in active_items]
    mean_msrp: float = float(np.mean(active_base_prices)) if active_base_prices else 1.0

    # Total inventory for the global block.
    total_inv: float = float(sum(inventory.values()))

    # Cash normaliser.
    if initial_cash is not None and initial_cash > 0:
        cash_norm = float(initial_cash)
    else:
        cash_norm = max(1.0, cash_val)

    # Seasonality lookup.
    season_months: dict[str, list[int]] = getattr(
        getattr(market, "params", None), "season_months", {}
    )
    current_month: int = market.date.month

    obs = np.zeros(observation_dim(K_active), dtype=np.float32)

    # ---------------------------------------------------------------------------
    # Per-SKU block
    # ---------------------------------------------------------------------------
    for slot_idx in range(K_active):
        base_offset = slot_idx * N_PER_SKU
        item_pos = slot_perm[slot_idx] if slot_idx < len(slot_perm) else slot_idx
        if item_pos >= n_active:
            continue
        pid = active_items[item_pos]

        inv = float(inventory.get(pid, 0))
        price = float(prices.get(pid, 1.0))
        msrp = float(base_prices.get(pid, prices.get(pid, 1.0)))
        cost = float(costs.get(pid, 0.0))
        pend = float(pending.get(pid, 0))

        # 0: inventory / per-SKU capacity
        obs[base_offset + 0] = float(np.clip(inv / per_sku_capacity, 0.0, 1.0))

        # 1: rolling-5-tick mean sales / per-sku capacity
        if sales_history is not None and pid in sales_history:
            hist = sales_history[pid]
            recent = list(hist)[-5:]
            mean_sales = float(np.mean(recent)) if recent else 0.0
        else:
            mean_sales = 0.0
        obs[base_offset + 1] = float(np.clip(mean_sales / per_sku_capacity, 0.0, 1.0))

        # 2: pending orders / per-sku capacity
        obs[base_offset + 2] = float(np.clip(pend / per_sku_capacity, 0.0, 1.0))

        # 3: price / MSRP
        obs[base_offset + 3] = float(price / max(1e-9, msrp))

        # 4: MSRP / mean MSRP
        obs[base_offset + 4] = float(msrp / max(1e-9, mean_msrp))

        # 5: unit_cost / MSRP
        obs[base_offset + 5] = float(np.clip(cost / max(1e-9, msrp), 0.0, 1.0))

        # 6-10: lifecycle stage one-hot
        if registry is not None:
            stage = registry.stage(pid)
        else:
            stage = None
        stage_idx = _STAGE_INDEX.get(stage or "introduction", 0)
        obs[base_offset + 6 + stage_idx] = 1.0

        # 11: ticks_since_activation, log-scaled and clipped to [0,1]
        tau = step - activation_tick.get(pid, 0)
        tau = max(0, tau)
        obs[base_offset + 11] = float(
            np.clip(math.log1p(tau) / _LOG_TAU_MAX, 0.0, 1.0)
        )

        # 12: in_season flag
        if registry is not None:
            seasonality = registry.seasonality(pid)
        else:
            seasonality = _ALL_SEASON
        if seasonality == _ALL_SEASON or seasonality is None:
            in_season = 1.0
        else:
            in_season = 1.0 if current_month in season_months.get(seasonality, []) else 0.0
        obs[base_offset + 12] = float(in_season)

        # 13: demand-units inventory feature
        if effective_rate is not None:
            rate = float(effective_rate.get(pid, 0.0))
            _epsilon = 1e-9
            inv_lt = float(np.clip(inv / max(rate, _epsilon), 0.0, max_inventory_lt))
            obs[base_offset + 13] = inv_lt / max(max_inventory_lt, _epsilon)

        # 14-17: central_table_snapshot[product_slot] block
        # supplier_count, min_price, mean_lead_time, mean_fill_rate
        if central_table is not None:
            _offers = central_table.snapshot_for_buyer(pid)
            if supplier_ids_for is not None:
                _allowed = set(supplier_ids_for.get(pid, []))
                _offers = [(sid, o) for sid, o in _offers if sid in _allowed]
            if _offers:
                n_sup = len(_offers)
                _prices = [o.list_price for _, o in _offers]
                _fill_rates = [o.fill_rate_recent for _, o in _offers]
                min_p = min(_prices)
                mean_fr = float(np.mean(_fill_rates)) if _fill_rates else 1.0
                # mean_lead_time: read from central_table if available,
                # else use 0.0 (not available in Offer; left as 0 for now).
                mean_lt = 0.0
                # Normalise.
                obs[base_offset + 14] = float(
                    np.clip(n_sup / _MAX_SUPPLIER_COUNT, 0.0, 1.0)
                )
                obs[base_offset + 15] = float(min_p / max(1e-9, msrp))
                obs[base_offset + 16] = float(
                    np.clip(mean_lt / max(_MAX_LEAD_TIME, 1e-9), 0.0, 1.0)
                )
                obs[base_offset + 17] = float(np.clip(mean_fr, 0.0, 1.0))
            # else: all four central-table slots remain 0.0

    # ---------------------------------------------------------------------------
    # Global block (appended after all per-SKU blocks)
    # ---------------------------------------------------------------------------
    global_offset = K_active * N_PER_SKU
    obs[global_offset + 0] = float(cash_val / cash_norm)
    obs[global_offset + 1] = float(np.clip(total_inv / max(1.0, capacity), 0.0, 1.0))
    angle = 2.0 * math.pi * step / 360.0
    obs[global_offset + 2] = float(math.sin(angle))
    obs[global_offset + 3] = float(math.cos(angle))

    return obs


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------


def decode_action(
    action_vec: np.ndarray,
    slot_perm: Sequence[int],
    node: Any,
    K_active: int,
    base_prices: dict[str, float],
    *,
    supplier_ids: list[str] | None = None,
    active_subset: Sequence[str] | None = None,
    effective_rate: dict[str, float] | None = None,
    target_centre_lead_times: int = 15,
    target_half_span_lead_times: int = 15,
    target_max_lead_times: int = 30,
    # Legacy store-compat kwarg (ignored on graph engine).
    store: Any = None,
) -> dict[str, Any]:
    """Decode a continuous action vector into the graph engine's action dict.

    Order-up-to decoder: the order half of the action vector selects an
    inventory target expressed in lead-times of expected demand.  The
    final order is emitted as ``{pid: [(supplier_id, qty), ...]}`` to
    match the ``IntermediatePolicy.decide()`` return format.

    The decoder emits per-supplier splits.  In the degenerate single-
    supplier case (the standard RL episode), each pid gets one order line
    ``[(supplier_id, qty)]`` of length 1.  When ``supplier_ids`` carries
    multiple entries, the total qty is split equally across suppliers that
    have stock.

    Parameters
    ----------
    action_vec:
        Numpy array of shape ``(2*K_active,)`` with values in ``[-1, 1]``.
        First K_active → price multipliers.  Second K_active → order scalars.
    slot_perm:
        Slot index → position in ``active_subset`` mapping.
    node:
        An ``IntermediateNode`` instance (or Store for backward-compat).
    K_active:
        Number of active SKU slots.
    base_prices:
        Dict mapping pid → MSRP.  Price decisions are applied as a
        multiplier on these base prices.
    supplier_ids:
        List of upstream supplier node ids (e.g. ``["F_P0001"]``).  When
        ``None``, defaults to a single synthetic supplier id ``"F_<pid>"``
        per product.
    active_subset:
        Ordered list of K_active product ids.  When ``None``, read from
        ``node.carried_products`` (sorted).
    effective_rate:
        Per-pid demand rate used to compute the order-up-to target.
    target_centre_lead_times, target_half_span_lead_times, target_max_lead_times:
        Order decoder parameters.

    Returns
    -------
    dict
        ``{"order": {pid: [(supplier_id, qty), ...]},
           "list_price": {pid: float},
           "min_order_imposed": {pid: int}}``
    """
    _node = node if node is not None else store

    # Determine active items.
    if active_subset is not None:
        active_items: list[str] = list(active_subset)
    elif hasattr(_node, "active_items"):
        active_items = list(_node.active_items)
    elif hasattr(_node, "carried_products"):
        active_items = sorted(_node.carried_products)
    else:
        active_items = []

    n_active = len(active_items)

    if hasattr(_node, "capacity") and not callable(getattr(_node, "capacity", None)):
        capacity: float = float(_node.capacity)
    else:
        capacity = float(max(1, n_active * 100))

    # Pending: normalise IntermediateNode's nested pending dict.
    raw_pending = getattr(_node, "pending", {})
    if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
        pending: dict[str, int] = {}
        for _sup_pend in raw_pending.values():
            for pid, qty in _sup_pend.items():
                pending[pid] = pending.get(pid, 0) + qty
    else:
        pending = dict(raw_pending) if raw_pending else {}

    if hasattr(_node, "inventory") and isinstance(_node.inventory, dict):
        inventory: dict[str, int] = _node.inventory
    else:
        inventory = {}

    total_inv: float = sum(float(v) for v in inventory.values())
    total_pending: float = sum(float(v) for v in pending.values())
    global_free_space: int = max(0, int(capacity - total_inv - total_pending))

    order_dict: dict[str, list[tuple[str, int]]] = {}
    price_dict: dict[str, float] = {}
    min_order_imposed: dict[str, int] = {}

    requested: dict[str, float] = {}
    per_sku_headroom: dict[str, int] = {}

    for slot_idx in range(K_active):
        item_pos = slot_perm[slot_idx] if slot_idx < len(slot_perm) else slot_idx
        if item_pos >= n_active:
            continue
        pid = active_items[item_pos]
        msrp = float(base_prices.get(pid, 1.0))

        # Price half: linear map [-1,1] → [0.5, 1.5] multiplier on MSRP.
        price_raw = float(np.clip(float(action_vec[slot_idx]), -1.0, 1.0))
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
            order_raw = float(np.clip(float(action_vec[K_active + slot_idx]), -1.0, 1.0))
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

    # Two-pass fair-share allocation.
    allocated = fair_share_allocate(requested, per_sku_headroom, global_free_space)

    # Build per-supplier splits.
    for pid in requested:
        qty = allocated.get(pid, 0)
        if supplier_ids is not None:
            # Use the first available supplier (degenerate: 1 supplier per product
            # in the standard RL episode). A future multi-supplier policy would
            # split across all supplier_ids.
            # For the standard single-factory case: supplier_id = "F_<pid>".
            _sup = next(
                (s for s in supplier_ids if pid in s or True),
                f"F_{pid}",
            )
            order_dict[pid] = [(_sup, qty)] if qty > 0 else []
        else:
            # Fallback: synthetic supplier id per product.
            order_dict[pid] = [(f"F_{pid}", qty)] if qty > 0 else []

    return {
        "order": order_dict,
        "list_price": price_dict,
        "min_order_imposed": min_order_imposed,
    }


# ---------------------------------------------------------------------------
# Rate primitive
# ---------------------------------------------------------------------------


def compute_effective_rate(
    sales_history: dict[str, deque],
    base_demand_prior: float,
) -> dict[str, float]:
    """Return effective rate per pid: max(rolling-5-mean of sales[pid], prior).

    Empty or missing history for a pid contributes a value equal to the prior.
    Partial windows (1-4 entries) use the partial-window mean, still clamped
    to the prior as a floor.

    Parameters
    ----------
    sales_history:
        Dict mapping pid → deque of per-tick sales values.
    base_demand_prior:
        Floor rate applied when the empirical rolling mean falls below it.

    Returns
    -------
    dict[str, float]
        One entry per key in ``sales_history``.
    """
    result: dict[str, float] = {}
    for pid, hist in sales_history.items():
        if not hist:
            result[pid] = base_demand_prior
        else:
            recent = list(hist)[-5:]
            empirical = sum(recent) / len(recent)
            result[pid] = max(empirical, base_demand_prior)
    return result


# ---------------------------------------------------------------------------
# Capacity allocator
# ---------------------------------------------------------------------------


def fair_share_allocate(
    requested: dict[str, float],
    per_sku_headroom: dict[str, int],
    global_free_space: int,
) -> dict[str, int]:
    """Two-pass capacity allocator.

    Pass 1: cap each request at the per-SKU physical headroom
            (capped[pid] = min(requested[pid], per_sku_headroom[pid])).
    Pass 2: if sum(capped) > global_free_space, scale every capped
            value proportionally by global_free_space / sum(capped).
    Return integer quantities (truncate).

    Parameters
    ----------
    requested:
        Dict mapping pid → desired float quantity for each SKU.
    per_sku_headroom:
        Dict mapping pid → per-SKU physical headroom.
    global_free_space:
        Total capacity available across all SKUs combined.

    Returns
    -------
    dict[str, int]
        One entry per key in ``requested`` with integer allocated quantities.
    """
    if not requested:
        return {}

    capped: dict[str, float] = {
        pid: min(float(qty), float(per_sku_headroom.get(pid, 0)))
        for pid, qty in requested.items()
    }

    total_capped = sum(capped.values())
    if total_capped > global_free_space:
        if total_capped <= 0:
            scale = 0.0
        else:
            scale = global_free_space / total_capped
        return {pid: int(v * scale) for pid, v in capped.items()}

    return {pid: int(v) for pid, v in capped.items()}


__all__ = [
    "encode_observation",
    "decode_action",
    "observation_dim",
    "action_dim",
    "compute_effective_rate",
    "fair_share_allocate",
    "N_PER_SKU",
    "N_GLOBAL",
]
