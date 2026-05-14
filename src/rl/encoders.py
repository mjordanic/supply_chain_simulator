"""Pure-function observation/action encoders for the RL env.

All functions are deterministic and stateless — no simulator state is
held, no I/O is performed.  The module is the single source of truth for
the observation layout and the action decoding contract.

Public surface
--------------
encode_observation(store, market, registry, step, slot_perm, K_active)
    → numpy float32 1-D array

decode_action(action_vec, slot_perm, store, K_active, base_prices)
    → action dict suitable for Policy.decide()

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

N_PER_SKU = 13

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

The inverse slot_perm maps slot indices back to active SKU product ids.
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

N_PER_SKU: int = 13
"""Number of features per active-SKU slot in the observation."""

N_GLOBAL: int = 4
"""Number of global features appended after all per-SKU blocks."""

# Pre-build a stage → one-hot index mapping from the canonical order.
_STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(CANONICAL_STAGES)}
_N_STAGES: int = len(CANONICAL_STAGES)  # 5

# Log scale normaliser for ticks_since_activation: clamp tau to [0, 360] then
# scale by log1p(360) so the feature lives in [0, 1].
_LOG_TAU_MAX: float = math.log1p(360.0)

# Seasonality labels that map to "in-season" during a tick.  The market's
# ``season_factor`` uses these to apply the peak multiplier; we mirror the
# logic here by treating the current month as in-season if the product's
# seasonality label appears in MarketParams.season_months for that month.
# ``all_season`` products are always in-season.
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
    store: Any,
    market: Any,
    registry: Any,
    step: int,
    slot_perm: Sequence[int],
    K_active: int,
    *,
    initial_cash: float | None = None,
    sales_history: dict[str, deque] | None = None,
) -> np.ndarray:
    """Encode the current simulator state into a flat float32 observation.

    Parameters
    ----------
    store:
        A ``Store`` instance.  Fields consulted: ``active_items``,
        ``inventory``, ``prices``, ``base_prices``, ``costs``, ``pending``,
        ``capacity``, ``balance``, ``activation_tick``.
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
        ``store.active_items``.  Slot i carries the SKU at
        ``store.active_items[slot_perm[i]]``.
    K_active:
        Number of active SKU slots.  Must match len(slot_perm).
    initial_cash:
        The opening balance for this episode, used to normalise the cash
        feature.  When None (e.g. during early ticks before the env wires
        it up), falls back to max(1, store.balance).
    sales_history:
        Optional dict mapping pid → deque of per-tick sales values.  The
        rolling-5-tick mean feature is computed from the last 5 entries.
        When None or the product has no history, the feature is 0.

    Returns
    -------
    numpy.ndarray
        Shape ``(observation_dim(K_active),)``, dtype float32.
    """
    active_items: list[str] = list(store.active_items)
    n_active = len(active_items)
    capacity: float = float(store.capacity)
    per_sku_capacity: float = max(1.0, capacity / max(1, K_active))

    # Mean MSRP across active SKUs — needed for the MSRP/mean-MSRP feature.
    base_prices: dict[str, float] = store.base_prices
    active_base_prices = [base_prices.get(pid, 1.0) for pid in active_items]
    mean_msrp: float = float(np.mean(active_base_prices)) if active_base_prices else 1.0

    # Normalise total inventory for the global block.
    total_inv: float = float(sum(store.inventory.values()))

    # Cash normaliser.
    if initial_cash is not None and initial_cash > 0:
        cash_norm = float(initial_cash)
    else:
        cash_norm = max(1.0, float(store.balance))

    # Seasonality lookup: which months are in-season for each seasonality label.
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
        # Map slot → active_items index via slot_perm.
        item_pos = slot_perm[slot_idx] if slot_idx < len(slot_perm) else slot_idx
        if item_pos >= n_active:
            # Slot beyond the actual active count — leave as zero padding.
            continue
        pid = active_items[item_pos]

        inv = float(store.inventory.get(pid, 0))
        price = float(store.prices.get(pid, 1.0))
        msrp = float(base_prices.get(pid, 1.0))
        cost = float(store.costs.get(pid, 0.0))
        pending = float(store.pending.get(pid, 0))

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
        obs[base_offset + 2] = float(np.clip(pending / per_sku_capacity, 0.0, 1.0))

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
        tau = step - store.activation_tick.get(pid, 0)
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

    # ---------------------------------------------------------------------------
    # Global block (appended after all per-SKU blocks)
    # ---------------------------------------------------------------------------
    global_offset = K_active * N_PER_SKU
    obs[global_offset + 0] = float(store.balance / cash_norm)
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
    store: Any,
    K_active: int,
    base_prices: dict[str, float],
    *,
    effective_rate: dict[str, float] | None = None,
    target_centre_lead_times: int = 15,
    target_half_span_lead_times: int = 15,
    target_max_lead_times: int = 30,
) -> dict[str, Any]:
    """Decode a continuous action vector into the simulator's action dict.

    Order-up-to decoder: the order half of the action vector selects an
    inventory target expressed in lead-times of expected demand.  The
    final quantity per SKU is:

        target_lt = clip(target_centre + order_raw * target_half_span,
                         0, target_max)
        requested  = max(0, target_lt * effective_rate[pid]
                            − (inventory[pid] + pending[pid]))
        qty        = fair_share_allocate(requested, per_sku_headroom,
                                         global_free_space)[pid]

    When ``effective_rate is None`` (backward-compat path for unit tests
    that don't supply a rate), every active SKU contributes zero requested
    quantity.

    Parameters
    ----------
    action_vec:
        Numpy array of shape ``(2*K_active,)`` with values in ``[-1, 1]``.
        First K_active → price multipliers.  Second K_active → order scalars.
    slot_perm:
        Slot index → position in ``store.active_items`` mapping (same as
        used in encode_observation).
    store:
        A ``Store`` instance.  ``store.active_items``, ``store.inventory``,
        ``store.pending``, ``store.capacity`` are consulted.
    K_active:
        Number of active SKU slots.
    base_prices:
        Dict mapping pid → MSRP.  Price decisions are applied as a
        multiplier on these base prices.
    effective_rate:
        Per-pid demand rate used to compute the order-up-to target.
        When ``None``, all order quantities are forced to zero.
    target_centre_lead_times:
        Inventory target (in lead-times) at ``order_raw = 0``.
    target_half_span_lead_times:
        Width of the action range around the centre.
    target_max_lead_times:
        Upper clip for the target.

    Returns
    -------
    dict
        ``{"order": {pid: int}, "price": {pid: float},
           "activate": [], "deactivate": [], "promotions": {}}``
    """
    active_items: list[str] = list(store.active_items)
    n_active = len(active_items)
    capacity: float = float(store.capacity)

    total_inv: float = sum(float(v) for v in store.inventory.values())
    total_pending: float = sum(float(v) for v in store.pending.values())
    global_free_space: int = max(0, int(capacity - total_inv - total_pending))

    order_dict: dict[str, int] = {}
    price_dict: dict[str, float] = {}

    # Accumulate per-SKU requested quantities (float) before allocation.
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
        price_mult = 0.5 + (price_raw + 1.0) * 0.5  # maps to [0.5, 1.5]
        price_dict[pid] = msrp * price_mult

        # Per-SKU shelf headroom (physical capacity available per SKU).
        inv = float(store.inventory.get(pid, 0))
        pend = float(store.pending.get(pid, 0))
        headroom = max(0, int(capacity - inv - pend))
        per_sku_headroom[pid] = headroom

        if effective_rate is None:
            # Backward-compat: no rate signal → zero order.
            requested[pid] = 0.0
        else:
            # Order half: clip(centre + raw*span, 0, max) lead-times.
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

    for pid in requested:
        order_dict[pid] = allocated.get(pid, 0)

    return {
        "order": order_dict,
        "price": price_dict,
        "activate": [],
        "deactivate": [],
        "promotions": {},
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
        Dict mapping pid → deque of per-tick sales values.  The env populates
        this with every catalog pid at reset (see RLEnv.reset).
    base_demand_prior:
        Floor rate applied when the empirical rolling mean falls below it.
        Set to 0.0 to disable the floor.

    Returns
    -------
    dict[str, float]
        One entry per key in ``sales_history``.
    """
    result: dict[str, float] = {}
    for pid, hist in sales_history.items():
        if not hist:
            # Empty history — prior is the only signal.
            result[pid] = base_demand_prior
        else:
            # Use the last 5 entries (partial window is fine).
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
    Return integer quantities (truncate, do not round up — under-allocation
    is safer than over-allocation against a hard capacity constraint).

    Parameters
    ----------
    requested:
        Dict mapping pid → desired float quantity for each SKU.
    per_sku_headroom:
        Dict mapping pid → per-SKU physical headroom (hard ceiling per SKU).
    global_free_space:
        Total capacity available across all SKUs combined.

    Returns
    -------
    dict[str, int]
        One entry per key in ``requested`` with integer allocated quantities.
        Guaranteed: ``output[pid] <= per_sku_headroom[pid]`` and
        ``sum(output.values()) <= global_free_space``.
    """
    if not requested:
        return {}

    # Pass 1: cap each request at per-SKU headroom.
    capped: dict[str, float] = {
        pid: min(float(qty), float(per_sku_headroom.get(pid, 0)))
        for pid, qty in requested.items()
    }

    # Pass 2: if total capped exceeds global free space, scale proportionally.
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
