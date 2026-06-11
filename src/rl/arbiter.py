"""Arbiter: deterministic joint-proposal reconciler for variable-K RL.

The Arbiter is a pure, torch-free module that projects a joint per-product
order proposal onto the feasible set defined by:

  - per-SKU capacity headroom (one cap per product),
  - global free space (shared capacity across all products), and
  - a cash budget (node cash × configured fraction, costed at central-table
    offer prices).

It is the single place within-tick resource contention is resolved,
replacing the engine's implicit pid-iteration-order cash starvation with a
deterministic, learnable rule.

Two allocation modes are provided behind a single ``mode`` flag:

proportional (default)
    Extends the two-pass fair-share allocator semantics already present in
    ``src/rl/encoders.fair_share_allocate`` with cash as a third binding
    resource. All proposals are scaled by the binding feasibility ratio; no
    product is starved entirely by this mode.

greedy
    Fills proposals in descending priority order until a resource
    (per-SKU headroom, global free space, or cash) is exhausted. Can starve
    low-priority products entirely.

Public API
----------
- ``allocate`` — the single entry point; takes proposals + constraints,
  returns integer allocations per product.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

__all__ = ["allocate", "arbitrate_orders"]

_Mode = Literal["proportional", "greedy"]


def allocate(
    proposed: dict[str, float],
    per_sku_headroom: dict[str, int],
    global_free_space: int,
    cash_budget: float,
    unit_prices: dict[str, float],
    priorities: dict[str, float] | None = None,
    *,
    mode: _Mode = "proportional",
) -> dict[str, int]:
    """Project per-product order proposals onto the feasible set.

    Parameters
    ----------
    proposed:
        Dict mapping product id → proposed order quantity (float ≥ 0).
        Products absent from this dict are not allocated.
    per_sku_headroom:
        Dict mapping product id → maximum units that can be added to that
        SKU slot (capacity headroom for this product alone).
    global_free_space:
        Total units of capacity available across *all* products combined.
    cash_budget:
        Total cash available to spend in this tick.
    unit_prices:
        Dict mapping product id → unit purchase price from the central table.
        Products missing from this dict default to price 0.0, which means
        "no cash constraint" for that product.
    priorities:
        Optional dict mapping product id → priority scalar (higher = served
        first). Used only in ``greedy`` mode; ignored in ``proportional``
        mode. Missing products default to 0.0.
    mode:
        Allocation mode: ``"proportional"`` (default) or ``"greedy"``.

    Returns
    -------
    dict[str, int]
        Integer allocations, one entry per key in ``proposed``.
        All values are ≥ 0 and satisfy the feasibility constraints.

    Notes
    -----
    - An empty ``proposed`` dict returns an empty dict immediately.
    - A zero or negative ``cash_budget`` or ``global_free_space`` causes all
      allocations to be 0 (zero-budget / zero-space edge case).
    - Zero proposals return zero allocations without error.
    """
    if not proposed:
        return {}

    if mode == "proportional":
        return _proportional(
            proposed, per_sku_headroom, global_free_space, cash_budget, unit_prices
        )
    elif mode == "greedy":
        return _greedy(
            proposed,
            per_sku_headroom,
            global_free_space,
            cash_budget,
            unit_prices,
            priorities or {},
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'proportional' or 'greedy'.")


# ---------------------------------------------------------------------------
# Shared arbitration helper
# ---------------------------------------------------------------------------


class _NodeSnapshot(Protocol):
    """Structural interface for the subset of node state arbitrate_orders reads."""

    inventory: dict[str, Any]
    pending: dict[str, Any]
    capacity: Any
    cash: Any


class _ArbiterConfig(Protocol):
    """Structural interface for the config fields arbitrate_orders reads."""

    arbiter_mode: str
    cash_budget_fraction: float


def arbitrate_orders(
    raw_orders: dict[str, list[tuple[str, float]]],
    active_subset: tuple[str, ...],
    node: _NodeSnapshot,
    unit_prices: dict[str, float],
    priorities: dict[str, float],
    config: _ArbiterConfig,
) -> dict[str, list[tuple[str, int]]]:
    """Resolve per-product order contention and return an arbitrated order dict.

    This is the single code path for arbitration dynamics. Both the training
    env (``RLEnv.step``) and ``RLNodePolicy`` call this function.

    Parameters
    ----------
    raw_orders:
        Per-product order lists as produced by ``decode_set_action``; each
        value is a list of ``(supplier_id, qty)`` tuples. The aggregate
        per-product quantity is summed to form proposals.
    active_subset:
        Ordered tuple of active product ids for the current episode tick.
    node:
        A duck-typed node object exposing ``.inventory``, ``.pending``,
        ``.capacity``, and ``.cash``.
    unit_prices:
        Dict mapping product id → unit purchase price (central-table offer).
    priorities:
        Dict mapping product id → priority scalar (only used in greedy mode).
    config:
        An object exposing ``.arbiter_mode`` and ``.cash_budget_fraction``.

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        Arbitrated order dict: each surviving product has one order line
        ``[(f"F_{pid}", qty)]`` with integer qty > 0; products with zero
        allocation return an empty list ``[]``.
    """
    # Sum per-product proposals from the raw order lists.
    proposed: dict[str, float] = {}
    for pid in active_subset:
        lines = raw_orders.get(pid, [])
        proposed[pid] = float(sum(qty for _, qty in lines))

    # Flatten pending (may be nested by supplier or flat by pid).
    inventory = dict(node.inventory)
    raw_pending = getattr(node, "pending", {})
    if raw_pending and isinstance(next(iter(raw_pending.values()), None), dict):
        pending: dict[str, int] = {}
        for sup_pend in raw_pending.values():
            for pid, qty in sup_pend.items():
                pending[pid] = pending.get(pid, 0) + qty
    else:
        pending = dict(raw_pending) if raw_pending else {}

    # Compute global free space and per-SKU headroom.
    capacity_val = float(getattr(node, "capacity", max(1, len(active_subset) * 100)))
    total_inv = sum(float(v) for v in inventory.values())
    total_pend = sum(float(v) for v in pending.values())
    global_free_space = max(0, int(capacity_val - total_inv - total_pend))

    per_sku_headroom: dict[str, int] = {}
    for pid in active_subset:
        inv_pid = float(inventory.get(pid, 0))
        pend_pid = float(pending.get(pid, 0))
        per_sku_headroom[pid] = max(0, int(capacity_val - inv_pid - pend_pid))

    cash_budget = float(node.cash) * config.cash_budget_fraction

    allocated = allocate(
        proposed=proposed,
        per_sku_headroom=per_sku_headroom,
        global_free_space=global_free_space,
        cash_budget=cash_budget,
        unit_prices=unit_prices,
        priorities=priorities,
        mode=config.arbiter_mode,  # type: ignore[arg-type]
    )

    # Build arbitrated order dict: one line per surviving product.
    order_dict: dict[str, list[tuple[str, int]]] = {}
    for pid in active_subset:
        qty = allocated.get(pid, 0)
        order_dict[pid] = [(f"F_{pid}", qty)] if qty > 0 else []

    return order_dict


# ---------------------------------------------------------------------------
# Proportional fair-share
# ---------------------------------------------------------------------------

def _proportional(
    proposed: dict[str, float],
    per_sku_headroom: dict[str, int],
    global_free_space: int,
    cash_budget: float,
    unit_prices: dict[str, float],
) -> dict[str, int]:
    """Three-pass proportional fair-share allocator.

    Pass 1: cap each proposal at the per-SKU headroom.
    Pass 2: if the sum of capped proposals exceeds global_free_space, scale
            all proportionally by global_free_space / sum.
    Pass 3: if the total cash cost exceeds cash_budget, scale all
            proportionally by cash_budget / total_cost, using the scaled
            quantities after pass 2.

    The binding constraint is the one that imposes the smallest scale factor
    (≤ 1.0). Trucate to integers at the end.
    """
    pids = list(proposed.keys())

    # Edge case: non-positive global capacity means nothing can be allocated.
    if global_free_space <= 0:
        return {pid: 0 for pid in pids}

    # Pass 1: per-SKU cap.
    capped: dict[str, float] = {}
    for pid in pids:
        headroom = float(per_sku_headroom.get(pid, 0))
        capped[pid] = min(float(proposed[pid]), max(0.0, headroom))

    # Pass 2: global space scale.
    total_capped = sum(capped.values())
    if total_capped > global_free_space:
        space_scale = global_free_space / total_capped if total_capped > 0 else 0.0
    else:
        space_scale = 1.0

    after_space: dict[str, float] = {pid: v * space_scale for pid, v in capped.items()}

    # Pass 3: cash budget scale.
    total_cost = sum(
        after_space[pid] * float(unit_prices.get(pid, 0.0)) for pid in pids
    )
    if cash_budget <= 0.0:
        cash_scale = 0.0
    elif total_cost > cash_budget and total_cost > 0.0:
        cash_scale = cash_budget / total_cost
    else:
        cash_scale = 1.0

    return {pid: int(v * cash_scale) for pid, v in after_space.items()}


# ---------------------------------------------------------------------------
# Priority greedy
# ---------------------------------------------------------------------------

def _greedy(
    proposed: dict[str, float],
    per_sku_headroom: dict[str, int],
    global_free_space: int,
    cash_budget: float,
    unit_prices: dict[str, float],
    priorities: dict[str, float],
) -> dict[str, int]:
    """Priority-greedy allocator.

    Products are processed in descending priority order (ties broken by pid
    for determinism). For each product, allocate the maximum feasible
    quantity given:

      - the per-SKU headroom remaining for that product,
      - the remaining global free space, and
      - the remaining cash budget.

    Products with lower priority may receive zero if any resource is
    exhausted by higher-priority products.
    """
    pids = list(proposed.keys())
    result: dict[str, int] = {pid: 0 for pid in pids}

    remaining_space = max(0, int(global_free_space))
    remaining_cash = max(0.0, float(cash_budget))

    # Sort by priority descending, then by pid ascending for determinism.
    sorted_pids = sorted(pids, key=lambda p: (-float(priorities.get(p, 0.0)), p))

    for pid in sorted_pids:
        if remaining_space <= 0 or remaining_cash < 0:
            # Resources exhausted; remaining products get 0.
            break

        headroom = max(0, int(per_sku_headroom.get(pid, 0)))
        qty_wanted = max(0.0, float(proposed[pid]))

        # Cap by per-SKU headroom and global space.
        qty = min(qty_wanted, float(headroom), float(remaining_space))

        # Cap by cash budget.
        price = float(unit_prices.get(pid, 0.0))
        if price > 0.0 and remaining_cash >= 0.0:
            affordable = remaining_cash / price
            qty = min(qty, affordable)
        # If price is 0, no cash constraint for this product.

        qty_int = int(qty)
        result[pid] = qty_int

        remaining_space -= qty_int
        cost = qty_int * price
        remaining_cash -= cost

    return result
