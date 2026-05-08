"""Pure ``init_store_state`` extraction (issue 05).

``Store.__init__`` previously inlined the whole step-0 setup: sample
``Distribution``-typed template fields against a per-instance
``init_rng``, pick the active SKU set, and allocate the initial stock
budget. The bit-identity contract — same ``(template, init_seed,
catalog)`` ⇒ identical step-0 state — was an *emergent* property of
that constructor.

This module lifts the deterministic step-0 resolution into one pure
function so the contract is testable as a single explicit assertion.
``Store.__init__`` delegates here; subsequent slices (06, 07, 08) plug
new authoring fields (``init_active_products``, ``init_freshness``,
``init_stock_share``) into the same seam.

Sampling order is **load-bearing**: any change shifts the ``init_rng``
draw sequence and breaks bit-identity across runs. The order below
matches the pre-extraction ``Store.__init__`` exactly: capacity →
init_balance → init_stock_pct → delivery_lag → holding_rate →
order_fee → init_active_count → ``init_rng.sample`` for the active SKU
pick.

Issue 08 adds an optional ``item_registry`` argument. When set, the
initial-stock budget is distributed across active SKUs proportionally
to their resolved ``init_stock_share`` weights (one resolved scalar per
``Ware``, shared across stores). When unset (T4 unit-test path), the
legacy even-split allocation is used. With the catalog-wide default of
``1.0`` and no per-``Ware`` overrides, weighted allocation collapses
to the legacy even-split, preserving bit-identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any

from src.sim.distributions import Distribution
from src.sim.scenario import StoreTemplate, Ware

if TYPE_CHECKING:  # avoid runtime cycle: ItemRegistry imports scenario.
    from src.sim.item_registry import ItemRegistry


def _maybe_sample(value: Any, rng: Random) -> Any:
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


@dataclass
class InitialStoreState:
    """Resolved step-0 state for one ``Store``.

    ``Distribution``-typed template fields have been sampled against the
    per-instance ``init_rng`` once, here. ``inventory`` is keyed by every
    catalog ``product_id`` (inactive Wares carry stock 0) so callers can
    iterate it in catalog order without consulting the catalog again.
    ``activation_tick`` is the seed map handed to ``Store``: empty under
    ``init_freshness="baseline"`` (initial active SKUs skip the hype
    window — ``Store.freshness_multiplier`` returns ``1.0`` for any pid
    not in the dict); ``{pid: 0 for pid in active_ids}`` under
    ``init_freshness="fresh"`` (grand-opening, full hype at step 0).
    """

    region: str
    capacity: float
    balance: float
    init_stock_pct: float
    delivery_lag: float
    holding_rate: float
    order_fee: float
    active_ids: list[str]
    inventory: dict[str, int]
    activation_tick: dict[str, int] = field(default_factory=dict)


def init_store_state(
    template: StoreTemplate,
    init_seed: int,
    catalog: list[Ware],
    item_registry: "ItemRegistry | None" = None,
) -> InitialStoreState:
    """Resolve a ``Store``'s step-0 state from ``(template, init_seed, catalog)``.

    The only RNG used is ``Random(init_seed)`` constructed here. Same
    inputs ⇒ identical output (bit-identity). ``item_registry`` is the
    issue-08 seam: when supplied, initial-stock allocation reads each
    active ``Ware``'s resolved ``init_stock_share`` as a normalised
    weight; when ``None`` (T4 unit-test path), the legacy even-split is
    used instead.
    """
    init_rng = Random(init_seed)

    region: str = template.region
    capacity = float(_maybe_sample(template.capacity, init_rng))
    balance = float(_maybe_sample(template.init_balance, init_rng))
    init_stock_pct = float(_maybe_sample(template.init_stock_pct, init_rng))
    delivery_lag = float(_maybe_sample(template.delivery_lag, init_rng))
    holding_rate = float(_maybe_sample(template.holding_rate, init_rng))
    order_fee = float(_maybe_sample(template.order_fee, init_rng))

    product_ids = [w.product_id for w in catalog]
    if template.init_active_products is not None:
        # Explicit per-template roster (issue 06). Skip both the
        # ``init_active_count`` sample and the ``init_rng.sample`` for
        # the active pick — the template authors which SKUs are active
        # literally. Validate against the catalog so a typo'd id fails
        # loudly rather than silently disappearing.
        catalog_ids = set(product_ids)
        unknown = [
            pid for pid in template.init_active_products if pid not in catalog_ids
        ]
        if unknown:
            raise ValueError(
                f"init_active_products references unknown product_ids: {unknown}"
            )
        active_ids = list(template.init_active_products)
        n_active = len(active_ids)
    else:
        n_active = int(_maybe_sample(template.init_active_count, init_rng))
        n_active = max(0, min(n_active, len(catalog)))
        active_ids = (
            init_rng.sample(product_ids, n_active) if n_active > 0 else []
        )

    # Issue 08: weighted stock allocation. With an ``item_registry`` and
    # positive resolved weights, distribute the budget proportionally to
    # ``init_stock_share`` (so staples can outweigh fashion within the
    # same budget). Without a registry, or when all active weights are
    # zero, fall back to the legacy even-split — verbatim port of the
    # ``StoreAgent.load_items`` policy, including the
    # ``min(per_item, total_stock - allocated)`` clamp that absorbs the
    # rounding remainder into the last active Ware.
    total_stock = int(capacity * init_stock_pct)
    active_set = set(active_ids)
    weights: dict[str, float] | None = None
    if item_registry is not None and n_active > 0:
        candidate = {
            pid: max(0.0, float(item_registry.init_stock_share(pid)))
            for pid in active_ids
        }
        if sum(candidate.values()) > 0.0:
            weights = candidate

    inventory: dict[str, int] = {}
    if weights is not None:
        sum_weights = sum(weights.values())
        allocated = 0
        for w in catalog:
            if w.product_id in active_set:
                share = weights[w.product_id] / sum_weights
                stock = int(total_stock * share)
                # Defensive clamp: ``floor`` of a non-negative partition
                # never overflows the budget, but keep the invariant
                # explicit so downstream readers don't have to re-derive
                # it.
                stock = min(stock, total_stock - allocated)
                allocated += stock
            else:
                stock = 0
            inventory[w.product_id] = stock
    else:
        per_item = total_stock // n_active if n_active > 0 else 0
        allocated = 0
        for w in catalog:
            if w.product_id in active_set:
                stock = min(per_item, total_stock - allocated)
                allocated += stock
            else:
                stock = 0
            inventory[w.product_id] = stock

    # Issue 07: compute activation_tick per init_freshness mode.
    # "baseline" leaves the dict empty so freshness_multiplier returns
    # 1.0 for initial active SKUs (the "no entry → 1" fallback in
    # Store.freshness_multiplier). "fresh" plants τ=0 for every initial
    # active SKU so the multiplier evaluates to 1 + α at step 0.
    if template.init_freshness == "fresh":
        activation_tick = {pid: 0 for pid in active_ids}
    else:
        activation_tick = {}

    return InitialStoreState(
        region=region,
        capacity=capacity,
        balance=balance,
        init_stock_pct=init_stock_pct,
        delivery_lag=delivery_lag,
        holding_rate=holding_rate,
        order_fee=order_fee,
        active_ids=active_ids,
        inventory=inventory,
        activation_tick=activation_tick,
    )


__all__ = ["InitialStoreState", "init_store_state"]
