"""Policy registry and fact-injection factory for setup-directory loading.

A ``{name: class}`` registry maps stable snake_case names to policy classes.
``build_policy(name, params, *, node, edges, policy_seed)`` resolves the
class, injects node/edge-derived facts (capacity, unit_cost, list_prices,
supplier ids, lead times) and the derived policy_seed, then splats remaining
``params`` as constructor kwargs.  ``params`` carries only true
hyperparameters — not injected facts.

Registry names (stable snake_case):
    order_up_to          → OrderUpToPolicy
    reorder_point        → ReorderPointPolicy
    periodic_order_up_to → PeriodicOrderUpToPolicy
    periodic_reorder     → PeriodicReorderPolicy
    single_supplier      → IntermediatePolicy.SingleSupplierAdapter
    static_factory       → StaticFactoryPolicy
    default_demand_sink  → DefaultDemandSinkPolicy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.sim.policy import (
    DefaultDemandSinkPolicy,
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
    StaticFactoryPolicy,
)

if TYPE_CHECKING:
    from src.sim.graph import EdgeSpec
    from src.sim.node import Node


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "order_up_to": OrderUpToPolicy,
    "reorder_point": ReorderPointPolicy,
    "periodic_order_up_to": PeriodicOrderUpToPolicy,
    "periodic_reorder": PeriodicReorderPolicy,
    "single_supplier": IntermediatePolicy.SingleSupplierAdapter,  # type: ignore[attr-defined]
    "static_factory": StaticFactoryPolicy,
    "default_demand_sink": DefaultDemandSinkPolicy,
}


def list_policy_names() -> list[str]:
    """Return the sorted list of registered policy names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Fact injection
# ---------------------------------------------------------------------------

def _average_lead_time(node_id: str, edges: list["EdgeSpec"]) -> int:
    """Return the mean default_lead_time of all edges with this node as buyer.

    Falls back to 2 ticks when no edges supply this node.
    """
    supplier_edges = [e for e in edges if e.buyer_id == node_id]
    if not supplier_edges:
        return 2
    return round(sum(e.default_lead_time for e in supplier_edges) / len(supplier_edges))


def build_policy(
    name: str,
    params: dict[str, Any],
    *,
    node: "Node",
    edges: list["EdgeSpec"],
    policy_seed: int | None,
) -> Any:
    """Resolve a policy class by name, inject node/edge facts, and construct it.

    Parameters
    ----------
    name:
        Registry key (snake_case). Raises ``ValueError`` for unknown names.
    params:
        Caller-supplied hyperparameters. Keys that conflict with injected
        facts are silently overridden by the injected value so the policy
        file stays minimal and can't accidentally diverge from the node.
    node:
        The node this policy will be attached to. Used to read capacity,
        unit_cost, list_prices, etc.
    edges:
        All edges in the scenario. Used to derive delivery_lag and supplier ids.
    policy_seed:
        Pre-derived per-node seed (``derive(world_seed, node_id, "policy")``).

    Returns
    -------
    A constructed ``NodePolicy`` instance.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown policy name {name!r}. "
            f"Registered names: {sorted(_REGISTRY)}"
        )

    cls = _REGISTRY[name]
    kwargs: dict[str, Any] = dict(params)  # shallow copy — don't mutate caller's dict
    kwargs["policy_seed"] = policy_seed

    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode

    if cls is StaticFactoryPolicy:
        if isinstance(node, FactoryNode):
            from src.sim.distributions import Distribution
            cap = node.capacity_per_tick
            if isinstance(cap, Distribution):
                # We can't inject a Distribution into an int parameter —
                # use 0 as a sentinel that the default produce-at-capacity
                # branch handles. The issue spec says scalars only in catalog.
                kwargs.setdefault("capacity_per_tick", 0)
            else:
                kwargs["capacity_per_tick"] = int(cap)
            kwargs["unit_cost"] = float(node.unit_cost)

    elif cls in (
        OrderUpToPolicy,
        ReorderPointPolicy,
        PeriodicOrderUpToPolicy,
        PeriodicReorderPolicy,
    ):
        if isinstance(node, IntermediateNode):
            # Inject the representative delivery lag from upstream edges.
            kwargs["delivery_lag"] = _average_lead_time(node.id, edges)
            # unit_cost for the inner policy's cash accounting.
            # IntermediateNode doesn't have a direct unit_cost — use
            # the mean of list_prices as a proxy, or 1.0 as fallback.
            if node.list_prices:
                mean_price = sum(node.list_prices.values()) / len(node.list_prices)
                kwargs.setdefault("unit_cost", mean_price)
            else:
                kwargs.setdefault("unit_cost", 1.0)

    elif cls is IntermediatePolicy.SingleSupplierAdapter:  # type: ignore[attr-defined]
        if isinstance(node, IntermediateNode):
            # Inject the single upstream supplier id when not overridden.
            supplier_ids = [e.supplier_id for e in edges if e.buyer_id == node.id]
            if supplier_ids and "supplier_id" not in kwargs:
                kwargs["supplier_id"] = supplier_ids[0]
            kwargs["delivery_lag"] = _average_lead_time(node.id, edges)
            if node.list_prices:
                mean_price = sum(node.list_prices.values()) / len(node.list_prices)
                kwargs.setdefault("unit_cost", mean_price)
            else:
                kwargs.setdefault("unit_cost", 1.0)

    # DefaultDemandSinkPolicy needs only policy_seed — already set above.

    return cls(**kwargs)


__all__ = ["build_policy", "list_policy_names"]
