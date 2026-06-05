"""Topology scaffolder — deterministic starter topology from a catalog.

``scaffold_topology(catalog, spec) -> {"nodes": [...], "edges": [...]}``
produces a ``nodes``/``edges`` block in the same YAML-serializable format
that ``load_setup`` reads from ``setup.yaml``.  Scaffolded and hand-written
setups are indistinguishable.

The generated topology follows a two-level factory→shop→sink DAG:

- One ``factory`` node per active product in the catalog.
- One ``intermediate`` (shop) node, shared across all products.
  If ``spec.shop_count > 1`` the products are spread evenly across shops.
- One ``demand_sink`` per active product × shop (sink_density=1 means one
  sink per product-shop pair; sink_density<1 collapses some sinks).

``spec`` fields
---------------
shop_count : int       — number of intermediate shop nodes (default 1)
sink_density : float   — fraction of (product × shop) pairs that get a
                         dedicated sink; 1.0 = full coverage (default 1.0)
region : str           — region applied to every node (default "US")
factory_capacity : int — capacity_per_tick for each factory (default 100)
shop_capacity : int    — capacity for each shop (default 500)
factory_lead_time : int— lead_time on factory→shop edges (default 2)
shop_lead_time : int   — lead_time on shop→sink edges (default 1)

Determinism guarantee
---------------------
Same catalog (same product ids, in the same order) + same spec always
produces the same nodes/edges list.  Node ids are stable snake_case strings
derived from the catalog product_ids and shop/sink indices so they read
naturally in a text editor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScaffoldSpec:
    """Parameters that drive topology generation.

    All fields have defaults so callers can construct a spec with a single
    override and get sensible values for everything else.
    """

    shop_count: int = 1
    sink_density: float = 1.0
    region: str = "US"
    factory_capacity: int = 100
    shop_capacity: int = 500
    factory_lead_time: int = 2
    shop_lead_time: int = 1

    def __post_init__(self) -> None:
        if self.shop_count < 1:
            raise ValueError(f"ScaffoldSpec.shop_count must be >= 1, got {self.shop_count}")
        if not (0.0 < self.sink_density <= 1.0):
            raise ValueError(
                f"ScaffoldSpec.sink_density must be in (0, 1], got {self.sink_density}"
            )


def _product_id_to_slug(pid: str) -> str:
    """Convert a product_id to a filesystem/YAML-safe slug for use in node ids."""
    return pid.lower().replace(" ", "_").replace("/", "_")


def scaffold_topology(
    catalog: list[Any],
    spec: "ScaffoldSpec | None" = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate a starter ``{nodes, edges}`` block from a catalog + spec.

    Parameters
    ----------
    catalog:
        List of ``Ware`` namedtuples (or any objects with ``product_id``,
        ``unit_cost``, and ``base_price`` attributes).
    spec:
        ``ScaffoldSpec`` controlling shop count, sink density, region,
        and node sizes.  Defaults to ``ScaffoldSpec()`` when ``None``.

    Returns
    -------
    A ``dict`` with keys ``"nodes"`` and ``"edges"``, each a list of
    plain-dict records in the same format ``load_setup`` reads from
    ``setup.yaml``.  The structure is YAML-serializable with ``pyyaml``
    and identical to what a human would hand-author.

    Raises
    ------
    ValueError
        If the catalog is empty or the spec is invalid.
    """
    if spec is None:
        spec = ScaffoldSpec()
    if not catalog:
        raise ValueError("scaffold_topology: catalog must be non-empty")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Factory nodes — one per catalog product
    # ------------------------------------------------------------------
    for ware in catalog:
        pid = ware.product_id
        slug = _product_id_to_slug(pid)
        unit_cost = float(getattr(ware, "unit_cost", 1.0))
        nodes.append(
            {
                "id": f"factory-{slug}",
                "type": "factory",
                "region": spec.region,
                "produces_product_id": pid,
                "unit_cost": unit_cost,
                "capacity_per_tick": spec.factory_capacity,
                "inventory": 0,
                "list_price": unit_cost,
                "cash": 0.0,
            }
        )

    # ------------------------------------------------------------------
    # 2. Intermediate (shop) nodes — spec.shop_count total, each carrying
    #    the products that are "assigned" to that shop by round-robin.
    # ------------------------------------------------------------------
    product_ids = [w.product_id for w in catalog]

    # Assign products to shops round-robin.
    shop_products: list[list[str]] = [[] for _ in range(spec.shop_count)]
    for i, pid in enumerate(product_ids):
        shop_products[i % spec.shop_count].append(pid)

    # Build list_prices map from catalog base_price
    price_by_pid: dict[str, float] = {
        w.product_id: float(getattr(w, "base_price", 1.0)) for w in catalog
    }

    shop_node_ids: list[str] = []
    for shop_idx in range(spec.shop_count):
        shop_id = f"shop-{shop_idx + 1}" if spec.shop_count > 1 else "shop-1"
        shop_node_ids.append(shop_id)
        carried = sorted(shop_products[shop_idx])  # deterministic order
        inventory: dict[str, int] = {pid: 0 for pid in carried}
        list_prices: dict[str, float] = {pid: price_by_pid[pid] for pid in carried}
        min_order_imposed: dict[str, int] = {pid: 0 for pid in carried}
        nodes.append(
            {
                "id": shop_id,
                "type": "intermediate",
                "region": spec.region,
                "carried_products": carried,
                "capacity": spec.shop_capacity,
                "tags": ["shop"],
                "inventory": inventory,
                "list_prices": list_prices,
                "min_order_imposed": min_order_imposed,
                "cash": 1000.0,
            }
        )

    # ------------------------------------------------------------------
    # 3. Demand-sink nodes — one per (product, shop) pair, subject to
    #    sink_density.  We use a deterministic decimation: keep a sink
    #    when its sequential index / total_pairs < sink_density.
    # ------------------------------------------------------------------
    # Build the full (shop, product) pairs in deterministic order.
    pairs: list[tuple[str, str]] = []
    for shop_idx, shop_id in enumerate(shop_node_ids):
        for pid in sorted(shop_products[shop_idx]):
            pairs.append((shop_id, pid))

    total_pairs = len(pairs)
    # Minimum 1 sink even if density rounds down to 0
    n_sinks = max(1, math.floor(total_pairs * spec.sink_density))

    # Spread n_sinks evenly across pairs — deterministic decimation.
    kept_pairs: list[tuple[str, str]] = []
    if n_sinks >= total_pairs:
        kept_pairs = list(pairs)
    else:
        # Evenly-spaced indices
        step = total_pairs / n_sinks
        for k in range(n_sinks):
            idx = int(k * step)
            kept_pairs.append(pairs[idx])

    for shop_id, pid in kept_pairs:
        slug = _product_id_to_slug(pid)
        sink_id = f"sink-{slug}" if spec.shop_count == 1 else f"sink-{slug}-{shop_id}"
        base_price = price_by_pid[pid]
        nodes.append(
            {
                "id": sink_id,
                "type": "demand_sink",
                "region": spec.region,
                "product_id": pid,
                "demand_dist": {"kind": "constant", "value": 10.0},
                "income_rate": base_price * 10.0,
                "cash": 1000.0,
            }
        )

    # ------------------------------------------------------------------
    # 4. Factory → shop edges (one per product, to the shop carrying it)
    # ------------------------------------------------------------------
    for shop_idx, shop_id in enumerate(shop_node_ids):
        for pid in shop_products[shop_idx]:
            slug = _product_id_to_slug(pid)
            edges.append(
                {
                    "supplier": f"factory-{slug}",
                    "buyer": shop_id,
                    "lead_time": spec.factory_lead_time,
                }
            )

    # ------------------------------------------------------------------
    # 5. Shop → sink edges
    # ------------------------------------------------------------------
    for shop_id, pid in kept_pairs:
        slug = _product_id_to_slug(pid)
        sink_id = f"sink-{slug}" if spec.shop_count == 1 else f"sink-{slug}-{shop_id}"
        edges.append(
            {
                "supplier": shop_id,
                "buyer": sink_id,
                "lead_time": spec.shop_lead_time,
            }
        )

    return {"nodes": nodes, "edges": edges}


__all__ = ["ScaffoldSpec", "scaffold_topology"]
