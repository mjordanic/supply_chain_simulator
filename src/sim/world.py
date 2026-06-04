"""``World``: LLM-generated world artifact consumed by ``Scenario`` authoring.

The ``World`` dataclass lives here — in ``src/sim/`` — so downstream layers
(``src/sim/world_loader.py``, scenario helpers) can import the persisted
artifact type without pulling in the full ``src/llm/`` pipeline.

``WorldBuilder`` (the LLM-driven constructor) stays in
``src/llm/world_builder.py``; it imports ``World`` from here.

On-disk format at ``data/worlds/<archetype>/world.json`` is unchanged.

Phase-4 addition: ``world_to_graph(world, *, sink_density)`` synthesises a
default graph topology from ``World.store_templates``. Each store template
expands to a 3-node sub-graph: ``FactoryNode → IntermediateNode → DemandSinkNode``.
One ``DemandSinkNode`` per product per ``sink_density`` fraction of the catalog.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.sim.scenario import (
    MarketParams,
    StoreTemplate,
    Ware,
    _ware_from_dict,
    _ware_to_dict,
)


@dataclass
class World:
    """LLM-generated world artifact consumed by ``Scenario`` authoring.

    Three structured fields: ``catalog`` (list of ``Ware``), ``market``
    (full ``MarketParams``), and ``store_templates`` (``dict[id,
    StoreTemplate]``). ``meta`` carries provenance for the cache:
    archetype, n_items, model id, builder version, build timestamp.

    Phase-4 addition: ``default_graph_topology`` is an optional dict
    that can carry LLM-authored hints for graph topology. Currently unused
    by the simulator; reserved for future ``world_to_graph`` extensions.
    """

    catalog: list[Ware]
    market: MarketParams
    store_templates: dict[str, StoreTemplate]
    meta: dict[str, Any] | None = None
    default_graph_topology: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly nested dict."""
        d: dict[str, Any] = {
            "market": self.market.to_dict(),
            "store_templates": {k: v.to_dict() for k, v in self.store_templates.items()},
            "meta": self.meta,
            "catalog": [_ware_to_dict(w) for w in self.catalog],
        }
        if self.default_graph_topology is not None:
            d["default_graph_topology"] = self.default_graph_topology
        return d

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise to JSON, optionally writing to ``path`` (creating dirs)."""
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            p = Path(path)
            # Ensure the cache directory exists.
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "World":
        """Inverse of ``to_dict``: rebuild catalog + market + templates."""
        return cls(
            catalog=[_ware_from_dict(w) for w in d["catalog"]],
            market=MarketParams.from_dict(d["market"]),
            store_templates={
                k: StoreTemplate.from_dict(v)
                for k, v in d["store_templates"].items()
            },
            meta=d.get("meta"),
            default_graph_topology=d.get("default_graph_topology"),
        )

    @classmethod
    def from_json(cls, source: str | Path) -> "World":
        """Load a ``World`` from a path or a JSON string.

        Accepts both ``Path`` and ``str``; if the string looks like a
        file path that exists, it's read as a file, otherwise it's
        treated as the raw JSON payload.
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            # ``Path`` raises ``OSError`` on absurdly long inputs — fall
            # back to "treat as raw JSON" in that case.
            p = Path(source)
            try:
                is_file = p.exists()
            except OSError:
                is_file = False
            text = p.read_text(encoding="utf-8") if is_file else source
        return cls.from_dict(json.loads(text))

    def catalog_df(self) -> Any:
        """One-row-per-catalog-Ware DataFrame view (for notebook inspection)."""
        import pandas as pd
        rows = []
        for w in self.catalog:
            rows.append({
                "product_id": w.product_id,
                "name": w.name,
                "category": w.category,
                "base_price": w.base_price,
                "unit_cost": w.unit_cost,
                # Convenience-derived margin column.
                "margin": w.base_price - w.unit_cost,
                "seasonality": w.seasonality,
                "freshness_alpha": w.freshness_alpha,
                "freshness_decay": w.freshness_decay,
                "init_stage": w.init_stage,
                "stage_change_probs": w.stage_change_probs,
                "init_stock_share": w.init_stock_share,
                "related_products": list(w.related_products),
            })
        return pd.DataFrame(rows)

    def store_templates_df(self) -> Any:
        """One-row-per-template DataFrame view."""
        import pandas as pd
        from dataclasses import fields
        rows = []
        for tmpl in self.store_templates.values():
            rows.append({f.name: getattr(tmpl, f.name) for f in fields(tmpl)})
        return pd.DataFrame(rows)

    def market_df(self) -> Any:
        """Single-row DataFrame view of ``MarketParams``."""
        import pandas as pd
        from dataclasses import fields
        row = {f.name: getattr(self.market, f.name) for f in fields(self.market)}
        return pd.DataFrame([row])

    def meta_df(self) -> Any:
        """Single-row DataFrame view of the build-provenance ``meta`` block."""
        import pandas as pd
        # Pinned column set so an empty / missing meta still produces
        # the same schema.
        _META_COLUMNS = ["archetype", "n_items", "model", "builder_version", "built_at"]
        if self.meta is None:
            return pd.DataFrame(columns=_META_COLUMNS)
        return pd.DataFrame([{k: self.meta.get(k) for k in _META_COLUMNS}])


def _deterministic_seed(name: str) -> int:
    """Return a stable 31-bit integer seed derived from *name*.

    Uses SHA-256 to avoid Python's randomised ``hash()`` which changes
    between processes (PYTHONHASHSEED).  The result is reproducible across
    runs and platforms.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Take the first 4 bytes as a big-endian unsigned int, then clamp to 31 bits.
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def world_to_graph(
    world: "World",
    *,
    sink_density: float = 1.0,
) -> dict[str, Any]:
    """Synthesise a default graph topology from ``world.store_templates``.

    Each ``StoreTemplate`` in ``world.store_templates`` expands to one
    3-node sub-graph:

        FactoryNode(``<tmpl_id>-factory-<pid>``) per product
            → IntermediateNode(``<tmpl_id>-shop``)
            → DemandSinkNode(``<tmpl_id>-sink-<pid>``) per product

    ``sink_density`` (0 < sink_density ≤ 1.0) controls how many catalog
    products get a demand-sink node. At ``sink_density=1.0`` (default) every
    catalog product gets a sink. Lower values reduce the number of sinks by
    selecting the first ``ceil(len(catalog) * sink_density)`` products.

    Returns a plain dict with keys:
    - ``"node_instances"``  — list of ``NodeInstance`` objects
    - ``"edges"``           — list of ``EdgeSpec`` objects
    - ``"template_ids"``    — list of template ids used as keys

    The returned dict can be passed to ``Scenario`` authoring helpers to
    build a full graph-mode ``Scenario`` from a ``World``.

    Parameters
    ----------
    world:
        LLM-generated world artifact with ``store_templates``.
    sink_density:
        Fraction of catalog products to create demand-sinks for.
        Default 1.0 = one sink per product.

    Returns
    -------
    dict
        Graph topology description with ``"node_instances"`` and ``"edges"`` lists.
    """
    import math

    from src.sim.distributions import Normal
    from src.sim.graph import EdgeSpec
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance

    if not (0 < sink_density <= 1.0):
        raise ValueError(
            f"sink_density must be in (0, 1.0], got {sink_density!r}"
        )

    catalog = world.catalog
    n_sink_products = max(1, math.ceil(len(catalog) * sink_density))
    sink_products = [w.product_id for w in catalog[:n_sink_products]]

    all_node_instances: list[NodeInstance] = []
    all_edges: list[EdgeSpec] = []

    for tmpl_id, tmpl in world.store_templates.items():
        region = tmpl.region

        # One factory per carried product. ``FactoryNode`` produces a single
        # product, so a multi-product shop needs one factory per product to
        # have an upstream source for every SKU it carries.
        catalog_by_pid = {w.product_id: w for w in catalog}
        factory_lead = (
            int(tmpl.delivery_lag) if isinstance(tmpl.delivery_lag, (int, float)) else 2
        )

        # One shop (intermediate) per template.
        shop_id = f"{tmpl_id}-shop"
        # All sink products are carried by this shop.
        carried = set(sink_products)
        base_price = catalog[0].base_price if catalog else 2.0
        shop = IntermediateNode(
            id=shop_id,
            region=region,
            init_seed=_deterministic_seed(shop_id),
            carried_products=carried,
            capacity=int(tmpl.capacity) if isinstance(tmpl.capacity, (int, float)) else 500,
            tags=["shop"],
            inventory={pid: 20 for pid in sink_products},
            pending={},
            list_prices={pid: base_price for pid in sink_products},
            min_order_imposed={pid: 0 for pid in sink_products},
            cash=float(tmpl.init_balance) if isinstance(tmpl.init_balance, (int, float)) else 1000.0,
        )
        shop_instance = NodeInstance(node=shop, init_seed=shop.init_seed)
        all_node_instances.append(shop_instance)

        # One factory per carried product, each feeding the shop.
        for pid in sink_products:
            item = catalog_by_pid.get(pid)
            unit_cost = item.unit_cost if item is not None else 1.0
            factory_id = f"{tmpl_id}-factory-{pid}"
            factory = FactoryNode(
                id=factory_id,
                region=region,
                init_seed=_deterministic_seed(factory_id),
                produces_product_id=pid,
                unit_cost=unit_cost,
                capacity_per_tick=50,
                inventory=100,
                list_price=unit_cost,
                cash=0.0,
            )
            all_node_instances.append(
                NodeInstance(node=factory, init_seed=factory.init_seed)
            )
            # Factory → shop edge.
            all_edges.append(
                EdgeSpec(
                    supplier_id=factory_id,
                    buyer_id=shop_id,
                    default_lead_time=factory_lead,
                )
            )

        # One sink per product for this template.
        for pid in sink_products:
            sink_id = f"{tmpl_id}-sink-{pid}"
            sink = DemandSinkNode(
                id=sink_id,
                region=region,
                init_seed=_deterministic_seed(sink_id),
                product_id=pid,
                demand_dist=Normal(mean=10.0, std=2.0),
                income_rate=200.0,
                cash=500.0,
                activation_tick={},
            )
            sink_instance = NodeInstance(node=sink, init_seed=sink.init_seed)
            all_node_instances.append(sink_instance)

            # Shop → sink edge.
            all_edges.append(
                EdgeSpec(
                    supplier_id=shop_id,
                    buyer_id=sink_id,
                    default_lead_time=1,
                )
            )

    return {
        "node_instances": all_node_instances,
        "edges": all_edges,
        "template_ids": list(world.store_templates.keys()),
    }


__all__ = ["World", "world_to_graph"]
