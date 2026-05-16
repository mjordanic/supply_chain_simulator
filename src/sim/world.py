"""``World``: LLM-generated world artifact consumed by ``Scenario`` authoring.

The ``World`` dataclass lives here — in ``src/sim/`` — so downstream layers
(``src/sim/world_loader.py``, scenario helpers) can import the persisted
artifact type without pulling in the full ``src/llm/`` pipeline.

``WorldBuilder`` (the LLM-driven constructor) stays in
``src/llm/world_builder.py``; it imports ``World`` from here.

On-disk format at ``data/worlds/<archetype>/world.json`` is unchanged.
"""

from __future__ import annotations

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
    """

    catalog: list[Ware]
    market: MarketParams
    store_templates: dict[str, StoreTemplate]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly nested dict."""
        return {
            "market": self.market.to_dict(),
            "store_templates": {k: v.to_dict() for k, v in self.store_templates.items()},
            "meta": self.meta,
            "catalog": [_ware_to_dict(w) for w in self.catalog],
        }

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


__all__ = ["World"]
