"""World loading for tuning studies.

Loads a cached LLM-built world from ``data/worlds/<archetype>/world.json`` (or
an explicit ``world_cache_path``). The default archetype is
``fashion_retail_250``. Falls back to a synthesised catalog only when no
cache is found — useful for CI smoke runs.

Returns ``(catalog, base_template, market_params, disruption_params)``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.sim.scenario import (
    DisruptionParams,
    MarketParams,
    StoreTemplate,
    Ware,
    load_catalog,
)
from src.tuning.episode import _default_disruption_params

if TYPE_CHECKING:
    from src.tuning.config import TuningConfig


def load_world(
    config: "TuningConfig",
) -> tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]:
    """Resolve catalog + base ``StoreTemplate`` (+ optional market / disruption params).

    Resolution order:
      1. ``config.world_cache_path`` if set and file exists.
      2. ``data/worlds/<config.world_archetype>/world.json``.
      3. Synthetic ``K_catalog`` fallback (no LLM required) — for CI.
    """
    if config.world_cache_path is not None:
        cache = Path(config.world_cache_path)
        if cache.exists():
            return _load_from_file(cache, config)
        print(
            f"[tuning] WARNING: world_cache_path={config.world_cache_path!r} not found; "
            "falling back to auto-lookup.",
            file=sys.stderr,
        )

    auto_path = Path("data/worlds") / config.world_archetype / "world.json"
    if auto_path.exists():
        return _load_from_file(auto_path, config)

    print(
        f"[tuning] No world cache found at {auto_path}. "
        "Generating a synthetic catalog (no LLM calls).",
        file=sys.stderr,
    )
    return _build_synthetic_catalog(config)


def _load_from_file(
    path: Path, config: "TuningConfig"
) -> tuple[list[Ware], StoreTemplate, MarketParams, DisruptionParams]:
    from src.llm.world_builder import World

    world = World.from_json(path)
    catalog = world.catalog

    if world.store_templates:
        key = next(iter(world.store_templates))
        tmpl = world.store_templates[key]
        tmpl = StoreTemplate(
            id=tmpl.id,
            region=tmpl.region,
            capacity=200,          # overridden per episode by sample_episode
            init_balance=20000.0,  # overridden per episode by sample_episode
            init_stock_pct=0.0,
            delivery_lag=config.delivery_lag,
            holding_rate=config.holding_rate,
            order_fee=config.order_fee,
            init_active_count=config.K_active,
        )
    else:
        tmpl = _default_template(config)

    market_params = world.market
    disruption_params = replace(
        _default_disruption_params(),
        regions=list(world.market.regions),
    )

    print(f"[tuning] Loaded world from {path} ({len(catalog)} products).", file=sys.stderr)
    return catalog, tmpl, market_params, disruption_params


def _build_synthetic_catalog(
    config: "TuningConfig",
) -> tuple[list[Ware], StoreTemplate, None, None]:
    n = config.K_catalog
    items = [
        {
            "name": f"Product {i}",
            "category": "General",
            "related_products": [],
            "base_price": float(10 + (i % 30)),
            "unit_cost": float(4 + (i % 10)),
            "seasonality": "all_season",
        }
        for i in range(n)
    ]
    catalog = load_catalog(items)
    tmpl = _default_template(config)
    print(f"[tuning] Synthetic catalog: {n} products.", file=sys.stderr)
    return catalog, tmpl, None, None


def _default_template(config: "TuningConfig") -> StoreTemplate:
    return StoreTemplate(
        id="tuning_default",
        region="US",
        capacity=200,
        init_balance=20000.0,
        init_stock_pct=0.0,
        delivery_lag=config.delivery_lag,
        holding_rate=config.holding_rate,
        order_fee=config.order_fee,
        init_active_count=config.K_active,
    )


__all__ = ["load_world"]
