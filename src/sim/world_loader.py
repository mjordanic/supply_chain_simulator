"""Canonical world loader for the sim layer.

``load_world(*, archetype, cache_path, delivery_lag, holding_rate, order_fee,
             K_active, synthetic_fallback, synthetic_catalog_size) -> tuple``
resolves a product world following the three-tier lookup:

  1. Explicit ``cache_path`` — when set and the file exists.
  2. ``data/worlds/<archetype>/world.json`` — auto-lookup by archetype label.
  3. Synthetic fallback — ``K_catalog``-item synthetic catalog, no LLM needed
     (gated by ``synthetic_fallback=True``; raises when ``False`` and neither
     path resolves).

Returns ``(catalog, base_template, market_params, disruption_params)`` where
``market_params`` and ``disruption_params`` are ``None`` for the synthetic
fallback (callers let ``sample_episode`` use its own defaults).

The sim layer is Config-agnostic: callers from tuning or RL supply their own
Config-adapter shims that unpack their respective config objects and forward
primitives here.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from src.sim.episode_sampler import default_disruption_params
from src.sim.scenario import (
    DisruptionParams,
    MarketParams,
    StoreTemplate,
    Ware,
    load_catalog,
)


def load_world(
    *,
    archetype: str,
    cache_path: str | None = None,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    K_active: int,
    synthetic_fallback: bool = True,
    synthetic_catalog_size: int = 100,
) -> tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]:
    """Resolve catalog + base ``StoreTemplate`` (+ optional market / disruption params).

    Parameters
    ----------
    archetype:
        Label for the world-builder cache auto-lookup
        (``data/worlds/<archetype>/world.json``).
    cache_path:
        Explicit path to a cached ``world.json``.  When set and the file
        exists, it is loaded unconditionally (tier 1).
    delivery_lag, holding_rate, order_fee, K_active:
        Non-episodic store-template knobs applied to the loaded template.
        Episodic fields (capacity, balance, active assortment) are always
        overridden by ``sample_episode`` and are set to placeholder values here.
    synthetic_fallback:
        When ``True`` (default), fall through to a synthesised catalog when
        neither tier-1 nor tier-2 paths resolve.  When ``False``, raises
        ``FileNotFoundError`` instead.
    synthetic_catalog_size:
        Number of products in the synthetic fallback catalog.

    Returns
    -------
    tuple of (catalog, base_template, market_params, disruption_params)
        ``market_params`` and ``disruption_params`` are ``None`` for the
        synthetic fallback so ``sample_episode`` uses its own defaults.
    """
    # Tier 1: explicit cache path.
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return _load_from_file(
                p,
                delivery_lag=delivery_lag,
                holding_rate=holding_rate,
                order_fee=order_fee,
                K_active=K_active,
                label="[sim.world_loader]",
            )
        print(
            f"[sim.world_loader] WARNING: cache_path={cache_path!r} not found; "
            "falling back to auto-lookup.",
            file=sys.stderr,
        )

    # Tier 2: auto-lookup by archetype.
    auto_path = Path("data/worlds") / archetype / "world.json"
    if auto_path.exists():
        return _load_from_file(
            auto_path,
            delivery_lag=delivery_lag,
            holding_rate=holding_rate,
            order_fee=order_fee,
            K_active=K_active,
            label="[sim.world_loader]",
        )

    # Tier 3: synthetic fallback.
    if not synthetic_fallback:
        raise FileNotFoundError(
            f"[sim.world_loader] No world found at cache_path={cache_path!r} or "
            f"data/worlds/{archetype}/world.json, and synthetic_fallback=False."
        )

    print(
        f"[sim.world_loader] No world cache found at {auto_path}. "
        "Generating a synthetic catalog (no LLM calls).",
        file=sys.stderr,
    )
    return _build_synthetic_catalog(
        delivery_lag=delivery_lag,
        holding_rate=holding_rate,
        order_fee=order_fee,
        K_active=K_active,
        synthetic_catalog_size=synthetic_catalog_size,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_from_file(
    path: Path,
    *,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    K_active: int,
    label: str = "[sim.world_loader]",
) -> tuple[list[Ware], StoreTemplate, MarketParams, DisruptionParams]:
    """Load catalog, base StoreTemplate, MarketParams, DisruptionParams from path."""
    from src.sim.world import World

    world = World.from_json(path)
    catalog = world.catalog

    if world.store_templates:
        key = next(iter(world.store_templates))
        tmpl = world.store_templates[key]
        base_template = StoreTemplate(
            id=tmpl.id,
            region=tmpl.region,
            capacity=200,          # overridden per episode by sample_episode
            init_balance=20000.0,  # overridden per episode by sample_episode
            init_stock_pct=0.0,
            delivery_lag=delivery_lag,
            holding_rate=holding_rate,
            order_fee=order_fee,
            init_active_count=K_active,
        )
    else:
        base_template = _default_template(
            delivery_lag=delivery_lag,
            holding_rate=holding_rate,
            order_fee=order_fee,
            K_active=K_active,
        )

    market_params = world.market
    disruption_params = replace(
        default_disruption_params(),
        regions=list(world.market.regions),
    )

    print(f"{label} Loaded world from {path} ({len(catalog)} products).", file=sys.stderr)
    return catalog, base_template, market_params, disruption_params


def _build_synthetic_catalog(
    *,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    K_active: int,
    synthetic_catalog_size: int,
) -> tuple[list[Ware], StoreTemplate, None, None]:
    """Build a synthetic catalog (no LLM) for CI / smoke runs."""
    n = synthetic_catalog_size
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
    base_template = _default_template(
        delivery_lag=delivery_lag,
        holding_rate=holding_rate,
        order_fee=order_fee,
        K_active=K_active,
    )
    print(f"[sim.world_loader] Synthetic catalog: {n} products.", file=sys.stderr)
    return catalog, base_template, None, None


def _default_template(
    *,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    K_active: int,
) -> StoreTemplate:
    return StoreTemplate(
        id="sim_default",
        region="US",
        capacity=200,
        init_balance=20000.0,
        init_stock_pct=0.0,
        delivery_lag=delivery_lag,
        holding_rate=holding_rate,
        order_fee=order_fee,
        init_active_count=K_active,
    )


__all__ = ["load_world"]
