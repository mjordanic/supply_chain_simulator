"""Tests for ``src.sim.world_loader`` (issue 05).

Covers the four acceptance criteria:
- Explicit ``cache_path`` resolves first when present.
- Archetype lookup (``data/worlds/<archetype>/world.json``) resolves second.
- ``synthetic_fallback=True`` produces a viable 4-tuple when neither path
  resolves; ``synthetic_fallback=False`` raises.
- Disruption-params regions adjustment matches ``world.market.regions``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.sim.world_loader import load_world


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_world_json(regions: list[str] | None = None) -> dict:
    """Return a minimal world.json dict for testing."""
    if regions is None:
        regions = ["US", "EU"]
    return {
        "catalog": [
            {
                "product_id": "P0000",
                "name": "Widget A",
                "category": "General",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            },
            {
                "product_id": "P0001",
                "name": "Widget B",
                "category": "General",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 6.0,
                "seasonality": "all_season",
            },
        ],
        "market": {
            "cycle_len": 365,
            "cycle_amp": 0.3,
            "init_demand": 1.0,
            "init_supply": 1.0,
            "peak_factor": 1.2,
            "off_factor": 0.7,
            "season_months": {"all_season": list(range(1, 13))},
            "regions": regions,
            "correlation": 0.7,
            "trend_update_interval": 20,
            "min_value": 0.2,
            "max_value": 2.0,
            "stage_multipliers": {
                "introduction": 0.7,
                "growth": 1.5,
                "maturity": 1.0,
                "decline": 0.2,
                "dead": 0.05,
            },
            "price_elasticity": -1.5,
            "promo_multiplier": 1.0,
            "demand_factor_min": 0.1,
            "supply_factor_min": 0.01,
            "cross_inv_lo": 0.3,
            "cross_inv_hi": 0.7,
            "cross_factor_range": [0.3, 1.6],
            "trend": {"type": "Constant", "value": 1.0},
            "demand_shock": {"type": "Normal", "mean": 0.0, "std": 0.01},
            "supply_shock": {"type": "Normal", "mean": 0.0, "std": 0.01},
            "base_demand": {"type": "Uniform", "low": 2, "high": 8},
        },
        "store_templates": {
            "main": {
                "id": "main",
                "region": regions[0],
                "capacity": 500,
                "init_balance": 50000.0,
                "init_stock_pct": 0.0,
                "delivery_lag": 3,
                "holding_rate": 0.01,
                "order_fee": 50.0,
                "init_active_count": 5,
            }
        },
        "_meta": {"archetype": "test", "version": "1"},
    }


# ---------------------------------------------------------------------------
# Tier 1: explicit cache_path
# ---------------------------------------------------------------------------

def test_explicit_cache_path_resolves_first():
    """When cache_path exists it is loaded unconditionally (tier 1)."""
    world_data = _minimal_world_json(regions=["US"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "world.json"
        path.write_text(json.dumps(world_data))

        catalog, base_tmpl, market_params, disruption_params = load_world(
            archetype="nonexistent_archetype",
            cache_path=str(path),
            delivery_lag=3,
            holding_rate=0.01,
            order_fee=50.0,
            K_active=2,
        )

    assert len(catalog) == 2
    assert market_params is not None
    assert disruption_params is not None
    assert base_tmpl is not None


# ---------------------------------------------------------------------------
# Tier 2: archetype auto-lookup
# ---------------------------------------------------------------------------

def test_archetype_lookup_resolves_second(tmp_path, monkeypatch):
    """When cache_path is None, ``data/worlds/<archetype>/world.json`` is tried."""
    world_data = _minimal_world_json(regions=["CA"])

    # Write the world file at the expected archetype path relative to cwd.
    worlds_dir = tmp_path / "data" / "worlds" / "test_arch"
    worlds_dir.mkdir(parents=True)
    (worlds_dir / "world.json").write_text(json.dumps(world_data))

    # Change cwd so the relative path resolves.
    monkeypatch.chdir(tmp_path)

    catalog, base_tmpl, market_params, disruption_params = load_world(
        archetype="test_arch",
        cache_path=None,
        delivery_lag=5,
        holding_rate=0.02,
        order_fee=10.0,
        K_active=2,
    )

    assert len(catalog) == 2
    assert market_params is not None
    # delivery_lag from the call should be applied to the base_template.
    assert base_tmpl.delivery_lag == 5


# ---------------------------------------------------------------------------
# Tier 3: synthetic fallback
# ---------------------------------------------------------------------------

def test_synthetic_fallback_true_produces_viable_tuple(monkeypatch, tmp_path):
    """When neither path resolves and synthetic_fallback=True, returns a 4-tuple."""
    monkeypatch.chdir(tmp_path)  # no data/worlds/ exists here

    catalog, base_tmpl, market_params, disruption_params = load_world(
        archetype="no_such_archetype",
        cache_path=None,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        K_active=3,
        synthetic_fallback=True,
        synthetic_catalog_size=20,
    )

    assert len(catalog) == 20
    assert base_tmpl is not None
    assert market_params is None   # synthetic fallback returns None
    assert disruption_params is None


def test_synthetic_fallback_false_raises(monkeypatch, tmp_path):
    """When neither path resolves and synthetic_fallback=False, raises FileNotFoundError."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_world(
            archetype="no_such_archetype",
            cache_path=None,
            delivery_lag=3,
            holding_rate=0.01,
            order_fee=50.0,
            K_active=3,
            synthetic_fallback=False,
        )


# ---------------------------------------------------------------------------
# Disruption-params regions adjustment
# ---------------------------------------------------------------------------

def test_disruption_params_regions_match_world_market_regions():
    """disruption_params.regions must match world.market.regions after loading."""
    world_regions = ["US", "EU", "APAC"]
    world_data = _minimal_world_json(regions=world_regions)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "world.json"
        path.write_text(json.dumps(world_data))

        _, _, market_params, disruption_params = load_world(
            archetype="any",
            cache_path=str(path),
            delivery_lag=3,
            holding_rate=0.01,
            order_fee=50.0,
            K_active=2,
        )

    assert disruption_params is not None
    assert set(disruption_params.regions) == set(world_regions)
    # market_params regions should also match
    assert set(market_params.regions) == set(world_regions)
