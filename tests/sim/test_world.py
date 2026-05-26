"""Tests for ``src.sim.world.World`` (issue 02).

Covers:
- JSON round-trip of a small synthetic ``World`` (catalog, market,
  store_templates, meta).
- Every ``data/worlds/<archetype>/world.json`` currently checked into
  the repo loads via ``World.from_json`` without regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import MarketParams, StoreTemplate, Ware
from src.sim.world import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_market() -> MarketParams:
    return MarketParams(
        cycle_len=12,
        cycle_amp=0.001,
        init_demand=0.5,
        init_supply=0.5,
        peak_factor=1.2,
        off_factor=0.8,
        season_months={"all": list(range(1, 13))},
        regions=["EU"],
        correlation=0.5,
        trend_update_interval=10,
        min_value=0.1,
        max_value=1.0,
        stage_multipliers={"maturity": 1.0},
        price_elasticity=-1.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.05),
        supply_shock=Normal(0.0, 0.05),
        base_demand=Uniform(2, 8),
    )


def _minimal_template(tid: str = "t1") -> StoreTemplate:
    return StoreTemplate(
        id=tid,
        region="EU",
        capacity=200.0,
        init_balance=5000.0,
        init_stock_pct=0.5,
        delivery_lag=1.0,
        holding_rate=0.01,
        order_fee=10.0,
        init_active_count=2,
    )


def _minimal_ware(pid: str = "P0000") -> Ware:
    return Ware(
        product_id=pid,
        name="Widget",
        category="Gadgets",
        related_products=[],
        base_price=20.0,
        unit_cost=10.0,
        seasonality="all_season",
    )


def _make_world(meta: dict[str, Any] | None = None) -> World:
    return World(
        catalog=[_minimal_ware("P0000"), _minimal_ware("P0001")],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template("t1")},
        meta=meta,
    )


# ---------------------------------------------------------------------------
# (a) JSON round-trip of a small synthetic World
# ---------------------------------------------------------------------------

def test_json_round_trip_with_meta() -> None:
    """Full ``to_json`` / ``from_json`` cycle preserves all fields."""
    meta = {
        "archetype": "test",
        "n_items": 2,
        "model": None,
        "builder_version": "1",
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    world = _make_world(meta)
    json_str = world.to_json()

    recovered = World.from_json(json_str)

    assert len(recovered.catalog) == 2
    assert recovered.catalog[0].product_id == "P0000"
    assert recovered.catalog[1].product_id == "P0001"
    assert recovered.market.price_elasticity == -1.0
    assert recovered.market.regions == ["EU"]
    assert set(recovered.store_templates.keys()) == {"t1"}
    assert recovered.store_templates["t1"].capacity == 200.0
    assert recovered.meta == meta


def test_json_round_trip_no_meta() -> None:
    """``meta=None`` survives the round-trip."""
    world = _make_world(meta=None)
    recovered = World.from_json(world.to_json())
    assert recovered.meta is None


def test_json_round_trip_via_file(tmp_path: Path) -> None:
    """``to_json(path=…)`` writes and ``from_json(path)`` reads correctly."""
    world = _make_world({"key": "val"})
    out = tmp_path / "world.json"
    world.to_json(path=out)

    recovered = World.from_json(out)
    assert recovered.meta == {"key": "val"}
    assert len(recovered.catalog) == 2


def test_round_trip_distribution_typed_fields() -> None:
    """Distribution-typed Ware fields survive serialisation."""
    ware = Ware(
        product_id="P0000",
        name="Widget",
        category="Gadgets",
        related_products=[],
        base_price=20.0,
        unit_cost=10.0,
        seasonality="all_season",
        freshness_alpha=Uniform(0.1, 0.3),
        freshness_decay=Normal(30.0, 5.0),
    )
    world = World(
        catalog=[ware],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template()},
    )
    recovered = World.from_json(world.to_json())
    alpha = recovered.catalog[0].freshness_alpha
    decay = recovered.catalog[0].freshness_decay
    assert isinstance(alpha, Uniform)
    assert alpha.low == 0.1
    assert isinstance(decay, Normal)
    assert decay.mean == 30.0


# ---------------------------------------------------------------------------
# (b) Every data/worlds/<archetype>/world.json loads without regression
# ---------------------------------------------------------------------------

def _collect_world_json_files() -> list[Path]:
    """Return all world.json files under ``data/worlds/``."""
    repo_root = Path(__file__).parent.parent.parent
    worlds_dir = repo_root / "data" / "worlds"
    if not worlds_dir.exists():
        return []
    return sorted(worlds_dir.rglob("world.json"))


_WORLD_FILES = _collect_world_json_files()


@pytest.mark.skipif(not _WORLD_FILES, reason="No data/worlds/<archetype>/world.json files checked in")
@pytest.mark.parametrize("path", _WORLD_FILES, ids=lambda p: p.parent.name)
def test_checked_in_world_loads_without_error(path: Path) -> None:
    """Each checked-in world.json round-trips through ``World.from_json``."""
    world = World.from_json(path)
    assert len(world.catalog) > 0
    assert world.market is not None
    assert isinstance(world.store_templates, dict)
