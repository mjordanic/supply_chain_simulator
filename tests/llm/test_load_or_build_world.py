"""Tests for load_or_build_world and LLMBuildAbortedError (issue 02).

Covers:
- cache hit: returns saved world, build_fn not called, no warning
- cache miss + auto_confirm=True: calls build_fn, writes file, returns world
- cache miss + auto_confirm=False + Enter ("") -> calls build_fn, writes file
- cache miss + auto_confirm=False + abort ("n") -> LLMBuildAbortedError, no build
- cache miss + auto_confirm=False + EOFError -> LLMBuildAbortedError, no build
- force_rebuild=True + existing cache + auto_confirm=True -> overwrites
- warning printed to stderr naming the path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.llm.world_builder import LLMBuildAbortedError, World, load_or_build_world
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.scenario import MarketParams, StoreTemplate, Ware


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_market() -> MarketParams:
    return MarketParams(
        cycle_len=12,
        cycle_amp=0.1,
        init_demand=50.0,
        init_supply=50.0,
        peak_factor=1.2,
        off_factor=0.8,
        season_months={"all": list(range(1, 13))},
        regions=["EU"],
        correlation=0.5,
        trend_update_interval=10,
        min_value=10.0,
        max_value=100.0,
        stage_multipliers={"maturity": 1.0},
        price_elasticity=-1.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        demand_divisor=100.0,
        supply_factor_min=0.01,
        supply_divisor=100.0,
        demand_range=(80.0, 120.0),
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 5.0),
        supply_shock=Normal(0.0, 5.0),
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


def _fixture_world() -> World:
    return World(
        catalog=[_minimal_ware("P0000")],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template()},
        meta={"archetype": "test"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_hit_returns_saved_world_without_calling_build_fn(tmp_path: Path) -> None:
    world = _fixture_world()
    cache_path = tmp_path / "myworld" / "world.json"
    world.to_json(path=cache_path)

    build_fn = MagicMock(return_value=world)
    result = load_or_build_world("myworld", build_fn, base_dir=tmp_path)

    build_fn.assert_not_called()
    assert result.meta == {"archetype": "test"}


def test_cache_hit_no_warning_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    world = _fixture_world()
    cache_path = tmp_path / "myworld" / "world.json"
    world.to_json(path=cache_path)

    load_or_build_world("myworld", MagicMock(return_value=world), base_dir=tmp_path)

    captured = capsys.readouterr()
    assert captured.err == ""


def test_cache_miss_auto_confirm_calls_build_fn_and_writes_file(tmp_path: Path) -> None:
    world = _fixture_world()
    build_fn = MagicMock(return_value=world)

    result = load_or_build_world("newworld", build_fn, base_dir=tmp_path, auto_confirm=True)

    build_fn.assert_called_once()
    expected_path = tmp_path / "newworld" / "world.json"
    assert expected_path.exists()
    assert result.meta == {"archetype": "test"}


def test_cache_miss_auto_confirm_file_content_matches_returned_world(tmp_path: Path) -> None:
    world = _fixture_world()
    load_or_build_world("newworld", MagicMock(return_value=world), base_dir=tmp_path, auto_confirm=True)

    file_data = json.loads((tmp_path / "newworld" / "world.json").read_text())
    assert file_data["meta"] == {"archetype": "test"}
    assert len(file_data["catalog"]) == 1


def test_cache_miss_auto_confirm_false_enter_calls_build_fn(tmp_path: Path) -> None:
    world = _fixture_world()
    build_fn = MagicMock(return_value=world)

    with patch("builtins.input", return_value=""):
        result = load_or_build_world("w", build_fn, base_dir=tmp_path, auto_confirm=False)

    build_fn.assert_called_once()
    assert (tmp_path / "w" / "world.json").exists()
    assert result.meta == {"archetype": "test"}


def test_cache_miss_auto_confirm_false_abort_raises_and_no_build(tmp_path: Path) -> None:
    world = _fixture_world()
    build_fn = MagicMock(return_value=world)

    with patch("builtins.input", return_value="n"):
        with pytest.raises(LLMBuildAbortedError):
            load_or_build_world("w", build_fn, base_dir=tmp_path, auto_confirm=False)

    build_fn.assert_not_called()
    assert not (tmp_path / "w" / "world.json").exists()


def test_cache_miss_auto_confirm_false_eoferror_raises_and_no_build(tmp_path: Path) -> None:
    world = _fixture_world()
    build_fn = MagicMock(return_value=world)

    with patch("builtins.input", side_effect=EOFError):
        with pytest.raises(LLMBuildAbortedError):
            load_or_build_world("w", build_fn, base_dir=tmp_path, auto_confirm=False)

    build_fn.assert_not_called()
    assert not (tmp_path / "w" / "world.json").exists()


def test_force_rebuild_overwrites_existing_cache(tmp_path: Path) -> None:
    old_world = _fixture_world()
    cache_path = tmp_path / "myworld" / "world.json"
    old_world.to_json(path=cache_path)

    new_world = World(
        catalog=[_minimal_ware("P0001")],
        market=_minimal_market(),
        store_templates={"t1": _minimal_template()},
        meta={"archetype": "new"},
    )
    build_fn = MagicMock(return_value=new_world)

    result = load_or_build_world(
        "myworld", build_fn, base_dir=tmp_path, force_rebuild=True, auto_confirm=True
    )

    build_fn.assert_called_once()
    assert result.meta == {"archetype": "new"}
    file_data = json.loads(cache_path.read_text())
    assert file_data["meta"] == {"archetype": "new"}


def test_warning_printed_to_stderr_names_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    world = _fixture_world()
    load_or_build_world("w", MagicMock(return_value=world), base_dir=tmp_path, auto_confirm=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    expected_path = str(tmp_path / "w" / "world.json")
    assert expected_path in captured.err


def test_llm_build_aborted_error_is_importable() -> None:
    assert issubclass(LLMBuildAbortedError, Exception)
