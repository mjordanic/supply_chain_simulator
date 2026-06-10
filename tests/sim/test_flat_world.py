"""Tests for the flat-world authoring helper (issue 01).

Gate contract: ``market.demand_multiplier(pid, region)`` must equal exactly
1.0 (bit-exact, no tolerance) for every product and region on every tick of a
full-length run. This is the off-switch that enables pure demand replay.
"""

from __future__ import annotations

from datetime import datetime
from random import Random

import pytest

from src.sim.distributions import Constant, Uniform
from src.sim.flat_world import flat_world
from src.sim.market import Market
from src.sim.node import DemandSinkNode
from src.sim.scenario import Ware


# ── helpers ──────────────────────────────────────────────────────────────────


def _wares(n: int) -> list[Ware]:
    """Build a minimal catalog of ``n`` products."""
    return [
        Ware(
            product_id=f"P{i:04d}",
            name=f"Item {i}",
            category="test",
            related_products=[],
            base_price=10.0,
            unit_cost=5.0,
            seasonality=None,
        )
        for i in range(n)
    ]


def _build_market(regions: list[str]) -> Market:
    market_params, _ = flat_world(regions=regions)
    return Market(market_params, Random(0), datetime(2024, 1, 1))


# ── AC1: one-call helper produces identity params ─────────────────────────────


def test_flat_world_returns_two_objects():
    """``flat_world`` returns a (MarketParams, ItemLifecycleParams) tuple."""
    from src.sim.scenario import ItemLifecycleParams, MarketParams

    result = flat_world(regions=["US"])
    assert len(result) == 2
    mp, lp = result
    assert isinstance(mp, MarketParams)
    assert isinstance(lp, ItemLifecycleParams)


# ── AC2: Gate test — bit-exact 1.0 over full run ─────────────────────────────


def test_demand_multiplier_is_exactly_one_single_region():
    """Single region: every pid, every tick → demand_multiplier == 1.0 exactly."""
    regions = ["US"]
    market = _build_market(regions)
    catalog = _wares(4)
    n_steps = 365

    for _ in range(n_steps):
        market.tick()
        for ware in catalog:
            for region in regions:
                mult = market.demand_multiplier(ware.product_id, region)
                assert mult == 1.0, (
                    f"demand_multiplier({ware.product_id!r}, {region!r}) "
                    f"= {mult!r} != 1.0 at step {market.current_step()}"
                )


def test_demand_multiplier_is_exactly_one_multi_region_multi_product():
    """Multi-region, multi-product: every multiplier is bit-exact 1.0."""
    regions = ["US", "EU", "APAC"]
    market = _build_market(regions)
    catalog = _wares(6)
    n_steps = 500

    for _ in range(n_steps):
        market.tick()
        for ware in catalog:
            for region in regions:
                mult = market.demand_multiplier(ware.product_id, region)
                assert mult == 1.0, (
                    f"demand_multiplier({ware.product_id!r}, {region!r}) "
                    f"= {mult!r} != 1.0 at step {market.current_step()}"
                )


# ── AC3: sink demand under flat world equals raw demand_dist draws ────────────


def test_sink_demand_equals_raw_draw_under_flat_world():
    """Under flat world, a sink's demand_target equals int(demand_dist.sample(rng)).

    Because multiplier == 1.0 exactly, demand_target = int(base * 1.0) = int(base).
    We verify that the output of demand_target matches what a reference RNG
    would have drawn in the same sequence, proving no hidden factor is active.
    """
    regions = ["US"]
    market_params, _ = flat_world(regions=regions)

    catalog = _wares(3)
    sink = DemandSinkNode(
        id="sink_0",
        region="US",
        init_seed=0,
        product_id="P0001",  # second product in catalog
        demand_dist=Uniform(10.0, 20.0),
        income_rate=1000.0,
        cash=1_000_000.0,
    )

    world_seed = 42
    n_steps = 100

    # Run the sink under flat world.
    market = Market(market_params, Random(world_seed), datetime(2024, 1, 1))
    actual_demands: list[int] = []
    rng_flat = Random(world_seed)
    for _ in range(n_steps):
        market.tick()
        result = sink.demand_target(
            tick=market.current_step() - 1,
            market=market,
            catalog=catalog,
            world_rng=rng_flat,
        )
        actual_demands.append(result)

    # Replay the same draws from a reference RNG (same seed).
    ref_rng = Random(world_seed)
    expected_demands: list[int] = []
    for _ in range(n_steps):
        for ware in catalog:
            draw = float(sink.demand_dist.sample(ref_rng))
            if ware.product_id == sink.product_id:
                expected_demands.append(max(0, int(draw)))

    assert actual_demands == expected_demands, (
        "Sink demands under flat world deviate from raw demand_dist draws — "
        "the multiplier chain is not identity."
    )
