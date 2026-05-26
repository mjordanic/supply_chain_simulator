"""CRN determinism gate for the Runner / Simulation refactor (issue 03).

Two tests pin the bit-identity contract that issues 06-08 must keep green.

1. ``test_runner_snapshot_equals_golden`` — snapshot test.
   Runs the canonical small scenario via ``Runner(scenario).run()`` and
   compares numeric/categorical columns against the pre-committed golden
   parquet at ``tests/sim/fixtures/runner_snapshot_pre.parquet``.

2. ``test_runner_run_equals_simulation_tick_loop`` — cross-path equivalence.
   Drives the same scenario via ``Runner`` *and* via the new
   ``build_world`` + ``Simulation.tick()`` loop, then asserts end-of-run
   state equality on a set of scalar and structured fields.

The canonical scenario is defined once in this module so both tests share
identical construction parameters.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.sim.distributions import Constant, Normal, Uniform
from src.sim.policy import OrderUpToPolicy
from src.sim.runner import Runner, Simulation, TickResult, build_world
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreTemplate,
    load_catalog,
    make_stores,
)


# ---------------------------------------------------------------------------
# Canonical small scenario — deterministic; matches the golden fixture
# ---------------------------------------------------------------------------

def _canonical_scenario() -> Scenario:
    """Return the canonical small scenario used to generate the golden fixture."""
    catalog = load_catalog([
        {
            "name": "Widget A",
            "category": "Widgets",
            "related_products": [],
            "base_price": 20.0,
            "unit_cost": 12.0,
            "seasonality": "all_season",
        },
        {
            "name": "Widget B",
            "category": "Widgets",
            "related_products": [["Widget A", 0.5]],
            "base_price": 30.0,
            "unit_cost": 18.0,
            "seasonality": "all_season",
        },
        {
            "name": "Widget C",
            "category": "Widgets",
            "related_products": [],
            "base_price": 15.0,
            "unit_cost": 8.0,
            "seasonality": "all_season",
        },
    ])
    market = MarketParams(
        cycle_len=365,
        cycle_amp=0.001,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.2,
        off_factor=0.7,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.7,
        trend_update_interval=20,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.7,
            "growth": 1.5,
            "maturity": 1.0,
            "decline": 0.2,
            "dead": 0.05,
        },
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.02),
        supply_shock=Normal(0.0, 0.02),
        base_demand=Uniform(2, 8),
    )
    disruption = DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.0),
        duration=Constant(1),
    )
    lifecycle_stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=lifecycle_stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in lifecycle_stages},
    )
    template = StoreTemplate(
        id="small",
        region="US",
        capacity=100,
        init_balance=5000.0,
        init_stock_pct=0.5,
        delivery_lag=2,
        holding_rate=0.005,
        order_fee=5.0,
        init_active_count=3,
    )
    stores = make_stores([(template, 42, OrderUpToPolicy())])
    return Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption,
        item_lifecycle=lifecycle,
        stores=stores,
        n_steps=30,
        start_date=datetime(2024, 1, 1),
        world_seed=1234,
    )


# ---------------------------------------------------------------------------
# Helper: flatten run_log → tidy DataFrame (same schema as the fixture)
# ---------------------------------------------------------------------------

def _run_log_to_df(run_log: dict) -> pd.DataFrame:
    n_steps = len(run_log["global"]["time"]["simulation_step"])
    store_ids = list(run_log["stores"].keys())
    pids = list(run_log["stores"][0]["products"].keys())
    rows = []
    for t in range(n_steps):
        for i in store_ids:
            for pid in pids:
                p = run_log["stores"][i]["products"][pid]
                rows.append({
                    "step": run_log["global"]["time"]["simulation_step"][t],
                    "store": i,
                    "pid": pid,
                    "inventory": p["inventory"][t],
                    "demand": p["demand"][t],
                    "sales": p["sales"][t],
                    "balance": run_log["stores"][i]["balance"][t],
                    "market_supply_US": run_log["global"]["market_supply"]["US"][t],
                    "market_demand_US": run_log["global"]["market_demand"]["US"][t],
                    "lifecycle_stage": run_log["global"]["products"][pid]["lifecycle_stage"][t],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: snapshot test against golden fixture
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "runner_snapshot_pre.parquet"

_NUMERIC_COLS = ["inventory", "demand", "sales", "balance", "market_supply_US", "market_demand_US"]
_EXACT_COLS = ["step", "store", "lifecycle_stage"]


def test_runner_snapshot_equals_golden() -> None:
    """Runner output is bit-identical to the pre-committed golden fixture.

    Numeric columns (float/int) are compared with ``DataFrame.equals``
    (requires identical dtypes and values, no tolerance). Categorical /
    integer columns are also compared with exact equality.
    """
    assert _FIXTURE_PATH.exists(), (
        f"Golden fixture not found: {_FIXTURE_PATH}. "
        "Run the snapshot generation script to create it."
    )
    golden = pd.read_parquet(_FIXTURE_PATH)

    scenario = _canonical_scenario()
    runner = Runner(scenario)
    run_log = runner.run()
    actual = _run_log_to_df(run_log)

    # Sort both DataFrames identically so row order doesn't matter.
    sort_keys = ["step", "store", "pid"]
    golden_sorted = golden.sort_values(sort_keys).reset_index(drop=True)
    actual_sorted = actual.sort_values(sort_keys).reset_index(drop=True)

    # Numeric columns: exact bit-for-bit equality after type alignment.
    for col in _NUMERIC_COLS:
        assert golden_sorted[col].equals(actual_sorted[col]), (
            f"Column '{col}' differs from golden fixture.\n"
            f"Max delta: {(golden_sorted[col] - actual_sorted[col]).abs().max()}"
        )

    # Exact columns.
    for col in _EXACT_COLS:
        assert (golden_sorted[col] == actual_sorted[col]).all(), (
            f"Column '{col}' has mismatches vs golden fixture."
        )


# ---------------------------------------------------------------------------
# Test 2: cross-path equivalence Runner vs Simulation tick loop
# ---------------------------------------------------------------------------

def test_runner_run_equals_simulation_tick_loop() -> None:
    """Runner.run() and build_world + Simulation.tick() share the same trajectory.

    Compares end-of-run state on:
    - world_rng.getstate() (full RNG state)
    - store.balance (scalar accounting)
    - store.inventory (per-pid dict)
    - store.demand (per-pid dict)
    - store.sales (per-pid dict)
    - market.market_state (regional supply/demand)
    - item_registry lifecycle stage map
    """
    scenario = _canonical_scenario()

    # Path A: Runner
    runner = Runner(scenario)
    runner_log = runner.run()

    # Path B: build_world + manual tick loop
    sim = build_world(scenario)
    for _ in range(scenario.n_steps):
        sim.tick()

    # --- RNG state ---
    assert runner.world_rng.getstate() == sim.world_rng.getstate(), (
        "world_rng state diverged between Runner and Simulation tick loop"
    )

    # --- Per-store state ---
    assert len(runner.stores) == len(sim.stores)
    for i, (rs, ss) in enumerate(zip(runner.stores, sim.stores)):
        assert rs.balance == pytest.approx(ss.balance, abs=1e-9), (
            f"Store {i} balance: Runner={rs.balance}, Sim={ss.balance}"
        )
        assert rs.inventory == ss.inventory, (
            f"Store {i} inventory diverged"
        )
        assert rs.demand == ss.demand, (
            f"Store {i} demand diverged"
        )
        assert rs.sales == ss.sales, (
            f"Store {i} sales diverged"
        )

    # --- Market state ---
    assert runner.market.market_state == sim.market.market_state, (
        "market.market_state diverged"
    )

    # --- Lifecycle stage map ---
    runner_stages = {
        pid: item.lifecycle_stage
        for pid, item in runner.item_registry.items.items()
    }
    sim_stages = {
        pid: item.lifecycle_stage
        for pid, item in sim.item_registry.items.items()
    }
    assert runner_stages == sim_stages, (
        "ItemRegistry lifecycle stages diverged"
    )
