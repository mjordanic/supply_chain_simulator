"""T9: Regression snapshot test (issue 10).

A canonical ``Scenario`` with a fixed ``world_seed`` runs to completion;
the resulting ``RunLog`` is hashed and compared against ``EXPECTED_HASH``.
CI fails if anyone changes simulator math without intentionally bumping
the snapshot. Updating the snapshot is a one-line change to
``EXPECTED_HASH``.

The scenario is intentionally small (30 steps, 2 stores, 4-product
catalog with cross-references, 1 region) so the test runs in well under
a second on CI, while still exercising every subsystem: Market trend /
seasonality / cross-demand, EventEngine spawn + duration, ItemRegistry
lifecycle ticks, Store accounting, and OrderUpToPolicy decide.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from src.sim.data_exporter import _jsonable
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.policy import OrderUpToPolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    load_catalog,
)


# Bump this constant when simulator numerics change intentionally. A
# failing assertion prints the observed digest so it can be copied here.
EXPECTED_HASH = "582f661b5589a8dc16807a11337e18eb96a83aafc2b893be6f2ceb789d4d5546"


N_STEPS = 30
WORLD_SEED = 12345


def _canonical_catalog():
    return load_catalog(
        [
            {
                "name": "Anchor",
                "category": "Hardware",
                "related_products": [],
                "base_price": 25.0,
                "unit_cost": 12.0,
                "seasonality": "all_season",
            },
            {
                "name": "Bracket",
                "category": "Hardware",
                "related_products": [["Anchor", 0.6]],
                "base_price": 18.0,
                "unit_cost": 9.0,
                "seasonality": "all_season",
            },
            {
                "name": "Lantern",
                "category": "Lighting",
                "related_products": [],
                "base_price": 40.0,
                "unit_cost": 22.0,
                "seasonality": "all_season",
            },
            {
                "name": "Sconce",
                "category": "Lighting",
                "related_products": [["Lantern", 0.4]],
                "base_price": 32.0,
                "unit_cost": 16.0,
                "seasonality": "all_season",
            },
        ]
    )


def _canonical_market():
    return MarketParams(
        cycle_len=180,
        cycle_amp=0.0015,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.25,
        off_factor=0.75,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.6,
        trend_update_interval=15,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.7,
            "growth": 1.4,
            "maturity": 1.0,
            "decline": 0.3,
            "dead": 0.05,
        },
        price_elasticity=-1.4,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.05,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.04),
        supply_shock=Normal(0.0, 0.04),
        base_demand=Uniform(2, 7),
    )


def _canonical_disruption():
    return DisruptionParams(
        event_prob=0.1,
        types=["natural_disaster", "economic_crisis"],
        regions=["US"],
        severity=Constant(0.01),
        duration=Constant(2),
    )


def _canonical_lifecycle():
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.05 for s in stages},
    )


def _canonical_template():
    return StoreTemplate(
        id="canon",
        region="US",
        capacity=150,
        init_balance=8000.0,
        init_stock_pct=0.5,
        delivery_lag=2,
        holding_rate=0.005,
        order_fee=8.0,
        init_active_count=3,
    )


def _canonical_policy(seed: int) -> OrderUpToPolicy:
    return OrderUpToPolicy(policy_seed=seed)


def _canonical_scenario() -> Scenario:
    template = _canonical_template()
    return Scenario(
        catalog=_canonical_catalog(),
        market=_canonical_market(),
        disruption=_canonical_disruption(),
        item_lifecycle=_canonical_lifecycle(),
        stores=[
            StoreInstance(
                template=template, init_seed=101, policy=_canonical_policy(seed=201)
            ),
            StoreInstance(
                template=template, init_seed=102, policy=_canonical_policy(seed=202)
            ),
        ],
        n_steps=N_STEPS,
        start_date=datetime(2024, 3, 1),
        world_seed=WORLD_SEED,
    )


def _hash_run_log(run_log: dict) -> str:
    """SHA-256 over a stable JSON encoding of the run log.

    ``_jsonable`` from ``data_exporter`` normalises datetimes / sets /
    tuples; ``sort_keys=True`` removes any insertion-order dependence
    inside the run log dicts.
    """
    payload = _jsonable(run_log)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def canonical_run_log():
    return Runner(_canonical_scenario()).run()


def test_run_log_hash_matches_snapshot(canonical_run_log):
    """T9: the canonical run log hashes to ``EXPECTED_HASH``.

    On intentional numeric change, copy the digest from the failure
    message into ``EXPECTED_HASH`` at the top of this file.
    """
    actual = _hash_run_log(canonical_run_log)
    assert actual == EXPECTED_HASH, (
        f"Regression snapshot drift detected.\n"
        f"  expected: {EXPECTED_HASH}\n"
        f"  actual:   {actual}\n"
        f"If the change is intentional, update EXPECTED_HASH in "
        f"tests/sim/test_regression_snapshot.py."
    )


def test_canonical_run_is_reproducible():
    """Two fresh runs of the canonical scenario hash to the same value.

    This protects ``test_run_log_hash_matches_snapshot`` from false
    positives caused by run-time non-determinism (e.g., a stray
    ``random.random()`` call against the global RNG).
    """
    first = _hash_run_log(Runner(_canonical_scenario()).run())
    second = _hash_run_log(Runner(_canonical_scenario()).run())
    assert first == second


def test_different_world_seed_changes_hash():
    """Sanity check that the hash actually depends on the run's numerics.

    Catches regressions where the hash function collapses to a constant
    (e.g., hashing only the run log shape, not its values).
    """
    base = _hash_run_log(Runner(_canonical_scenario()).run())
    perturbed_scenario = _canonical_scenario()
    perturbed_scenario.world_seed = WORLD_SEED + 1
    perturbed = _hash_run_log(Runner(perturbed_scenario).run())
    assert base != perturbed
