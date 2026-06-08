"""T9: Regression snapshot test (issue 10 / issue 11).

A canonical graph-mode ``Scenario`` with a fixed ``world_seed`` runs to
completion; the resulting ``RunLog`` is hashed and compared against
``EXPECTED_HASH``.  CI fails if anyone changes simulator math without
intentionally bumping the snapshot.  Updating the snapshot is a one-line
change to ``EXPECTED_HASH``.

The scenario is intentionally small (30 steps, 2 shops, 4-product catalog
with cross-references, 1 region) so the test runs in well under a second
on CI, while still exercising every subsystem: Market trend / seasonality /
cross-demand, EventEngine spawn + duration, ItemRegistry lifecycle ticks,
allocation (FCFS), and OrderUpToPolicy / DefaultDemandSinkPolicy decide.

Issue 11: The legacy ``Store`` engine has been retired.  The canonical
scenario is now a graph-mode scenario with two 3-node sub-graphs:
    FactoryNode -> IntermediateNode -> DemandSinkNode (per product)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from src.sim.data_exporter import _jsonable
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import DefaultDemandSinkPolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    load_catalog,
)


# Bump this constant when simulator numerics change intentionally. A
# failing assertion prints the observed digest so it can be copied here.
# Re-baselined for ADR 0019: each tick snapshot now carries the per-tick flow
# log (``node_flows`` + ``purchases``); the underlying cash/inventory numerics
# are unchanged.
EXPECTED_HASH = "5d6e24a4dfec5a5f85b5b03397710f3223859286a9011670cd7e385dd7d3e323"


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


def _canonical_scenario() -> Scenario:
    """Build a canonical graph-mode scenario for regression testing.

    Layout: two 3-node sub-graphs sharing the same catalog and market.
    Each sub-graph: FactoryNode -> IntermediateNode -> DemandSinkNode (x4 products).
    """
    catalog = _canonical_catalog()
    pids = [w.product_id for w in catalog]

    nodes: list[NodeInstance] = []
    edges: list[EdgeSpec] = []

    for store_n in [1, 2]:
        f_id = f"f-store{store_n}"
        shop_id = f"shop-store{store_n}"

        f = FactoryNode(
            id=f_id,
            region="US",
            init_seed=100 + store_n,
            produces_product_id=pids[0],
            unit_cost=12.0,
            capacity_per_tick=50,
            inventory=200,
            list_price=12.0,
            cash=0.0,
        )
        f.policy = StaticFactoryPolicy(
            capacity_per_tick=50,
            unit_cost=12.0,
            policy_seed=200 + store_n,
        )

        shop = IntermediateNode(
            id=shop_id,
            region="US",
            init_seed=102 + store_n,
            carried_products=set(pids),
            capacity=600,
            tags=["shop"],
            inventory={pid: 30 for pid in pids},
            pending={},
            list_prices={pids[i]: catalog[i].base_price for i in range(len(pids))},
            min_order_imposed={pid: 0 for pid in pids},
            cash=8000.0,
        )
        shop.policy = OrderUpToPolicy(
            cover_horizon_ticks=14,
            safety_lead_pct_of_lag=1 / 3,
            delivery_lag=2,
            unit_cost=12.0,
            list_price_out=25.0,
            policy_seed=300 + store_n,
        )

        nodes.append(NodeInstance(node=f, init_seed=100 + store_n))
        nodes.append(NodeInstance(node=shop, init_seed=102 + store_n))
        edges.append(
            EdgeSpec(
                supplier_id=f_id,
                buyer_id=shop_id,
                default_lead_time=2,
            )
        )

        for i, pid in enumerate(pids):
            sink_id = f"sink-store{store_n}-{pid}"
            sink = DemandSinkNode(
                id=sink_id,
                region="US",
                init_seed=200 + i * 10 + store_n,
                product_id=pid,
                demand_dist=Uniform(2.0, 7.0),
                income_rate=100.0,
                cash=500.0,
            )
            sink.policy = DefaultDemandSinkPolicy(
                policy_seed=400 + i * 10 + store_n
            )
            nodes.append(NodeInstance(node=sink, init_seed=200 + i * 10 + store_n))
            edges.append(
                EdgeSpec(
                    supplier_id=shop_id,
                    buyer_id=sink_id,
                    default_lead_time=1,
                )
            )

    return Scenario(
        catalog=catalog,
        market=_canonical_market(),
        disruption=_canonical_disruption(),
        item_lifecycle=_canonical_lifecycle(),
        n_steps=N_STEPS,
        start_date=datetime(2024, 3, 1),
        world_seed=WORLD_SEED,
        nodes=nodes,
        edges=edges,
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
