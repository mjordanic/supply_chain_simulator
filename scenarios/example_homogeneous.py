"""Example: a homogeneous fleet of stores on the graph engine.

Demonstrates a 3-store topology where each "store" expands to a
3-node sub-graph: FactoryNode → IntermediateNode → DemandSinkNode.
Each sub-graph uses the same policy configuration but a distinct seed
so each store's step-0 active SKU set and stock allocation is distinct.

Run it directly::

    uv run python scenarios/example_homogeneous.py

or hand the path to the CLI shim::

    uv run python main.py scenarios/example_homogeneous.py

Phase-4 (issue 11): migrated from the legacy Store engine to the
multi-echelon graph engine. No HeuristicPolicy; uses
StaticFactoryPolicy + DefaultDemandSinkPolicy + OrderUpToPolicy.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Standalone execution: ensure the project root is importable so
# ``src.sim.*`` resolves regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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


# Five-product toy catalog. ``load_catalog`` assigns ``P0000``–``P0004`` ids.
_CATALOG = load_catalog(
    [
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
            # Cross-product correlation with Widget A — feeds into
            # ``Market.cross_demand_factor``.
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
        {
            "name": "Gadget A",
            "category": "Gadgets",
            "related_products": [],
            "base_price": 25.0,
            "unit_cost": 15.0,
            "seasonality": "all_season",
        },
        {
            "name": "Gadget B",
            "category": "Gadgets",
            "related_products": [],
            "base_price": 18.0,
            "unit_cost": 10.0,
            "seasonality": "all_season",
        },
    ]
)


# Hand-authored ``MarketParams`` — domain knobs + math defaults flat
# at one level. All seasonal months pooled into ``all_season`` so the
# example doesn't exercise seasonality.
_MARKET = MarketParams(
    cycle_len=365,
    cycle_amp=0.3,
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
    # Constant trend => no drift; the cycle and shocks dominate.
    trend=Constant(1.0),
    demand_shock=Normal(0.0, 0.01),
    supply_shock=Normal(0.0, 0.01),
    base_demand=Uniform(2, 8),
)


# Low-probability disruption events with fixed severity / duration.
_DISRUPTION = DisruptionParams(
    event_prob=0.05,
    types=["natural_disaster", "economic_crisis"],
    regions=["US"],
    severity=Constant(0.01),
    duration=Constant(3),
)


# Canonical 5-stage product lifecycle list.
_LIFECYCLE_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
# Strict-terminal lifecycle: every transition probability is zero so
# every product stays at its ``init_stage`` for the whole run.
_LIFECYCLE = ItemLifecycleParams(
    stages=_LIFECYCLE_STAGES,
    init_stage="maturity",
    default_stage_change_probs={s: 0.0 for s in _LIFECYCLE_STAGES},
)


_PIDS = [w.product_id for w in _CATALOG]
_PRIMARY_PID = _PIDS[0]


def _make_store_subgraph(store_index: int, seed_offset: int):
    """Build one (factory, shop, sinks) sub-graph for store ``store_index``.

    Returns (node_instances, edges).
    """
    base_seed = (store_index + 1) * 1000 + seed_offset

    factory_id = f"store{store_index}-factory"
    shop_id = f"store{store_index}-shop"

    factory = FactoryNode(
        id=factory_id,
        region="US",
        init_seed=base_seed,
        produces_product_id=_PRIMARY_PID,
        unit_cost=_CATALOG[0].unit_cost,
        capacity_per_tick=100,
        inventory=200,
        list_price=_CATALOG[0].unit_cost,
        cash=0.0,
    )
    shop = IntermediateNode(
        id=shop_id,
        region="US",
        init_seed=base_seed + 1,
        carried_products=set(_PIDS),
        capacity=200,
        tags=["shop"],
        inventory={pid: 20 for pid in _PIDS},
        pending={},
        list_prices={w.product_id: w.base_price for w in _CATALOG},
        min_order_imposed={pid: 0 for pid in _PIDS},
        cash=10000.0,
    )

    node_instances = [
        NodeInstance(node=factory, init_seed=factory.init_seed,
                     policy=StaticFactoryPolicy(capacity_per_tick=100, unit_cost=_CATALOG[0].unit_cost)),
        NodeInstance(node=shop, init_seed=shop.init_seed,
                     policy=OrderUpToPolicy(policy_seed=base_seed + 2)),
    ]
    edges = [
        EdgeSpec(supplier_id=factory_id, buyer_id=shop_id, default_lead_time=2),
    ]

    for pid in _PIDS:
        sink_id = f"store{store_index}-sink-{pid}"
        sink = DemandSinkNode(
            id=sink_id,
            region="US",
            init_seed=base_seed + 3 + _PIDS.index(pid),
            product_id=pid,
            demand_dist=Normal(mean=5.0, std=1.0),
            income_rate=200.0,
            cash=500.0,
            activation_tick={},
        )
        node_instances.append(
            NodeInstance(node=sink, init_seed=sink.init_seed,
                         policy=DefaultDemandSinkPolicy())
        )
        edges.append(
            EdgeSpec(supplier_id=shop_id, buyer_id=sink_id, default_lead_time=1)
        )

    return node_instances, edges


# Build 3 store sub-graphs (store0, store1, store2).
_ALL_NODES: list[NodeInstance] = []
_ALL_EDGES: list[EdgeSpec] = []
for _i in range(3):
    _ni, _e = _make_store_subgraph(_i, seed_offset=0)
    _ALL_NODES.extend(_ni)
    _ALL_EDGES.extend(_e)


# Scenario top-level binding — the CLI shim and standalone ``main``
# both pick this up.
scenario = Scenario(
    catalog=_CATALOG,
    market=_MARKET,
    disruption=_DISRUPTION,
    item_lifecycle=_LIFECYCLE,
    stores=[],
    nodes=_ALL_NODES,
    edges=_ALL_EDGES,
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and print a brief summary."""
    from src.sim.data_exporter import DataExporter
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "example_homogeneous"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Ran {scenario.n_steps} steps with {len(scenario.nodes)} nodes")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
