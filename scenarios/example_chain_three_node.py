"""Phase-1 chain smoke scenario: F1 → S1 → Sink1.

A 3-node single-product supply chain:
- ``factory-1``  (FactoryNode)        — produces 50 units/tick at cost 5.0
- ``shop-1``     (IntermediateNode)   — orders from factory via SingleSupplierAdapter,
                                        sells to sink at price 8.0
- ``sink-1``     (DemandSinkNode)     — demands ~10 units/tick, income_rate=200

After 180 ticks the chain should show non-trivial dynamics: factory produces,
shop orders using textbook rate-estimate / safety-horizon math, sink generates
demand and buys.

Run::

    uv run python scenarios/example_chain_three_node.py

Output is written to ``runs/example_chain_three_node/``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.sim.distributions import Constant, Normal
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    IntermediatePolicy,
    StaticFactoryPolicy,
)
from src.sim.runner import GraphRunner, build_graph_world
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    Ware,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

WORLD_SEED = 42
N_STEPS = 180
PRODUCT_UNIT_COST = 5.0
FACTORY_LIST_PRICE = 5.0
SHOP_LIST_PRICE = 8.0
FACTORY_CAPACITY = 50
SINK_DEMAND_MEAN = 10
SINK_DEMAND_STD = 2
SINK_INCOME_RATE = 200.0


def _build_scenario() -> Scenario:
    catalog = load_catalog([
        {
            "name": "Widget",
            "category": "general",
            "related_products": [],
            "base_price": SHOP_LIST_PRICE,
            "unit_cost": PRODUCT_UNIT_COST,
            "seasonality": "all_season",
        }
    ])
    pid = catalog[0].product_id  # "P0000"

    # Attach Phase-1 concrete policies to nodes.
    factory_policy = StaticFactoryPolicy(
        capacity_per_tick=FACTORY_CAPACITY,
        unit_cost=PRODUCT_UNIT_COST,
        policy_seed=1,
    )
    shop_policy = IntermediatePolicy.SingleSupplierAdapter(
        supplier_id="factory-1",
        cover_horizon_ticks=14,
        safety_lead_pct_of_lag=1 / 3,
        delivery_lag=2,
        unit_cost=PRODUCT_UNIT_COST,
        list_price_out=SHOP_LIST_PRICE,
        policy_seed=2,
    )
    sink_policy = DefaultDemandSinkPolicy(policy_seed=3)

    factory = FactoryNode(
        id="factory-1",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=PRODUCT_UNIT_COST,
        capacity_per_tick=FACTORY_CAPACITY,
        inventory=100,
        list_price=FACTORY_LIST_PRICE,
        cash=0.0,
    )
    factory.policy = factory_policy

    shop = IntermediateNode(
        id="shop-1",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=500,
        tags=["shop"],
        inventory={pid: 20},
        pending={},
        list_prices={pid: SHOP_LIST_PRICE},
        min_order_imposed={pid: 0},
        cash=500.0,
    )
    shop.policy = shop_policy

    sink = DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=3,
        product_id=pid,
        demand_dist=Normal(mean=SINK_DEMAND_MEAN, std=SINK_DEMAND_STD),
        income_rate=SINK_INCOME_RATE,
        cash=1000.0,
    )
    sink.policy = sink_policy

    edges = [
        EdgeSpec(supplier_id="factory-1", buyer_id="shop-1", default_lead_time=2),
        EdgeSpec(supplier_id="shop-1", buyer_id="sink-1", default_lead_time=1),
    ]

    lifecycle = ItemLifecycleParams(
        stages=["introduction", "growth", "maturity", "decline", "dead"],
        init_stage="maturity",
        default_stage_change_probs={
            s: 0.0
            for s in ["introduction", "growth", "maturity", "decline", "dead"]
        },
    )
    disruption = DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.0),
        duration=Constant(1),
    )
    market = MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=100,
        min_value=0.5,
        max_value=2.0,
        stage_multipliers={
            "introduction": 1.0,
            "growth": 1.0,
            "maturity": 1.0,
            "decline": 1.0,
            "dead": 0.0,
        },
        price_elasticity=0.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(SINK_DEMAND_MEAN),
    )

    return Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption,
        item_lifecycle=lifecycle,
        stores=[],
        n_steps=N_STEPS,
        start_date=datetime(2024, 1, 1),
        world_seed=WORLD_SEED,
        nodes=[
            NodeInstance(node=factory, init_seed=1),
            NodeInstance(node=shop, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=edges,
    )


# Module-level ``scenario`` so the CLI shim (``main.py``) can load and run
# this file like any other scenario. ``main()`` below preserves the
# standalone smoke-test output when the file is executed directly.
scenario = _build_scenario()


def main() -> None:
    runner = GraphRunner(scenario)
    log = runner.run()

    # Write output to runs/example_chain_three_node/.
    out_dir = _PROJECT_ROOT / "runs" / "example_chain_three_node"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"Chain scenario complete: {N_STEPS} ticks, world_seed={WORLD_SEED}")
    print(f"Run log written to: {log_path}")

    # Print a brief summary.
    last_tick = log["ticks"][-1] if log["ticks"] else {}
    print("\nFinal state (last tick):")
    for node_id, cash in last_tick.get("node_cash", {}).items():
        inv = last_tick.get("node_inventory", {}).get(node_id, "N/A")
        print(f"  {node_id}: cash={cash:.2f}, inventory={inv}")


if __name__ == "__main__":
    main()
