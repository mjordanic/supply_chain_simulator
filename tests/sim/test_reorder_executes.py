"""Regression: a shop's textbook policy must actually place reorders that
execute through the ``Runner``, and every carried product must have an
upstream producer.

Guards two bugs that together produced zero reordering in the rewritten
notebooks (notebook 03 in particular):

1. The runner did not inject ``direct_supplier_ids`` into the intermediate
   observation, so ``MultiSupplierTextbookPolicy`` fell back to the central
   table — which lists the shop's *own* published offers — and routed every
   reorder to the shop itself, where the runner's safety filter dropped them.
2. ``world_to_graph`` built a single factory producing only the first catalog
   product, leaving every other carried product without a producer, so even
   correctly-routed reorders filled nothing.

The scenario mirrors ``notebooks/03-run-and-inspect-simulation.ipynb``: it
loads the committed ``fashion_retail_20`` world (no LLM / API key needed) and
runs the graph engine end-to-end.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from src.sim.distributions import Uniform
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import DefaultDemandSinkPolicy, OrderUpToPolicy, StaticFactoryPolicy
from src.sim.runner import Runner
from src.sim.scenario import DisruptionParams, ItemLifecycleParams, Scenario
from src.sim.world import World, world_to_graph

WORLD_PATH = Path("data/worlds/fashion_retail_20/world.json")


def _world() -> World:
    world = World.from_json(WORLD_PATH)
    _base = next(iter(world.store_templates.values()))
    template = replace(
        _base,
        id="flagship_20_0",
        capacity=400,
        init_balance=500_000.0,
        init_active_count=10,
        init_freshness="baseline",
        init_stock_pct=0.0,
        order_fee=10.0,
    )
    return replace(world, store_templates={"flagship_20_0": template})


def _build_scenario(world: World) -> Scenario:
    topology = world_to_graph(world, sink_density=1.0)
    for ni in topology["node_instances"]:
        node = ni.node
        if isinstance(node, FactoryNode):
            ni.policy = StaticFactoryPolicy(
                capacity_per_tick=node.capacity_per_tick, unit_cost=node.unit_cost
            )
        elif isinstance(node, IntermediateNode):
            ni.policy = OrderUpToPolicy(policy_seed=1000, cover_horizon_ticks=30)
        elif isinstance(node, DemandSinkNode):
            ni.policy = DefaultDemandSinkPolicy()
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return Scenario.from_world(
        world,
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=world.market.regions,
            severity=Uniform(0.005, 0.02),
            duration=Uniform(5, 20),
        ),
        item_lifecycle=ItemLifecycleParams(
            stages=stages,
            init_stage="maturity",
            default_stage_change_probs={s: 0.0 for s in stages},
        ),
        nodes=topology["node_instances"],
        edges=topology["edges"],
        n_steps=40,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


@pytest.mark.skipif(
    not WORLD_PATH.exists(), reason="fashion_retail_20 world fixture missing"
)
def test_every_carried_product_has_a_factory() -> None:
    """``world_to_graph`` gives each carried product its own producer (Bug 2)."""
    topology = world_to_graph(_world(), sink_density=1.0)
    shop = next(
        ni.node
        for ni in topology["node_instances"]
        if isinstance(ni.node, IntermediateNode)
    )
    produced = {
        ni.node.produces_product_id
        for ni in topology["node_instances"]
        if isinstance(ni.node, FactoryNode)
    }
    assert shop.carried_products <= produced


@pytest.mark.skipif(
    not WORLD_PATH.exists(), reason="fashion_retail_20 world fixture missing"
)
def test_shop_reorders_and_replenishes() -> None:
    """The shop places reorders that execute, and a drained SKU recovers."""
    scenario = _build_scenario(_world())
    run_log = Runner(scenario).run()

    shop_id = next(
        ni.node.id for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)
    )

    # Bug 1: at least one tick books a non-zero, executed order for the shop.
    total_ordered = sum(
        sum(tick.get("node_orders", {}).get(shop_id, {}).values())
        for tick in run_log["ticks"]
    )
    assert total_ordered > 0, "shop never placed an executable reorder"

    # Bug 2: at least one carried SKU recovers after dropping to its trough —
    # i.e. a delivered reorder actually replenished the shelf, not orders that
    # route correctly but fill nothing.
    inv_by_pid: dict[str, list[int]] = {}
    for tick in run_log["ticks"]:
        for pid, qty in tick.get("node_inventory", {}).get(shop_id, {}).items():
            inv_by_pid.setdefault(pid, []).append(qty)

    def recovers(series: list[int]) -> bool:
        trough = min(series)
        trough_idx = series.index(trough)
        return any(q > trough for q in series[trough_idx + 1 :])

    assert any(recovers(s) for s in inv_by_pid.values()), (
        "no carried product was replenished after draining"
    )
