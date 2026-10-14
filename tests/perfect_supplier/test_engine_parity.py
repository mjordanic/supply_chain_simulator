"""The lean simulator and ``src/sim`` agree on one demand path.

Both run the same textbook rule against the same replayed demand, with one
factory per product (the engine charges one fee per supplier edge, so the fee
is zero here; the shared fee is tested in ``test_sim``).
"""

import dataclasses
from datetime import datetime

import numpy as np
import pytest

from src.perfect_supplier.sim import simulate
from src.perfect_supplier.textbook import FAMILIES, TextbookParams, TextbookPolicy
from src.perfect_supplier.world import HISTORY_DAYS, demand_path, sample_world
from src.sim.distributions import Constant
from src.sim.flat_world import flat_world
from src.sim.graph import EdgeSpec
from src.sim.node import FactoryNode, IntermediateNode
from src.sim.policy import (
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
    StaticFactoryPolicy,
)
from src.sim.replay_demand_sink import ReplayDemandSinkNode
from src.sim.runner import Runner
from src.sim.scenario import DisruptionParams, NodeInstance, Scenario, Ware

ENGINE = {
    "order_up_to": OrderUpToPolicy,
    "reorder_point": ReorderPointPolicy,
    "periodic_order_up_to": PeriodicOrderUpToPolicy,
    "periodic_reorder": PeriodicReorderPolicy,
}


def _engine_run(world, demand, policy):
    pids = [f"P{p}" for p in range(world.n_products)]
    market, lifecycle = flat_world(regions=["US"])
    wares = [
        Ware(
            product_id=pid, name=pid, category="x", base_price=world.price, unit_cost=world.unit_cost,
            seasonality="default", related_products=[], init_stage=None, stage_change_probs=None,
            freshness_alpha=None, freshness_decay=None, init_stock_share=1.0,
        )
        for pid in pids
    ]
    nodes, edges = [], []
    for i, pid in enumerate(pids):
        factory = FactoryNode(
            id=f"factory-{pid}", region="US", init_seed=i, produces_product_id=pid,
            unit_cost=world.unit_cost, capacity_per_tick=Constant(100_000), inventory=100_000,
            list_price=world.unit_cost, cash=0.0,
        )
        nodes.append(NodeInstance(node=factory, init_seed=i, policy=StaticFactoryPolicy(
            capacity_per_tick=100_000, unit_cost=world.unit_cost, policy_seed=i)))
        edges.append(EdgeSpec(supplier_id=factory.id, buyer_id="shop", default_lead_time=world.delivery_lag))
    shop = IntermediateNode(
        id="shop", region="US", init_seed=100, carried_products=set(pids), capacity=world.capacity,
        tags=["shop"], inventory={pid: int(q) for pid, q in zip(pids, world.initial_on_hand)},
        list_prices={pid: world.price for pid in pids}, min_order_imposed={pid: 0 for pid in pids},
        cash=world.initial_cash, holding_rate=world.holding_rate, order_fee=0.0,
    )
    nodes.append(NodeInstance(node=shop, init_seed=100, policy=None))
    for i, pid in enumerate(pids):
        sink = ReplayDemandSinkNode(
            id=f"sink-{pid}", region="US", init_seed=200 + i, product_id=pid,
            series=[int(x) for x in demand[HISTORY_DAYS:, i]], n_steps=world.n_days,
            income_rate=1e9, cash=1e12, demand_dist=Constant(0),
        )
        nodes.append(NodeInstance(node=sink, init_seed=200 + i, policy=None))
        edges.append(EdgeSpec(supplier_id="shop", buyer_id=sink.id, default_lead_time=0))
    scenario = Scenario(
        catalog=wares, market=market,
        disruption=DisruptionParams(event_prob=0.0, types=["natural_disaster"], regions=["US"],
                                    severity=Constant(0.0), duration=Constant(1)),
        item_lifecycle=lifecycle, nodes=nodes, edges=edges, n_steps=world.n_days,
        start_date=datetime(2024, 1, 1), world_seed=1,
    )
    # The lean shop has HISTORY_DAYS of sales before day 0; give the engine rule the same.
    for i, pid in enumerate(pids):
        history = [int(x) for x in demand[:HISTORY_DAYS, i]]
        policy.sales_log[pid] = list(history)
        policy.inv_before_settle_log[pid] = [x + 10**6 for x in history]
    log = Runner(scenario, policy_overrides={"shop": policy}).run()
    on_hand = np.array([[t["node_inventory"]["shop"][pid] for pid in pids] for t in log["ticks"]])
    cash = np.array([t["node_cash"]["shop"] for t in log["ticks"]])
    return on_hand, cash


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [1, 2])
def test_lean_simulator_matches_the_engine(family, seed):
    world = dataclasses.replace(sample_world(seed), order_fee=0.0, n_days=120)
    demand = demand_path(world)
    params = TextbookParams(family=family, stockout_safety_bonus_pct_of_lag=0.5)
    lean = simulate(world, TextbookPolicy(params), demand=demand).frame
    engine_policy = ENGINE[family](
        delivery_lag=world.delivery_lag, stockout_safety_bonus_pct_of_lag=0.5, policy_seed=0
    )
    on_hand, cash = _engine_run(world, demand, engine_policy)
    lean_on_hand = lean.pivot(index="day", columns="product", values="on_hand_close").to_numpy()
    lean_cash = lean.groupby("day")["cash_close"].first().to_numpy()
    np.testing.assert_array_equal(lean_on_hand, on_hand)
    np.testing.assert_allclose(lean_cash, cash, rtol=1e-9)
