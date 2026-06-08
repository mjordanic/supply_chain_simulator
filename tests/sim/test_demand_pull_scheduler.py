"""Tests for the demand-pull topological scheduler (issue 02).

Covers build_demand_pull_schedule (pure graph.py) and the unified
_run_demand_pull_schedule runner integration.
"""

from __future__ import annotations

from datetime import datetime
from random import Random

import pytest

from src.sim.graph import EdgeSpec, build_demand_pull_schedule, build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chain_graph():
    """factory → shop → sink (strict linear chain)."""
    nodes = ["factory", "shop", "sink"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=1),
        EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=1),
    ]
    return build_graph(nodes, edges)


def _build_lateral_graph():
    """factory → wh1 → wh2 → sink, lateral wh1 → wh2 makes wh2 depend on wh1."""
    nodes = ["factory", "wh1", "wh2", "sink"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="factory", buyer_id="wh2", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="wh2", default_lead_time=1),  # lateral
        EdgeSpec(supplier_id="wh2", buyer_id="sink", default_lead_time=1),
    ]
    return build_graph(nodes, edges)


def _build_diamond_graph():
    """factory → wh1, factory → wh2, wh1 → sink, wh2 → sink.
    wh1 and wh2 are incomparable peers (diamond shape).
    """
    nodes = ["factory", "wh1", "wh2", "sink"]
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="wh1", default_lead_time=1),
        EdgeSpec(supplier_id="factory", buyer_id="wh2", default_lead_time=1),
        EdgeSpec(supplier_id="wh1", buyer_id="sink", default_lead_time=1),
        EdgeSpec(supplier_id="wh2", buyer_id="sink", default_lead_time=1),
    ]
    return build_graph(nodes, edges)


# ---------------------------------------------------------------------------
# build_demand_pull_schedule: strict chain
# ---------------------------------------------------------------------------

class TestDemandPullScheduleChain:
    def test_chain_yields_singleton_ready_sets(self):
        """A strict chain must yield one node per ready-set (no shuffle effect)."""
        g = _build_chain_graph()
        schedule = build_demand_pull_schedule(g)
        assert len(schedule) == 3
        for ready_set in schedule:
            assert len(ready_set) == 1

    def test_chain_order_is_sink_first_factory_last(self):
        """Sinks (graph-terminals) must appear first; factories last."""
        g = _build_chain_graph()
        schedule = build_demand_pull_schedule(g)
        node_order = [s[0] for s in schedule]
        assert node_order == ["sink", "shop", "factory"]

    def test_chain_covers_all_nodes(self):
        """All nodes must appear in exactly one ready-set."""
        g = _build_chain_graph()
        schedule = build_demand_pull_schedule(g)
        all_nodes = [n for s in schedule for n in s]
        assert sorted(all_nodes) == sorted(["factory", "shop", "sink"])

    def test_chain_deterministic(self):
        """Same graph → identical schedule on repeated calls."""
        g = _build_chain_graph()
        s1 = build_demand_pull_schedule(g)
        s2 = build_demand_pull_schedule(g)
        assert s1 == s2


# ---------------------------------------------------------------------------
# build_demand_pull_schedule: lateral edge forces dependency order
# ---------------------------------------------------------------------------

class TestDemandPullScheduleLateral:
    def test_lateral_edge_forces_wh2_before_wh1(self):
        """Lateral wh1→wh2 means wh2 buys from wh1, so wh2 must be processed
        before wh1 in demand-pull order (wh2 is closer to the sink on the
        supply chain)."""
        g = _build_lateral_graph()
        schedule = build_demand_pull_schedule(g)
        all_nodes_flat = [n for s in schedule for n in s]
        assert all_nodes_flat.index("wh2") < all_nodes_flat.index("wh1"), (
            "wh2 (buyer of wh1 laterally) must come before wh1 in demand-pull order"
        )

    def test_lateral_graph_sink_is_first(self):
        """Sink (graph-terminal) must be in the first ready-set."""
        g = _build_lateral_graph()
        schedule = build_demand_pull_schedule(g)
        assert "sink" in schedule[0]

    def test_lateral_graph_factory_is_last(self):
        """Factory (graph-source) must be in the last ready-set."""
        g = _build_lateral_graph()
        schedule = build_demand_pull_schedule(g)
        assert "factory" in schedule[-1]

    def test_lateral_graph_covers_all_nodes(self):
        g = _build_lateral_graph()
        schedule = build_demand_pull_schedule(g)
        all_nodes = [n for s in schedule for n in s]
        assert sorted(all_nodes) == ["factory", "sink", "wh1", "wh2"]


# ---------------------------------------------------------------------------
# build_demand_pull_schedule: diamond (incomparable peers)
# ---------------------------------------------------------------------------

class TestDemandPullScheduleDiamond:
    def test_diamond_peers_in_same_ready_set(self):
        """wh1 and wh2 are incomparable (neither is a supplier of the other),
        so they must appear in the same ready-set."""
        g = _build_diamond_graph()
        schedule = build_demand_pull_schedule(g)
        # Find which ready-set contains wh1
        for ready_set in schedule:
            if "wh1" in ready_set:
                assert "wh2" in ready_set, (
                    "Incomparable peers wh1 and wh2 must be in the same ready-set"
                )
                break

    def test_diamond_sink_first_factory_last(self):
        g = _build_diamond_graph()
        schedule = build_demand_pull_schedule(g)
        all_nodes_flat = [n for s in schedule for n in s]
        assert all_nodes_flat.index("sink") < all_nodes_flat.index("factory")

    def test_diamond_three_ready_sets(self):
        """Diamond must yield 3 ready-sets: [sink], [wh1,wh2], [factory]."""
        g = _build_diamond_graph()
        schedule = build_demand_pull_schedule(g)
        assert len(schedule) == 3
        assert len(schedule[0]) == 1  # sink
        assert len(schedule[1]) == 2  # wh1, wh2
        assert len(schedule[2]) == 1  # factory


# ---------------------------------------------------------------------------
# Shuffle seam: same topology + seed → same order; different seed → different
# ---------------------------------------------------------------------------

class TestDemandPullShuffleSeed:
    def test_same_seed_same_order(self):
        """Applying the same RNG seed to each ready-set gives the same total order."""
        g = _build_diamond_graph()
        schedule = build_demand_pull_schedule(g)

        def _apply_shuffle(seed: int) -> list[str]:
            rng = Random(seed)
            order: list[str] = []
            for ready_set in schedule:
                shuffled = list(ready_set)
                rng.shuffle(shuffled)
                order.extend(shuffled)
            return order

        order1 = _apply_shuffle(42)
        order2 = _apply_shuffle(42)
        assert order1 == order2, "Same seed must produce identical node order"

    def test_different_seeds_may_give_different_peer_order(self):
        """Different seeds should (with high probability) produce a different
        order within a multi-node ready-set."""
        g = _build_diamond_graph()
        schedule = build_demand_pull_schedule(g)

        # Find the ready-set with multiple nodes
        peer_set = next(s for s in schedule if len(s) > 1)
        assert len(peer_set) == 2  # sanity check

        orders = set()
        for seed in range(20):
            rng = Random(seed)
            shuffled = list(peer_set)
            rng.shuffle(shuffled)
            orders.add(tuple(shuffled))
        # With 20 seeds over 2 elements, we expect both permutations to appear
        assert len(orders) == 2, (
            "Different seeds should produce both permutations of 2-element peer set"
        )


# ---------------------------------------------------------------------------
# Integration: Runner uses demand-pull schedule, chain unaffected in shape
# ---------------------------------------------------------------------------

def _minimal_market_params():
    from src.sim.distributions import Constant
    from src.sim.scenario import MarketParams
    return MarketParams(
        cycle_len=365, cycle_amp=0.0, init_demand=1.0, init_supply=1.0,
        peak_factor=1.0, off_factor=1.0, season_months={}, regions=["US"],
        correlation=0.0, trend_update_interval=100, min_value=0.5, max_value=2.0,
        stage_multipliers={
            "introduction": 1.0, "growth": 1.0, "maturity": 1.0,
            "decline": 1.0, "dead": 0.0,
        },
        price_elasticity=0.0, promo_multiplier=1.0,
        demand_factor_min=0.1, supply_factor_min=0.01,
        cross_inv_lo=0.3, cross_inv_hi=0.7, cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0), demand_shock=Constant(0.0), supply_shock=Constant(0.0),
        base_demand=Constant(10),
    )


def _minimal_disruption_params():
    from src.sim.distributions import Constant
    from src.sim.scenario import DisruptionParams
    return DisruptionParams(
        event_prob=0.0, types=["natural_disaster"], regions=["US"],
        severity=Constant(0.1), duration=Constant(1),
    )


def _minimal_lifecycle_params():
    from src.sim.scenario import ItemLifecycleParams
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages, init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _minimal_catalog():
    from src.sim.scenario import Ware, load_catalog
    return load_catalog([{
        "name": "Widget", "category": "test", "related_products": [],
        "base_price": 10.0, "unit_cost": 5.0, "seasonality": "all_season",
    }])


class TestRunnerDemandPullIntegration:
    """Integration tests: Runner uses demand-pull schedule."""

    def _build_chain_scenario(self, n_steps: int = 20, world_seed: int = 42):
        from src.sim.distributions import Constant
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.scenario import NodeInstance, Scenario

        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        factory = FactoryNode(
            id="factory-1", region="US", init_seed=1,
            produces_product_id=pid, unit_cost=5.0,
            capacity_per_tick=100, inventory=200, list_price=5.0, cash=0.0,
        )
        shop = IntermediateNode(
            id="shop-1", region="US", init_seed=2,
            carried_products={pid}, capacity=500, tags=["shop"],
            inventory={pid: 50}, pending={},
            list_prices={pid: 8.0}, min_order_imposed={pid: 0}, cash=500.0,
        )
        sink = DemandSinkNode(
            id="sink-1", region="US", init_seed=3,
            product_id=pid, demand_dist=Constant(5),
            income_rate=100.0, cash=1000.0,
        )
        edges = [
            EdgeSpec(supplier_id="factory-1", buyer_id="shop-1", default_lead_time=2),
            EdgeSpec(supplier_id="shop-1", buyer_id="sink-1", default_lead_time=1),
        ]
        return Scenario(
            catalog=catalog,
            market=_minimal_market_params(),
            disruption=_minimal_disruption_params(),
            item_lifecycle=_minimal_lifecycle_params(),
            n_steps=n_steps,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
            nodes=[
                NodeInstance(node=factory, init_seed=1),
                NodeInstance(node=shop, init_seed=2),
                NodeInstance(node=sink, init_seed=3),
            ],
            edges=edges,
        )

    def test_chain_runs_without_error(self):
        from src.sim.runner import Runner
        scenario = self._build_chain_scenario()
        log = Runner(scenario).run()
        assert log["n_steps"] == 20

    def test_chain_is_bit_deterministic(self):
        """Same seed → identical cash sequence."""
        from src.sim.runner import Runner
        s1 = self._build_chain_scenario(world_seed=99)
        s2 = self._build_chain_scenario(world_seed=99)
        log1 = Runner(s1).run()
        log2 = Runner(s2).run()
        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["node_cash"] == t2["node_cash"]

    def test_chain_no_negative_inventory(self):
        from src.sim.runner import Runner
        from src.sim.node import FactoryNode, IntermediateNode
        scenario = self._build_chain_scenario(n_steps=50)
        log = Runner(scenario).run()
        for tick_log in log["ticks"]:
            for node_id, inv in tick_log["node_inventory"].items():
                if isinstance(inv, dict):
                    for p, qty in inv.items():
                        assert qty >= 0

    def test_tick_world_and_decide_settle_still_work(self):
        """The RL two-phase tick API must remain functional."""
        from src.sim.runner import build_world
        scenario = self._build_chain_scenario(n_steps=5)
        sim = build_world(scenario)
        for _ in range(5):
            current_tick = sim.tick_world()
            sim.tick_decide_and_settle(current_tick)

    def test_schedule_cached_on_simulation(self):
        """build_world must cache the demand-pull schedule on the Simulation."""
        from src.sim.runner import build_world
        scenario = self._build_chain_scenario()
        sim = build_world(scenario)
        assert hasattr(sim, "schedule"), "Simulation must have a 'schedule' attribute"
        assert isinstance(sim.schedule, list)
        assert len(sim.schedule) > 0

    def test_cash_conservation(self):
        """Cash + inventory-in-transit must equal initial + income (ADR 0013).

        Uses zero holding_rate and order_fee so void charges do not appear in
        the balance.  The full void-accounting conservation identity (ADR 0013
        Rule 5) is tested separately in issue 06.
        """
        from src.sim.node import FactoryNode, IntermediateNode
        from src.sim.runner import build_world
        scenario = self._build_chain_scenario(n_steps=30)
        # Zero out holding/fee so this classic conservation test stays valid;
        # void charges are tested by the Rule-5 identity test in issue 06.
        for ni in scenario.nodes:
            if isinstance(ni.node, IntermediateNode):
                ni.node.holding_rate = 0.0
                ni.node.order_fee = 0.0
        sim = build_world(scenario)
        initial_total = sum(getattr(n, "cash", 0.0) for n in sim.nodes.values())
        income_per_tick = sum(
            getattr(n, "income_rate", 0.0) for n in sim.nodes.values()
        )
        factory_cash_0 = {
            nid: n.cash for nid, n in sim.nodes.items() if isinstance(n, FactoryNode)
        }
        for t in range(1, 31):
            sim.tick()
            current_total = sum(
                getattr(n, "cash", 0.0) for n in sim.nodes.values()
            )
            in_transit = sum(
                factory_cash_0[nid] - sim.nodes[nid].cash for nid in factory_cash_0
            )
            conserved = current_total + in_transit
            expected = initial_total + t * income_per_tick
            assert abs(conserved - expected) < 1e-6, (
                f"Cash conservation violated at tick {t}: got {conserved:.6f}, "
                f"expected {expected:.6f}"
            )


class TestRunnerLateralGraph:
    """Integration: lateral graph settles by supply dependency, not by shuffle."""

    def _build_lateral_scenario(self, n_steps: int = 20, world_seed: int = 42):
        """factory → wh1 and factory → wh2, wh1 → wh2 (lateral), wh2 → sink."""
        from src.sim.distributions import Constant
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.scenario import NodeInstance, Scenario

        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        factory = FactoryNode(
            id="factory", region="US", init_seed=1,
            produces_product_id=pid, unit_cost=5.0,
            capacity_per_tick=200, inventory=500, list_price=5.0, cash=0.0,
        )
        wh1 = IntermediateNode(
            id="wh1", region="US", init_seed=2,
            carried_products={pid}, capacity=1000, tags=["warehouse"],
            inventory={pid: 200}, pending={},
            list_prices={pid: 7.0}, min_order_imposed={pid: 0}, cash=5000.0,
        )
        wh2 = IntermediateNode(
            id="wh2", region="US", init_seed=3,
            carried_products={pid}, capacity=1000, tags=["warehouse"],
            inventory={pid: 200}, pending={},
            list_prices={pid: 9.0}, min_order_imposed={pid: 0}, cash=5000.0,
        )
        sink = DemandSinkNode(
            id="sink", region="US", init_seed=4,
            product_id=pid, demand_dist=Constant(5),
            income_rate=200.0, cash=5000.0,
        )
        edges = [
            EdgeSpec(supplier_id="factory", buyer_id="wh1", default_lead_time=1),
            EdgeSpec(supplier_id="factory", buyer_id="wh2", default_lead_time=1),
            EdgeSpec(supplier_id="wh1", buyer_id="wh2", default_lead_time=1),  # lateral
            EdgeSpec(supplier_id="wh2", buyer_id="sink", default_lead_time=1),
        ]
        return Scenario(
            catalog=catalog,
            market=_minimal_market_params(),
            disruption=_minimal_disruption_params(),
            item_lifecycle=_minimal_lifecycle_params(),
            n_steps=n_steps,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
            nodes=[
                NodeInstance(node=factory, init_seed=1),
                NodeInstance(node=wh1, init_seed=2),
                NodeInstance(node=wh2, init_seed=3),
                NodeInstance(node=sink, init_seed=4),
            ],
            edges=edges,
        )

    def test_lateral_graph_runs_without_error(self):
        from src.sim.runner import Runner
        scenario = self._build_lateral_scenario()
        log = Runner(scenario).run()
        assert log["n_steps"] == 20

    def test_lateral_graph_is_bit_deterministic(self):
        from src.sim.runner import Runner
        s1 = self._build_lateral_scenario(world_seed=7)
        s2 = self._build_lateral_scenario(world_seed=7)
        log1 = Runner(s1).run()
        log2 = Runner(s2).run()
        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["node_cash"] == t2["node_cash"]

    def test_lateral_graph_no_negative_inventory(self):
        from src.sim.runner import Runner
        scenario = self._build_lateral_scenario(n_steps=30)
        log = Runner(scenario).run()
        for tick_log in log["ticks"]:
            for node_id, inv in tick_log["node_inventory"].items():
                if isinstance(inv, dict):
                    for p, qty in inv.items():
                        assert qty >= 0

    def test_lateral_graph_cash_conservation(self):
        from src.sim.node import FactoryNode, IntermediateNode
        from src.sim.runner import build_world
        scenario = self._build_lateral_scenario(n_steps=20)
        # Zero out holding/fee so void charges don't appear in this balance;
        # full void-accounting Rule-5 conservation is tested in issue 06.
        for ni in scenario.nodes:
            if isinstance(ni.node, IntermediateNode):
                ni.node.holding_rate = 0.0
                ni.node.order_fee = 0.0
        sim = build_world(scenario)
        initial_total = sum(getattr(n, "cash", 0.0) for n in sim.nodes.values())
        income_per_tick = sum(
            getattr(n, "income_rate", 0.0) for n in sim.nodes.values()
        )
        factory_cash_0 = {
            nid: n.cash for nid, n in sim.nodes.items() if isinstance(n, FactoryNode)
        }
        for t in range(1, 21):
            sim.tick()
            current_total = sum(
                getattr(n, "cash", 0.0) for n in sim.nodes.values()
            )
            in_transit = sum(
                factory_cash_0[nid] - sim.nodes[nid].cash for nid in factory_cash_0
            )
            conserved = current_total + in_transit
            expected = initial_total + t * income_per_tick
            assert abs(conserved - expected) < 1e-6


class TestUnlaggedDemandSignal:
    """Issue 03: observed_sales carries current-tick (un-lagged) demand.

    The demand-pull walk processes sinks before intermediates, so by the time
    an intermediate node is processed its ``observed_sales`` already reflects
    all sales that happened in the same tick.
    """

    def _build_simple_chain_sim(self, demand: int = 5, init_stock: int = 100):
        """factory → shop → sink; shop starts with init_stock units."""
        from src.sim.distributions import Constant
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.runner import build_world
        from src.sim.scenario import NodeInstance, Scenario

        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        factory = FactoryNode(
            id="factory", region="US", init_seed=1,
            produces_product_id=pid, unit_cost=1.0,
            capacity_per_tick=200, inventory=500, list_price=1.0, cash=0.0,
        )
        shop = IntermediateNode(
            id="shop", region="US", init_seed=2,
            carried_products={pid}, capacity=1000, tags=["shop"],
            inventory={pid: init_stock}, pending={},
            list_prices={pid: 2.0}, min_order_imposed={pid: 0}, cash=5000.0,
        )
        sink = DemandSinkNode(
            id="sink", region="US", init_seed=3,
            product_id=pid, demand_dist=Constant(demand),
            income_rate=500.0, cash=5000.0,
        )
        edges = [
            EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=1),
            EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=1),
        ]
        scenario = Scenario(
            catalog=catalog,
            market=_minimal_market_params(),
            disruption=_minimal_disruption_params(),
            item_lifecycle=_minimal_lifecycle_params(),
            n_steps=10,
            start_date=datetime(2024, 1, 1),
            world_seed=42,
            nodes=[
                NodeInstance(node=factory, init_seed=1),
                NodeInstance(node=shop, init_seed=2),
                NodeInstance(node=sink, init_seed=3),
            ],
            edges=edges,
        )
        return build_world(scenario), pid

    def test_observed_sales_present_in_intermediate_obs(self):
        """observed_sales is injected into the intermediate observation each tick."""
        from src.sim.runner import _run_demand_pull_schedule
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import IntermediateNode, FactoryNode
        from src.sim.observation import build_intermediate_obs

        sim, pid = self._build_simple_chain_sim(demand=5, init_stock=50)

        # Manually run one tick's world phase to get the table.
        sim.market.tick()
        sim.event_engine.tick(sim.market)
        current_tick = sim.market.current_step()

        # Publish offers.
        table = CentralTable()
        for nid, node in sim.nodes.items():
            if isinstance(node, FactoryNode):
                table.publish(nid, node.produces_product_id,
                              Offer(available_qty=node.inventory, list_price=node.list_price, min_order=0))
            elif isinstance(node, IntermediateNode):
                for p in node.carried_products:
                    table.publish(nid, p,
                                  Offer(available_qty=node.inventory.get(p, 0),
                                        list_price=node.list_prices.get(p, 0.0), min_order=0))

        # Reset accumulators.
        sim._last_tick_orders = {}
        sim._tick_sales = {}

        # Capture observed_sales passed to shop using a spy policy.
        captured_obs: list[dict] = []

        class SpyPolicy:
            def decide(self, obs, table):
                captured_obs.append(dict(obs))
                return {"order": {}, "list_price": {}, "min_order_imposed": {}}

        sim.nodes["shop"].policy = SpyPolicy()

        _run_demand_pull_schedule(sim, table, current_tick)

        assert len(captured_obs) == 1
        obs = captured_obs[0]
        assert "observed_sales" in obs, "observed_sales must be injected into intermediate obs"
        # Sink buys demand=5 units; shop sold 5 units to sink this tick.
        assert obs["observed_sales"].get(pid, 0) == 5, (
            f"expected 5 current-tick sales for pid={pid}, got {obs['observed_sales']}"
        )

    def test_observed_sales_zero_when_no_downstream_buyers(self):
        """observed_sales is {} (not lagged) when the shop has no downstream demand."""
        from src.sim.runner import _run_demand_pull_schedule
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import IntermediateNode, FactoryNode

        # Build a sim where the sink has zero demand.
        sim, pid = self._build_simple_chain_sim(demand=0, init_stock=50)

        sim.market.tick()
        sim.event_engine.tick(sim.market)
        current_tick = sim.market.current_step()

        table = CentralTable()
        for nid, node in sim.nodes.items():
            if isinstance(node, FactoryNode):
                table.publish(nid, node.produces_product_id,
                              Offer(available_qty=node.inventory, list_price=node.list_price, min_order=0))
            elif isinstance(node, IntermediateNode):
                for p in node.carried_products:
                    table.publish(nid, p,
                                  Offer(available_qty=node.inventory.get(p, 0),
                                        list_price=node.list_prices.get(p, 0.0), min_order=0))

        sim._last_tick_orders = {}
        sim._tick_sales = {}

        captured_obs: list[dict] = []

        class SpyPolicy:
            def decide(self, obs, table):
                captured_obs.append(dict(obs))
                return {"order": {}, "list_price": {}, "min_order_imposed": {}}

        sim.nodes["shop"].policy = SpyPolicy()

        _run_demand_pull_schedule(sim, table, current_tick)

        assert len(captured_obs) == 1
        assert captured_obs[0].get("observed_sales", {}) == {}, (
            "observed_sales should be empty dict when no downstream sales occurred"
        )

    def test_intermediate_reorders_from_same_tick_sales_on_lateral_graph(self):
        """End-to-end: on a lateral graph, shop reorders in response to same-tick sales.

        factory → wh → sink.  wh starts with exactly one period of stock
        (init_stock == demand).  After one tick the sink buys all available
        units; observed_sales for wh equals demand; the textbook policy,
        seeing sales=demand and position falling to 0, must place a reorder.
        """
        from src.sim.distributions import Constant
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.policy import MultiSupplierTextbookPolicy
        from src.sim.runner import build_world
        from src.sim.scenario import NodeInstance, Scenario

        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        demand = 5
        factory = FactoryNode(
            id="factory", region="US", init_seed=1,
            produces_product_id=pid, unit_cost=1.0,
            capacity_per_tick=200, inventory=500, list_price=1.0, cash=0.0,
        )
        wh = IntermediateNode(
            id="wh", region="US", init_seed=2,
            carried_products={pid}, capacity=2000, tags=["warehouse"],
            inventory={pid: demand},  # exactly one tick's worth of stock
            pending={},
            list_prices={pid: 2.0}, min_order_imposed={pid: 0}, cash=50_000.0,
        )
        sink = DemandSinkNode(
            id="sink", region="US", init_seed=3,
            product_id=pid, demand_dist=Constant(demand),
            income_rate=5000.0, cash=50_000.0,
        )
        # Attach a textbook policy to wh that will reorder once it sees sales.
        wh_policy = MultiSupplierTextbookPolicy(
            delivery_lag=1,
            unit_cost=1.0,
            list_price_out=2.0,
            cover_horizon_ticks=5,
            safety_lead_pct_of_lag=0.0,
        )
        edges = [
            EdgeSpec(supplier_id="factory", buyer_id="wh", default_lead_time=1),
            EdgeSpec(supplier_id="wh", buyer_id="sink", default_lead_time=1),
        ]
        scenario = Scenario(
            catalog=catalog,
            market=_minimal_market_params(),
            disruption=_minimal_disruption_params(),
            item_lifecycle=_minimal_lifecycle_params(),
            n_steps=5,
            start_date=datetime(2024, 1, 1),
            world_seed=7,
            nodes=[
                NodeInstance(node=factory, init_seed=1),
                NodeInstance(node=wh, init_seed=2, policy=wh_policy),
                NodeInstance(node=sink, init_seed=3),
            ],
            edges=edges,
        )
        sim = build_world(scenario)
        sim.tick()

        # After tick 1, the sink bought from wh; wh should have placed a reorder.
        orders_tick1 = sim._last_tick_orders.get("wh", {})
        assert orders_tick1.get(pid, 0) > 0, (
            f"wh must place a reorder on tick 1 after observing same-tick sales; "
            f"got orders={orders_tick1}"
        )
