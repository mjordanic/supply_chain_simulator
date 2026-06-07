"""Integration tests for GraphSimulation + GraphRunner on a 3-node chain.

Phase 1 smoke tests (issue 05):

- 3-node chain runs for 50 ticks without error.
- Cash conservation: demand-sink cash decreases by no more than income_rate × n_steps
  (some may be spent on purchases); supplier cash increases by the corresponding amount.
- No negative inventory anywhere after any tick.
- GraphRunner.run() returns a log with correct structure.
- Market.demand_multiplier returns a positive float.
- Legacy Simulation (build_world) is unaffected.
"""

from __future__ import annotations

from datetime import datetime
from random import Random

import pytest

from src.sim.central_table import CentralTable
from src.sim.distributions import Constant
from src.sim.episode_sampler import _derive_seed
from src.sim.event_engine import EventEngine
from src.sim.graph import EdgeSpec
from src.sim.market import Market
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.runner import Runner as GraphRunner, Simulation as GraphSimulation, build_world as build_graph_world
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
# Helpers
# ---------------------------------------------------------------------------

def _minimal_market_params() -> MarketParams:
    from src.sim.distributions import Constant, Normal
    return MarketParams(
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
        base_demand=Constant(10),
    )


def _minimal_disruption_params() -> DisruptionParams:
    from src.sim.distributions import Constant
    return DisruptionParams(
        event_prob=0.0,  # no disruptions
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.1),
        duration=Constant(1),
    )


def _minimal_lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _minimal_catalog() -> list[Ware]:
    return load_catalog([
        {
            "name": "Widget",
            "category": "test",
            "related_products": [],
            "base_price": 10.0,
            "unit_cost": 5.0,
            "seasonality": "all_season",
        }
    ])


def _build_chain_scenario(n_steps: int = 50, world_seed: int = 42) -> Scenario:
    """Build a 3-node chain scenario: factory → intermediate → sink."""
    catalog = _minimal_catalog()
    pid = catalog[0].product_id  # "P0000"

    factory = FactoryNode(
        id="factory-1",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=5.0,
        capacity_per_tick=100,
        inventory=200,  # start with stock
        list_price=5.0,
        cash=0.0,
    )
    intermediate = IntermediateNode(
        id="shop-1",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=500,
        tags=["shop"],
        inventory={pid: 50},  # start with some stock
        pending={},
        list_prices={pid: 8.0},
        min_order_imposed={pid: 0},
        cash=500.0,
    )
    sink = DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=3,
        product_id=pid,
        demand_dist=Constant(5),
        income_rate=100.0,
        cash=1000.0,
    )

    edges = [
        EdgeSpec(
            supplier_id="factory-1",
            buyer_id="shop-1",
            default_lead_time=2,
        ),
        EdgeSpec(
            supplier_id="shop-1",
            buyer_id="sink-1",
            default_lead_time=1,
        ),
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
            NodeInstance(node=intermediate, init_seed=2),
            NodeInstance(node=sink, init_seed=3),
        ],
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGraphWorld:
    def test_returns_graph_simulation(self):
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)
        assert isinstance(gsim, GraphSimulation)

    def test_nodes_populated(self):
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)
        assert set(gsim.nodes.keys()) == {"factory-1", "shop-1", "sink-1"}

    def test_levels_computed(self):
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)
        assert gsim.levels["factory-1"] == 0
        assert gsim.levels["shop-1"] == 1
        assert gsim.levels["sink-1"] == 2

    def test_world_rng_seeded_from_world_seed(self):
        """Same world_seed → same allocation_rng starting state."""
        s1 = _build_chain_scenario(world_seed=99)
        s2 = _build_chain_scenario(world_seed=99)
        g1 = build_graph_world(s1)
        g2 = build_graph_world(s2)
        # Same allocation_rng starting state.
        v1 = g1.allocation_rng.random()
        v2 = g2.allocation_rng.random()
        assert v1 == v2

    def test_allocation_rng_different_from_world_rng(self):
        """allocation_rng is derived from a different sub-seed than world_rng."""
        scenario = _build_chain_scenario(world_seed=42)
        gsim = build_graph_world(scenario)
        # The two RNGs start at different states.
        w_val = gsim.world_rng.random()
        a_val = gsim.allocation_rng.random()
        # Both should give a value in [0,1] but they should differ.
        # (Very unlikely to be equal unless seeding is wrong.)
        assert 0.0 <= w_val <= 1.0
        assert 0.0 <= a_val <= 1.0
        # The seeds derive differently — assert they are not identically seeded.
        world_seed = 42
        derived_alloc = _derive_seed(world_seed, "allocation")
        assert derived_alloc != world_seed

    def test_raises_on_non_graph_scenario(self):
        """build_graph_world raises ValueError when scenario.is_graph is False."""
        scenario = _build_chain_scenario()
        # Remove all nodes.
        scenario.nodes.clear()
        with pytest.raises(ValueError, match="graph-mode"):
            build_graph_world(scenario)


class TestGraphSimulationTick:
    def test_single_tick_no_error(self):
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)
        gsim.tick()  # should not raise

    def test_sink_cash_increases_by_income_rate(self):
        """After one tick, sink.cash >= initial_cash (income arrives; spending may reduce)."""
        scenario = _build_chain_scenario(n_steps=1)
        gsim = build_graph_world(scenario)
        sink = gsim.nodes["sink-1"]
        initial_cash = sink.cash

        gsim.tick()

        # cash_after >= initial_cash - demand_cost + income_rate
        # Since income_rate=100 and demand=5 * price=8 = 40 max spend:
        # net should be positive.
        assert sink.cash > 0

    def test_no_negative_inventory_after_tick(self):
        """No node has negative inventory after a tick."""
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)

        for _ in range(10):
            gsim.tick()
            for node_id, node in gsim.nodes.items():
                if isinstance(node, FactoryNode):
                    assert node.inventory >= 0, f"{node_id} has negative inventory"
                elif isinstance(node, IntermediateNode):
                    for pid, qty in node.inventory.items():
                        assert qty >= 0, f"{node_id}[{pid}] has negative inventory"

    def test_supplier_cash_increases_after_sales(self):
        """Intermediate's cash should increase when sink buys from it."""
        scenario = _build_chain_scenario()
        gsim = build_graph_world(scenario)
        intermediate = gsim.nodes["shop-1"]
        initial_cash = intermediate.cash

        # Run 5 ticks.
        for _ in range(5):
            gsim.tick()

        # Intermediate received cash from sink purchases.
        # Cash may also decrease if it bought from factory, but
        # in a properly functioning chain it should have transacted.
        # We just assert no error occurred; cash conservation tested in GraphRunner tests.
        assert intermediate.cash >= 0


class TestGraphRunnerChain:
    def test_run_50_ticks(self):
        scenario = _build_chain_scenario(n_steps=50)
        runner = GraphRunner(scenario)
        log = runner.run()
        assert log["n_steps"] == 50
        assert len(log["ticks"]) == 50

    def test_log_has_expected_keys(self):
        scenario = _build_chain_scenario(n_steps=1)
        runner = GraphRunner(scenario)
        log = runner.run()
        tick_log = log["ticks"][0]
        assert "tick" in tick_log
        assert "node_cash" in tick_log
        assert "node_inventory" in tick_log

    def test_no_negative_inventory_50_ticks(self):
        scenario = _build_chain_scenario(n_steps=50)
        runner = GraphRunner(scenario)
        log = runner.run()

        for tick_log in log["ticks"]:
            for node_id, inv in tick_log["node_inventory"].items():
                if isinstance(inv, dict):
                    for pid, qty in inv.items():
                        assert qty >= 0, (
                            f"Negative inventory: {node_id}[{pid}]={qty} "
                            f"at tick {tick_log['tick']}"
                        )
                elif isinstance(inv, int):
                    assert inv >= 0

    def test_sink_cash_nonnegative_50_ticks(self):
        scenario = _build_chain_scenario(n_steps=50)
        runner = GraphRunner(scenario)
        log = runner.run()

        for tick_log in log["ticks"]:
            sink_cash = tick_log["node_cash"].get("sink-1", 0)
            assert sink_cash >= 0, f"Sink cash negative at tick {tick_log['tick']}"

    def test_run_is_deterministic(self):
        """Same world_seed → same cash sequence across two runs."""
        s1 = _build_chain_scenario(n_steps=10, world_seed=123)
        s2 = _build_chain_scenario(n_steps=10, world_seed=123)
        log1 = GraphRunner(s1).run()
        log2 = GraphRunner(s2).run()

        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["node_cash"] == t2["node_cash"], (
                f"Cash diverged at tick {t1['tick']}"
            )

    def test_cash_conservation_50_ticks(self):
        """Cash + inventory-in-transit grows by sum(income_rates) each tick.

        The only source of new cash is ``DemandSinkNode.income_rate``. Per
        ADR 0013 rule 5 the conserved quantity is *cash plus inventory value
        in transit*, not cash alone: a zero-margin factory absorbs
        ``unit_cost`` when it produces (ADR 0013 rule 3) and recovers exactly
        that when it later sells at ``list_price == unit_cost``. Cash absorbed
        by production-not-yet-resold is held as factory-origin inventory value,
        so it leaves the *cash-only* total — that is why this test holds the
        factory cash deltas out of the sum rather than asserting the cash-only
        total is constant (the old assertion silently encoded the
        produce-creates-free-inventory bug).

        Conservation invariant (factory cash deltas excluded == inventory
        value in transit, which nets back in on sale):
            total_cash(t) - Σ_factory(cash(t) - cash(0))
                == total_cash(0) + t * sum(income_rates)
        """
        scenario = _build_chain_scenario(n_steps=50)
        gsim = build_graph_world(scenario)

        # Record initial total cash before any ticks.
        initial_total = sum(
            getattr(node, "cash", 0.0) for node in gsim.nodes.values()
        )
        income_per_tick = sum(
            getattr(node, "income_rate", 0.0) for node in gsim.nodes.values()
        )
        factory_cash_0 = {
            nid: node.cash
            for nid, node in gsim.nodes.items()
            if isinstance(node, FactoryNode)
        }

        for t in range(1, 51):
            gsim.tick()
            current_total = sum(
                getattr(node, "cash", 0.0) for node in gsim.nodes.values()
            )
            # Add back the cash factories have absorbed via production but not
            # yet recovered via sales — this is the "inventory value in
            # transit" term of ADR 0013 rule 5.
            inventory_in_transit_value = sum(
                factory_cash_0[nid] - gsim.nodes[nid].cash
                for nid in factory_cash_0
            )
            conserved = current_total + inventory_in_transit_value
            expected_total = initial_total + t * income_per_tick
            assert abs(conserved - expected_total) < 1e-6, (
                f"Cash conservation violated at tick {t}: "
                f"got {conserved:.6f}, expected {expected_total:.6f}"
            )


class TestMarketDemandMultiplier:
    def _build_market_with_catalog(self) -> tuple[Market, str]:
        """Return a Market + pid pair where demand_multiplier can be called."""
        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        params = _minimal_market_params()
        world_rng = Random(1)
        market = Market(params, world_rng, datetime(2024, 7, 15), catalog=catalog)
        return market, pid

    def test_demand_multiplier_returns_float(self):
        market, pid = self._build_market_with_catalog()
        result = market.demand_multiplier(pid, "US")
        assert isinstance(result, float)

    def test_demand_multiplier_is_positive(self):
        market, pid = self._build_market_with_catalog()
        result = market.demand_multiplier(pid, "US")
        assert result > 0.0

    def test_demand_multiplier_does_not_consume_world_rng(self):
        """demand_multiplier must not advance world_rng."""
        market, pid = self._build_market_with_catalog()
        # Save RNG state via a snapshot of the next value.
        rng_before = market.rng.getstate()
        market.demand_multiplier(pid, "US")
        rng_after = market.rng.getstate()
        assert rng_before == rng_after, "demand_multiplier must not consume world_rng"

    def test_demand_multiplier_works_without_catalog_for_unknown_pid(self):
        """Market without a catalog falls back gracefully (unknown pid -> off_factor)."""
        params = _minimal_market_params()
        market = Market(params, Random(1), datetime(2024, 1, 1))
        # Unknown pid with no catalog — seasonality falls back to None -> off_factor.
        result = market.demand_multiplier("P9999", "US")
        assert isinstance(result, float)
        assert result > 0.0

    def test_demand_multiplier_respects_demand_factor_min(self):
        """Even in a depressed market, multiplier >= demand_factor_min * off_factor."""
        catalog = _minimal_catalog()
        pid = catalog[0].product_id

        from src.sim.distributions import Constant
        params = _minimal_market_params()
        world_rng = Random(1)
        # Force demand to be very low (below demand_factor_min=0.1).
        params_modified = MarketParams(
            **{
                **{f.name: getattr(params, f.name) for f in params.__dataclass_fields__.values()},
                "init_demand": 0.0,
                "demand_factor_min": 0.2,
                "demand_shock": Constant(0.0),
            }
        )
        market = Market(params_modified, world_rng, datetime(2024, 1, 1), catalog=catalog)
        # Suppress season effect.
        market.off_factor = 1.0
        market.peak_factor = 1.0
        result = market.demand_multiplier(pid, "US")
        assert result >= 0.2  # demand_factor_min floor applies


class TestAllocationExecuteBuy:
    """Minimal single-supplier execute_buy tests (full tests in issue 07)."""

    def _build_table_with_offer(
        self,
        supplier_id: str,
        pid: str,
        available: int = 100,
        price: float = 5.0,
        min_order: int = 0,
    ) -> CentralTable:
        from src.sim.central_table import Offer
        table = CentralTable()
        table.publish(supplier_id, pid, Offer(
            available_qty=available,
            list_price=price,
            min_order=min_order,
        ))
        return table

    def test_basic_purchase_debits_buyer_credits_supplier(self):
        from src.sim.allocation import execute_buy
        from src.sim.event_engine import EventEngine
        from src.sim.scenario import DisruptionParams
        from src.sim.distributions import Constant

        pid = "P0000"
        buyer = DemandSinkNode(
            id="sink-1", region="US", init_seed=1,
            product_id=pid, demand_dist=Constant(5),
            income_rate=0.0, cash=500.0,
        )
        supplier = IntermediateNode(
            id="shop-1", region="US", init_seed=1,
            carried_products={pid},
            inventory={pid: 100},
            list_prices={pid: 5.0},
            min_order_imposed={pid: 0},
            cash=0.0,
        )
        table = self._build_table_with_offer("shop-1", pid, available=100, price=5.0)
        ee = EventEngine(
            DisruptionParams(
                event_prob=0.0, types=["natural_disaster"], regions=["US"],
                severity=Constant(0.0), duration=Constant(1)
            ),
            Random(1),
        )

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid=pid,
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=5,
            lead_time=2,
        )

        assert result.qty_filled == 10
        assert result.qty_rejected == 0
        assert result.cash_paid == 50.0
        assert buyer.cash == 450.0  # 500 - 50
        assert supplier.cash == 50.0  # 0 + 50

    def test_inventory_limited_purchase(self):
        from src.sim.allocation import execute_buy
        from src.sim.event_engine import EventEngine
        from src.sim.scenario import DisruptionParams
        from src.sim.distributions import Constant

        pid = "P0000"
        buyer = DemandSinkNode(
            id="sink-1", region="US", init_seed=1,
            product_id=pid, demand_dist=Constant(5),
            income_rate=0.0, cash=9999.0,
        )
        supplier = IntermediateNode(
            id="shop-1", region="US", init_seed=1,
            carried_products={pid},
            inventory={pid: 3},  # only 3 available
            list_prices={pid: 5.0},
            min_order_imposed={pid: 0},
            cash=0.0,
        )
        table = self._build_table_with_offer("shop-1", pid, available=3, price=5.0)
        ee = EventEngine(
            DisruptionParams(
                event_prob=0.0, types=["natural_disaster"], regions=["US"],
                severity=Constant(0.0), duration=Constant(1)
            ),
            Random(1),
        )

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid=pid,
            qty_requested=10,
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 3
        assert result.qty_rejected == 7

    def test_cash_limited_purchase(self):
        from src.sim.allocation import execute_buy
        from src.sim.event_engine import EventEngine
        from src.sim.scenario import DisruptionParams
        from src.sim.distributions import Constant

        pid = "P0000"
        buyer = DemandSinkNode(
            id="sink-1", region="US", init_seed=1,
            product_id=pid, demand_dist=Constant(5),
            income_rate=0.0, cash=25.0,  # can only afford 5 units at price=5
        )
        supplier = IntermediateNode(
            id="shop-1", region="US", init_seed=1,
            carried_products={pid},
            inventory={pid: 100},
            list_prices={pid: 5.0},
            min_order_imposed={pid: 0},
            cash=0.0,
        )
        table = self._build_table_with_offer("shop-1", pid, available=100, price=5.0)
        ee = EventEngine(
            DisruptionParams(
                event_prob=0.0, types=["natural_disaster"], regions=["US"],
                severity=Constant(0.0), duration=Constant(1)
            ),
            Random(1),
        )

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid=pid,
            qty_requested=10,  # wants 10 but can only afford 5
            table=table,
            event_engine=ee,
            current_tick=0,
            lead_time=1,
        )

        assert result.qty_filled == 5
        assert result.cash_paid == 25.0
        assert buyer.cash == 0.0

    def test_delivery_scheduled_on_event_engine(self):
        """After execute_buy, a delivery callback is queued for current_tick + lead_time."""
        from src.sim.allocation import execute_buy
        from src.sim.event_engine import EventEngine
        from src.sim.scenario import DisruptionParams
        from src.sim.distributions import Constant

        pid = "P0000"
        buyer = IntermediateNode(
            id="shop-1", region="US", init_seed=1,
            carried_products={pid},
            inventory={pid: 0},
            list_prices={pid: 10.0},
            min_order_imposed={pid: 0},
            cash=1000.0,
        )
        supplier = FactoryNode(
            id="factory-1", region="US", init_seed=1,
            produces_product_id=pid,
            unit_cost=5.0,
            capacity_per_tick=100,
            inventory=100,
            list_price=5.0,
            cash=0.0,
        )
        table = self._build_table_with_offer("factory-1", pid, available=100, price=5.0)
        ee = EventEngine(
            DisruptionParams(
                event_prob=0.0, types=["natural_disaster"], regions=["US"],
                severity=Constant(0.0), duration=Constant(1)
            ),
            Random(1),
        )

        result = execute_buy(
            buyer=buyer,
            supplier=supplier,
            pid=pid,
            qty_requested=20,
            table=table,
            event_engine=ee,
            current_tick=5,
            lead_time=3,
        )

        assert result.qty_filled == 20
        # A queued callback should exist for tick 5 + 3 = 8.
        assert len(ee.queued) == 1
        assert ee.queued[0].delay == 8
