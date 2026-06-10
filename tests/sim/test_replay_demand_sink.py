"""Tests for ReplayDemandSinkNode (issue 02).

Acceptance criteria:
1. Under a flat world, per-tick sink consumption equals the replay series
   exactly, end-to-end through a real Runner.run().
2. With a non-flat market, replayed demand scales by the same multiplier
   chain as a stochastic sink.
3. CRN-paired: a graph mixing one replay sink with stochastic sinks yields
   bit-identical trajectories for the stochastic parts vs the same graph
   with the replay sink replaced by a stochastic sink (same seeds).
4. A series shorter than n_steps raises at build time with a clear error.
5. In-sim lost sales against replayed demand appear in the rejection log
   (unmet_demand).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.sim.distributions import Constant, Uniform
from src.sim.flat_world import flat_world
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.replay_demand_sink import ReplayDemandSinkNode
from src.sim.runner import Runner, build_world
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
# Shared helpers
# ---------------------------------------------------------------------------


def _catalog(n: int = 1) -> list[Ware]:
    return load_catalog(
        [
            {
                "name": f"Item{i}",
                "category": "test",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 1.0,
                "seasonality": "all_season",
            }
            for i in range(n)
        ]
    )


def _disruption_params() -> DisruptionParams:
    return DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.0),
        duration=Constant(1),
    )


def _lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _flat_market_params() -> MarketParams:
    market_params, _ = flat_world(regions=["US"])
    return market_params


def _build_flat_scenario(
    series: list[int],
    n_steps: int,
    world_seed: int = 42,
    factory_inventory: int = 10_000,
    shop_inventory: int = 10_000,
    sink_cash: float = 1_000_000.0,
) -> Scenario:
    """Build a minimal factory→shop→replay-sink scenario under flat world."""
    catalog = _catalog(1)
    pid = catalog[0].product_id

    factory = FactoryNode(
        id="factory",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=1.0,
        capacity_per_tick=factory_inventory,
        inventory=factory_inventory,
        list_price=1.0,
        cash=0.0,
    )
    shop = IntermediateNode(
        id="shop",
        region="US",
        init_seed=2,
        carried_products={pid},
        capacity=1_000_000,
        tags=["shop"],
        inventory={pid: shop_inventory},
        pending={},
        list_prices={pid: 2.0},
        min_order_imposed={pid: 0},
        cash=1_000_000.0,
    )
    sink = ReplayDemandSinkNode(
        id="sink",
        region="US",
        init_seed=3,
        product_id=pid,
        series=series,
        n_steps=n_steps,
        income_rate=1_000_000.0,
        cash=sink_cash,
    )
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=0),
        EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=0),
    ]
    return Scenario(
        catalog=catalog,
        market=_flat_market_params(),
        disruption=_disruption_params(),
        item_lifecycle=_lifecycle_params(),
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


# ---------------------------------------------------------------------------
# AC4: build-time validation — too-short series raises
# ---------------------------------------------------------------------------


class TestBuildTimeValidation:
    def test_series_shorter_than_n_steps_raises(self):
        """A series with fewer entries than n_steps must raise ValueError at construction."""
        catalog = _catalog(1)
        pid = catalog[0].product_id
        with pytest.raises(ValueError, match="series"):
            ReplayDemandSinkNode(
                id="sink",
                region="US",
                init_seed=0,
                product_id=pid,
                series=[10, 20],  # only 2 entries
                n_steps=5,  # needs 5
                income_rate=0.0,
                cash=0.0,
            )

    def test_series_equal_to_n_steps_is_ok(self):
        """A series whose length equals n_steps must not raise."""
        catalog = _catalog(1)
        pid = catalog[0].product_id
        node = ReplayDemandSinkNode(
            id="sink",
            region="US",
            init_seed=0,
            product_id=pid,
            series=[10, 20, 30],
            n_steps=3,
            income_rate=0.0,
            cash=0.0,
        )
        assert node.series == [10, 20, 30]

    def test_series_longer_than_n_steps_is_ok(self):
        """A series longer than n_steps is fine (excess entries are unused)."""
        catalog = _catalog(1)
        pid = catalog[0].product_id
        node = ReplayDemandSinkNode(
            id="sink",
            region="US",
            init_seed=0,
            product_id=pid,
            series=[10, 20, 30, 40, 50],
            n_steps=3,
            income_rate=0.0,
            cash=0.0,
        )
        assert node is not None


# ---------------------------------------------------------------------------
# AC1: exact replay under flat world through a real Runner.run()
# ---------------------------------------------------------------------------


class TestExactReplayFlatWorld:
    def test_sink_consumption_equals_series(self):
        """Under flat world, per-tick sink demand equals series[tick] exactly.

        We measure demand via the flow log's sink demand entries, since sinks
        have zero inventory and the demand_target is the exogenous signal.
        """
        series = [5, 10, 3, 7, 2]
        n_steps = len(series)
        scenario = _build_flat_scenario(series=series, n_steps=n_steps)

        log = Runner(scenario).run()

        for i, tick_log in enumerate(log["ticks"]):
            # The flow log records exogenous demand for sinks.
            sink_flows = [
                f
                for f in tick_log["node_flows"]
                if f["node_id"] == "sink"
            ]
            assert len(sink_flows) == 1, f"tick {i}: expected 1 sink flow entry"
            recorded_demand = sink_flows[0]["demand"]
            assert recorded_demand == series[i], (
                f"tick {i}: expected demand {series[i]}, got {recorded_demand}"
            )

    def test_exact_replay_zero_demand(self):
        """A zero in the series produces zero demand (no purchases)."""
        series = [0, 5, 0]
        scenario = _build_flat_scenario(series=series, n_steps=3)
        log = Runner(scenario).run()

        tick0_flows = [
            f for f in log["ticks"][0]["node_flows"] if f["node_id"] == "sink"
        ]
        assert tick0_flows[0]["demand"] == 0

        tick2_flows = [
            f for f in log["ticks"][2]["node_flows"] if f["node_id"] == "sink"
        ]
        assert tick2_flows[0]["demand"] == 0


# ---------------------------------------------------------------------------
# AC2: non-flat market scales replayed demand by the multiplier chain
# ---------------------------------------------------------------------------


class TestReplayWithMultiplierChain:
    def _build_stochastic_market(self) -> MarketParams:
        return MarketParams(
            cycle_len=365,
            cycle_amp=0.0,
            init_demand=2.0,  # multiplier ≠ 1.0
            init_supply=1.0,
            peak_factor=1.0,
            off_factor=1.0,
            season_months={},
            regions=["US"],
            correlation=0.0,
            trend_update_interval=100,
            min_value=0.0,
            max_value=10.0,
            stage_multipliers={
                "introduction": 1.0,
                "growth": 1.0,
                "maturity": 1.0,
                "decline": 1.0,
                "dead": 1.0,
            },
            price_elasticity=0.0,
            promo_multiplier=1.0,
            demand_factor_min=0.0,
            supply_factor_min=0.0,
            cross_inv_lo=0.0,
            cross_inv_hi=1.0,
            cross_factor_range=(1.0, 1.0),
            trend=Constant(1.0),
            demand_shock=Constant(0.0),
            supply_shock=Constant(0.0),
            base_demand=Constant(1.0),
        )

    def test_replay_demand_scales_by_multiplier(self):
        """With init_demand=2.0, replayed demand should be scaled by ~2.0.

        We can't predict the exact integer output (int(base * 2.0) may round),
        but we can confirm the demand_target > the raw series value.
        """
        from random import Random

        from src.sim.market import Market

        series = [10] * 5
        n_steps = 5
        catalog = _catalog(1)
        pid = catalog[0].product_id

        # Manually invoke demand_target to verify the multiplier is applied.
        market_params = self._build_stochastic_market()
        market = Market(market_params, Random(42), datetime(2024, 1, 1))
        market.tick()

        node = ReplayDemandSinkNode(
            id="sink",
            region="US",
            init_seed=0,
            product_id=pid,
            series=series,
            n_steps=n_steps,
            income_rate=0.0,
            cash=0.0,
        )
        rng = Random(42)
        result = node.demand_target(
            tick=0,
            market=market,
            catalog=catalog,
            world_rng=rng,
        )
        # multiplier is 2.0, series[0]=10, so result should be 20
        assert result == 20, f"expected 20 with 2x multiplier, got {result}"

    def test_replay_matches_stochastic_multiplier_application(self):
        """ReplayDemandSinkNode applies multiplier the same way as DemandSinkNode.

        Use a Constant distribution on the stochastic sink (so the raw draw
        equals the replay series value) and confirm both produce the same output.
        """
        from random import Random

        from src.sim.market import Market

        catalog = _catalog(1)
        pid = catalog[0].product_id
        series_val = 10

        market_params = self._build_stochastic_market()

        stochastic_sink = DemandSinkNode(
            id="s_stoch",
            region="US",
            init_seed=0,
            product_id=pid,
            demand_dist=Constant(series_val),
            income_rate=0.0,
            cash=0.0,
        )
        replay_sink = ReplayDemandSinkNode(
            id="s_replay",
            region="US",
            init_seed=0,
            product_id=pid,
            series=[series_val] * 5,
            n_steps=5,
            income_rate=0.0,
            cash=0.0,
        )

        for seed in range(5):
            m_stoch = Market(market_params, Random(seed), datetime(2024, 1, 1))
            m_replay = Market(market_params, Random(seed), datetime(2024, 1, 1))
            m_stoch.tick()
            m_replay.tick()

            rng_stoch = Random(seed + 100)
            rng_replay = Random(seed + 100)

            result_stoch = stochastic_sink.demand_target(
                tick=0, market=m_stoch, catalog=catalog, world_rng=rng_stoch
            )
            result_replay = replay_sink.demand_target(
                tick=0, market=m_replay, catalog=catalog, world_rng=rng_replay
            )
            assert result_stoch == result_replay, (
                f"seed={seed}: stochastic={result_stoch}, replay={result_replay} differ"
            )


# ---------------------------------------------------------------------------
# AC3: CRN-paired test — stochastic sinks stay bit-identical when one replay
#       sink is mixed into the graph
# ---------------------------------------------------------------------------


class TestCRNPairedReplay:
    """A graph mixing one replay sink + stochastic sinks must yield bit-identical
    stochastic trajectories compared to the same graph with the replay sink
    replaced by a stochastic sink (same seeds).
    """

    def _build_mixed_scenario(
        self,
        *,
        use_replay: bool,
        world_seed: int = 42,
        n_steps: int = 10,
    ) -> Scenario:
        """Two-sink, one-shop, one-factory graph.

        sink-A: always stochastic (Uniform(5, 15)) — uses 1 world_rng draw per tick.
        sink-B: replay (use_replay=True) or stochastic Uniform(5, 15) (use_replay=False).

        Both variants of sink-B must burn exactly 1 world_rng draw per catalog item
        per tick so that sink-A sees an identical RNG stream in both cases.
        """
        from src.sim.distributions import Normal

        catalog = _catalog(1)
        pid = catalog[0].product_id
        market_params, _ = flat_world(regions=["US"])

        factory = FactoryNode(
            id="factory",
            region="US",
            init_seed=1,
            produces_product_id=pid,
            unit_cost=1.0,
            capacity_per_tick=1000,
            inventory=1000,
            list_price=1.0,
            cash=0.0,
        )
        shop = IntermediateNode(
            id="shop",
            region="US",
            init_seed=2,
            carried_products={pid},
            capacity=10_000,
            tags=["shop"],
            inventory={pid: 1000},
            pending={},
            list_prices={pid: 2.0},
            min_order_imposed={pid: 0},
            cash=1_000_000.0,
        )
        # sink-A uses Uniform so it draws exactly 1 world_rng value per tick.
        sink_a = DemandSinkNode(
            id="sink-a",
            region="US",
            init_seed=5,
            product_id=pid,
            demand_dist=Uniform(5.0, 15.0),
            income_rate=500.0,
            cash=100_000.0,
        )
        if use_replay:
            # sink-B replay: burns exactly 1 world_rng draw per catalog item.
            sink_b: DemandSinkNode = ReplayDemandSinkNode(
                id="sink-b",
                region="US",
                init_seed=6,
                product_id=pid,
                series=[10] * n_steps,
                n_steps=n_steps,
                income_rate=500.0,
                cash=100_000.0,
            )
        else:
            # sink-B stochastic: Uniform also burns exactly 1 world_rng draw per item.
            sink_b = DemandSinkNode(
                id="sink-b",
                region="US",
                init_seed=6,
                product_id=pid,
                demand_dist=Uniform(5.0, 15.0),
                income_rate=500.0,
                cash=100_000.0,
            )

        edges = [
            EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=0),
            EdgeSpec(supplier_id="shop", buyer_id="sink-a", default_lead_time=0),
            EdgeSpec(supplier_id="shop", buyer_id="sink-b", default_lead_time=0),
        ]
        return Scenario(
            catalog=catalog,
            market=market_params,
            disruption=_disruption_params(),
            item_lifecycle=_lifecycle_params(),
            n_steps=n_steps,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
            nodes=[
                NodeInstance(node=factory, init_seed=1),
                NodeInstance(node=shop, init_seed=2),
                NodeInstance(node=sink_a, init_seed=5),
                NodeInstance(node=sink_b, init_seed=6),
            ],
            edges=edges,
        )

    def test_stochastic_sink_trajectory_identical_with_replay_mixed_in(self):
        """sink-a's demand trajectory is bit-identical whether sink-b is replay or stochastic.

        Both sink-B variants burn exactly 1 world_rng draw per catalog item per tick
        (Uniform.sample draws once; ReplayDemandSinkNode also burns once and discards).
        Result: sink-a sees the same world_rng stream in both cases.
        """
        n_steps = 20

        scenario_stoch = self._build_mixed_scenario(use_replay=False, n_steps=n_steps)
        scenario_replay = self._build_mixed_scenario(use_replay=True, n_steps=n_steps)

        log_stoch = Runner(scenario_stoch).run()
        log_replay = Runner(scenario_replay).run()

        # Extract per-tick demand for sink-a from the flow log.
        def sink_a_demands(log):
            return [
                next(
                    f["demand"]
                    for f in tick["node_flows"]
                    if f["node_id"] == "sink-a"
                )
                for tick in log["ticks"]
            ]

        demands_stoch = sink_a_demands(log_stoch)
        demands_replay = sink_a_demands(log_replay)

        assert demands_stoch == demands_replay, (
            "sink-a demands differ between stochastic and replay configurations — "
            "CRN invariant violated.\n"
            f"stochastic: {demands_stoch}\nreplay:     {demands_replay}"
        )


# ---------------------------------------------------------------------------
# AC5: in-sim lost sales appear in the rejection log (unmet_demand)
# ---------------------------------------------------------------------------


class TestLostSalesInRejectionLog:
    def test_replay_lost_sales_logged_as_unmet_demand(self):
        """When replay demand exceeds available stock, rejection log records unmet_demand."""
        series = [100]  # demand 100 units but only 5 available
        n_steps = 1
        scenario = _build_flat_scenario(
            series=series,
            n_steps=n_steps,
            factory_inventory=0,
            shop_inventory=5,
        )
        log = Runner(scenario).run()
        tick0 = log["ticks"][0]
        unmet = [r for r in tick0["rejections"] if r["supplier_id"] is None]
        assert len(unmet) == 1
        assert unmet[0]["buyer_id"] == "sink"
        assert unmet[0]["reason"] == "unmet_demand"
        assert unmet[0]["qty_rejected"] == 95  # 100 - 5

    def test_no_lost_sales_when_series_zero(self):
        """Zero demand → no unmet_demand entries."""
        series = [0]
        scenario = _build_flat_scenario(
            series=series, n_steps=1, shop_inventory=0
        )
        log = Runner(scenario).run()
        unmet = [
            r
            for r in log["ticks"][0]["rejections"]
            if r["supplier_id"] is None
        ]
        assert unmet == []


# ---------------------------------------------------------------------------
# Subclass visibility — node type in the roster
# ---------------------------------------------------------------------------


class TestNodeType:
    def test_replay_sink_has_distinct_node_type(self):
        """ReplayDemandSinkNode._node_type must be distinguishable from 'demand_sink'."""
        catalog = _catalog(1)
        pid = catalog[0].product_id
        node = ReplayDemandSinkNode(
            id="sink",
            region="US",
            init_seed=0,
            product_id=pid,
            series=[1, 2, 3],
            n_steps=3,
            income_rate=0.0,
            cash=0.0,
        )
        # Must be a string; the value is 'replay_demand_sink' per design.
        assert isinstance(node._node_type, str)
        assert node._node_type != "demand_sink"
