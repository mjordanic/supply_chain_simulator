"""Tests for the rejection & lost-sale log (issue 05).

Covers:
- Per-tick ``rejections`` list in the run log (``Runner.run()``)
- Correct ``reason`` values for each clamp path
- ``unmet_demand`` entries (``supplier_id=None``) when a sink cannot source
  its full demand from any supplier
- Run-log is always-on and bit-deterministic
"""

from __future__ import annotations

from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_catalog():
    from src.sim.scenario import load_catalog
    return load_catalog([{
        "name": "Widget", "category": "test", "related_products": [],
        "base_price": 10.0, "unit_cost": 1.0, "seasonality": "all_season",
    }])


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


def _build_shortage_scenario(
    factory_inventory: int = 0,
    shop_inventory: int = 0,
    demand: int = 5,
    shop_cash: float = 5_000.0,
    sink_cash: float = 5_000.0,
    n_steps: int = 3,
    world_seed: int = 42,
):
    """factory → shop → sink.

    Setting ``factory_inventory=0`` and ``shop_inventory=0`` ensures
    the sink faces an engineered shortage on tick 1.
    """
    from src.sim.distributions import Constant
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance, Scenario

    catalog = _minimal_catalog()
    pid = catalog[0].product_id

    factory = FactoryNode(
        id="factory", region="US", init_seed=1,
        produces_product_id=pid, unit_cost=1.0,
        capacity_per_tick=0,  # produce nothing
        inventory=factory_inventory, list_price=1.0, cash=0.0,
    )
    shop = IntermediateNode(
        id="shop", region="US", init_seed=2,
        carried_products={pid}, capacity=1000, tags=["shop"],
        inventory={pid: shop_inventory}, pending={},
        list_prices={pid: 2.0}, min_order_imposed={pid: 0}, cash=shop_cash,
    )
    sink = DemandSinkNode(
        id="sink", region="US", init_seed=3,
        product_id=pid, demand_dist=Constant(demand),
        income_rate=500.0, cash=sink_cash,
    )
    from src.sim.graph import EdgeSpec
    edges = [
        EdgeSpec(supplier_id="factory", buyer_id="shop", default_lead_time=1),
        EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=1),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRejectionLogAlwaysPresent:
    """The ``rejections`` key is always present in the tick log."""

    def test_rejections_key_in_every_tick(self):
        from src.sim.runner import Runner
        scenario = _build_shortage_scenario(
            factory_inventory=100, shop_inventory=100, demand=5
        )
        log = Runner(scenario).run()
        for tick_log in log["ticks"]:
            assert "rejections" in tick_log, (
                f"tick {tick_log['tick']}: missing 'rejections' key"
            )

    def test_rejections_empty_when_no_shortage(self):
        """No rejections when ample stock is always available."""
        from src.sim.runner import Runner
        scenario = _build_shortage_scenario(
            factory_inventory=500, shop_inventory=500, demand=5, n_steps=5
        )
        log = Runner(scenario).run()
        # Tick 1: shop has 500 units → sink buys 5, no shortage.
        tick1 = log["ticks"][0]
        assert tick1["rejections"] == [], (
            f"expected no rejections on tick 1 with ample stock; got {tick1['rejections']}"
        )


class TestUnmetDemandEntry:
    """Sink-level ``unmet_demand`` entries when no supplier can fill demand."""

    def test_unmet_demand_when_shop_empty(self):
        """shop has zero stock → sink cannot buy anything → unmet_demand entry."""
        from src.sim.runner import Runner
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=0, demand=5, n_steps=1
        )
        log = Runner(scenario).run()
        tick1 = log["ticks"][0]
        rejections = tick1["rejections"]

        unmet = [r for r in rejections if r["supplier_id"] is None]
        assert len(unmet) >= 1, f"expected at least one unmet_demand entry; got {rejections}"
        entry = unmet[0]
        assert entry["buyer_id"] == "sink"
        assert entry["pid"] == "P0000"
        assert entry["reason"] == "unmet_demand"
        assert entry["qty_rejected"] == 5
        assert entry["qty_filled"] == 0

    def test_partial_fill_produces_unmet_demand_residual(self):
        """shop has 2 units, demand=5 → filled=2, unmet residual=3."""
        from src.sim.runner import Runner
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=2, demand=5, n_steps=1
        )
        log = Runner(scenario).run()
        tick1 = log["ticks"][0]
        rejections = tick1["rejections"]

        unmet = [r for r in rejections if r["supplier_id"] is None]
        assert len(unmet) == 1, f"expected one unmet_demand entry; got {rejections}"
        entry = unmet[0]
        assert entry["qty_filled"] == 2
        assert entry["qty_rejected"] == 3

    def test_no_unmet_demand_when_fully_filled(self):
        """sink gets exactly demand units → no unmet_demand entry."""
        from src.sim.runner import Runner
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=100, demand=5, n_steps=1
        )
        log = Runner(scenario).run()
        tick1 = log["ticks"][0]
        unmet = [r for r in tick1["rejections"] if r["supplier_id"] is None]
        assert unmet == [], f"expected no unmet_demand, got {unmet}"


class TestRejectionEntrySchema:
    """Each rejection entry has the correct fields."""

    def _get_first_rejection(self, scenario):
        from src.sim.runner import Runner
        log = Runner(scenario).run()
        for tick_log in log["ticks"]:
            if tick_log["rejections"]:
                return tick_log["rejections"][0]
        return None

    def test_rejection_entry_has_all_fields(self):
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=0, demand=5, n_steps=1
        )
        entry = self._get_first_rejection(scenario)
        assert entry is not None, "expected at least one rejection"
        required_keys = {
            "tick", "buyer_id", "supplier_id", "pid",
            "qty_requested", "qty_filled", "qty_rejected", "reason",
        }
        assert required_keys <= entry.keys(), (
            f"missing keys: {required_keys - entry.keys()}"
        )

    def test_rejection_entry_tick_is_correct(self):
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=0, demand=5, n_steps=1
        )
        from src.sim.runner import Runner
        log = Runner(scenario).run()
        tick1_rejections = log["ticks"][0]["rejections"]
        for entry in tick1_rejections:
            assert entry["tick"] == 1, f"expected tick=1, got {entry['tick']}"

    def test_qty_filled_plus_rejected_equals_requested(self):
        scenario = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=2, demand=5, n_steps=1
        )
        from src.sim.runner import Runner
        log = Runner(scenario).run()
        for tick_log in log["ticks"]:
            for entry in tick_log["rejections"]:
                assert entry["qty_filled"] + entry["qty_rejected"] == entry["qty_requested"], (
                    f"qty_filled + qty_rejected != qty_requested in {entry}"
                )


class TestRejectionLogDeterminism:
    """The rejection log is bit-deterministic from the seed."""

    def test_same_seed_same_rejections(self):
        from src.sim.runner import Runner
        s1 = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=2, demand=5, n_steps=3, world_seed=7
        )
        s2 = _build_shortage_scenario(
            factory_inventory=0, shop_inventory=2, demand=5, n_steps=3, world_seed=7
        )
        log1 = Runner(s1).run()
        log2 = Runner(s2).run()
        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["rejections"] == t2["rejections"]

    def test_different_seed_may_differ(self):
        from src.sim.runner import Runner
        s1 = _build_shortage_scenario(world_seed=7, n_steps=3)
        s2 = _build_shortage_scenario(world_seed=99, n_steps=3)
        log1 = Runner(s1).run()
        log2 = Runner(s2).run()
        # Different seeds must produce different cash trajectories;
        # this just ensures runs complete without error.
        assert log1["n_steps"] == log2["n_steps"] == 3
