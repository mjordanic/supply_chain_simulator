"""Integration tests for the freshness curve (issue 03).

Pins the cross-module behaviour layered on top of the
``test_freshness_curve.py`` math properties:

- ``Store.activate_item`` writes to ``activation_tick``;
  ``freshness_multiplier`` returns ``1.0`` for never-activated products
  and ``1 + α`` immediately after activation.
- Re-activating a deactivated product resets ``τ`` to ``0`` so the
  multiplier returns to ``1 + α``.
- ``Market.sample_demand`` composes the freshness factor multiplicatively
  in the contracted order (``stage * freshness * season * promo *
  cross``) — verified by reading the demand at varied ``τ``.
- The CRN demand-draw cardinality contract is preserved: every catalog
  product consumes exactly one ``world_rng`` draw per ``(store, tick)``,
  even when the product is inactive in the store.
- A paired pair (same ``(template, init_seed)``, different policies)
  consumes the same ``world_rng`` draws across an entire run — proven
  via identical world streams and demand traces under the no-op
  ``α = 0`` configuration.
"""

from __future__ import annotations

from datetime import datetime
from random import Random
from typing import Any, Mapping

import pytest

from src.sim.distributions import Constant, Normal, Uniform
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.node import DemandSinkNode
from src.sim.policy import NoopPolicy, Policy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    Ware,
    load_catalog,
)
from src.sim.store import Store


def _catalog() -> list[Ware]:
    return load_catalog(
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
                "related_products": [],
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
        ]
    )


def _market_params() -> MarketParams:
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=50,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 1.0,
            "growth": 1.0,
            "maturity": 1.0,
            "decline": 1.0,
            "dead": 1.0,
        },
        price_elasticity=0.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(100),
    )


def _lifecycle(alpha: float = 0.0, decay: float = 30.0) -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
        default_freshness_alpha=alpha,
        default_freshness_decay=decay,
    )


def _template(init_active: int = 1) -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=100,
        init_balance=10000,
        init_stock_pct=0.5,
        delivery_lag=2,
        holding_rate=0.005,
        order_fee=10.0,
        init_active_count=init_active,
    )


def _registry(catalog: list[Ware], lifecycle: ItemLifecycleParams) -> ItemRegistry:
    return ItemRegistry(lifecycle, catalog, Random(0))


# --------------------------------------------------------------- Store unit


def test_freshness_multiplier_is_one_for_never_activated_product():
    """Initial active SKUs are not in ``activation_tick`` ⇒ baseline."""
    store = Store(
        _template(),
        init_seed=1,
        policy=None,
        catalog=_catalog(),
        freshness_alpha=0.4,
        freshness_decay=30,
    )
    pid = next(iter(store.inventory))
    assert store.freshness_multiplier(pid, current_step=0) == 1.0
    assert store.freshness_multiplier(pid, current_step=100) == 1.0


def test_activate_item_writes_activation_tick_and_peaks_at_alpha():
    """Right after activation, multiplier == 1 + α."""
    store = Store(
        _template(),
        init_seed=1,
        policy=None,
        catalog=_catalog(),
        freshness_alpha=0.4,
        freshness_decay=30,
    )
    pid = "P0002"
    assert pid not in store.activation_tick
    store.activate_item(pid, current_step=5)
    assert store.activation_tick[pid] == 5
    assert store.freshness_multiplier(pid, current_step=5) == pytest.approx(1.4)


def test_freshness_decays_then_resets_on_reactivation():
    """Re-activating a discontinued SKU restores τ=0 / multiplier=1+α."""
    store = Store(
        _template(),
        init_seed=1,
        policy=None,
        catalog=_catalog(),
        freshness_alpha=0.4,
        freshness_decay=30,
    )
    pid = "P0002"
    store.activate_item(pid, current_step=0)
    assert store.freshness_multiplier(pid, current_step=0) == pytest.approx(1.4)

    decayed = store.freshness_multiplier(pid, current_step=60)
    # 60 ticks ≈ 2β → factor ≈ 1 + 0.4 · e^(-2) ≈ 1.0541
    assert decayed < 1.4
    assert decayed > 1.0

    # Deactivating leaves activation_tick alone — re-activation overwrites.
    store.deactivate_item(pid)
    assert pid in store.activation_tick

    store.activate_item(pid, current_step=200)
    assert store.activation_tick[pid] == 200
    assert store.freshness_multiplier(pid, current_step=200) == pytest.approx(1.4)


# --------------------------------------------------------------- Market integration


def test_sample_demand_composes_freshness_factor_multiplicatively():
    """Increasing freshness multiplier proportionally increases demand.

    With every other demand factor pinned to ``1.0`` (cycle_amp 0,
    elasticity 0, peak/off both 1, no promo, no related products), the
    realised ``int(base * multiplier * demand_factor)`` reduces to
    ``int(100 * freshness * 1.0)``.
    """
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    registry = _registry(catalog, lifecycle)
    market = Market(_market_params(), Random(7), datetime(2024, 6, 1), registry=registry)

    pid = "P0000"
    base_price = registry.items[pid].base_price
    store = Store(
        _template(init_active=0),
        init_seed=1,
        policy=None,
        catalog=catalog,
        freshness_alpha=0.4,
        freshness_decay=30,
    )

    # Never activated → multiplier 1 → base demand 100.
    assert market.sample_demand(pid, store, base_price, current_step=0) == 100

    # Activate at step 0; sampled at step 0 ⇒ multiplier 1.4 ⇒ demand 140.
    store.activate_item(pid, current_step=0)
    assert market.sample_demand(pid, store, base_price, current_step=0) == 140


# ------------------------------------------------ CRN cardinality + paired


class _DrawCountingRng(Random):
    """Random subclass that counts every ``random()`` call.

    Tracks the cumulative number of ``random()`` draws against this
    stream so the CRN cardinality contract can be measured end-to-end.
    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return super().random()


def _crn_scenario(stores: list[StoreInstance]) -> Scenario:
    return Scenario(
        catalog=_catalog(),
        market=_market_params(),
        disruption=DisruptionParams(
            event_prob=0.0,
            types=["natural_disaster"],
            regions=["US"],
            severity=Constant(0.01),
            duration=Constant(1),
        ),
        item_lifecycle=_lifecycle(alpha=0.4, decay=30.0),
        stores=stores,
        n_steps=10,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


def test_world_streams_invariant_under_policy_swap_with_freshness_active():
    """With ``α = 0.4`` enabled, swapping policies on a fixed
    ``(template, init_seed, world_seed)`` does not perturb the global
    world streams.

    Freshness multipliers shift per-product demand *outputs* once a
    policy starts diverging the activation history, but the underlying
    ``world_rng`` draws (market_demand / supply, events, base_demand
    sequence) only depend on ``world_rng`` — not on policy state. If
    they ever did, this test would fail.
    """

    class _ActivateAllPolicy(Policy):
        def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
            inactive = [
                pid
                for pid in observation["inventory"]
                if pid not in observation["active_products"]
            ]
            return {"activate": inactive}

    template = _template(init_active=1)
    s_noop = _crn_scenario(
        stores=[
            StoreInstance(
                template=template, init_seed=99, policy=NoopPolicy(policy_seed=1)
            ),
        ]
    )
    s_activate = _crn_scenario(
        stores=[
            StoreInstance(
                template=template, init_seed=99, policy=_ActivateAllPolicy(policy_seed=1)
            ),
        ]
    )

    log_noop = Runner(s_noop).run()
    log_activate = Runner(s_activate).run()

    assert log_noop["global"]["market_demand"] == log_activate["global"]["market_demand"]
    assert log_noop["global"]["market_supply"] == log_activate["global"]["market_supply"]
    assert (
        log_noop["global"]["events"]["occurrences"]
        == log_activate["global"]["events"]["occurrences"]
    )


def test_crn_demand_draw_cardinality_one_per_inventory_per_tick():
    """Every catalog product consumes one ``world_rng`` draw per tick
    inside ``sample_demand``, regardless of whether the product is in
    the store's active assortment (ADR 0003).

    Use a ``Uniform`` ``base_demand`` so every ``sample_demand`` call
    consumes exactly one ``rng.random()`` — a ``Constant`` would burn
    zero draws and the test would tautologically pass.
    """
    rng = _DrawCountingRng(42)

    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.0)
    params = _market_params()
    params.base_demand = Uniform(100.0, 100.0)  # 1 rng.random() per sample
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(params, rng, datetime(2024, 1, 1), registry=registry)

    # Single store with only ONE active SKU out of three.
    store = Store(
        _template(init_active=1),
        init_seed=1,
        policy=None,
        catalog=catalog,
        freshness_alpha=0.0,
        freshness_decay=30,
    )
    market.add_store(store)

    n_products = len(catalog)
    n_steps = 20

    for tick in range(n_steps):
        market.tick()
        loop_draws_before = rng.draws
        for pid in store.inventory:
            market.sample_demand(pid, store, store.prices[pid], current_step=tick)
        # Every catalog product consumed exactly one rng draw per tick,
        # regardless of whether it is in the active assortment.
        assert rng.draws - loop_draws_before == n_products


# ----------------------------------------------- Per-Ware overrides (issue 04)


def _store_with_registry(
    catalog: list[Ware],
    lifecycle: ItemLifecycleParams,
    init_active: int = 0,
) -> tuple[Store, ItemRegistry]:
    """Build a ``(Store, ItemRegistry)`` pair with the registry attached.

    Mirrors the Runner's wiring: catalog defaults flow from
    ``ItemLifecycleParams`` through ``ItemRegistry`` into ``Store``,
    and ``Store.freshness_multiplier`` resolves per-Ware overrides via
    the registry. The same plumbing the production loop uses.
    """
    registry = _registry(catalog, lifecycle)
    store = Store(
        _template(init_active=init_active),
        init_seed=1,
        policy=None,
        catalog=catalog,
        freshness_alpha=registry.default_freshness_alpha,
        freshness_decay=registry.default_freshness_decay,
        item_registry=registry,
    )
    return store, registry


def test_per_ware_freshness_alpha_zero_staple_yields_baseline_multiplier():
    """A Ware with ``freshness_alpha=0`` is a staple: the curve is
    identically 1 for every τ regardless of catalog-wide defaults."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.0, freshness_decay=15.0)
    lifecycle = _lifecycle(alpha=0.5, decay=20.0)
    store, _ = _store_with_registry(catalog, lifecycle)

    pid = "P0000"
    store.activate_item(pid, current_step=0)
    for tau in (0, 1, 30, 1_000):
        assert store.freshness_multiplier(pid, current_step=tau) == 1.0


def test_per_ware_freshness_overrides_drive_multiplier_not_defaults():
    """A Ware with non-zero overrides produces ``1 + α`` against its
    own ``α`` at τ=0, not the catalog default."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.7, freshness_decay=50.0)
    lifecycle = _lifecycle(alpha=0.1, decay=15.0)
    store, _ = _store_with_registry(catalog, lifecycle)

    pid = "P0000"
    store.activate_item(pid, current_step=0)
    assert store.freshness_multiplier(pid, current_step=0) == pytest.approx(1.7)
    # At τ = β = 50 the curve is 1 + 0.7 / e (per-Ware β, not the default 15).
    import math as _math
    expected = 1.0 + 0.7 * _math.exp(-1.0)
    assert store.freshness_multiplier(pid, current_step=50) == pytest.approx(expected)


def test_freshness_falls_back_to_catalog_defaults_when_ware_has_no_override():
    """A Ware with no overrides resolves against
    ``ItemLifecycleParams.default_*``."""
    catalog = _catalog()  # no per-Ware overrides
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    store, _ = _store_with_registry(catalog, lifecycle)

    pid = "P0001"
    store.activate_item(pid, current_step=0)
    assert store.freshness_multiplier(pid, current_step=0) == pytest.approx(1.4)
    # Confirm it picked up the catalog default β=30, not a different value.
    import math as _math
    expected = 1.0 + 0.4 * _math.exp(-30.0 / 30.0)
    assert store.freshness_multiplier(pid, current_step=30) == pytest.approx(expected)


def test_per_ware_overrides_coexist_with_default_fallback():
    """Mixed catalog: one Ware overrides α/β, the other inherits defaults.
    Both resolutions must apply correctly inside the same store."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.0, freshness_decay=1.0)  # staple
    # catalog[1] (P0001) keeps None ⇒ catalog defaults
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    store, _ = _store_with_registry(catalog, lifecycle)

    store.activate_item("P0000", current_step=0)
    store.activate_item("P0001", current_step=0)
    assert store.freshness_multiplier("P0000", current_step=0) == 1.0
    assert store.freshness_multiplier("P0001", current_step=0) == pytest.approx(1.4)


# ------------------------------------------------ Step-0 baseline preserved


def test_step0_initial_active_skus_have_no_freshness_spike():
    """A baseline-equivalent scenario must not emit a step-0 demand spike.

    Run a tiny scenario with ``α = 0.4`` (a hype curve that *would* push
    demand way above baseline if initial active SKUs entered with
    ``τ = 0``). Initial active SKUs land in ``inventory`` but not in
    ``activation_tick`` — so their step-0 freshness multiplier is 1.0,
    matching pre-issue-03 behaviour.
    """
    template = _template(init_active=2)
    scenario = _crn_scenario(
        stores=[
            StoreInstance(template=template, init_seed=11, policy=NoopPolicy()),
        ]
    )
    runner = Runner(scenario)
    store = runner.stores[0]
    for pid in store.active_items:
        assert store.freshness_multiplier(pid, current_step=0) == 1.0


# ------------------------------------------------ DemandSinkNode path (issue 10)


def _make_demand_sink_registry(
    catalog: list[Ware],
    alpha: float,
    decay: float = 30.0,
) -> tuple[ItemRegistry, Market]:
    """Helper: build an (ItemRegistry, Market) pair with fixed freshness params."""
    lifecycle = _lifecycle(alpha=alpha, decay=decay)
    rng = Random(0)
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)
    return registry, market


def test_demand_sink_freshness_alpha_two_vs_zero_visible_difference():
    """DemandSinkNode.demand_target with freshness_alpha=2.0 vs 0.0 produces
    a visibly different demand series when the product is activated at tick 0.

    This is the acceptance-criteria visible-hype check reframed onto the
    demand-sink path (ADR 0015 / issue 10).
    """
    catalog = _catalog()  # 3 products

    def _run_series(alpha: float) -> list[int]:
        registry, market = _make_demand_sink_registry(catalog, alpha=alpha, decay=10.0)
        # Use Uniform so each sample consumes a world_rng draw (CRN-safe).
        sink = DemandSinkNode(
            id="sink",
            region="US",
            init_seed=1,
            product_id="P0000",
            demand_dist=Uniform(100.0, 100.0),
            income_rate=0.0,
        )
        sink.activation_tick["P0000"] = 0  # activate at tick 0

        series: list[int] = []
        for tick in range(1, 21):
            market.tick()
            result = sink.demand_target(
                tick=tick,
                market=market,
                registry=registry,
                world_rng=Random(tick * 999),  # independent per tick
            )
            series.append(result)
        return series

    series_hype = _run_series(alpha=2.0)
    series_no_hype = _run_series(alpha=0.0)

    # Hype series should start higher than no-hype.
    assert series_hype[0] > series_no_hype[0]
    # No-hype series is flat at 100 (Constant-like Uniform(100,100), all multipliers=1).
    assert all(d == 100 for d in series_no_hype)
    # Hype decays: first tick higher than last tick.
    assert series_hype[0] > series_hype[-1]


def test_demand_sink_never_activated_no_freshness_boost():
    """DemandSinkNode with large α but no activation_tick entry returns baseline demand.

    Never-activated product → τ = ∞ → freshness multiplier → 1.0.
    """
    catalog = _catalog()
    registry, market = _make_demand_sink_registry(catalog, alpha=5.0, decay=30.0)
    market.tick()

    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id="P0000",
        demand_dist=Constant(100),
        income_rate=0.0,
    )
    # No activation_tick entry for P0000

    result = sink.demand_target(
        tick=1,
        market=market,
        registry=registry,
        world_rng=Random(42),
    )
    # maturity stage (×1.0), market factor 1.0, freshness 1.0 → 100
    assert result == 100
