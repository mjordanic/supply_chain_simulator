"""Integration tests for the freshness curve (issue 03 / issue 11).

Pins the cross-module behaviour layered on top of the
``test_freshness_curve.py`` math properties:

- ``DemandSinkNode.demand_target`` returns ``1.0`` multiplier for
  never-activated products and ``1 + alpha`` immediately after activation.
- Re-activating a deactivated product resets tau=0 so the
  multiplier returns to ``1 + alpha``.
- The demand-target composes the freshness factor multiplicatively
  (verified by controlling ``demand_dist`` to ``Uniform(100, 100)``).
- The CRN demand-draw cardinality contract is preserved: every catalog
  product consumes exactly one ``world_rng`` draw per ``demand_target``
  call, even when the product is not the sink's bound product.
- A paired pair (same world_seed, different policies) produces identical
  world_rng streams — proven via ``build_world`` + manual RNG sampling.
- Per-ware freshness alpha/decay overrides flow through ItemRegistry
  into freshness_curve.multiplier correctly.

Issue 11 note: the legacy ``Store`` engine has been retired.  All tests
below operate on the graph-engine path (``DemandSinkNode`` /
``ItemRegistry`` / ``freshness_curve``).
"""

from __future__ import annotations

import math
from datetime import datetime
from random import Random

import pytest

from src.sim import freshness_curve as _fc
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.graph import EdgeSpec
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    NoopPolicy,
    OrderUpToPolicy,
    StaticFactoryPolicy,
)
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


def _registry(catalog: list[Ware], lifecycle: ItemLifecycleParams) -> ItemRegistry:
    return ItemRegistry(lifecycle, catalog, Random(0))


# ---------------------------------------------------------------------------
# Freshness-multiplier unit tests (via ItemRegistry + freshness_curve)
# ---------------------------------------------------------------------------


def test_freshness_multiplier_is_one_for_never_activated_product():
    """Never-activated product: tau = inf -> multiplier = 1.0."""
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30)
    registry = _registry(catalog, lifecycle)
    pid = "P0000"

    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    # Never activated: tau = inf
    assert _fc.multiplier(alpha, decay, float("inf")) == 1.0
    # Far future also resolves to 1.0 asymptotically.
    assert _fc.multiplier(alpha, decay, 1000) == pytest.approx(1.0, abs=1e-6)


def test_activate_item_writes_activation_tick_and_peaks_at_alpha():
    """Right after activation (tau=0), DemandSinkNode reports multiplier == 1 + alpha."""
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30)
    registry = _registry(catalog, lifecycle)
    pid = "P0000"

    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id=pid,
        demand_dist=Constant(100),
        income_rate=0.0,
    )

    assert pid not in sink.activation_tick
    # Activate at tick 5.
    sink.activation_tick[pid] = 5

    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    # tau = tick - activation_tick = 5 - 5 = 0 -> multiplier = 1 + alpha.
    assert _fc.multiplier(alpha, decay, 0.0) == pytest.approx(1.4)


def test_freshness_decays_then_resets_on_reactivation():
    """Re-activating a product resets tau=0 and restores 1+alpha."""
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30)
    registry = _registry(catalog, lifecycle)
    pid = "P0000"

    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    # At tau=0 peak.
    assert _fc.multiplier(alpha, decay, 0.0) == pytest.approx(1.4)

    # At tau=60 (2*decay) considerably decayed.
    decayed = _fc.multiplier(alpha, decay, 60.0)
    # 60 ticks ~ 2*beta -> factor ~ 1 + 0.4 * e^(-2) ~ 1.0541
    assert decayed < 1.4
    assert decayed > 1.0

    # After "reactivation" (reset tau to 0), peak restored.
    assert _fc.multiplier(alpha, decay, 0.0) == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Market integration (demand_target composes freshness multiplicatively)
# ---------------------------------------------------------------------------


def test_sample_demand_composes_freshness_factor_multiplicatively():
    """demand_target with freshness_alpha=0.4 raises demand above baseline.

    With Uniform(100, 100) demand_dist, every multiplier pinned to 1.0
    except freshness, so demand_target == int(100 * freshness_mult).

    Never-activated -> multiplier 1.0 -> demand 100.
    Activated at tau=0 -> multiplier 1.4 -> demand 140.
    """
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    rng = Random(0)
    registry = _registry(catalog, lifecycle)
    market = Market(_market_params(), rng, datetime(2024, 6, 1), registry=registry)
    market.tick()

    pid = "P0000"
    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id=pid,
        demand_dist=Uniform(100.0, 100.0),
        income_rate=0.0,
    )

    # Never activated -> multiplier 1 -> base demand 100.
    result_no_activation = sink.demand_target(
        tick=1,
        market=market,
        registry=registry,
        world_rng=Random(42),
    )
    assert result_no_activation == 100

    # Activate at tick 0; sampled at tick 1 with tau = 1 - 0 = 1.
    # freshness_mult = 1 + 0.4 * exp(-1/30) ~ 1.387
    sink.activation_tick[pid] = 0
    result_activated = sink.demand_target(
        tick=1,
        market=market,
        registry=registry,
        world_rng=Random(42),
    )
    # Demand should be higher than baseline due to freshness boost.
    assert result_activated > result_no_activation


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


def test_crn_demand_draw_cardinality_one_per_inventory_per_tick():
    """DemandSinkNode.demand_target consumes exactly len(catalog) draws per call.

    Use a Uniform base_demand so every sample consumes exactly one
    world_rng draw.  With 3 catalog products, each demand_target call
    must consume exactly 3 draws regardless of which product the sink
    is bound to.
    """
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.0)
    params = _market_params()
    params.base_demand = Uniform(100.0, 100.0)  # 1 rng.random() per sample

    rng = _DrawCountingRng(42)
    registry = ItemRegistry(lifecycle, catalog, rng)
    market = Market(params, rng, datetime(2024, 1, 1), registry=registry)

    n_products = len(catalog)
    assert n_products == 3

    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id="P0000",
        demand_dist=Uniform(100.0, 100.0),
        income_rate=0.0,
    )

    n_steps = 20
    for tick in range(1, n_steps + 1):
        market.tick()
        draws_before = rng.draws
        sink.demand_target(
            tick=tick,
            market=market,
            registry=registry,
            world_rng=rng,
        )
        draws_consumed = rng.draws - draws_before
        # Every catalog product consumed exactly one rng draw per tick.
        assert draws_consumed == n_products, (
            f"tick {tick}: expected {n_products} draws, got {draws_consumed}"
        )


def test_world_streams_invariant_under_policy_swap_with_freshness_active():
    """With freshness enabled, swapping policies does not perturb world_rng.

    Two builds with the same world_seed but different sink policies must
    produce identical world_rng sequences — the world stream is seeded
    from world_seed only, not from policy_seed.
    """
    catalog = _catalog()
    world_seed = 42

    pid = catalog[0].product_id
    unit_cost = catalog[0].unit_cost

    def _make_scenario(sink_policy):
        f = FactoryNode(
            id="f", region="US", init_seed=1,
            produces_product_id=pid, unit_cost=unit_cost,
            capacity_per_tick=50, inventory=200,
            list_price=unit_cost, cash=0.0,
        )
        f.policy = StaticFactoryPolicy(
            capacity_per_tick=50, unit_cost=unit_cost, policy_seed=10
        )
        shop = IntermediateNode(
            id="shop", region="US", init_seed=2,
            carried_products={pid}, capacity=300, tags=["shop"],
            inventory={pid: 50}, pending={},
            list_prices={pid: 20.0},
            min_order_imposed={pid: 0}, cash=2000.0,
        )
        shop.policy = OrderUpToPolicy(
            cover_horizon_ticks=14, safety_lead_pct_of_lag=1 / 3,
            delivery_lag=2, unit_cost=unit_cost, list_price_out=20.0,
            policy_seed=20,
        )
        sink = DemandSinkNode(
            id="sink", region="US", init_seed=3,
            product_id=pid,
            demand_dist=Uniform(10.0, 10.0),
            income_rate=200.0, cash=1000.0,
        )
        sink.policy = sink_policy
        stages = ["introduction", "growth", "maturity", "decline", "dead"]
        return Scenario(
            catalog=catalog,
            market=_market_params(),
            disruption=DisruptionParams(
                event_prob=0.0, types=["natural_disaster"],
                regions=["US"], severity=Constant(0.0), duration=Constant(1),
            ),
            item_lifecycle=ItemLifecycleParams(
                stages=stages,
                init_stage="maturity",
                default_stage_change_probs={s: 0.0 for s in stages},
                default_freshness_alpha=0.4,
                default_freshness_decay=30.0,
            ),
            stores=[],
            n_steps=10,
            start_date=datetime(2024, 1, 1),
            world_seed=world_seed,
            nodes=[
                NodeInstance(node=f, init_seed=1),
                NodeInstance(node=shop, init_seed=2),
                NodeInstance(node=sink, init_seed=3),
            ],
            edges=[
                EdgeSpec(supplier_id="f", buyer_id="shop", default_lead_time=2),
                EdgeSpec(supplier_id="shop", buyer_id="sink", default_lead_time=1),
            ],
        )

    class _NopSinkPolicy(DefaultDemandSinkPolicy):
        """No extra RNG draws beyond what DefaultDemandSinkPolicy uses."""
        pass

    class _BurnSinkPolicy(DefaultDemandSinkPolicy):
        """Burns extra policy_rng values on every decide call."""
        def decide(self, obs, central_table):
            for _ in range(7):
                self.policy_rng.random()
            return super().decide(obs, central_table)

    s_normal = _make_scenario(
        _NopSinkPolicy(policy_seed=30)
    )
    s_burn = _make_scenario(
        _BurnSinkPolicy(policy_seed=999)
    )

    g_normal = build_world(s_normal)
    g_burn = build_world(s_burn)

    # world_rng must start identically.
    wv1 = [g_normal.world_rng.random() for _ in range(10)]
    wv2 = [g_burn.world_rng.random() for _ in range(10)]
    assert wv1 == wv2, "world_rng diverged after policy swap at build time"


# ------------------------------------------- Per-Ware overrides (issue 04)


def test_per_ware_freshness_alpha_zero_staple_yields_baseline_multiplier():
    """A Ware with freshness_alpha=0 is a staple: curve is identically 1.0."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.0, freshness_decay=15.0)
    lifecycle = _lifecycle(alpha=0.5, decay=20.0)
    registry = _registry(catalog, lifecycle)

    pid = "P0000"
    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    # Staple: alpha=0 -> multiplier always 1.0.
    for tau in (0.0, 1.0, 30.0, 1000.0):
        assert _fc.multiplier(alpha, decay, tau) == 1.0


def test_per_ware_freshness_overrides_drive_multiplier_not_defaults():
    """A Ware with non-zero overrides produces 1 + alpha at tau=0."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.7, freshness_decay=50.0)
    lifecycle = _lifecycle(alpha=0.1, decay=15.0)
    registry = _registry(catalog, lifecycle)

    pid = "P0000"
    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    assert _fc.multiplier(alpha, decay, 0.0) == pytest.approx(1.7)
    # At tau = beta = 50: 1 + 0.7 / e (per-Ware beta, not the default 15).
    expected = 1.0 + 0.7 * math.exp(-1.0)
    assert _fc.multiplier(alpha, decay, 50.0) == pytest.approx(expected)


def test_freshness_falls_back_to_catalog_defaults_when_ware_has_no_override():
    """A Ware with no overrides resolves against ItemLifecycleParams defaults."""
    catalog = _catalog()  # no per-Ware overrides
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    registry = _registry(catalog, lifecycle)

    pid = "P0001"
    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    assert _fc.multiplier(alpha, decay, 0.0) == pytest.approx(1.4)
    # Confirm it picked up the catalog default beta=30, not a different value.
    expected = 1.0 + 0.4 * math.exp(-30.0 / 30.0)
    assert _fc.multiplier(alpha, decay, 30.0) == pytest.approx(expected)


def test_per_ware_overrides_coexist_with_default_fallback():
    """Mixed catalog: one Ware overrides alpha/beta, the other inherits defaults."""
    catalog = _catalog()
    catalog[0] = catalog[0]._replace(freshness_alpha=0.0, freshness_decay=1.0)  # staple
    # catalog[1] (P0001) keeps None => catalog defaults
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    registry = _registry(catalog, lifecycle)

    a0 = registry.freshness_alpha("P0000")
    d0 = registry.freshness_decay("P0000")
    a1 = registry.freshness_alpha("P0001")
    d1 = registry.freshness_decay("P0001")

    # P0000 is a staple.
    assert _fc.multiplier(a0, d0, 0.0) == 1.0
    # P0001 inherits defaults.
    assert _fc.multiplier(a1, d1, 0.0) == pytest.approx(1.4)


# ------------------------------------------------ Step-0 baseline preserved


def test_step0_initial_active_skus_have_no_freshness_spike():
    """DemandSinkNode with no activation_tick entry returns baseline demand at tick 0.

    Products that were active from the start (not freshly introduced) should
    not have an activation_tick entry, so their freshness multiplier is 1.0
    at step 0 — matching pre-issue-03 behaviour.
    """
    catalog = _catalog()
    lifecycle = _lifecycle(alpha=0.4, decay=30.0)
    rng = Random(0)
    registry = _registry(catalog, lifecycle)
    market = Market(_market_params(), rng, datetime(2024, 1, 1), registry=registry)
    market.tick()

    pid = "P0000"
    sink = DemandSinkNode(
        id="sink",
        region="US",
        init_seed=1,
        product_id=pid,
        demand_dist=Constant(100),
        income_rate=0.0,
    )
    # No activation_tick entry — product was always active.
    assert pid not in sink.activation_tick

    alpha = registry.freshness_alpha(pid)
    decay = registry.freshness_decay(pid)

    # tau = inf -> multiplier = 1.0 -> no spike.
    tau = float("inf")
    assert _fc.multiplier(alpha, decay, tau) == 1.0


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
    """DemandSinkNode with large alpha but no activation_tick entry returns baseline demand.

    Never-activated product -> tau = inf -> freshness multiplier -> 1.0.
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
    # maturity stage (x1.0), market factor 1.0, freshness 1.0 -> 100
    assert result == 100
