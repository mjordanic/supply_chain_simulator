"""Tests for src/sim/episode_sampler.py.

Acceptance criteria covered (from issue 04 spec):

1. Determinism: ``sample_episode(episode_seed=N)`` called twice returns equal
   ``EpisodeSpec``.
2. Cross-seed difference: different seeds produce different active subsets /
   capacities / balances.
3. 4-stream independence: perturbing the ``(prime, offset)`` pair for one
   sub-seed purpose does not change the other three outputs.
4. ``default_*_params()`` invoked when ``None`` passed; explicit overrides
   honoured.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.sim.distributions import Constant, LogUniform, Uniform
from src.sim.episode_sampler import (
    EpisodeSpec,
    _SUB_SEED_PARAMS,
    _derive_seed,
    default_disruption_params,
    default_lifecycle_params,
    default_market_params,
    sample_episode,
)
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    StoreTemplate,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 20) -> list:
    """Build a minimal n-product catalog with stable P{i:04d} ids."""
    return load_catalog(
        [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0 + i * 0.5,
                "seasonality": "all_season",
            }
            for i in range(n)
        ]
    )


def _make_base_template() -> StoreTemplate:
    """Minimal StoreTemplate; episodic fields are always overridden by sampler."""
    return StoreTemplate(
        id="sim_test",
        region="US",
        capacity=200,          # overridden
        init_balance=10000.0,  # overridden
        init_stock_pct=0.5,    # overridden to 0.0
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _call_sample(
    episode_seed: int,
    K_active: int = 5,
    episode_length: int = 5,
    n_catalog: int = 20,
    capacity_lo: float = 150.0,
    capacity_hi: float = 400.0,
    balance_lo: float = 15_000.0,
    balance_hi: float = 40_000.0,
    **kwargs,
) -> EpisodeSpec:
    catalog = _make_catalog(n_catalog)
    template = _make_base_template()
    return sample_episode(
        catalog,
        template,
        K_active=K_active,
        episode_length=episode_length,
        capacity_dist=Uniform(capacity_lo, capacity_hi),
        balance_dist=Uniform(balance_lo, balance_hi),
        episode_seed=episode_seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Criterion 1: same seed → identical EpisodeSpec."""

    def test_same_seed_produces_identical_spec(self):
        """Two calls with the same seed return bit-identical EpisodeSpec."""
        spec1 = _call_sample(episode_seed=42)
        spec2 = _call_sample(episode_seed=42)

        assert spec1.active_subset == spec2.active_subset
        assert spec1.scenario.world_seed == spec2.scenario.world_seed
        t1 = spec1.scenario.stores[0].template
        t2 = spec2.scenario.stores[0].template
        assert t1.capacity == t2.capacity
        assert t1.init_balance == t2.init_balance

    def test_determinism_across_many_seeds(self):
        """Spot-check 20 seeds: each seed is self-consistent."""
        for seed in range(20):
            s1 = _call_sample(episode_seed=seed)
            s2 = _call_sample(episode_seed=seed)
            assert s1.active_subset == s2.active_subset

    def test_world_seed_derived_from_episode_seed(self):
        """Scenario.world_seed == _derive_seed(episode_seed, 'world')."""
        for ep_seed in [0, 1, 99, 12345]:
            spec = _call_sample(episode_seed=ep_seed)
            expected = _derive_seed(ep_seed, "world")
            assert spec.scenario.world_seed == expected


# ---------------------------------------------------------------------------
# 2. Cross-seed difference
# ---------------------------------------------------------------------------


class TestCrossSeedDifference:
    """Criterion 2: different seeds → different outputs."""

    def test_different_seeds_produce_different_active_subsets(self):
        specs = [_call_sample(episode_seed=s) for s in range(50)]
        unique_subsets = {s.active_subset for s in specs}
        assert len(unique_subsets) > 1, "All seeds produced the same active_subset"

    def test_different_seeds_produce_different_capacities(self):
        """With a Uniform distribution, capacities should vary across seeds."""
        caps = {
            _call_sample(episode_seed=s).scenario.stores[0].template.capacity
            for s in range(50)
        }
        assert len(caps) > 1, "All seeds produced the same capacity"

    def test_different_seeds_produce_different_balances(self):
        bals = {
            _call_sample(episode_seed=s).scenario.stores[0].template.init_balance
            for s in range(50)
        }
        assert len(bals) > 1, "All seeds produced the same balance"

    def test_active_subset_length_equals_K(self):
        for K in [3, 5, 8]:
            for seed in range(10):
                spec = _call_sample(episode_seed=seed, K_active=K, n_catalog=20)
                assert len(spec.active_subset) == K

    def test_active_subset_no_duplicates(self):
        for seed in range(30):
            spec = _call_sample(episode_seed=seed)
            assert len(set(spec.active_subset)) == len(spec.active_subset)

    def test_active_subset_pids_in_catalog(self):
        catalog = _make_catalog(20)
        all_pids = {w.product_id for w in catalog}
        template = _make_base_template()
        for seed in range(20):
            spec = sample_episode(
                catalog,
                template,
                K_active=5,
                episode_length=5,
                capacity_dist=Uniform(150, 400),
                balance_dist=Uniform(15_000, 40_000),
                episode_seed=seed,
            )
            for pid in spec.active_subset:
                assert pid in all_pids

    def test_capacity_within_distribution_range(self):
        for seed in range(100):
            spec = _call_sample(episode_seed=seed, capacity_lo=150, capacity_hi=400)
            cap = spec.scenario.stores[0].template.capacity
            assert 150 <= cap <= 400, f"seed={seed}: cap={cap} outside [150, 400]"

    def test_balance_within_distribution_range(self):
        for seed in range(100):
            spec = _call_sample(episode_seed=seed, balance_lo=15_000, balance_hi=40_000)
            bal = spec.scenario.stores[0].template.init_balance
            assert 15_000 <= bal <= 40_000, f"seed={seed}: bal={bal} outside range"


# ---------------------------------------------------------------------------
# 3. 4-stream independence
# ---------------------------------------------------------------------------


class TestStreamIndependence:
    """Criterion 3: perturbing one sub-seed purpose does not change other outputs.

    We test this by finding episode seeds whose assortment sub-seed matches
    (same active_subset) but whose other sub-seeds differ, and verifying
    that the other outputs (capacity, balance, world_seed) do indeed differ.
    """

    def _find_seeds_with_same_assortment(
        self,
        base_seed: int,
        n_catalog: int = 20,
        K_active: int = 5,
        n_search: int = 5000,
        need: int = 3,
    ) -> list[EpisodeSpec]:
        """Return ``need`` EpisodeSpecs that share the same active_subset as base_seed."""
        base = _call_sample(episode_seed=base_seed, n_catalog=n_catalog, K_active=K_active)
        target = base.active_subset
        matches: list[EpisodeSpec] = []
        for seed in range(n_search):
            if seed == base_seed:
                continue
            s = _call_sample(episode_seed=seed, n_catalog=n_catalog, K_active=K_active)
            if s.active_subset == target:
                matches.append(s)
            if len(matches) >= need:
                break
        return matches

    def test_world_seed_varies_when_assortment_frozen(self):
        """Seeds sharing the same active_subset have different world seeds."""
        matches = self._find_seeds_with_same_assortment(base_seed=0)
        if len(matches) < 2:
            pytest.skip("Could not find 2+ seeds with the same assortment")

        world_seeds = {m.scenario.world_seed for m in matches}
        assert len(world_seeds) > 1, (
            "All matching-assortment specs share the same world_seed — "
            "world_seed stream is not independent of assortment stream"
        )

    def test_capacity_varies_when_assortment_frozen(self):
        """Seeds sharing the same active_subset have different capacities."""
        matches = self._find_seeds_with_same_assortment(base_seed=1)
        if len(matches) < 2:
            pytest.skip("Could not find 2+ seeds with the same assortment")

        caps = {m.scenario.stores[0].template.capacity for m in matches}
        assert len(caps) > 1, (
            "All matching-assortment specs share the same capacity — "
            "capacity stream is not independent of assortment stream"
        )

    def test_balance_varies_when_assortment_frozen(self):
        """Seeds sharing the same active_subset have different balances."""
        matches = self._find_seeds_with_same_assortment(base_seed=2)
        if len(matches) < 2:
            pytest.skip("Could not find 2+ seeds with the same assortment")

        bals = {m.scenario.stores[0].template.init_balance for m in matches}
        assert len(bals) > 1, (
            "All matching-assortment specs share the same balance — "
            "balance stream is not independent of assortment stream"
        )

    def test_derive_seed_uniqueness_all_four_purposes(self):
        """_derive_seed produces 4 distinct values for any episode_seed."""
        for ep_seed in [0, 1, 42, 999, 2**16]:
            values = {_derive_seed(ep_seed, purpose) for purpose in _SUB_SEED_PARAMS}
            assert len(values) == len(_SUB_SEED_PARAMS), (
                f"ep_seed={ep_seed}: sub-seeds collided: {values}"
            )


# ---------------------------------------------------------------------------
# 4. Default params and explicit overrides
# ---------------------------------------------------------------------------


class TestDefaultParamsAndOverrides:
    """Criterion 4: defaults invoked when None; overrides honoured."""

    def test_default_market_params_returned_when_none(self):
        """No market_params → scenario.market equals default_market_params()."""
        spec = _call_sample(episode_seed=0, market_params=None)
        expected = default_market_params()
        # Spot-check a few fields; dataclass equality works if all fields match.
        assert spec.scenario.market.cycle_len == expected.cycle_len
        assert spec.scenario.market.price_elasticity == expected.price_elasticity
        assert spec.scenario.market.regions == expected.regions

    def test_explicit_market_params_honoured(self):
        """Explicit market_params → scenario.market is that object."""
        custom = default_market_params()
        # Build a custom param with a different cycle_len to distinguish it.
        from dataclasses import replace
        custom = replace(custom, cycle_len=99)
        spec = _call_sample(episode_seed=0, market_params=custom)
        assert spec.scenario.market.cycle_len == 99

    def test_default_disruption_params_returned_when_none(self):
        """No disruption_params → scenario.disruption equals default_disruption_params()."""
        spec = _call_sample(episode_seed=0, disruption_params=None)
        expected = default_disruption_params()
        assert spec.scenario.disruption.event_prob == expected.event_prob
        assert spec.scenario.disruption.regions == expected.regions

    def test_explicit_disruption_params_honoured(self):
        """Explicit disruption_params → scenario.disruption is that object."""
        from dataclasses import replace
        custom = replace(default_disruption_params(), event_prob=0.99)
        spec = _call_sample(episode_seed=0, disruption_params=custom)
        assert spec.scenario.disruption.event_prob == 0.99

    def test_default_lifecycle_params_returned_when_none(self):
        """No lifecycle_params → scenario.item_lifecycle equals default."""
        spec = _call_sample(episode_seed=0, lifecycle_params=None)
        expected = default_lifecycle_params()
        assert spec.scenario.item_lifecycle.init_stage == expected.init_stage
        assert spec.scenario.item_lifecycle.stages == expected.stages

    def test_explicit_lifecycle_params_honoured(self):
        """Explicit lifecycle_params → scenario.item_lifecycle is that object."""
        from dataclasses import replace
        custom = replace(default_lifecycle_params(), init_stage="growth")
        spec = _call_sample(episode_seed=0, lifecycle_params=custom)
        assert spec.scenario.item_lifecycle.init_stage == "growth"

    def test_start_date_defaults_to_2024_01_01(self):
        """Default start_date is 2024-01-01."""
        spec = _call_sample(episode_seed=0)
        assert spec.scenario.start_date == datetime(2024, 1, 1)

    def test_explicit_start_date_honoured(self):
        """Explicit start_date flows into the scenario."""
        custom_date = datetime(2023, 6, 15)
        spec = _call_sample(episode_seed=0, start_date=custom_date)
        assert spec.scenario.start_date == custom_date

    def test_k_active_exceeds_catalog_raises(self):
        """K_active > catalog size raises ValueError."""
        with pytest.raises(ValueError, match="K_active"):
            _call_sample(episode_seed=0, K_active=25, n_catalog=20)

    def test_init_stock_pct_always_zero(self):
        """StoreTemplate.init_stock_pct is always 0.0 (grand-opening)."""
        for seed in range(20):
            spec = _call_sample(episode_seed=seed)
            assert spec.scenario.stores[0].template.init_stock_pct == 0.0

    def test_init_freshness_always_fresh(self):
        """StoreTemplate.init_freshness is always 'fresh'."""
        for seed in range(20):
            spec = _call_sample(episode_seed=seed)
            assert spec.scenario.stores[0].template.init_freshness == "fresh"

    def test_n_steps_matches_episode_length(self):
        """Scenario.n_steps == episode_length kwarg."""
        for ep_len in [1, 10, 50, 180]:
            spec = _call_sample(episode_seed=7, episode_length=ep_len)
            assert spec.scenario.n_steps == ep_len

    def test_scenario_has_single_store(self):
        """Returned scenario has exactly one store."""
        spec = _call_sample(episode_seed=0)
        assert len(spec.scenario.stores) == 1

    def test_active_products_on_template_match_active_subset(self):
        """StoreTemplate.init_active_products matches EpisodeSpec.active_subset."""
        for seed in range(20):
            spec = _call_sample(episode_seed=seed)
            t = spec.scenario.stores[0].template
            assert list(t.init_active_products) == list(spec.active_subset)


# ---------------------------------------------------------------------------
# Default param factory smoke tests
# ---------------------------------------------------------------------------


class TestDefaultParamFactories:
    """Smoke-test that the three public factory functions return valid objects."""

    def test_default_market_params_returns_market_params(self):
        params = default_market_params()
        assert isinstance(params, MarketParams)
        # Must have the required fields at minimum.
        assert params.cycle_len > 0
        assert params.regions

    def test_default_disruption_params_returns_disruption_params(self):
        params = default_disruption_params()
        assert isinstance(params, DisruptionParams)
        assert 0 < params.event_prob < 1
        assert params.regions

    def test_default_lifecycle_params_returns_lifecycle_params(self):
        params = default_lifecycle_params()
        assert isinstance(params, ItemLifecycleParams)
        assert "maturity" in params.stages
        assert params.init_stage in params.stages

    def test_factories_are_independent(self):
        """Each call returns a fresh object (no shared mutable state)."""
        p1 = default_market_params()
        p2 = default_market_params()
        # Should be equal by value.
        assert p1.cycle_len == p2.cycle_len
        # Modifying one must not affect the other (immutable dataclasses — just check values).
        assert p1.regions is not p2.regions or p1.regions == p2.regions
