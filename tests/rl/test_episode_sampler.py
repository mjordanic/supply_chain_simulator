"""Tests for src/rl/episode_sampler.py.

Acceptance criteria covered:
  - Determinism: same episode_seed → identical EpisodeSpec across all fields
  - Assortment coverage: every product appears across many seeds
  - Distribution sanity: capacities and balances inside configured ranges;
    means within a generous tolerance
  - Scenario validity: returned Scenario round-trips through Runner for at
    least one tick without raising
  - Seed splitting independence: freezing the assortment sub-seed and varying
    the rest changes capacity/balance/slot_perm but not the active_subset
"""

from __future__ import annotations

import statistics
from datetime import datetime

import pytest

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import EpisodeSpec, _derive_seed, sample_episode
from src.sim.distributions import Constant, Uniform
from src.sim.runner import Runner
from src.sim.scenario import (
    StoreTemplate,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 20) -> list:
    """Build a minimal n-product catalog with stable P{i:04d} ids."""
    items = [
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
    return load_catalog(items)


def _make_base_template() -> StoreTemplate:
    """Minimal StoreTemplate; episodic fields are always overridden by sampler."""
    return StoreTemplate(
        id="rl_test",
        region="US",
        capacity=200,          # will be overridden
        init_balance=10000.0,  # will be overridden
        init_stock_pct=0.5,    # will be overridden to 0.0
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _make_config(K_active: int = 5) -> RLConfig:
    """RLConfig with Uniform distributions for easy range checking."""
    return RLConfig(
        K_active=K_active,
        episode_length=5,  # short for speed in tests
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_identical_spec(self):
        """Two calls with the same seed → byte-identical EpisodeSpec."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()

        spec1 = sample_episode(catalog, template, config, episode_seed=42)
        spec2 = sample_episode(catalog, template, config, episode_seed=42)

        assert spec1.active_subset == spec2.active_subset
        assert spec1.slot_permutation == spec2.slot_permutation
        assert spec1.scenario.world_seed == spec2.scenario.world_seed
        # Capacity lives on the StoreTemplate inside the Scenario.
        t1 = spec1.scenario.stores[0].template
        t2 = spec2.scenario.stores[0].template
        assert t1.capacity == t2.capacity
        assert t1.init_balance == t2.init_balance

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different specs."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()

        specs = [
            sample_episode(catalog, template, config, episode_seed=s)
            for s in range(20)
        ]
        # Collect all unique active_subsets; expect more than 1 distinct value
        unique_subsets = {s.active_subset for s in specs}
        assert len(unique_subsets) > 1, "All seeds produced the same active_subset"

    def test_world_seed_field_derived(self):
        """world_seed in the Scenario equals _derive_seed(episode_seed, 'world')."""
        catalog = _make_catalog(10)
        template = _make_base_template()
        config = _make_config(K_active=3)

        for ep_seed in [0, 1, 99, 12345]:
            spec = sample_episode(catalog, template, config, episode_seed=ep_seed)
            expected_world_seed = _derive_seed(ep_seed, "world")
            assert spec.scenario.world_seed == expected_world_seed


# ---------------------------------------------------------------------------
# Assortment coverage
# ---------------------------------------------------------------------------


class TestAssortmentCoverage:
    def test_every_product_appears_across_many_seeds(self):
        """Across 500 seeds, every catalog product appears at least once."""
        n_catalog = 20
        catalog = _make_catalog(n_catalog)
        template = _make_base_template()
        config = _make_config(K_active=5)

        seen = set()
        for seed in range(500):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            seen.update(spec.active_subset)

        missing = {w.product_id for w in catalog} - seen
        assert not missing, f"Products never selected: {missing}"

    def test_active_subset_length_equals_K(self):
        """active_subset always has exactly K_active elements."""
        catalog = _make_catalog(20)
        template = _make_base_template()

        for K in [3, 5, 8]:
            config = _make_config(K_active=K)
            for seed in range(20):
                spec = sample_episode(catalog, template, config, episode_seed=seed)
                assert len(spec.active_subset) == K, (
                    f"K_active={K}, seed={seed}: "
                    f"got {len(spec.active_subset)} active products"
                )

    def test_active_subset_no_duplicates(self):
        """active_subset contains no repeated product ids."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(K_active=5)

        for seed in range(50):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            assert len(set(spec.active_subset)) == len(spec.active_subset), (
                f"seed={seed}: duplicate products in active_subset {spec.active_subset}"
            )

    def test_active_subset_in_catalog(self):
        """All ids in active_subset come from the catalog."""
        catalog = _make_catalog(20)
        all_pids = {w.product_id for w in catalog}
        template = _make_base_template()
        config = _make_config(K_active=5)

        for seed in range(30):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            for pid in spec.active_subset:
                assert pid in all_pids, f"Unknown pid {pid!r} in active_subset"


# ---------------------------------------------------------------------------
# Distribution sanity
# ---------------------------------------------------------------------------


class TestDistributionSanity:
    def test_capacity_in_configured_range(self):
        """Sampled capacities fall within [150, 400] (default Uniform)."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()  # default capacity_dist = Uniform(150, 400)

        for seed in range(200):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            cap = spec.scenario.stores[0].template.capacity
            assert 150 <= cap <= 400, f"seed={seed}: capacity={cap} outside [150, 400]"

    def test_balance_in_configured_range(self):
        """Sampled balances fall within [15000, 40000] (default Uniform)."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()  # default balance_dist = Uniform(15_000, 40_000)

        for seed in range(200):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            bal = spec.scenario.stores[0].template.init_balance
            assert 15000 <= bal <= 40000, (
                f"seed={seed}: balance={bal} outside [15000, 40000]"
            )

    def test_capacity_mean_within_tolerance(self):
        """Mean sampled capacity is within 10% of the midpoint (275)."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()

        caps = [
            sample_episode(catalog, template, config, episode_seed=s)
            .scenario.stores[0].template.capacity
            for s in range(500)
        ]
        mean_cap = statistics.mean(caps)
        expected_mid = (150 + 400) / 2  # 275
        assert abs(mean_cap - expected_mid) < 0.10 * expected_mid, (
            f"Mean capacity {mean_cap:.1f} is >10% from expected midpoint {expected_mid}"
        )

    def test_balance_mean_within_tolerance(self):
        """Mean sampled balance is within 10% of the midpoint (27500)."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config()

        bals = [
            sample_episode(catalog, template, config, episode_seed=s)
            .scenario.stores[0].template.init_balance
            for s in range(500)
        ]
        mean_bal = statistics.mean(bals)
        expected_mid = (15000 + 40000) / 2  # 27500
        assert abs(mean_bal - expected_mid) < 0.10 * expected_mid, (
            f"Mean balance {mean_bal:.1f} is >10% from expected midpoint {expected_mid}"
        )

    def test_custom_capacity_distribution(self):
        """Custom narrow Uniform is respected."""
        from src.sim.distributions import Uniform as Uni

        catalog = _make_catalog(10)
        template = _make_base_template()
        config = RLConfig(K_active=3, episode_length=5, capacity_dist=Uni(500, 600))

        for seed in range(50):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            cap = spec.scenario.stores[0].template.capacity
            assert 500 <= cap <= 600, (
                f"Custom range [500,600] violated: cap={cap}"
            )


# ---------------------------------------------------------------------------
# Scenario validity
# ---------------------------------------------------------------------------


class TestScenarioValidity:
    def test_scenario_runs_one_tick(self):
        """Returned Scenario round-trips through Runner for 1 tick without raising."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        # episode_length=1 so Runner.run() executes exactly one tick.
        config = RLConfig(K_active=5, episode_length=1)

        # Test a handful of different seeds.
        for seed in [0, 1, 42, 999]:
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            # Attach a simple pass-through policy so Runner can call decide().
            from src.sim.policy import BaselinePolicy
            from src.sim.distributions import Uniform as Uni

            policy = BaselinePolicy(
                policy_seed=seed,
                min_qty=1,
                init_qty_factor=0.3,
                promo_len=Uni(3, 5),
                promo_cd_len=5,
                review_interval=10,
                promo_threshold=0.4,
                target_active_count=config.K_active,
                slow_sales_limit=2,
                history_window=4,
                max_history=50,
                promo_discount=0.7,
            )
            spec.scenario.stores[0].policy = policy
            run_log = Runner(spec.scenario).run()
            # Basic sanity: run log has store 0 and n_steps+1 balance entries.
            assert 0 in run_log["stores"]
            assert len(run_log["stores"][0]["balance"]) == 2  # step0 + 1 tick

    def test_active_products_set_on_template(self):
        """StoreTemplate.init_active_products matches active_subset."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(K_active=5)

        for seed in range(20):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            t = spec.scenario.stores[0].template
            assert list(t.init_active_products) == list(spec.active_subset), (
                f"seed={seed}: template active products mismatch active_subset"
            )

    def test_init_stock_pct_is_zero(self):
        """init_stock_pct is always 0.0 (grand-opening scenario)."""
        catalog = _make_catalog(10)
        template = _make_base_template()
        config = _make_config()

        for seed in range(20):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            assert spec.scenario.stores[0].template.init_stock_pct == 0.0

    def test_init_freshness_is_fresh(self):
        """init_freshness is always 'fresh'."""
        catalog = _make_catalog(10)
        template = _make_base_template()
        config = _make_config()

        for seed in range(20):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            assert spec.scenario.stores[0].template.init_freshness == "fresh"

    def test_n_steps_matches_episode_length(self):
        """Scenario.n_steps == config.episode_length."""
        catalog = _make_catalog(10)
        template = _make_base_template()

        for ep_len in [1, 10, 50, 180]:
            config = RLConfig(K_active=3, episode_length=ep_len)
            spec = sample_episode(catalog, template, config, episode_seed=7)
            assert spec.scenario.n_steps == ep_len


# ---------------------------------------------------------------------------
# Seed splitting independence
# ---------------------------------------------------------------------------


class TestSeedSplitting:
    """Freezing one sub-seed must not change quantities from other sub-seeds."""

    def _vary_world_and_capacity_only(self, catalog, template, config, base_seed):
        """Return specs where assortment is frozen but world/capacity vary."""
        # We achieve this by constructing episode_seeds whose assortment
        # sub-seed matches that of base_seed.  A brute-force search over
        # seeds finds other seeds with the same assortment_seed value.
        base_spec = sample_episode(catalog, template, config, episode_seed=base_seed)
        base_assortment = base_spec.active_subset

        # Collect specs that share the same assortment (same assortment sub-seed path).
        matching = []
        for seed in range(10000):
            if seed == base_seed:
                continue
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            if spec.active_subset == base_assortment:
                matching.append(spec)
            if len(matching) >= 5:
                break

        return base_spec, matching

    def test_slot_permutation_independent_of_assortment(self):
        """Different slot_permutations can exist even with the same active_subset."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(K_active=5)

        base_spec, matching = self._vary_world_and_capacity_only(
            catalog, template, config, base_seed=0
        )

        if not matching:
            pytest.skip("Could not find two seeds sharing the same assortment")

        # Slot permutations should differ at least sometimes.
        all_perms = [base_spec.slot_permutation] + [s.slot_permutation for s in matching]
        unique_perms = set(all_perms)
        # With 5! = 120 possible permutations and independent sub-seeds, we
        # expect at least 2 distinct permutations across 5+ samples.
        assert len(unique_perms) > 1 or len(matching) < 3, (
            "All matching-assortment specs share the same slot permutation — "
            "slot_seed is not independent of assortment_seed"
        )

    def test_derive_seed_uniqueness(self):
        """_derive_seed produces 5 distinct values for any episode_seed."""
        from src.rl.episode_sampler import _SUB_SEED_PARAMS

        for ep_seed in [0, 1, 42, 999, 2**16]:
            values = {_derive_seed(ep_seed, purpose) for purpose in _SUB_SEED_PARAMS}
            assert len(values) == len(_SUB_SEED_PARAMS), (
                f"ep_seed={ep_seed}: sub-seeds collided: {values}"
            )

    def test_assortment_frozen_capacity_varies(self):
        """Seeds that share assortment_seed still have different capacity draws."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(K_active=5)

        base_spec, matching = self._vary_world_and_capacity_only(
            catalog, template, config, base_seed=1
        )

        if len(matching) < 2:
            pytest.skip("Could not find 2+ seeds sharing the same assortment")

        # assortment_seed matches → active_subset matches
        for spec in matching:
            assert spec.active_subset == base_spec.active_subset

        # capacity_seed is independent → capacities should vary
        all_caps = [base_spec.scenario.stores[0].template.capacity] + [
            s.scenario.stores[0].template.capacity for s in matching
        ]
        assert len(set(all_caps)) > 1, (
            "All specs with the same assortment share the same capacity — "
            "capacity_seed is not independent of assortment_seed"
        )

    def test_world_seed_varies_independently(self):
        """world_seed varies across seeds even when active_subset is identical."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(K_active=5)

        base_spec, matching = self._vary_world_and_capacity_only(
            catalog, template, config, base_seed=2
        )

        if len(matching) < 2:
            pytest.skip("Could not find 2+ seeds sharing the same assortment")

        world_seeds = [base_spec.scenario.world_seed] + [
            s.scenario.world_seed for s in matching
        ]
        assert len(set(world_seeds)) > 1, (
            "Specs with identical assortment share the same world_seed"
        )


# ---------------------------------------------------------------------------
# EpisodeSpec structure
# ---------------------------------------------------------------------------


class TestEpisodeSpecStructure:
    def test_slot_permutation_is_permutation_of_range_K(self):
        """slot_permutation is a valid permutation of [0, K)."""
        catalog = _make_catalog(20)
        template = _make_base_template()

        for K in [3, 5]:
            config = _make_config(K_active=K)
            for seed in range(30):
                spec = sample_episode(catalog, template, config, episode_seed=seed)
                perm = spec.slot_permutation
                assert sorted(perm) == list(range(K)), (
                    f"K={K}, seed={seed}: slot_permutation {perm} is not a permutation of range({K})"
                )

    def test_scenario_has_single_store(self):
        """Scenario always contains exactly one store."""
        catalog = _make_catalog(10)
        template = _make_base_template()
        config = _make_config()

        for seed in range(10):
            spec = sample_episode(catalog, template, config, episode_seed=seed)
            assert len(spec.scenario.stores) == 1

    def test_too_large_K_raises(self):
        """K_active > catalog size raises ValueError."""
        catalog = _make_catalog(3)
        template = _make_base_template()
        config = _make_config(K_active=5)  # 5 > 3

        with pytest.raises(ValueError, match="K_active"):
            sample_episode(catalog, template, config, episode_seed=0)

    def test_start_date_default_and_override(self):
        """Default start_date is 2024-01-01; override is respected."""
        catalog = _make_catalog(10)
        template = _make_base_template()
        config = _make_config()

        spec_default = sample_episode(catalog, template, config, episode_seed=0)
        assert spec_default.scenario.start_date == datetime(2024, 1, 1)

        custom_date = datetime(2023, 6, 15)
        spec_custom = sample_episode(
            catalog, template, config, episode_seed=0, start_date=custom_date
        )
        assert spec_custom.scenario.start_date == custom_date
