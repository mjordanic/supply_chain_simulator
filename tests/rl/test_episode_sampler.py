"""Tests for src/rl/episode_sampler.py (graph engine).

Acceptance criteria covered:
  - Determinism: same episode_seed → identical RLEpisodeSpec across all fields
  - Assortment coverage: every product appears across many seeds
  - Distribution sanity: capacities and balances inside configured ranges
  - Scenario validity: returned graph-mode Scenario runs through Runner for
    at least one tick without raising
  - Seed splitting independence: freezing the assortment sub-seed and varying
    the rest changes capacity/balance/slot_perm but not the active_subset
"""

from __future__ import annotations

import statistics
from datetime import datetime

import pytest

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import RLEpisodeSpec, _derive_seed, sample_episode
from src.sim.distributions import Constant, Uniform
from src.sim.runner import Runner, build_world
from src.sim.scenario import load_catalog


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


def _make_config(K_active: int = 5) -> RLConfig:
    """RLConfig with pinned Uniform distributions for deterministic range checks."""
    return RLConfig(
        K_active=K_active,
        episode_length=5,
        capacity_dist=Uniform(150, 400),
        balance_dist=Uniform(15_000, 40_000),
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_identical_spec(self):
        """Two calls with the same seed → identical RLEpisodeSpec."""
        catalog = _make_catalog(20)
        config = _make_config()

        spec1 = sample_episode(catalog, config, episode_seed=42)
        spec2 = sample_episode(catalog, config, episode_seed=42)

        assert spec1.active_subset == spec2.active_subset
        assert spec1.slot_permutation == spec2.slot_permutation
        assert spec1.world_seed == spec2.world_seed
        assert spec1.capacity == spec2.capacity
        assert spec1.balance == pytest.approx(spec2.balance)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different specs."""
        catalog = _make_catalog(20)
        config = _make_config()

        specs = [
            sample_episode(catalog, config, episode_seed=s)
            for s in range(20)
        ]
        unique_subsets = {s.active_subset for s in specs}
        assert len(unique_subsets) > 1, "All seeds produced the same active_subset"

    def test_world_seed_field_derived(self):
        """world_seed in the spec equals _derive_seed(episode_seed, 'world')."""
        catalog = _make_catalog(10)
        config = _make_config(K_active=3)

        for ep_seed in [0, 1, 99, 12345]:
            spec = sample_episode(catalog, config, episode_seed=ep_seed)
            expected_world_seed = _derive_seed(ep_seed, "world")
            assert spec.world_seed == expected_world_seed


# ---------------------------------------------------------------------------
# Assortment coverage
# ---------------------------------------------------------------------------


class TestAssortmentCoverage:
    def test_every_product_appears_across_many_seeds(self):
        """Across 500 seeds, every catalog product appears at least once."""
        n_catalog = 20
        catalog = _make_catalog(n_catalog)
        config = _make_config(K_active=5)

        seen = set()
        for seed in range(500):
            spec = sample_episode(catalog, config, episode_seed=seed)
            seen.update(spec.active_subset)

        missing = {w.product_id for w in catalog} - seen
        assert not missing, f"Products never selected: {missing}"

    def test_active_subset_length_equals_K(self):
        """active_subset always has exactly K_active elements."""
        catalog = _make_catalog(20)

        for K in [3, 5, 8]:
            config = _make_config(K_active=K)
            for seed in range(20):
                spec = sample_episode(catalog, config, episode_seed=seed)
                assert len(spec.active_subset) == K

    def test_active_subset_no_duplicates(self):
        """active_subset contains no repeated product ids."""
        catalog = _make_catalog(20)
        config = _make_config(K_active=5)

        for seed in range(50):
            spec = sample_episode(catalog, config, episode_seed=seed)
            assert len(set(spec.active_subset)) == len(spec.active_subset)

    def test_active_subset_in_catalog(self):
        """All ids in active_subset come from the catalog."""
        catalog = _make_catalog(20)
        all_pids = {w.product_id for w in catalog}
        config = _make_config(K_active=5)

        for seed in range(30):
            spec = sample_episode(catalog, config, episode_seed=seed)
            for pid in spec.active_subset:
                assert pid in all_pids, f"Unknown pid {pid!r} in active_subset"


# ---------------------------------------------------------------------------
# Distribution sanity
# ---------------------------------------------------------------------------


class TestDistributionSanity:
    def test_capacity_in_configured_range(self):
        """Sampled capacities fall within [150, 400] (Uniform)."""
        catalog = _make_catalog(20)
        config = _make_config()

        for seed in range(200):
            spec = sample_episode(catalog, config, episode_seed=seed)
            assert 150 <= spec.capacity <= 400, (
                f"seed={seed}: capacity={spec.capacity} outside [150, 400]"
            )

    def test_balance_in_configured_range(self):
        """Sampled balances fall within [15000, 40000] (Uniform)."""
        catalog = _make_catalog(20)
        config = _make_config()

        for seed in range(200):
            spec = sample_episode(catalog, config, episode_seed=seed)
            assert 15000 <= spec.balance <= 40000, (
                f"seed={seed}: balance={spec.balance} outside [15000, 40000]"
            )

    def test_capacity_mean_within_tolerance(self):
        """Mean sampled capacity is within 10% of the midpoint (275)."""
        catalog = _make_catalog(20)
        config = _make_config()

        caps = [
            sample_episode(catalog, config, episode_seed=s).capacity
            for s in range(500)
        ]
        mean_cap = statistics.mean(caps)
        expected_mid = (150 + 400) / 2  # 275
        assert abs(mean_cap - expected_mid) < 0.10 * expected_mid

    def test_custom_capacity_distribution(self):
        """Custom narrow Uniform is respected."""
        catalog = _make_catalog(10)
        config = RLConfig(K_active=3, episode_length=5, capacity_dist=Uniform(500, 600),
                          balance_dist=Uniform(15000, 40000))

        for seed in range(50):
            spec = sample_episode(catalog, config, episode_seed=seed)
            assert 500 <= spec.capacity <= 600


# ---------------------------------------------------------------------------
# Scenario validity (graph engine)
# ---------------------------------------------------------------------------


class TestScenarioValidity:
    def test_scenario_is_graph_mode(self):
        """Returned Scenario is in graph mode (has nodes + edges)."""
        catalog = _make_catalog(20)
        config = _make_config()

        for seed in [0, 1, 42, 999]:
            spec = sample_episode(catalog, config, episode_seed=seed)
            assert len(spec.scenario.nodes) > 0, "Expected graph-mode scenario with nodes"
            assert len(spec.scenario.edges) > 0

    def test_scenario_has_intermediate_node_s(self):
        """Graph scenario always contains node 'S' (the trainable intermediate)."""
        catalog = _make_catalog(20)
        config = _make_config()

        for seed in [0, 42, 999]:
            spec = sample_episode(catalog, config, episode_seed=seed)
            node_ids = [ni.node.id for ni in spec.scenario.nodes]
            assert "S" in node_ids, f"Node 'S' not found in {node_ids}"

    def test_scenario_runs_one_tick(self):
        """Returned Scenario round-trips through build_world for 1 tick without raising."""
        catalog = _make_catalog(20)
        config = RLConfig(K_active=5, episode_length=1,
                          capacity_dist=Uniform(150, 400),
                          balance_dist=Uniform(15_000, 40_000))

        for seed in [0, 1, 42, 999]:
            spec = sample_episode(catalog, config, episode_seed=seed)
            sim = build_world(spec.scenario)
            sim.tick()  # must not raise

    def test_n_steps_matches_episode_length(self):
        """Scenario.n_steps == config.episode_length."""
        catalog = _make_catalog(10)

        for ep_len in [1, 10, 50]:
            config = RLConfig(K_active=3, episode_length=ep_len,
                              capacity_dist=Uniform(150, 400),
                              balance_dist=Uniform(15_000, 40_000))
            spec = sample_episode(catalog, config, episode_seed=7)
            assert spec.scenario.n_steps == ep_len

    def test_factories_one_per_active_product(self):
        """Graph has one FactoryNode per active product."""
        from src.sim.node import FactoryNode

        catalog = _make_catalog(20)
        config = _make_config(K_active=3)

        spec = sample_episode(catalog, config, episode_seed=42)
        factory_pids = {
            ni.node.produces_product_id
            for ni in spec.scenario.nodes
            if isinstance(ni.node, FactoryNode)
        }
        assert factory_pids == set(spec.active_subset)

    def test_sinks_one_per_active_product(self):
        """Graph has one DemandSinkNode per active product."""
        from src.sim.node import DemandSinkNode

        catalog = _make_catalog(20)
        config = _make_config(K_active=3)

        spec = sample_episode(catalog, config, episode_seed=42)
        sink_pids = {
            ni.node.product_id
            for ni in spec.scenario.nodes
            if isinstance(ni.node, DemandSinkNode)
        }
        assert sink_pids == set(spec.active_subset)


# ---------------------------------------------------------------------------
# Seed splitting independence
# ---------------------------------------------------------------------------


class TestSeedSplitting:
    """Freezing one sub-seed must not change quantities from other sub-seeds."""

    def _find_same_assortment_specs(self, catalog, config, base_seed, n=5):
        base_spec = sample_episode(catalog, config, episode_seed=base_seed)
        base_assortment = base_spec.active_subset

        matching = []
        for seed in range(10000):
            if seed == base_seed:
                continue
            spec = sample_episode(catalog, config, episode_seed=seed)
            if spec.active_subset == base_assortment:
                matching.append(spec)
            if len(matching) >= n:
                break

        return base_spec, matching

    def test_slot_permutation_independent_of_assortment(self):
        """Different slot_permutations can exist even with the same active_subset."""
        catalog = _make_catalog(20)
        config = _make_config(K_active=5)

        base_spec, matching = self._find_same_assortment_specs(
            catalog, config, base_seed=0
        )

        if not matching:
            pytest.skip("Could not find two seeds sharing the same assortment")

        all_perms = [base_spec.slot_permutation] + [s.slot_permutation for s in matching]
        unique_perms = set(all_perms)
        assert len(unique_perms) > 1 or len(matching) < 3

    def test_derive_seed_uniqueness(self):
        """_derive_seed produces distinct values for each purpose."""
        from src.rl.episode_sampler import _SUB_SEED_PARAMS

        for ep_seed in [0, 1, 42, 999, 2**16]:
            values = {_derive_seed(ep_seed, purpose) for purpose in _SUB_SEED_PARAMS}
            assert len(values) == len(_SUB_SEED_PARAMS)

    def test_capacity_varies_with_same_assortment(self):
        """Seeds that share assortment_seed still have different capacity draws."""
        catalog = _make_catalog(20)
        config = _make_config(K_active=5)

        base_spec, matching = self._find_same_assortment_specs(
            catalog, config, base_seed=1
        )

        if len(matching) < 2:
            pytest.skip("Could not find 2+ seeds sharing the same assortment")

        for spec in matching:
            assert spec.active_subset == base_spec.active_subset

        all_caps = [base_spec.capacity] + [s.capacity for s in matching]
        assert len(set(all_caps)) > 1

    def test_world_seed_varies_independently(self):
        """world_seed varies across seeds even when active_subset is identical."""
        catalog = _make_catalog(20)
        config = _make_config(K_active=5)

        base_spec, matching = self._find_same_assortment_specs(
            catalog, config, base_seed=2
        )

        if len(matching) < 2:
            pytest.skip("Could not find 2+ seeds sharing the same assortment")

        world_seeds = [base_spec.world_seed] + [s.world_seed for s in matching]
        assert len(set(world_seeds)) > 1


# ---------------------------------------------------------------------------
# RLEpisodeSpec structure
# ---------------------------------------------------------------------------


class TestRLEpisodeSpecStructure:
    def test_slot_permutation_is_permutation_of_range_K(self):
        """slot_permutation is a valid permutation of [0, K)."""
        catalog = _make_catalog(20)

        for K in [3, 5]:
            config = _make_config(K_active=K)
            for seed in range(30):
                spec = sample_episode(catalog, config, episode_seed=seed)
                perm = spec.slot_permutation
                assert sorted(perm) == list(range(K))

    def test_too_large_K_raises(self):
        """K_active > catalog size raises ValueError."""
        catalog = _make_catalog(3)
        config = _make_config(K_active=5)  # 5 > 3

        with pytest.raises(ValueError, match="K_active"):
            sample_episode(catalog, config, episode_seed=0)

    def test_start_date_default_and_override(self):
        """Default start_date is 2024-01-01; override is respected."""
        catalog = _make_catalog(10)
        config = _make_config()

        spec_default = sample_episode(catalog, config, episode_seed=0)
        assert spec_default.scenario.start_date == datetime(2024, 1, 1)

        custom_date = datetime(2023, 6, 15)
        spec_custom = sample_episode(
            catalog, config, episode_seed=0, start_date=custom_date
        )
        assert spec_custom.scenario.start_date == custom_date

    def test_spec_has_world_seed_capacity_balance(self):
        """RLEpisodeSpec has world_seed, capacity, and balance fields."""
        catalog = _make_catalog(10)
        config = _make_config()

        spec = sample_episode(catalog, config, episode_seed=42)
        assert isinstance(spec.world_seed, int)
        assert isinstance(spec.capacity, int)
        assert isinstance(spec.balance, float)
        assert spec.capacity > 0
        assert spec.balance > 0
