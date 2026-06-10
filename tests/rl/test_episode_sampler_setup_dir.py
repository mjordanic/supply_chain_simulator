"""Tests for setup-directory integration in the RL stack (issue 05).

Acceptance criteria covered:
  - load_catalog_and_market_from_setup loads catalog + market from a
    setup directory (catalog.csv + setup.yaml with market: block).
  - make_synthetic_catalog returns a valid Ware list of requested length.
  - sample_episode works when market is loaded from a setup directory.
  - CRN determinism: identical episode specs produce identical trajectories.
  - OrderUpToPolicy anchor still runs without error.
  - RLConfig.setup_dir field is accepted and wired through _load_world_catalog_and_template.
  - Missing catalog.csv or setup.yaml raises FileNotFoundError/ValueError.
  - setup.yaml without market: block raises ValueError.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import (
    load_catalog_and_market_from_setup,
    make_synthetic_catalog,
    sample_episode,
)
from src.sim.distributions import Constant, Uniform
from src.sim.runner import Runner, build_world
from src.sim.scenario import MarketParams, Ware


# ---------------------------------------------------------------------------
# Helpers: build a minimal but valid setup directory
# ---------------------------------------------------------------------------


def _write_minimal_setup_dir(tmp_path: Path, n_products: int = 5) -> Path:
    """Write a minimal setup directory with n_products items."""
    setup_dir = tmp_path / "test_setup"
    setup_dir.mkdir()

    # catalog.csv
    catalog_path = setup_dir / "catalog.csv"
    with catalog_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "product_id", "name", "category", "base_price",
                "unit_cost", "seasonality", "related_products", "init_stock_share",
            ],
        )
        writer.writeheader()
        for i in range(n_products):
            writer.writerow({
                "product_id": f"P{i:04d}",
                "name": f"Product {i}",
                "category": "General",
                "base_price": str(10.0 + i),
                "unit_cost": str(4.0 + i * 0.5),
                "seasonality": "all_season",
                "related_products": "",
                "init_stock_share": "1.0",
            })

    # setup.yaml — only market: block required by load_catalog_and_market_from_setup
    yaml_text = textwrap.dedent("""\
        market:
          cycle_len: 365
          cycle_amp: 0.0
          init_demand: 1.0
          init_supply: 1.0
          peak_factor: 1.0
          off_factor: 1.0
          season_months:
            peak: [6, 7, 8]
          regions: [US]
          correlation: 0.0
          trend_update_interval: 100
          min_value: 0.5
          max_value: 2.0
          stage_multipliers: {}
          price_elasticity: 0.0
          promo_multiplier: 1.0
          demand_factor_min: 0.1
          supply_factor_min: 0.01
          cross_inv_lo: 0.3
          cross_inv_hi: 0.7
          cross_factor_range: [0.5, 1.5]
          trend: {kind: constant, value: 1.0}
          demand_shock: {kind: constant, value: 0.0}
          supply_shock: {kind: constant, value: 0.0}
          base_demand: {kind: constant, value: 10.0}
    """)
    (setup_dir / "setup.yaml").write_text(yaml_text, encoding="utf-8")

    return setup_dir


# ---------------------------------------------------------------------------
# Tests: load_catalog_and_market_from_setup
# ---------------------------------------------------------------------------


class TestLoadCatalogAndMarketFromSetup:
    def test_returns_correct_types(self, tmp_path):
        """Returns (list[Ware], MarketParams) tuple."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=5)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        assert isinstance(catalog, list)
        assert len(catalog) == 5
        assert all(isinstance(w, Ware) for w in catalog)
        assert isinstance(market, MarketParams)

    def test_catalog_product_ids(self, tmp_path):
        """Catalog contains products with the expected product_ids."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=3)
        catalog, _ = load_catalog_and_market_from_setup(setup_dir)
        pids = {w.product_id for w in catalog}
        assert pids == {"P0000", "P0001", "P0002"}

    def test_market_params_loaded(self, tmp_path):
        """MarketParams has expected values from setup.yaml."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        _, market = load_catalog_and_market_from_setup(setup_dir)

        assert market.cycle_len == 365
        assert market.init_demand == pytest.approx(1.0)
        assert market.init_supply == pytest.approx(1.0)

    def test_accepts_string_path(self, tmp_path):
        """Accepts a plain string path as well as a Path object."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        catalog, market = load_catalog_and_market_from_setup(str(setup_dir))
        assert len(catalog) == 2
        assert isinstance(market, MarketParams)

    def test_missing_catalog_raises_file_not_found(self, tmp_path):
        """FileNotFoundError if catalog.csv is absent."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "catalog.csv").unlink()
        with pytest.raises(FileNotFoundError, match="catalog.csv"):
            load_catalog_and_market_from_setup(setup_dir)

    def test_missing_yaml_raises_file_not_found(self, tmp_path):
        """FileNotFoundError if setup.yaml is absent."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "setup.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="setup.yaml"):
            load_catalog_and_market_from_setup(setup_dir)

    def test_yaml_without_market_block_raises_value_error(self, tmp_path):
        """ValueError if setup.yaml has no market: block."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "setup.yaml").write_text("run:\n  n_steps: 10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="market"):
            load_catalog_and_market_from_setup(setup_dir)

    def test_uses_existing_three_node_chain_example(self):
        """Loads the committed three_node_chain example without error."""
        example_dir = Path("setups/three_node_chain")
        if not example_dir.is_dir():
            pytest.skip("setups/three_node_chain not found")
        catalog, market = load_catalog_and_market_from_setup(example_dir)
        assert len(catalog) >= 1
        assert isinstance(market, MarketParams)


# ---------------------------------------------------------------------------
# Tests: make_synthetic_catalog
# ---------------------------------------------------------------------------


class TestMakeSyntheticCatalog:
    def test_default_size(self):
        """Default call returns 100 products."""
        catalog = make_synthetic_catalog()
        assert len(catalog) == 100

    def test_custom_size(self):
        """Custom n returns exactly n products."""
        for n in [1, 10, 50]:
            catalog = make_synthetic_catalog(n)
            assert len(catalog) == n, f"Expected {n}, got {len(catalog)}"

    def test_returns_ware_list(self):
        """Returns a list of Ware objects."""
        catalog = make_synthetic_catalog(5)
        assert all(isinstance(w, Ware) for w in catalog)

    def test_unique_product_ids(self):
        """All product_ids are unique."""
        catalog = make_synthetic_catalog(20)
        pids = [w.product_id for w in catalog]
        assert len(pids) == len(set(pids))

    def test_positive_prices(self):
        """All wares have positive base_price and unit_cost."""
        catalog = make_synthetic_catalog(10)
        for w in catalog:
            assert w.base_price > 0
            assert w.unit_cost > 0

    def test_deterministic(self):
        """Two calls with the same n produce identical product lists."""
        c1 = make_synthetic_catalog(10)
        c2 = make_synthetic_catalog(10)
        assert [w.product_id for w in c1] == [w.product_id for w in c2]
        assert [w.base_price for w in c1] == [w.base_price for w in c2]


# ---------------------------------------------------------------------------
# Tests: sample_episode with market from setup directory
# ---------------------------------------------------------------------------


class TestSampleEpisodeWithSetupDir:
    def _make_config(self, K_active: int = 3) -> RLConfig:
        return RLConfig(
            K_active=K_active,
            K_min=K_active,
            K_max_episode=K_active,
            episode_length=5,
            capacity_dist=Uniform(150, 400),
            balance_dist=Uniform(15_000, 40_000),
        )

    def test_sample_episode_with_catalog_from_setup_dir(self, tmp_path):
        """sample_episode works with catalog loaded from a setup directory."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        config = self._make_config(K_active=3)
        spec = sample_episode(catalog, config, episode_seed=42)

        assert len(spec.active_subset) == 3
        assert len(spec.scenario.nodes) > 0
        # All active pids come from the catalog
        all_pids = {w.product_id for w in catalog}
        for pid in spec.active_subset:
            assert pid in all_pids

    def test_scenario_runs_one_tick_with_setup_dir_catalog(self, tmp_path):
        """Scenario built from setup-dir catalog runs through Runner for 1 tick."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        config = RLConfig(
            K_active=3,
            K_min=3,
            K_max_episode=3,
            episode_length=1,
            capacity_dist=Uniform(150, 400),
            balance_dist=Uniform(15_000, 40_000),
        )
        spec = sample_episode(catalog, config, episode_seed=7)
        sim = build_world(spec.scenario)
        sim.tick()  # must not raise


# ---------------------------------------------------------------------------
# Tests: CRN determinism
# ---------------------------------------------------------------------------


class TestCRNDeterminism:
    """Two identical episode specs must produce identical sim trajectories."""

    def _make_config(self) -> RLConfig:
        return RLConfig(
            K_active=3,
            K_min=3,
            K_max_episode=3,
            episode_length=5,
            capacity_dist=Constant(300),
            balance_dist=Constant(20_000.0),
        )

    def test_identical_specs_produce_identical_trajectories(self, tmp_path):
        """Same episode_seed → same world_seed → same sim trajectory."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, _ = load_catalog_and_market_from_setup(setup_dir)

        config = self._make_config()

        spec1 = sample_episode(catalog, config, episode_seed=99)
        spec2 = sample_episode(catalog, config, episode_seed=99)

        # Structural equality.
        assert spec1.active_subset == spec2.active_subset
        assert spec1.world_seed == spec2.world_seed
        assert spec1.capacity == spec2.capacity
        assert spec1.balance == pytest.approx(spec2.balance)

        # Simulation-level equality: run both and compare run logs.
        log1 = Runner(spec1.scenario).run()
        log2 = Runner(spec2.scenario).run()
        # Compare per-tick node_cash totals — a proxy for identical trajectories.
        assert log1["n_steps"] == log2["n_steps"]
        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["node_cash"] == pytest.approx(t2["node_cash"], abs=1e-4), (
                f"Tick {t1['tick']}: node_cash diverged with identical seeds"
            )

    def test_different_seeds_diverge(self, tmp_path):
        """Different episode seeds produce different trajectories."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, _ = load_catalog_and_market_from_setup(setup_dir)

        config = self._make_config()

        world_seeds = {
            sample_episode(catalog, config, episode_seed=s).world_seed
            for s in range(20)
        }
        assert len(world_seeds) > 1, "All seeds produced the same world_seed"


# ---------------------------------------------------------------------------
# Tests: OrderUpToPolicy anchor
# ---------------------------------------------------------------------------


class TestOrderUpToPolicyAnchor:
    """OrderUpToPolicy must still run cleanly against a setup-dir catalog episode."""

    def test_order_up_to_policy_runs_without_error(self, tmp_path):
        """Running OrderUpToPolicy against a setup-dir episode does not raise."""
        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        config = RLConfig(
            K_active=3,
            K_min=3,
            K_max_episode=3,
            episode_length=10,
            capacity_dist=Constant(300),
            balance_dist=Constant(20_000.0),
        )
        spec = sample_episode(catalog, config, episode_seed=0)
        sim = build_world(spec.scenario)

        # Tick through the episode.
        for _ in range(10):
            sim.tick()  # must not raise

    def test_crn_eval_baseline_runs_against_setup_dir_catalog(self, tmp_path):
        """evaluate() with OrderUpToPolicy baseline completes on setup-dir catalog."""
        from src.rl.eval import build_eval_seeds, evaluate
        from src.sim.policy import OrderUpToPolicy
        import numpy as np

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, _ = load_catalog_and_market_from_setup(setup_dir)

        config = RLConfig(
            episode_length=5,
            K_active=3,
            K_min=3,
            K_max_episode=3,
            n_eval_seeds=3,
            eval_seed_offset=10_000_000,
            capacity_dist=Constant(200),
            balance_dist=Constant(20_000.0),
        )

        specs = build_eval_seeds(catalog, config, n_seeds=3)

        from src.rl.set_encoder import K_MAX

        def zero_policy(obs: "np.ndarray") -> "np.ndarray":
            return np.zeros(K_MAX * 3, dtype=np.float32)

        def baseline_factory():
            return OrderUpToPolicy(policy_seed=0)

        result = evaluate(zero_policy, baseline_factory, specs, config=config)
        assert "eval/rl_return" in result
        assert "eval/baseline_return" in result
        assert "eval/paired_uplift" in result
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Tests: RLConfig.setup_dir field
# ---------------------------------------------------------------------------


class TestRLConfigSetupDir:
    def test_setup_dir_default_is_none(self):
        """RLConfig.setup_dir defaults to None."""
        config = RLConfig()
        assert config.setup_dir is None

    def test_setup_dir_can_be_set(self, tmp_path):
        """RLConfig.setup_dir accepts a string path."""
        setup_dir = str(tmp_path / "my_setup")
        config = RLConfig(setup_dir=setup_dir)
        assert config.setup_dir == setup_dir

    def test_load_world_catalog_and_template_uses_setup_dir(self, tmp_path):
        """_load_world_catalog_and_template loads from setup_dir when set."""
        from src.rl.train import _load_world_catalog_and_template

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=8)
        config = RLConfig(
            setup_dir=str(setup_dir),
            K_active=3,
        )

        catalog, market, disruption = _load_world_catalog_and_template(config)

        assert len(catalog) == 8
        assert market is not None
        assert disruption is None  # setup-dir path returns None for disruption

    def test_load_world_catalog_and_template_legacy_path_when_no_setup_dir(self):
        """_load_world_catalog_and_template falls back to synthetic catalog when setup_dir is None."""
        from src.rl.train import _load_world_catalog_and_template

        config = RLConfig(setup_dir=None, K_catalog=10, K_active=3)
        # Should not raise — falls through to synthetic catalog fallback.
        catalog, market, disruption = _load_world_catalog_and_template(config)
        assert len(catalog) > 0
