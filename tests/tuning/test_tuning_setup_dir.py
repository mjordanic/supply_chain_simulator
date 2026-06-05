"""Tests for setup-directory integration in the tuning stack (issue 06).

Acceptance criteria covered:
  - TuningConfig.setup_dir field is accepted and defaults to None.
  - load_catalog_and_market_from_setup loads catalog + market from a
    setup directory (catalog.csv + setup.yaml with market: block).
  - make_synthetic_catalog returns a valid Ware list of requested length.
  - sample_episode builds a graph-mode scenario (NodeInstance/EdgeSpec terms).
  - Per-trial CRN determinism: same spec → identical run trajectories.
  - evaluate_policy_normalised works on graph-mode TuningEpisodeSpec.
  - TuningEpisodeSpec carries capacity and balance fields.
  - Missing catalog.csv or setup.yaml raises FileNotFoundError/ValueError.
  - setup.yaml without market: block raises ValueError.
  - OrderUpToPolicy anchor runs against a setup-dir graph episode.
  - study.py _build_eval_specs uses setup_dir when TuningConfig.setup_dir set.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from src.tuning.config import TuningConfig


# ---------------------------------------------------------------------------
# Helpers: build a minimal but valid setup directory
# ---------------------------------------------------------------------------


def _write_minimal_setup_dir(tmp_path: Path, n_products: int = 5) -> Path:
    """Write a minimal setup directory with n_products items."""
    setup_dir = tmp_path / "test_setup"
    setup_dir.mkdir(parents=True, exist_ok=True)

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

    # setup.yaml — only market: block required
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
# Tests: TuningConfig.setup_dir field
# ---------------------------------------------------------------------------


class TestTuningConfigSetupDir:
    def test_setup_dir_default_is_none(self):
        """TuningConfig.setup_dir defaults to None."""
        config = TuningConfig()
        assert config.setup_dir is None

    def test_setup_dir_can_be_set(self, tmp_path):
        """TuningConfig.setup_dir accepts a string path."""
        setup_dir = str(tmp_path / "my_setup")
        config = TuningConfig(setup_dir=setup_dir)
        assert config.setup_dir == setup_dir

    def test_setup_dir_is_frozen(self):
        """TuningConfig is frozen — setup_dir cannot be mutated."""
        config = TuningConfig(setup_dir="/some/path")
        with pytest.raises((TypeError, AttributeError)):
            config.setup_dir = "/other/path"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: load_catalog_and_market_from_setup (tuning module)
# ---------------------------------------------------------------------------


class TestLoadCatalogAndMarketFromSetup:
    def test_returns_correct_types(self, tmp_path):
        """Returns (list[Ware], MarketParams) tuple."""
        from src.tuning.episode import load_catalog_and_market_from_setup
        from src.sim.scenario import MarketParams, Ware

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=5)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        assert isinstance(catalog, list)
        assert len(catalog) == 5
        assert all(isinstance(w, Ware) for w in catalog)
        assert isinstance(market, MarketParams)

    def test_catalog_product_ids(self, tmp_path):
        """Catalog contains products with the expected product_ids."""
        from src.tuning.episode import load_catalog_and_market_from_setup

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=3)
        catalog, _ = load_catalog_and_market_from_setup(setup_dir)
        pids = {w.product_id for w in catalog}
        assert pids == {"P0000", "P0001", "P0002"}

    def test_accepts_string_path(self, tmp_path):
        """Accepts a plain string path as well as a Path object."""
        from src.tuning.episode import load_catalog_and_market_from_setup

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        catalog, market = load_catalog_and_market_from_setup(str(setup_dir))
        assert len(catalog) == 2

    def test_missing_catalog_raises_file_not_found(self, tmp_path):
        """FileNotFoundError if catalog.csv is absent."""
        from src.tuning.episode import load_catalog_and_market_from_setup

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "catalog.csv").unlink()
        with pytest.raises(FileNotFoundError, match="catalog.csv"):
            load_catalog_and_market_from_setup(setup_dir)

    def test_missing_yaml_raises_file_not_found(self, tmp_path):
        """FileNotFoundError if setup.yaml is absent."""
        from src.tuning.episode import load_catalog_and_market_from_setup

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "setup.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="setup.yaml"):
            load_catalog_and_market_from_setup(setup_dir)

    def test_yaml_without_market_block_raises_value_error(self, tmp_path):
        """ValueError if setup.yaml has no market: block."""
        from src.tuning.episode import load_catalog_and_market_from_setup

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=2)
        (setup_dir / "setup.yaml").write_text("run:\n  n_steps: 10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="market"):
            load_catalog_and_market_from_setup(setup_dir)


# ---------------------------------------------------------------------------
# Tests: make_synthetic_catalog (tuning module)
# ---------------------------------------------------------------------------


class TestMakeSyntheticCatalog:
    def test_default_size(self):
        """Default call returns 100 products."""
        from src.tuning.episode import make_synthetic_catalog

        catalog = make_synthetic_catalog()
        assert len(catalog) == 100

    def test_custom_size(self):
        """Custom n returns exactly n products."""
        from src.tuning.episode import make_synthetic_catalog

        for n in [1, 10, 50]:
            catalog = make_synthetic_catalog(n)
            assert len(catalog) == n

    def test_returns_ware_list(self):
        """Returns a list of Ware objects."""
        from src.tuning.episode import make_synthetic_catalog
        from src.sim.scenario import Ware

        catalog = make_synthetic_catalog(5)
        assert all(isinstance(w, Ware) for w in catalog)

    def test_deterministic(self):
        """Two calls with the same n produce identical product lists."""
        from src.tuning.episode import make_synthetic_catalog

        c1 = make_synthetic_catalog(10)
        c2 = make_synthetic_catalog(10)
        assert [w.product_id for w in c1] == [w.product_id for w in c2]


# ---------------------------------------------------------------------------
# Tests: TuningEpisodeSpec graph-mode
# ---------------------------------------------------------------------------


class TestTuningEpisodeSpecGraphMode:
    """sample_episode now builds a graph-mode Scenario in NodeInstance/EdgeSpec terms."""

    def _make_catalog(self, n: int = 10):
        from src.sim.scenario import load_catalog
        return load_catalog([
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i * 0.3),
                "seasonality": "all_season",
            }
            for i in range(n)
        ])

    def _make_config(self, K_active: int = 3, episode_length: int = 5) -> TuningConfig:
        from src.sim.distributions import Constant
        return TuningConfig(
            K_active=K_active,
            episode_length=episode_length,
            n_trials=1,
            n_search_seeds=2,
        )

    def test_spec_scenario_is_graph_mode(self, tmp_path):
        """sample_episode returns a spec whose scenario is a graph (has nodes+edges)."""
        from src.tuning.episode import sample_episode

        catalog = self._make_catalog(10)
        config = self._make_config(K_active=3)

        spec = sample_episode(catalog, config, episode_seed=42)
        # Graph-mode scenario has node_instances.
        assert spec.scenario.nodes is not None
        assert len(spec.scenario.nodes) > 0

    def test_spec_carries_capacity_and_balance(self, tmp_path):
        """TuningEpisodeSpec has capacity and balance fields (not buried in stores[0])."""
        from src.tuning.episode import sample_episode

        catalog = self._make_catalog(10)
        config = self._make_config(K_active=3)

        spec = sample_episode(catalog, config, episode_seed=42)
        assert hasattr(spec, "capacity")
        assert hasattr(spec, "balance")
        assert spec.capacity > 0
        assert spec.balance > 0

    def test_spec_active_subset_comes_from_catalog(self):
        """active_subset pids all come from the catalog."""
        from src.tuning.episode import sample_episode

        catalog = self._make_catalog(10)
        config = self._make_config(K_active=3)

        spec = sample_episode(catalog, config, episode_seed=42)
        all_pids = {w.product_id for w in catalog}
        for pid in spec.active_subset:
            assert pid in all_pids

    def test_scenario_runs_one_tick(self):
        """Graph scenario built by sample_episode runs through Runner for 1 tick."""
        from src.tuning.episode import sample_episode
        from src.sim.runner import build_world

        catalog = self._make_catalog(10)
        config = TuningConfig(K_active=3, episode_length=1, n_trials=1, n_search_seeds=1)

        spec = sample_episode(catalog, config, episode_seed=7)
        sim = build_world(spec.scenario)
        sim.tick()  # must not raise


# ---------------------------------------------------------------------------
# Tests: CRN determinism for tuning
# ---------------------------------------------------------------------------


class TestTuningCRNDeterminism:
    """Two identical episode specs produce identical run trajectories."""

    def test_identical_specs_produce_identical_trajectories(self, tmp_path):
        """Same episode_seed → same world_seed → same sim trajectory."""
        from src.tuning.episode import sample_episode
        from src.sim.runner import Runner

        catalog_raw = [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i * 0.3),
                "seasonality": "all_season",
            }
            for i in range(10)
        ]
        from src.sim.scenario import load_catalog
        catalog = load_catalog(catalog_raw)

        config = TuningConfig(
            K_active=3,
            episode_length=5,
            n_trials=1,
            n_search_seeds=2,
        )

        spec1 = sample_episode(catalog, config, episode_seed=99)
        spec2 = sample_episode(catalog, config, episode_seed=99)

        assert spec1.active_subset == spec2.active_subset
        assert spec1.capacity == spec2.capacity
        assert spec1.balance == pytest.approx(spec2.balance)

        log1 = Runner(spec1.scenario).run()
        log2 = Runner(spec2.scenario).run()

        assert log1["n_steps"] == log2["n_steps"]
        for t1, t2 in zip(log1["ticks"], log2["ticks"]):
            assert t1["node_cash"] == pytest.approx(t2["node_cash"], abs=1e-4)

    def test_different_seeds_diverge(self):
        """Different episode seeds produce different capacity/balance draws."""
        from src.tuning.episode import sample_episode
        from src.sim.scenario import load_catalog

        catalog = load_catalog([
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i * 0.3),
                "seasonality": "all_season",
            }
            for i in range(10)
        ])

        config = TuningConfig(K_active=3, episode_length=5, n_trials=1, n_search_seeds=2)

        world_seeds = {
            sample_episode(catalog, config, episode_seed=s).scenario.world_seed
            for s in range(20)
        }
        assert len(world_seeds) > 1


# ---------------------------------------------------------------------------
# Tests: evaluate_policy_normalised with graph-mode specs
# ---------------------------------------------------------------------------


class TestEvaluatePolicyGraphMode:
    """evaluate_policy_normalised works with graph-mode TuningEpisodeSpec."""

    def _make_catalog(self, n: int = 15):
        from src.sim.scenario import load_catalog
        return load_catalog([
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i * 0.3),
                "seasonality": "all_season",
            }
            for i in range(n)
        ])

    def _make_specs(self, n: int = 2):
        from src.tuning.episode import sample_episode
        catalog = self._make_catalog()
        config = TuningConfig(
            K_active=5,
            episode_length=5,
            n_trials=1,
            n_search_seeds=2,
        )
        return [
            sample_episode(catalog, config, episode_seed=12_000_000 + i)
            for i in range(n)
        ]

    def test_evaluate_returns_expected_keys(self):
        """evaluate_policy_normalised returns all expected KPI keys."""
        from src.sim.policy import OrderUpToPolicy
        from src.tuning.evaluator import evaluate_policy_normalised

        specs = self._make_specs(2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)

        expected_keys = {
            "mean_normalised_return",
            "per_seed_net_profit",
            "per_seed_initial_cash",
            "per_seed_capacity",
            "per_seed_service_level",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key!r}"

    def test_evaluate_crn_determinism(self):
        """Same specs + same policy → bit-identical results."""
        from src.sim.policy import OrderUpToPolicy
        from src.tuning.evaluator import evaluate_policy_normalised

        specs = self._make_specs(2)
        result1 = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)
        result2 = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)

        assert result1["mean_normalised_return"] == result2["mean_normalised_return"]
        assert result1["per_seed_net_profit"] == result2["per_seed_net_profit"]

    def test_capacity_and_initial_cash_are_positive(self):
        """per_seed_capacity and per_seed_initial_cash are positive."""
        from src.sim.policy import OrderUpToPolicy
        from src.tuning.evaluator import evaluate_policy_normalised

        specs = self._make_specs(2)
        result = evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs)

        for cap in result["per_seed_capacity"]:
            assert cap > 0
        for cash in result["per_seed_initial_cash"]:
            assert cash > 0


# ---------------------------------------------------------------------------
# Tests: OrderUpToPolicy anchor against graph-mode setup-dir catalog
# ---------------------------------------------------------------------------


class TestOrderUpToPolicyAnchorGraphMode:
    """OrderUpToPolicy runs cleanly on graph-mode tuning episodes."""

    def test_order_up_to_policy_runs_without_error(self, tmp_path):
        """run_policy_episode with OrderUpToPolicy does not raise on graph spec."""
        from src.sim.policy import OrderUpToPolicy
        from src.tuning.episode import sample_episode
        from src.tuning.rollout import run_policy_episode

        from src.sim.scenario import load_catalog
        catalog = load_catalog([
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(10 + i),
                "unit_cost": float(4 + i * 0.3),
                "seasonality": "all_season",
            }
            for i in range(10)
        ])

        config = TuningConfig(K_active=3, episode_length=10, n_trials=1, n_search_seeds=1)

        spec = sample_episode(catalog, config, episode_seed=0)
        metrics = run_policy_episode(OrderUpToPolicy(), spec)
        assert isinstance(metrics, dict)
        assert "net_profit" in metrics

    def test_order_up_to_policy_runs_on_setup_dir_catalog(self, tmp_path):
        """run_policy_episode on setup-dir catalog does not raise."""
        from src.sim.policy import OrderUpToPolicy
        from src.tuning.episode import load_catalog_and_market_from_setup, sample_episode
        from src.tuning.rollout import run_policy_episode

        setup_dir = _write_minimal_setup_dir(tmp_path, n_products=10)
        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        config = TuningConfig(K_active=3, episode_length=5, n_trials=1, n_search_seeds=1)

        spec = sample_episode(catalog, config, episode_seed=0, market_params=market)
        metrics = run_policy_episode(OrderUpToPolicy(), spec)
        assert isinstance(metrics, dict)
        assert "net_profit" in metrics


# ---------------------------------------------------------------------------
# Tests: _build_eval_specs in study.py uses setup_dir when set
# ---------------------------------------------------------------------------


class TestBuildEvalSpecsWithSetupDir:
    """_build_eval_specs (used by run_study) picks up catalog from setup_dir."""

    def test_run_study_with_setup_dir_config(self, tmp_path):
        """run_study smoke-runs when TuningConfig.setup_dir is set."""
        from src.tuning.study import run_study
        from src.tuning.search_spaces import order_up_to_space

        setup_dir = _write_minimal_setup_dir(tmp_path / "world", n_products=10)

        config = TuningConfig(
            n_trials=1,
            n_search_seeds=1,
            n_holdout_seeds=1,
            episode_length=5,
            K_active=3,
            setup_dir=str(setup_dir),
        )

        # Note: run_study loads catalog from setup_dir when config.setup_dir is set.
        # Here we test that run_study also works when catalog is passed explicitly
        # and market_params come from a setup dir loaded externally.
        from src.tuning.episode import load_catalog_and_market_from_setup

        catalog, market = load_catalog_and_market_from_setup(setup_dir)

        study = run_study(
            order_up_to_space,
            catalog=catalog,
            tuning_config=config,
            study_name="setup_dir_smoke",
            output_dir=str(tmp_path / "out"),
            market_params=market,
        )
        assert study is not None
        assert len(study.trials) == 1

    def test_cli_accepts_setup_dir_flag(self, tmp_path):
        """The CLI (src.tuning.study) accepts --setup-dir and runs without error."""
        import subprocess
        import sys

        setup_dir = _write_minimal_setup_dir(tmp_path / "world", n_products=10)
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "src.tuning.study",
                "--policy", "order_up_to",
                "--trials", "1",
                "--study-name", "cli_setup_dir",
                "--n-search-seeds", "1",
                "--n-holdout-seeds", "1",
                "--top-k", "1",
                "--episode-length", "5",
                "--setup-dir", str(setup_dir),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        study_dir = out_dir / "cli_setup_dir"
        assert (study_dir / "trials.parquet").exists()
        assert (study_dir / "study.json").exists()
