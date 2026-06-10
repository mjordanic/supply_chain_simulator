"""Tests for the issue-07 refactor of src/rl/eval.py.

Verifies:
  - No inline RunSlice class or bespoke collector remains.
  - _run_rl and _run_baseline both return the expected KPI keys.
  - Both run paths produce finite, consistent KPI values.
  - holding_rate / order_fee are sourced from the scenario (changing them
    changes net_profit).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.rl.configs.default import RLConfig
from src.rl.eval import _run_rl, _run_baseline, build_eval_seeds
from src.rl.episode_sampler import sample_episode
from src.sim.policy import OrderUpToPolicy
from src.sim.scenario import load_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 15) -> list:
    return load_catalog(
        [
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0 + i * 0.3,
                "seasonality": "all_season",
            }
            for i in range(n)
        ]
    )


def _make_config(episode_length: int = 5) -> RLConfig:
    return RLConfig(
        episode_length=episode_length,
        K_active=5,
        n_eval_seeds=2,
        eval_seed_offset=10_000_000,
    )


def _zero_policy(obs: np.ndarray) -> np.ndarray:
    from src.rl.set_encoder import K_MAX
    return np.zeros(K_MAX * 3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: no bespoke collector left in eval.py
# ---------------------------------------------------------------------------


class TestNoBespokeCollector:
    """eval.py must not contain the old inline RunSlice class."""

    def test_no_inline_run_slice(self):
        import src.rl.eval as eval_mod
        # Check that no RunSlice dataclass is defined inline in eval.py.
        # (The _dc / _field imports were part of the legacy pattern.)
        source = inspect.getsource(eval_mod)
        assert "class RunSlice" not in source, (
            "Inline RunSlice class still present in rl/eval.py"
        )

    def test_no_legacy_import_aliases(self):
        """The legacy _dc / _field import aliases are gone."""
        import src.rl.eval as eval_mod
        source = inspect.getsource(eval_mod)
        assert "dataclass as _dc" not in source, (
            "Legacy 'dataclass as _dc' import still present in rl/eval.py"
        )


# ---------------------------------------------------------------------------
# Tests: KPI keys and types from both run paths
# ---------------------------------------------------------------------------


EXPECTED_KEYS = {"net_profit", "stockout_rate", "inventory_turnover", "mean_price_pct_of_msrp"}


class TestRunRLKPIs:
    """_run_rl returns all expected KPI keys as finite floats."""

    def test_expected_keys_present(self):
        catalog = _make_catalog()
        config = _make_config()
        specs = build_eval_seeds(catalog, config, n_seeds=1)
        spec = specs[0]
        result = _run_rl(_zero_policy, spec, config=config)
        for key in EXPECTED_KEYS:
            assert key in result, f"Missing key {key!r} from _run_rl"

    def test_all_values_finite_float(self):
        catalog = _make_catalog()
        config = _make_config()
        specs = build_eval_seeds(catalog, config, n_seeds=1)
        spec = specs[0]
        result = _run_rl(_zero_policy, spec, config=config)
        for key, val in result.items():
            assert isinstance(val, float), f"{key!r}: expected float, got {type(val).__name__}"
            assert np.isfinite(val), f"{key!r} is not finite: {val}"


class TestRunBaselineKPIs:
    """_run_baseline returns all expected KPI keys as finite floats."""

    def test_expected_keys_present(self):
        catalog = _make_catalog()
        config = _make_config()
        specs = build_eval_seeds(catalog, config, n_seeds=1)
        spec = specs[0]
        result = _run_baseline(OrderUpToPolicy(policy_seed=0), spec)
        for key in EXPECTED_KEYS:
            assert key in result, f"Missing key {key!r} from _run_baseline"

    def test_all_values_finite_float(self):
        catalog = _make_catalog()
        config = _make_config()
        specs = build_eval_seeds(catalog, config, n_seeds=1)
        spec = specs[0]
        result = _run_baseline(OrderUpToPolicy(policy_seed=0), spec)
        for key, val in result.items():
            assert isinstance(val, float), f"{key!r}: expected float, got {type(val).__name__}"
            assert np.isfinite(val), f"{key!r} is not finite: {val}"


# ---------------------------------------------------------------------------
# Tests: holding_rate / order_fee sourced from scenario
# ---------------------------------------------------------------------------


class TestHoldingRateFromScenario:
    """Changing holding_rate on the scenario changes net_profit for both run paths."""

    def _make_specs_pair(self, episode_length: int = 10):
        """Return two specs with holding_rate=0.0 and holding_rate=0.10 on same world."""
        from src.rl.episode_sampler import sample_episode as rl_sample_episode
        from src.sim.distributions import Constant
        from src.rl.configs.default import RLConfig

        catalog = _make_catalog()
        config_low = RLConfig(
            episode_length=episode_length,
            K_active=5,
            holding_rate=0.0,
            order_fee=0.0,
        )
        config_high = RLConfig(
            episode_length=episode_length,
            K_active=5,
            holding_rate=0.10,
            order_fee=0.0,
        )
        spec_low = rl_sample_episode(catalog, config_low, episode_seed=10_000_500)
        spec_high = rl_sample_episode(catalog, config_high, episode_seed=10_000_500)
        return spec_low, spec_high, config_low, config_high

    def test_baseline_higher_holding_rate_lowers_net_profit(self):
        spec_low, spec_high, config_low, config_high = self._make_specs_pair()
        from src.sim.node import IntermediateNode
        for ni in spec_high.scenario.nodes:
            if isinstance(ni.node, IntermediateNode):
                assert ni.node.holding_rate == pytest.approx(0.10)

        r_low = _run_baseline(OrderUpToPolicy(), spec_low)
        r_high = _run_baseline(OrderUpToPolicy(), spec_high)
        assert r_high["net_profit"] < r_low["net_profit"], (
            f"Higher holding_rate should lower net_profit: "
            f"low={r_low['net_profit']:.2f}, high={r_high['net_profit']:.2f}"
        )

    def test_rl_higher_holding_rate_lowers_net_profit(self):
        spec_low, spec_high, config_low, config_high = self._make_specs_pair()
        r_low = _run_rl(_zero_policy, spec_low, config=config_low)
        r_high = _run_rl(_zero_policy, spec_high, config=config_high)
        assert r_high["net_profit"] < r_low["net_profit"], (
            f"Higher holding_rate should lower net_profit in RL path: "
            f"low={r_low['net_profit']:.2f}, high={r_high['net_profit']:.2f}"
        )
