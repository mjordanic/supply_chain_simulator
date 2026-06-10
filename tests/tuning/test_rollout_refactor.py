"""Tests for the issue-07 refactor of src/tuning/rollout.py.

Verifies:
  - run_policy_episode returns all expected KPI keys.
  - No bespoke collector (RunSlice, _TrackingDemandSinkNode) remains.
  - holding_rate / order_fee are sourced from the scenario's IntermediateNode,
    not from a hardcoded default in rollout.py.
  - KPI values are value-preserving across two identical calls (CRN).
"""

from __future__ import annotations

import inspect

import pytest

from src.sim.distributions import Constant
from src.sim.policy import OrderUpToPolicy
from src.sim.scenario import load_catalog
from src.tuning.config import TuningConfig
from src.tuning.episode import sample_episode
from src.tuning.rollout import run_policy_episode


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


def _make_config(episode_length: int = 5) -> TuningConfig:
    return TuningConfig(
        episode_length=episode_length,
        K_active=5,
        n_trials=1,
        n_search_seeds=2,
        init_stock_pct_dist=Constant(0.5),
    )


def _make_spec(seed: int = 12_000_000, episode_length: int = 5):
    catalog = _make_catalog()
    config = _make_config(episode_length=episode_length)
    return sample_episode(catalog, config, episode_seed=seed)


# ---------------------------------------------------------------------------
# Tests: no bespoke collector left in rollout.py
# ---------------------------------------------------------------------------


class TestNoBespokeCollector:
    """rollout.py must not contain the old bespoke collector code."""

    def test_no_run_slice_class(self):
        import src.tuning.rollout as rollout_mod
        source = inspect.getsource(rollout_mod)
        assert "RunSlice" not in source, "RunSlice still present in rollout.py"

    def test_no_tracking_sink_node(self):
        import src.tuning.rollout as rollout_mod
        source = inspect.getsource(rollout_mod)
        assert "_TrackingDemandSinkNode" not in source, (
            "_TrackingDemandSinkNode still present in rollout.py"
        )

    def test_no_record_active_subset(self):
        import src.tuning.rollout as rollout_mod
        source = inspect.getsource(rollout_mod)
        assert "_record_active_subset" not in source, (
            "_record_active_subset still present in rollout.py"
        )


# ---------------------------------------------------------------------------
# Tests: KPI keys and types
# ---------------------------------------------------------------------------


class TestRunPolicyEpisodeKPIs:
    """run_policy_episode returns all expected KPI keys as floats."""

    EXPECTED_KEYS = {
        "service_level",
        "stockout_rate",
        "inventory_turnover",
        "mean_price_pct_of_msrp",
        "revenue",
        "net_profit",
    }

    def test_all_expected_keys_present(self):
        spec = _make_spec()
        result = run_policy_episode(OrderUpToPolicy(), spec)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key {key!r}"

    def test_all_values_are_floats(self):
        spec = _make_spec()
        result = run_policy_episode(OrderUpToPolicy(), spec)
        for key, val in result.items():
            assert isinstance(val, float), (
                f"Value for {key!r} is {type(val).__name__}, expected float"
            )


# ---------------------------------------------------------------------------
# Tests: CRN determinism
# ---------------------------------------------------------------------------


class TestCRNDeterminism:
    """Two identical calls yield bit-identical KPIs."""

    def test_identical_calls_produce_identical_results(self):
        spec = _make_spec()
        r1 = run_policy_episode(OrderUpToPolicy(policy_seed=0), spec)
        r2 = run_policy_episode(OrderUpToPolicy(policy_seed=0), spec)
        assert r1 == r2, f"CRN diverged: {r1} vs {r2}"


# ---------------------------------------------------------------------------
# Tests: holding_rate / order_fee sourced from scenario
# ---------------------------------------------------------------------------


class TestHoldingRateFromScenario:
    """Changing holding_rate/order_fee on the scenario changes net_profit."""

    def test_higher_holding_rate_lowers_net_profit(self):
        """A scenario with a higher holding_rate yields lower net_profit."""
        catalog = _make_catalog()
        config_low = TuningConfig(
            episode_length=10,
            K_active=5,
            n_trials=1,
            n_search_seeds=2,
            init_stock_pct_dist=Constant(0.5),
            holding_rate=0.0,   # no holding cost
            order_fee=0.0,
        )
        config_high = TuningConfig(
            episode_length=10,
            K_active=5,
            n_trials=1,
            n_search_seeds=2,
            init_stock_pct_dist=Constant(0.5),
            holding_rate=0.10,  # high holding cost
            order_fee=0.0,
        )
        spec_low = sample_episode(catalog, config_low, episode_seed=12_000_100)
        spec_high = sample_episode(catalog, config_high, episode_seed=12_000_100)

        # Verify both scenarios have the same world_seed (same episode) but
        # different holding_rate on the intermediate node.
        from src.sim.node import IntermediateNode
        for ni in spec_low.scenario.nodes:
            if isinstance(ni.node, IntermediateNode):
                assert ni.node.holding_rate == pytest.approx(0.0)
        for ni in spec_high.scenario.nodes:
            if isinstance(ni.node, IntermediateNode):
                assert ni.node.holding_rate == pytest.approx(0.10)

        r_low = run_policy_episode(OrderUpToPolicy(), spec_low)
        r_high = run_policy_episode(OrderUpToPolicy(), spec_high)

        # Higher holding_rate incurs more cost → lower net_profit.
        assert r_high["net_profit"] < r_low["net_profit"], (
            f"Expected higher holding_rate to lower net_profit: "
            f"low={r_low['net_profit']:.2f}, high={r_high['net_profit']:.2f}"
        )
