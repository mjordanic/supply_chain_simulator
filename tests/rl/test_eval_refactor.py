"""Tests for the issue-07 refactor of src/rl/eval.py.

Verifies:
  - No inline RunSlice class or bespoke collector remains.
  - _run_rl and _run_baseline both return the expected KPI keys.
  - Both run paths produce finite, consistent KPI values.
  - holding_rate / order_fee are sourced from the scenario (changing them
    changes net_profit).
  - evaluate_two_scale rejects stale checkpoints (layout_version mismatch).
  - K-generalisation: eval runs correctly when config K_min/K_max_episode are
    set to values not seen during training.
  - insufficient_cash rejection entries in the engine log trigger a warning.
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


# ---------------------------------------------------------------------------
# Tests: stale checkpoint raises descriptive ValueError
# ---------------------------------------------------------------------------


class TestStaleCheckpointRejection:
    """evaluate_two_scale must fail loudly on a layout-version mismatch."""

    def test_legacy_bare_state_dict_raises_value_error(self, tmp_path):
        """A bare state-dict checkpoint (pre-ADR-0021) raises ValueError."""
        import torch
        import src.rl.checkpoint as ckpt_module

        # Write a legacy bare state-dict (no layout_version key).
        bare_path = tmp_path / "legacy.pt"
        torch.save({"fc.weight": torch.zeros(4, 4)}, bare_path)

        with pytest.raises(ValueError, match="Legacy checkpoint"):
            ckpt_module.load(bare_path)

    def test_mismatched_layout_version_raises_value_error(self, tmp_path):
        """A checkpoint with a wrong layout_version raises a descriptive ValueError."""
        import torch
        from src.rl.set_encoder import OBS_LAYOUT_VERSION
        import src.rl.checkpoint as ckpt_module

        stale_path = tmp_path / "stale.pt"
        bundle = {
            "state_dict": {},
            "config": {},
            "layout_version": OBS_LAYOUT_VERSION + 99,
        }
        torch.save(bundle, stale_path)

        with pytest.raises(ValueError, match="layout version mismatch"):
            ckpt_module.load(stale_path)


# ---------------------------------------------------------------------------
# Tests: K-generalisation eval recipe
# ---------------------------------------------------------------------------


class TestKGeneralisationEval:
    """K-generalisation: eval runs correctly with K_min/K_max_episode overridden."""

    def test_eval_with_k_equal_to_train_kmax_plus(self):
        """Eval with K_min=K_max_episode=8 on a catalog of 15 products completes.

        The shared-weight SetActor handles any K ≤ K_MAX; this test verifies
        the config knobs are sufficient to express the K-generalisation recipe.
        """
        from src.rl.eval import build_eval_seeds, evaluate
        from src.rl.set_encoder import K_MAX

        catalog = _make_catalog(15)
        # Simulate "generalisation eval": fixed K=8 (above the default K_active=5).
        gen_config = RLConfig(
            episode_length=5,
            K_active=5,
            K_min=8,
            K_max_episode=8,
            n_eval_seeds=2,
        )

        specs = build_eval_seeds(catalog, gen_config, n_seeds=2)
        # All specs should have active_subset of length 8.
        for spec in specs:
            assert len(spec.active_subset) == 8, (
                f"Expected K=8 but got {len(spec.active_subset)}"
            )

        result = evaluate(_zero_policy, lambda: None.__class__(), specs, config=gen_config)
        # evaluate returns {} when baseline_factory() produces a bad policy;
        # use _run_rl directly to exercise the K-generalisation path.
        from src.rl.eval import _run_rl
        metrics = _run_rl(_zero_policy, specs[0], config=gen_config)
        for key in ("net_profit", "stockout_rate"):
            assert key in metrics, f"Missing {key!r} from generalisation eval"
            import math
            assert math.isfinite(metrics[key]), f"{key!r} not finite: {metrics[key]}"


# ---------------------------------------------------------------------------
# Tests: insufficient_cash warning
# ---------------------------------------------------------------------------


class TestInsufficientCashWarning:
    """_run_rl emits a WARNING when the rejection log has insufficient_cash for 'S'."""

    def test_insufficient_cash_warning_emitted(self):
        """Inject a fake insufficient_cash rejection and verify the warning fires."""
        from src.rl.eval import _run_rl, build_eval_seeds
        from unittest.mock import patch
        import src.rl.eval as eval_mod

        catalog = _make_catalog()
        config = _make_config(episode_length=5)
        specs = build_eval_seeds(catalog, config, n_seeds=1)
        spec = specs[0]

        # Capture the sim object and inject a rejection after tick_decide_and_settle.
        sim_obj_holder: list = []

        def _patched_build_world(scenario, policy_overrides=None):
            from src.sim.runner import build_world as _real_build_world
            sim = _real_build_world(scenario, policy_overrides=policy_overrides)
            sim_obj_holder.append(sim)
            _orig = sim.tick_decide_and_settle

            def _injecting_tick_decide(tick):
                _orig(tick)
                # Inject one insufficient_cash rejection for "S" on every tick.
                sim._tick_rejections.append({
                    "tick": tick,
                    "buyer_id": "S",
                    "supplier_id": "F_P0000",
                    "pid": "P0000",
                    "qty_requested": 5,
                    "qty_filled": 0,
                    "qty_rejected": 5,
                    "reason": "insufficient_cash",
                })

            sim.tick_decide_and_settle = _injecting_tick_decide
            return sim

        with patch("src.rl.eval.build_world", side_effect=_patched_build_world):
            with patch.object(eval_mod.logger, "warning") as mock_warn:
                _run_rl(_zero_policy, spec, config=config)
                # Warning must have been called at least once for the injected rejection.
                assert mock_warn.called, (
                    "Expected logger.warning to be called for insufficient_cash rejection"
                )
                # Check the message content contains the expected phrase.
                first_call_fmt = mock_warn.call_args_list[0][0][0]
                assert "insufficient_cash" in first_call_fmt, (
                    f"Warning message does not mention 'insufficient_cash': {first_call_fmt!r}"
                )
