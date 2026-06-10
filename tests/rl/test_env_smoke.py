"""Smoke tests for src/rl/env.py (RLEnv) on the graph engine.

Acceptance criteria:
  - reset() returns an observation matching observation_space.shape
  - One full episode (episode_length step() calls) runs end-to-end without raising
  - terminated becomes True exactly at step episode_length
  - Same reset(seed=s) from two RLEnv instances produces identical first-observation tensors
  - Reward across one episode is a finite float each tick
  - RLIntermediatePolicy.decide() raises RuntimeError if called without set_pending_action
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.sim.policy import RLIntermediatePolicy
from src.sim.scenario import load_catalog


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 20) -> list:
    """Build a minimal n-product catalog with stable P{i:04d} ids."""
    items = [
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
    return load_catalog(items)


def _make_config(episode_length: int = 10) -> RLConfig:
    """Short episode for fast tests."""
    from src.sim.distributions import Uniform
    return RLConfig(
        episode_length=episode_length,
        K_active=5,
        capacity_dist=Uniform(150, 400),
        balance_dist=Uniform(15_000, 40_000),
    )


def _make_env(episode_length: int = 10) -> RLEnv:
    return RLEnv(
        catalog=_make_catalog(20),
        config=_make_config(episode_length),
    )


# ---------------------------------------------------------------------------
# RLIntermediatePolicy unit tests
# ---------------------------------------------------------------------------


class TestRLIntermediatePolicy:
    def test_set_then_decide_returns_action(self):
        policy = RLIntermediatePolicy()
        action_dict = {
            "order": {"P0000": [("F_P0000", 10)]},
            "list_price": {"P0000": 12.0},
            "min_order_imposed": {"P0000": 0},
        }
        policy.set_pending_action(action_dict)
        result = policy.decide({}, None)
        assert result is action_dict

    def test_decide_without_set_raises(self):
        policy = RLIntermediatePolicy()
        with pytest.raises(RuntimeError, match="set_pending_action"):
            policy.decide({}, None)

    def test_decide_clears_pending(self):
        """Calling decide() twice in a row raises on the second call."""
        policy = RLIntermediatePolicy()
        policy.set_pending_action({
            "order": {}, "list_price": {}, "min_order_imposed": {}
        })
        policy.decide({}, None)
        with pytest.raises(RuntimeError):
            policy.decide({}, None)

    def test_does_not_consume_policy_rng(self):
        """RLIntermediatePolicy.policy_rng state is unchanged after set/decide cycle."""
        policy = RLIntermediatePolicy()
        rng = policy.policy_rng
        state_before = rng.getstate()
        policy.set_pending_action({
            "order": {}, "list_price": {}, "min_order_imposed": {}
        })
        policy.decide({}, None)
        state_after = rng.getstate()
        assert state_before == state_after, "RLIntermediatePolicy.decide() must not draw from policy_rng"


# ---------------------------------------------------------------------------
# RLEnv — observation space
# ---------------------------------------------------------------------------


class TestObservationSpace:
    def test_reset_returns_obs_matching_space(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape, (
            f"obs.shape={obs.shape}, expected {env.observation_space.shape}"
        )
        assert obs.dtype == np.float32

    def test_step_returns_obs_matching_space(self):
        env = _make_env()
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_observation_space_shape_matches_config(self):
        from src.rl.set_encoder import K_MAX, F
        config = _make_config()
        env = RLEnv(catalog=_make_catalog(20), config=config)
        assert env.observation_space.shape == (K_MAX * F,), (
            f"Expected obs shape ({K_MAX * F},), got {env.observation_space.shape}"
        )
        assert env.action_space.shape == (K_MAX * 3,), (
            f"Expected action shape ({K_MAX * 3},), got {env.action_space.shape}"
        )

    def test_action_space_bounds(self):
        env = _make_env()
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RLEnv — full episode run
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_one_full_episode_no_raise(self):
        """30 step() calls complete without raising."""
        from src.sim.distributions import Uniform
        env = RLEnv(
            catalog=_make_catalog(20),
            config=RLConfig(episode_length=30, K_active=5,
                            capacity_dist=Uniform(150, 400),
                            balance_dist=Uniform(15_000, 40_000)),
        )
        env.reset(seed=7)
        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True

    def test_short_episode_terminated_at_episode_length(self):
        """terminated becomes True exactly at step episode_length."""
        episode_length = 15
        env = RLEnv(
            catalog=_make_catalog(20),
            config=_make_config(episode_length=episode_length),
        )
        env.reset(seed=99)
        for step_i in range(episode_length):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if step_i < episode_length - 1:
                assert not terminated, f"terminated early at step {step_i + 1}"
            else:
                assert terminated, "terminated must be True at last step"

    def test_truncated_never_set(self):
        """truncated is always False."""
        env = _make_env(episode_length=5)
        env.reset(seed=10)
        for _ in range(5):
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            assert truncated is False

    def test_step_before_reset_raises(self):
        """Calling step() before reset() raises RuntimeError."""
        env = _make_env()
        with pytest.raises(RuntimeError):
            env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# RLEnv — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_two_envs_identical_first_obs(self):
        """Two RLEnv instances reset with the same seed produce identical obs."""
        catalog = _make_catalog(20)
        config = _make_config(episode_length=10)

        env1 = RLEnv(catalog=catalog, config=config)
        env2 = RLEnv(catalog=catalog, config=config)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        np.testing.assert_array_equal(obs1, obs2, err_msg="Observations from same seed differ")

    def test_different_seeds_different_obs(self):
        """Different seeds should (with overwhelming probability) differ."""
        env = _make_env()
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        assert not np.allclose(obs_a, obs_b), "Different seeds produced identical observations"

    def test_full_episode_trajectory_determinism(self):
        """Two envs with same seed and same actions produce identical observations."""
        catalog = _make_catalog(20)
        config = _make_config(episode_length=5)

        env1 = RLEnv(catalog=catalog, config=config)
        env2 = RLEnv(catalog=catalog, config=config)

        from random import Random
        from src.rl.set_encoder import K_MAX
        rng = Random(555)
        action_sequence = [
            np.array([rng.uniform(-1, 1) for _ in range(K_MAX * 3)], dtype=np.float32)
            for _ in range(config.episode_length)
        ]

        env1.reset(seed=555)
        env2.reset(seed=555)

        for action in action_sequence:
            obs1, r1, t1, tr1, info1 = env1.step(action)
            obs2, r2, t2, tr2, info2 = env2.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == pytest.approx(r2, abs=1e-6)
            assert t1 == t2


# ---------------------------------------------------------------------------
# RLEnv — reward accounting
# ---------------------------------------------------------------------------


class TestRewardAccounting:
    def test_reward_is_scalar(self):
        """Reward returned by step() is a plain Python float, not an array."""
        env = _make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_rewards_are_finite(self):
        """All rewards across an episode are finite."""
        env = _make_env(episode_length=10)
        env.reset(seed=42)
        for _ in range(10):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_info_contains_expected_keys(self):
        """info dict has step, cash, active_products, inventory."""
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "step" in info
        assert "cash" in info
        assert "active_products" in info
        assert "inventory" in info


# ---------------------------------------------------------------------------
# RLEnv — multiple reset cycles
# ---------------------------------------------------------------------------


class TestMultipleResets:
    def test_env_reusable_across_resets(self):
        """Resetting and running multiple episodes in sequence does not raise."""
        env = _make_env(episode_length=5)
        for seed in [1, 2, 3]:
            env.reset(seed=seed)
            for _ in range(5):
                env.step(env.action_space.sample())

    def test_reset_restores_step_count(self):
        """After reset, _step_count is 0."""
        env = _make_env(episode_length=5)
        env.reset(seed=10)
        for _ in range(5):
            env.step(env.action_space.sample())
        assert env._step_count == 5
        env.reset(seed=11)
        assert env._step_count == 0


# ---------------------------------------------------------------------------
# RLEnv — graph-engine specific
# ---------------------------------------------------------------------------


class TestGraphEngineIntegration:
    def test_node_s_is_present_after_reset(self):
        """After reset, simulation has an IntermediateNode with id 'S'."""
        env = _make_env()
        env.reset(seed=42)
        assert env._sim is not None
        assert "S" in env._sim.nodes

    def test_rl_policy_is_attached_to_node_s(self):
        """RLIntermediatePolicy is attached to node S after reset."""
        env = _make_env()
        env.reset(seed=42)
        node_s = env._sim.nodes["S"]
        assert node_s.policy is env._rl_policy
        assert isinstance(env._rl_policy, RLIntermediatePolicy)

    def test_active_subset_matches_scenario_nodes(self):
        """active_subset matches the products carried by node S."""
        env = _make_env()
        env.reset(seed=42)
        node_s = env._sim.nodes["S"]
        assert set(env._active_subset) == node_s.carried_products

    def test_base_demand_prior_is_positive_after_reset(self):
        """After reset, env._get_base_demand_prior() returns a positive float."""
        env = _make_env()
        env.reset(seed=42)
        # _base_demand_prior is now computed on demand; _get_base_demand_prior() must be positive.
        prior = env._get_base_demand_prior()
        assert prior > 0, f"base_demand_prior={prior} (expected > 0)"
