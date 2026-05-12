"""Smoke tests for src/rl/env.py (RLEnv).

Acceptance criteria:
  - reset() returns an observation matching observation_space.shape
  - One full episode (episode_length step() calls) runs end-to-end without raising
  - terminated becomes True exactly at step episode_length
  - Same reset(seed=s) from two RLEnv instances produces identical first-observation tensors
  - Reward across one episode accumulates consistently with balance evolution
  - RLPolicy.decide() raises RuntimeError if called without set_pending_action
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.sim.policy import RLPolicy
from src.sim.scenario import StoreTemplate, load_catalog


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


def _make_base_template() -> StoreTemplate:
    return StoreTemplate(
        id="rl_smoke_test",
        region="US",
        capacity=200,
        init_balance=20000.0,
        init_stock_pct=0.0,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _make_config(episode_length: int = 10) -> RLConfig:
    """Short episode for fast tests."""
    return RLConfig(
        episode_length=episode_length,
        K_active=5,
    )


def _make_env(episode_length: int = 10) -> RLEnv:
    return RLEnv(
        catalog=_make_catalog(20),
        base_template=_make_base_template(),
        config=_make_config(episode_length),
    )


# ---------------------------------------------------------------------------
# RLPolicy unit tests
# ---------------------------------------------------------------------------


class TestRLPolicy:
    def test_set_then_decide_returns_action(self):
        policy = RLPolicy()
        action_dict = {
            "order": {"P0000": 10},
            "price": {"P0000": 12.0},
            "activate": [],
            "deactivate": [],
            "promotions": {},
        }
        policy.set_pending_action(action_dict)
        result = policy.decide({})
        assert result is action_dict

    def test_decide_without_set_raises(self):
        policy = RLPolicy()
        with pytest.raises(RuntimeError, match="set_pending_action"):
            policy.decide({})

    def test_decide_clears_pending(self):
        """Calling decide() twice in a row raises on the second call."""
        policy = RLPolicy()
        policy.set_pending_action({"order": {}, "price": {}, "activate": [], "deactivate": [], "promotions": {}})
        policy.decide({})
        with pytest.raises(RuntimeError):
            policy.decide({})

    def test_does_not_consume_policy_rng(self):
        """RLPolicy.policy_rng state is unchanged after set/decide cycle."""
        from random import Random
        policy = RLPolicy()
        # Capture initial RNG state by recording a draw, then reset.
        rng = policy.policy_rng
        state_before = rng.getstate()
        policy.set_pending_action({"order": {}, "price": {}, "activate": [], "deactivate": [], "promotions": {}})
        policy.decide({})
        state_after = rng.getstate()
        assert state_before == state_after, "RLPolicy.decide() must not draw from policy_rng"


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
        from src.rl.encoders import observation_dim, action_dim
        config = _make_config()
        env = RLEnv(catalog=_make_catalog(20), base_template=_make_base_template(), config=config)
        assert env.observation_space.shape == (observation_dim(config.K_active),)
        assert env.action_space.shape == (action_dim(config.K_active),)

    def test_action_space_bounds(self):
        env = _make_env()
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RLEnv — full episode run
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_one_full_episode_no_raise(self):
        """180 step() calls complete without raising."""
        env = RLEnv(
            catalog=_make_catalog(20),
            base_template=_make_base_template(),
            config=RLConfig(episode_length=180, K_active=5),
        )
        env.reset(seed=7)
        for _ in range(180):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True

    def test_short_episode_terminated_at_episode_length(self):
        """terminated becomes True exactly at step episode_length."""
        episode_length = 15
        env = RLEnv(
            catalog=_make_catalog(20),
            base_template=_make_base_template(),
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
        """truncated is always False (no time-limit truncation in this env)."""
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
        template = _make_base_template()
        config = _make_config(episode_length=10)

        env1 = RLEnv(catalog=catalog, base_template=template, config=config)
        env2 = RLEnv(catalog=catalog, base_template=template, config=config)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        np.testing.assert_array_equal(obs1, obs2, err_msg="Observations from same seed differ")

    def test_different_seeds_different_obs(self):
        """Different seeds should (with overwhelming probability) differ."""
        env = _make_env()
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        # Not checking strict equality — this should almost always differ.
        # If they're equal it would indicate a serious bug in sampling.
        assert not np.allclose(obs_a, obs_b), "Different seeds produced identical observations"

    def test_full_episode_trajectory_determinism(self):
        """Two envs with same seed and same actions produce identical observations."""
        catalog = _make_catalog(20)
        template = _make_base_template()
        config = _make_config(episode_length=5)

        env1 = RLEnv(catalog=catalog, base_template=template, config=config)
        env2 = RLEnv(catalog=catalog, base_template=template, config=config)

        from random import Random
        rng = Random(555)
        action_sequence = [
            np.array([rng.uniform(-1, 1) for _ in range(config.K_active * 2)], dtype=np.float32)
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
# RLEnv — reward / balance accounting
# ---------------------------------------------------------------------------


class TestRewardAccounting:
    def test_cumulative_reward_equals_balance_delta(self):
        """Sum of rewards across one episode equals the terminal balance minus the opening balance."""
        env = _make_env(episode_length=10)
        obs, _ = env.reset(seed=42)
        initial_balance = env._store.balance

        cumulative_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, _, _, info = env.step(action)
            cumulative_reward += reward

        terminal_balance = env._store.balance
        expected_delta = terminal_balance - initial_balance

        assert cumulative_reward == pytest.approx(expected_delta, abs=1.0), (
            f"Cumulative reward {cumulative_reward:.2f} != balance delta {expected_delta:.2f}"
        )

    def test_reward_is_scalar(self):
        """Reward returned by step() is a plain Python float, not an array."""
        env = _make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_info_contains_expected_keys(self):
        """info dict has step, balance, active_products, per_sku."""
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "step" in info
        assert "balance" in info
        assert "active_products" in info
        assert "per_sku" in info
        # per_sku should have at least one entry
        assert len(info["per_sku"]) > 0

    def test_per_sku_info_keys(self):
        """Each per_sku entry has the required accounting fields."""
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        required = {"sales", "demand", "inventory", "revenue", "total_cost", "holding_cost", "price", "is_active"}
        for pid, sku_info in info["per_sku"].items():
            assert required <= set(sku_info.keys()), f"{pid} missing keys: {required - set(sku_info.keys())}"


# ---------------------------------------------------------------------------
# RLEnv — assortment and promotion freeze
# ---------------------------------------------------------------------------


class TestAssortmentAndPromoFreeze:
    def test_active_products_frozen_across_episode(self):
        """The active assortment does not change across steps."""
        env = _make_env(episode_length=20)
        env.reset(seed=55)
        initial_active = list(env._store.active_items)

        for _ in range(20):
            env.step(env.action_space.sample())

        final_active = list(env._store.active_items)
        assert initial_active == final_active, (
            f"Assortment changed during episode: {initial_active} → {final_active}"
        )

    def test_no_promotions_in_store(self):
        """Store.promotions remains empty throughout the episode."""
        env = _make_env(episode_length=10)
        env.reset(seed=77)
        for _ in range(10):
            env.step(env.action_space.sample())
            assert env._store.promotions == {}, "Promotions should be disabled"


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
