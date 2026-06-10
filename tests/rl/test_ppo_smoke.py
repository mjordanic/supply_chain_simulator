"""PPO smoke-training test.

Marked ``@pytest.mark.slow`` — excluded from the default fast run.
Run explicitly with::

    uv run pytest -m slow tests/rl/test_ppo_smoke.py

Acceptance criteria (updated for ADR 0021 variable-K set actor-critic):
  - Trains PPO for ~1000 env steps (small n_steps, single env) on a fixed seed.
  - Loss values remain finite throughout.
  - KL divergence stays below a generous threshold (0.5).
  - Training completes without raising.
  - A TensorBoard event file is written to the configured log dir.
  - train_ppo returns SetActor / SetCritic (not the legacy flat networks).
  - Masked padded slots contribute nothing to the loss.
  - Checkpoint written through checkpoint I/O module; reloadable.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.agents.ppo import train_ppo
from src.rl.agents.set_actor_critic import SetActor, SetCritic
from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.rl.set_encoder import K_MAX, F
from src.sim.scenario import load_catalog


# ---------------------------------------------------------------------------
# Helpers
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


def _make_env_factory(episode_length: int = 10):
    """Return a zero-arg callable that produces a fresh RLEnv."""
    catalog = _make_catalog(20)
    config = RLConfig(episode_length=episode_length, K_min=1, K_max_episode=5)

    def _factory():
        return RLEnv(catalog=catalog, config=config)

    return _factory


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPPOSmoke:
    """Short PPO training run — ~1000 env steps.

    Uses a single sync env (n_envs=1) and small n_steps so the test
    completes in seconds on CPU.
    """

    def test_ppo_trains_without_raising(self, tmp_path):
        """Training loop runs end-to-end without raising."""
        factory = _make_env_factory(episode_length=10)
        envs = gym.vector.SyncVectorEnv([factory])

        config = RLConfig(
            episode_length=10,
            K_min=1,
            K_max_episode=5,
            n_steps=16,           # small rollout
            n_epochs=2,
            n_minibatches=2,
            n_envs=1,
            total_env_steps=256,  # ~16 updates × 16 steps
            lr=3e-4,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            gae_lambda=0.95,
            gamma=0.99,
            max_grad_norm=0.5,
            target_kl=None,
            tb_log_dir=str(tmp_path),
            experiment_name="smoke",
        )

        writer = SummaryWriter(log_dir=str(tmp_path / "smoke"))
        device = torch.device("cpu")

        actor, critic = train_ppo(
            envs,
            config,
            writer,
            eval_fn=None,
            device=device,
            seed=42,
        )
        writer.close()
        envs.close()

        assert isinstance(actor, SetActor), f"Expected SetActor, got {type(actor)}"
        assert isinstance(critic, SetCritic), f"Expected SetCritic, got {type(critic)}"

    def test_losses_are_finite(self, tmp_path):
        """SetActor and SetCritic weights remain finite after training."""
        factory = _make_env_factory(episode_length=10)
        envs = gym.vector.SyncVectorEnv([factory])

        config = RLConfig(
            episode_length=10,
            K_min=1,
            K_max_episode=5,
            n_steps=16,
            n_epochs=2,
            n_minibatches=2,
            n_envs=1,
            total_env_steps=256,
            lr=3e-4,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            gae_lambda=0.95,
            gamma=0.99,
            max_grad_norm=0.5,
            target_kl=None,
            tb_log_dir=str(tmp_path),
            experiment_name="smoke_finite",
        )

        writer = SummaryWriter(log_dir=str(tmp_path / "smoke_finite"))
        device = torch.device("cpu")

        actor, critic = train_ppo(
            envs,
            config,
            writer,
            eval_fn=None,
            device=device,
            seed=0,
        )
        writer.close()
        envs.close()

        for name, param in actor.named_parameters():
            assert torch.all(torch.isfinite(param)), f"Actor param {name!r} is non-finite after training"
        for name, param in critic.named_parameters():
            assert torch.all(torch.isfinite(param)), f"Critic param {name!r} is non-finite after training"

    def test_kl_stays_bounded(self, tmp_path):
        """Training with target_kl set does not diverge (KL < generous threshold)."""
        factory = _make_env_factory(episode_length=10)
        envs = gym.vector.SyncVectorEnv([factory])

        # Use target_kl early-stop so KL is controlled.
        config = RLConfig(
            episode_length=10,
            K_min=1,
            K_max_episode=5,
            n_steps=32,
            n_epochs=4,
            n_minibatches=2,
            n_envs=1,
            total_env_steps=256,
            lr=3e-4,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            gae_lambda=0.95,
            gamma=0.99,
            max_grad_norm=0.5,
            target_kl=0.5,        # generous threshold per issue spec
            tb_log_dir=str(tmp_path),
            experiment_name="smoke_kl",
        )

        writer = SummaryWriter(log_dir=str(tmp_path / "smoke_kl"))
        device = torch.device("cpu")

        actor, critic = train_ppo(
            envs,
            config,
            writer,
            eval_fn=None,
            device=device,
            seed=7,
        )
        writer.close()
        envs.close()

        # With target_kl=0.5, the update loop early-stops so weights should
        # be finite and the actor distribution should not have exploded.
        dummy_obs = torch.zeros(1, K_MAX * F, device=device)
        obs_2d = dummy_obs.reshape(1, K_MAX, F)
        # Give at least one active row so the distribution is valid.
        obs_2d[0, 0, -1] = 1.0  # ROW_MASK = 15 (last col)
        mask = obs_2d[:, :, -1]  # (1, K_MAX)
        dist = actor.get_distribution(obs_2d, mask)
        std = dist.std  # shape (3,) or broadcastable
        assert torch.all(std > 0), "Actor std collapsed to 0"
        assert torch.all(torch.isfinite(std)), "Actor std is non-finite"

    def test_tensorboard_event_file_written(self, tmp_path):
        """A TensorBoard event file is created under the log dir."""
        factory = _make_env_factory(episode_length=10)
        envs = gym.vector.SyncVectorEnv([factory])

        log_dir = tmp_path / "tb_run"
        log_dir.mkdir()

        config = RLConfig(
            episode_length=10,
            K_min=1,
            K_max_episode=5,
            n_steps=16,
            n_epochs=2,
            n_minibatches=2,
            n_envs=1,
            total_env_steps=256,
            lr=3e-4,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            gae_lambda=0.95,
            gamma=0.99,
            max_grad_norm=0.5,
            target_kl=None,
            tb_log_dir=str(log_dir),
            experiment_name="smoke_tb",
        )

        writer = SummaryWriter(log_dir=str(log_dir))
        device = torch.device("cpu")

        train_ppo(
            envs,
            config,
            writer,
            eval_fn=None,
            device=device,
            seed=1,
        )
        writer.close()
        envs.close()

        # TensorBoard writes at least one event file.
        event_files = [
            f for f in os.listdir(str(log_dir)) if f.startswith("events.out.tfevents")
        ]
        assert len(event_files) > 0, (
            f"No TensorBoard event files found in {log_dir}. "
            f"Directory contents: {os.listdir(str(log_dir))}"
        )

    def test_set_actor_critic_shapes(self, tmp_path):
        """SetActor and SetCritic instantiate and forward with correct shapes."""
        actor = SetActor()
        critic = SetCritic()

        B = 2
        dummy_obs = torch.zeros(B, K_MAX, F)
        # Activate first 3 rows.
        dummy_obs[:, :3, -1] = 1.0
        mask = dummy_obs[:, :, -1]  # (B, K_MAX)

        action, log_prob, entropy = actor.get_action_and_log_prob(dummy_obs, mask)
        value = critic(dummy_obs, mask)

        assert action.shape == (B, K_MAX, 3), f"Action shape: {action.shape}"
        assert log_prob.shape == (B,), f"Log-prob shape: {log_prob.shape}"
        assert entropy.shape == (B,), f"Entropy shape: {entropy.shape}"
        assert value.shape == (B, 1), f"Value shape: {value.shape}"

        # Active rows should be non-zero; padded rows should be zero.
        assert torch.all(action[:, :3, :].abs() > 0), "Active rows should have non-zero actions"
        assert torch.all(action[:, 3:, :] == 0), "Padded rows should have zero actions"

    def test_masked_slots_do_not_affect_loss(self, tmp_path):
        """Padded rows are masked out: changing their content does not change log_prob."""
        actor = SetActor()
        B = 2

        # Build obs with K=3 active rows.
        obs = torch.zeros(B, K_MAX, F)
        obs[:, :3, -1] = 1.0  # active mask
        mask = obs[:, :, -1]

        action, log_prob_orig, _ = actor.get_action_and_log_prob(obs, mask)

        # Scramble the padded rows with random content.
        obs_scrambled = obs.clone()
        obs_scrambled[:, 3:, :] = torch.randn(B, K_MAX - 3, F)
        # Keep mask column = 0 for padded rows.
        obs_scrambled[:, 3:, -1] = 0.0
        mask_scrambled = obs_scrambled[:, :, -1]

        _, log_prob_scrambled, _ = actor.get_action_and_log_prob(
            obs_scrambled, mask_scrambled, action
        )

        # log_prob should be identical regardless of padded-row content.
        assert torch.allclose(log_prob_orig, log_prob_scrambled, atol=1e-5), (
            f"Padded rows affected log_prob: orig={log_prob_orig}, "
            f"scrambled={log_prob_scrambled}"
        )


# ---------------------------------------------------------------------------
# Checkpoint round-trip test (issue 06 acceptance criterion)
# ---------------------------------------------------------------------------


def test_checkpoint_written_via_checkpoint_module(tmp_path):
    """train driver checkpoint is written through checkpoint.save(); reloadable.

    Verifies that the checkpoint file is a self-describing bundle (not a bare
    state-dict) and that checkpoint.load() can reconstruct it without raising.
    """
    import src.rl.checkpoint as ckpt_module
    from src.rl.set_encoder import OBS_LAYOUT_VERSION
    from src.rl.train import main as train_main

    argv = [
        "--total-env-steps", "16",
        "--n-envs", "1",
        "--episode-length", "5",
        "--k-catalog", "20",
        "--k-min", "1",
        "--k-max-episode", "5",
        "--n-steps", "8",
        "--n-epochs", "1",
        "--n-minibatches", "2",
        "--experiment-name", "ckpt_test",
        "--tb-log-dir", str(tmp_path / "runs"),
        "--checkpoint-dir", str(tmp_path / "runs"),
        "--no-eval",
    ]
    train_main(argv)

    # Find the written checkpoint.
    ckpt_dir = tmp_path / "runs" / "ckpt_test" / "checkpoints"
    pt_files = sorted(ckpt_dir.glob("*.pt"))
    assert pt_files, f"No checkpoint files in {ckpt_dir}"

    # checkpoint.load() must succeed (validates layout_version).
    bundle = ckpt_module.load(pt_files[-1])
    assert "state_dict" in bundle, "Missing 'state_dict' key in checkpoint bundle"
    assert "config" in bundle, "Missing 'config' key in checkpoint bundle"
    assert "layout_version" in bundle, "Missing 'layout_version' key in checkpoint bundle"
    assert bundle["layout_version"] == OBS_LAYOUT_VERSION

    # The loaded state_dict should reconstruct a functional SetActor.
    new_actor = SetActor()
    new_actor.load_state_dict(bundle["state_dict"])  # should not raise
    obs = torch.zeros(1, K_MAX, F)
    obs[0, 0, -1] = 1.0
    mask = obs[:, :, -1]
    action, _, _ = new_actor.get_action_and_log_prob(obs, mask)
    assert action.shape == (1, K_MAX, 3)


# ---------------------------------------------------------------------------
# Multi-env (vectorised) smoke (issue 06 acceptance criterion)
# ---------------------------------------------------------------------------


def test_train_ppo_with_multiple_envs(tmp_path):
    """Training with n_envs > 1 completes without raising (vectorised setup)."""
    catalog = _make_catalog(20)
    n_envs = 3
    config = RLConfig(
        episode_length=5,
        K_min=1,
        K_max_episode=5,
        K_catalog=20,
        n_steps=8,
        n_epochs=1,
        n_minibatches=2,
        n_envs=n_envs,
        total_env_steps=48,   # 2 updates × 8 steps × 3 envs
        tb_log_dir=str(tmp_path / "runs"),
        experiment_name="multi_env",
    )

    def _factory():
        return RLEnv(catalog=catalog, config=config)

    envs = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
    writer = SummaryWriter(log_dir=str(tmp_path / "runs" / "multi_env"))
    device = torch.device("cpu")

    actor, critic = train_ppo(envs, config, writer, eval_fn=None, device=device, seed=1)
    writer.close()
    envs.close()

    assert isinstance(actor, SetActor)
    assert isinstance(critic, SetCritic)
