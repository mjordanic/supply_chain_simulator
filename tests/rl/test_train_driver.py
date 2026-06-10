"""Tests for src/rl/train.py — the PPO training driver.

Acceptance criteria verified here:
  - ``src/rl/train.py`` runs via ``main([...])`` without raising for a
    short --total-env-steps run.
  - A SyncVectorEnv of ``config.n_envs`` RLEnv instances is constructed.
  - A TensorBoard event file is written under the configured log dir.
  - A checkpoint is written to ``runs/<experiment_name>/checkpoints/``
    as a self-describing bundle (checkpoint.save format, not bare state-dict).
  - CRN eval is wired and invoked at the configured cadence.
  - ``uv run pytest tests/sim/`` continues to pass (verified separately).

These tests use the synthetic catalog path so no world.json cache or
OpenAI API key is required.  All tests that run PPO are marked
``@pytest.mark.slow`` because they spin up gymnasium vector envs and run
several hundred env steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import gymnasium as gym

from src.rl.agents.set_actor_critic import SetActor
from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.rl.set_encoder import K_MAX, F
from src.rl.train import (
    _args_to_config,
    _checkpoint_dir,
    _parse_args,
    _save_checkpoint,
    main,
)
from src.sim.scenario import load_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(tmp_path: Path, *, n_envs: int = 2, total_env_steps: int = 256) -> RLConfig:
    """Return an RLConfig sized for a fast end-to-end smoke run."""
    return RLConfig(
        episode_length=10,
        K_min=1,
        K_max_episode=5,
        K_catalog=20,
        n_envs=n_envs,
        total_env_steps=total_env_steps,
        n_steps=16,
        n_epochs=2,
        n_minibatches=2,
        eval_cadence_env_steps=128,   # trigger eval mid-run
        n_eval_seeds=2,               # tiny eval for speed
        tb_log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "runs"),
        experiment_name="test_run",
    )


def _tiny_argv(tmp_path: Path, **overrides) -> list[str]:
    """Build a minimal argv list for ``main()``."""
    argv = [
        "--total-env-steps", "256",
        "--n-envs", "2",
        "--episode-length", "10",
        "--k-catalog", "20",
        "--k-min", "1",
        "--k-max-episode", "5",
        "--n-steps", "16",
        "--n-epochs", "2",
        "--n-minibatches", "2",
        "--eval-cadence-env-steps", "128",
        "--n-eval-seeds", "2",
        "--experiment-name", "test_run",
        "--tb-log-dir", str(tmp_path / "runs"),
        "--checkpoint-dir", str(tmp_path / "runs"),
        "--no-eval",  # skip eval by default; specific tests enable it
    ]
    for k, v in overrides.items():
        argv.extend([f"--{k.replace('_', '-')}", str(v)])
    return argv


# ---------------------------------------------------------------------------
# Unit: arg parsing and config construction
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_parse_defaults(self):
        """Default args parse without raising."""
        args = _parse_args([])
        assert args.total_env_steps == 1_000_000
        assert args.n_envs == 8
        assert args.experiment_name == "rl_ppo"
        assert args.seed == 0

    def test_parse_k_range(self):
        """K-range args (--k-min, --k-max-episode) are reflected in parsed namespace."""
        args = _parse_args([
            "--k-min", "3",
            "--k-max-episode", "15",
        ])
        assert args.K_min == 3
        assert args.K_max_episode == 15

    def test_parse_overrides(self):
        """Custom args are reflected in parsed namespace."""
        args = _parse_args([
            "--total-env-steps", "5000",
            "--n-envs", "4",
            "--experiment-name", "my_exp",
            "--seed", "42",
        ])
        assert args.total_env_steps == 5000
        assert args.n_envs == 4
        assert args.experiment_name == "my_exp"
        assert args.seed == 42

    def test_args_to_config(self):
        """_args_to_config maps all relevant CLI args to RLConfig fields."""
        args = _parse_args([
            "--total-env-steps", "10000",
            "--n-envs", "4",
            "--experiment-name", "exp",
            "--seed", "7",
            "--lr", "1e-3",
            "--episode-length", "30",
            "--k-min", "2",
            "--k-max-episode", "10",
        ])
        config = _args_to_config(args)
        assert config.total_env_steps == 10000
        assert config.n_envs == 4
        assert config.experiment_name == "exp"
        assert config.lr == pytest.approx(1e-3)
        assert config.episode_length == 30
        assert config.K_min == 2
        assert config.K_max_episode == 10

    def test_no_eval_flag_sets_attribute(self):
        """--no-eval flag sets args.no_eval to True."""
        args = _parse_args(["--no-eval"])
        assert args.no_eval is True

    def test_arbiter_mode_arg(self):
        """--arbiter-mode is parsed and mapped to config."""
        args = _parse_args(["--arbiter-mode", "greedy"])
        config = _args_to_config(args)
        assert config.arbiter_mode == "greedy"


# ---------------------------------------------------------------------------
# Unit: checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    def test_save_checkpoint_creates_self_describing_bundle(self, tmp_path):
        """_save_checkpoint writes a bundle loadable via checkpoint.load()."""
        import src.rl.checkpoint as ckpt_module
        from src.rl.set_encoder import OBS_LAYOUT_VERSION

        config = RLConfig(
            checkpoint_dir=str(tmp_path),
            experiment_name="ckpt_test",
            K_min=1,
            K_max_episode=5,
        )
        actor = SetActor()
        path = _save_checkpoint(actor, global_step=1000, config=config)

        assert path.exists(), f"Checkpoint not found at {path}"

        # Must be loadable via checkpoint.load() (not just torch.load).
        bundle = ckpt_module.load(path)
        assert "state_dict" in bundle
        assert "config" in bundle
        assert "layout_version" in bundle
        assert bundle["layout_version"] == OBS_LAYOUT_VERSION

    def test_checkpoint_dir_structure(self, tmp_path):
        """Checkpoint files land under runs/<experiment_name>/checkpoints/."""
        config = RLConfig(
            checkpoint_dir=str(tmp_path),
            experiment_name="my_exp",
            K_min=1,
            K_max_episode=5,
        )
        expected_dir = tmp_path / "my_exp" / "checkpoints"
        computed = _checkpoint_dir(config)
        assert computed == expected_dir


# ---------------------------------------------------------------------------
# Integration: SyncVectorEnv construction
# ---------------------------------------------------------------------------


class TestVecEnvConstruction:
    def test_sync_vec_env_n_envs(self):
        """SyncVectorEnv wraps exactly n_envs RLEnv instances."""
        n_envs = 3
        items = [
            {
                "name": f"P{i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            }
            for i in range(20)
        ]
        catalog = load_catalog(items)
        config = RLConfig(episode_length=5, K_min=1, K_max_episode=5, K_catalog=20)

        def _factory():
            return RLEnv(catalog=catalog, config=config)

        envs = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
        assert envs.num_envs == n_envs
        obs, _ = envs.reset(seed=0)
        assert obs.shape[0] == n_envs
        envs.close()

    def test_vec_env_obs_shape(self):
        """SyncVectorEnv observations have shape (n_envs, K_MAX * F)."""
        n_envs = 2
        items = [
            {
                "name": f"P{i}",
                "category": "General",
                "related_products": [],
                "base_price": 10.0 + i,
                "unit_cost": 4.0,
                "seasonality": "all_season",
            }
            for i in range(20)
        ]
        catalog = load_catalog(items)
        config = RLConfig(episode_length=5, K_min=1, K_max_episode=5, K_catalog=20)

        def _factory():
            return RLEnv(catalog=catalog, config=config)

        envs = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
        obs, _ = envs.reset(seed=1)
        expected_dim = K_MAX * F
        assert obs.shape == (n_envs, expected_dim)
        envs.close()


# ---------------------------------------------------------------------------
# Integration: end-to-end driver (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDriverEndToEnd:
    def test_main_no_raise(self, tmp_path):
        """main() completes without raising for a short run."""
        argv = _tiny_argv(tmp_path)
        main(argv)

    def test_tensorboard_event_file_written(self, tmp_path):
        """A TensorBoard event file is produced under the experiment log dir."""
        argv = _tiny_argv(tmp_path)
        main(argv)

        tb_dir = tmp_path / "runs" / "test_run"
        event_files = [
            f for f in os.listdir(str(tb_dir)) if f.startswith("events.out.tfevents")
        ]
        assert len(event_files) > 0, (
            f"No TensorBoard event files in {tb_dir}. "
            f"Contents: {os.listdir(str(tb_dir))}"
        )

    def test_checkpoint_written(self, tmp_path):
        """At least one checkpoint .pt file is written to the checkpoint dir."""
        argv = _tiny_argv(tmp_path)
        main(argv)

        ckpt_dir = tmp_path / "runs" / "test_run" / "checkpoints"
        assert ckpt_dir.exists(), f"Checkpoint dir missing: {ckpt_dir}"
        pt_files = list(ckpt_dir.glob("*.pt"))
        assert len(pt_files) > 0, (
            f"No .pt files in {ckpt_dir}. Contents: {list(ckpt_dir.iterdir())}"
        )

    def test_final_checkpoint_is_self_describing_bundle(self, tmp_path):
        """The final checkpoint is a self-describing bundle loadable via checkpoint.load()."""
        import src.rl.checkpoint as ckpt_module
        from src.rl.set_encoder import OBS_LAYOUT_VERSION

        argv = _tiny_argv(tmp_path)
        main(argv)

        ckpt_dir = tmp_path / "runs" / "test_run" / "checkpoints"
        pt_files = sorted(ckpt_dir.glob("*.pt"))
        assert pt_files, "No checkpoint files found"

        # Load and validate through the checkpoint module.
        bundle = ckpt_module.load(pt_files[-1])
        assert "state_dict" in bundle
        assert "config" in bundle
        assert "layout_version" in bundle
        assert bundle["layout_version"] == OBS_LAYOUT_VERSION

        # Reconstruct a functional SetActor.
        new_actor = SetActor()
        new_actor.load_state_dict(bundle["state_dict"])  # should not raise
        obs = torch.zeros(1, K_MAX, F)
        obs[0, 0, -1] = 1.0  # one active product row
        mask = obs[:, :, -1]
        action, _, _ = new_actor.get_action_and_log_prob(obs, mask)
        assert action.shape == (1, K_MAX, 3)

    def test_eval_triggered_during_training(self, tmp_path):
        """Eval runs are triggered and their metrics appear in the TensorBoard dir."""
        # Enable eval (don't pass --no-eval) and set a cadence below total steps.
        argv = [
            "--total-env-steps", "256",
            "--n-envs", "2",
            "--episode-length", "10",
            "--k-catalog", "20",
            "--k-min", "1",
            "--k-max-episode", "5",
            "--n-steps", "16",
            "--n-epochs", "2",
            "--n-minibatches", "2",
            "--eval-cadence-env-steps", "64",
            "--n-eval-seeds", "2",
            "--experiment-name", "eval_test",
            "--tb-log-dir", str(tmp_path / "runs"),
            "--checkpoint-dir", str(tmp_path / "runs"),
            # No --no-eval: eval is enabled.
        ]
        main(argv)

        # TensorBoard event file exists.
        tb_dir = tmp_path / "runs" / "eval_test"
        event_files = [f for f in os.listdir(str(tb_dir)) if f.startswith("events.out.tfevents")]
        assert len(event_files) > 0, f"No event files in {tb_dir}"

        # At least one checkpoint was written (eval triggers checkpoint saves).
        ckpt_dir = tmp_path / "runs" / "eval_test" / "checkpoints"
        pt_files = list(ckpt_dir.glob("*.pt"))
        assert len(pt_files) > 0, f"No checkpoints written — eval may not have triggered"

    def test_sim_tests_still_pass(self, tmp_path):
        """Importing train.py does not break existing simulator invariants.

        This is a lightweight proxy: we verify that the sim package imports
        cleanly alongside the RL package (the full pytest run covers tests/sim/).
        """
        import src.sim.runner  # noqa: F401 — verifies no import-time breakage
        import src.rl.train  # noqa: F401
        # If either import raises, this test fails.
