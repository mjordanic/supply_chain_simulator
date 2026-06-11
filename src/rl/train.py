"""Training driver for the supply-chain RL policy framework.

This script wires together every component built in issues 01-07 into a
single end-to-end PPO training run.  It is intentionally thin: all
algorithm logic lives in ``src/rl/agents/ppo.py``; this file only
constructs the pieces and calls ``train_ppo``.

Usage
-----
Quick smoke run (no world cache needed — uses a synthetic catalog)::

    uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2

Full run against a setup directory::

    uv run python -m src.rl.train \\
        --total-env-steps 1000000 \\
        --experiment-name my_run \\
        --setup-dir setups/my_scenario/

Architecture
------------
1. Load (or synthesise) the product catalog and market params.
2. Build a ``gymnasium.vector.SyncVectorEnv`` of ``n_envs`` ``RLEnv``
   instances — each env runs the full simulator, so envs are independent.
3. Build a ``SummaryWriter`` at ``runs/<experiment_name>/``.
4. Build the CRN-paired eval callback and pass it to ``train_ppo`` as
   ``eval_fn`` so evaluations are triggered by the PPO loop at the
   configured cadence.
5. Call ``train_ppo`` — it owns all rollout collection, GAE, and
   minibatch updates.
6. On return, save the final SetActor checkpoint through the checkpoint
   I/O module (self-describing bundle with layout-version validation).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from src.rl.agents.ppo import train_ppo
from src.rl.agents.set_actor_critic import SetActor
import src.rl.checkpoint as ckpt_module
from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.rl.eval import build_eval_seeds, evaluate
from src.rl.episode_sampler import load_catalog_and_market_from_setup, make_synthetic_catalog
from src.sim.scenario import DisruptionParams, MarketParams
from src.sim.policy import OrderUpToPolicy


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def _load_world_catalog_and_template(
    config: RLConfig,
) -> tuple[list[Any], MarketParams | None, DisruptionParams | None]:
    """Return ``(catalog, market_params, disruption_params)``.

    Resolution order:
    1. If ``config.setup_dir`` is set: load catalog + market from the setup
       directory.
    2. Otherwise: build a synthetic catalog (no LLM required).
    """
    if config.setup_dir is not None:
        catalog, market = load_catalog_and_market_from_setup(config.setup_dir)
        print(
            f"[train] Loaded catalog ({len(catalog)} products) + market "
            f"from setup directory: {config.setup_dir}",
            file=sys.stderr,
        )
        return catalog, market, None

    catalog = make_synthetic_catalog(config.K_catalog)
    print(
        f"[train] Synthetic catalog: {config.K_catalog} products.",
        file=sys.stderr,
    )
    return catalog, None, None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_dir(config: RLConfig) -> Path:
    """Return ``runs/<experiment_name>/checkpoints/``."""
    return Path(config.checkpoint_dir) / config.experiment_name / "checkpoints"


def _save_checkpoint(
    actor: SetActor,
    global_step: int,
    config: RLConfig,
) -> Path:
    """Save ``actor`` as a self-describing checkpoint bundle; return the path written."""
    ckpt_dir = _checkpoint_dir(config)
    path = ckpt_dir / f"actor_step{global_step:010d}.pt"
    config_snapshot = {
        "K_min": config.K_min,
        "K_max_episode": config.K_max_episode,
        "K_catalog": config.K_catalog,
        "arbiter_mode": config.arbiter_mode,
        "cash_budget_fraction": config.cash_budget_fraction,
        "episode_length": config.episode_length,
        "lr": config.lr,
        "n_steps": config.n_steps,
        "n_epochs": config.n_epochs,
        "n_minibatches": config.n_minibatches,
        "clip_coef": config.clip_coef,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "gae_lambda": config.gae_lambda,
        "gamma": config.gamma,
        "total_env_steps": config.total_env_steps,
        "experiment_name": config.experiment_name,
    }
    ckpt_module.save(actor.state_dict(), config_snapshot, path)
    return path


# ---------------------------------------------------------------------------
# Eval callback factory
# ---------------------------------------------------------------------------


def _make_eval_fn(
    catalog: list[Any],
    config: RLConfig,
    writer: SummaryWriter,
    device: torch.device,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
) -> Any:
    """Return a closure ``(actor, global_step) → dict[str, float]`` usable as ``eval_fn``."""
    from src.rl.set_encoder import K_MAX, F, ROW_MASK

    eval_specs = build_eval_seeds(
        catalog,
        config,
        market_params=market_params,
        disruption_params=disruption_params,
    )

    def _eval_fn(actor: SetActor, global_step: int) -> dict[str, float]:
        actor.eval()

        def _rl_policy(obs_np: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            obs_2d = obs_t.reshape(1, K_MAX, F)
            mask = obs_2d[:, :, ROW_MASK]
            with torch.no_grad():
                action_2d, _, _ = actor.get_action_and_log_prob(obs_2d, mask)
            return action_2d.reshape(K_MAX * 3).cpu().numpy()

        def _baseline_factory():
            return OrderUpToPolicy()

        metrics = evaluate(
            rl_policy_fn=_rl_policy,
            baseline_policy_factory=_baseline_factory,
            eval_specs=eval_specs,
            config=config,
        )

        _save_checkpoint(actor, global_step, config)

        actor.train()
        return metrics

    return _eval_fn


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO training driver for the supply-chain RL env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required / important knobs exposed via CLI.
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps to train for.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs in the SyncVectorEnv.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rl_ppo",
        help="Experiment label; used for the TensorBoard sub-dir and checkpoint prefix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for torch and numpy inside the training loop.",
    )

    # World / catalog.
    parser.add_argument(
        "--setup-dir",
        type=str,
        default=None,
        help="Path to a setup directory (catalog.csv + setup.yaml).",
    )

    # Episode / env knobs (variable-K range replaces fixed --k-active).
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument(
        "--k-min", type=int, default=1, dest="K_min",
        help="Minimum K sampled per episode (inclusive).",
    )
    parser.add_argument(
        "--k-max-episode", type=int, default=20, dest="K_max_episode",
        help="Maximum K sampled per episode (inclusive).",
    )
    parser.add_argument("--k-catalog", type=int, default=100, dest="K_catalog")
    parser.add_argument(
        "--arbiter-mode", type=str, default="proportional",
        choices=["proportional", "greedy"],
        help="Arbiter allocation mode.",
    )
    parser.add_argument(
        "--cash-budget-fraction", type=float, default=1.0,
        help="Cash budget = node.cash * this fraction.",
    )
    parser.add_argument("--delivery-lag", type=int, default=3)
    parser.add_argument("--holding-rate", type=float, default=0.01)
    parser.add_argument("--order-fee", type=float, default=50.0)

    # PPO hyperparameters.
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # Eval.
    parser.add_argument("--eval-cadence-env-steps", type=int, default=50_000)
    parser.add_argument("--n-eval-seeds", type=int, default=32)
    parser.add_argument("--no-eval", action="store_true", help="Disable CRN eval during training.")

    # Output.
    parser.add_argument("--tb-log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="runs")

    return parser.parse_args(argv)


def _args_to_config(args: argparse.Namespace) -> RLConfig:
    """Map parsed args onto an ``RLConfig`` dataclass."""
    return RLConfig(
        episode_length=args.episode_length,
        K_min=args.K_min,
        K_max_episode=args.K_max_episode,
        K_catalog=args.K_catalog,
        arbiter_mode=args.arbiter_mode,
        cash_budget_fraction=args.cash_budget_fraction,
        delivery_lag=args.delivery_lag,
        holding_rate=args.holding_rate,
        order_fee=args.order_fee,
        lr=args.lr,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        n_envs=args.n_envs,
        total_env_steps=args.total_env_steps,
        eval_cadence_env_steps=args.eval_cadence_env_steps,
        n_eval_seeds=args.n_eval_seeds,
        tb_log_dir=args.tb_log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        setup_dir=args.setup_dir,
    )


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse args, build all components, and run PPO training."""
    args = _parse_args(argv)
    config = _args_to_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 1. Load catalog (and optional market params from setup dir).
    # ------------------------------------------------------------------
    catalog, market_params, disruption_params = (
        _load_world_catalog_and_template(config)
    )

    # ------------------------------------------------------------------
    # 2. Build SyncVectorEnv.
    # ------------------------------------------------------------------
    def _env_factory():
        return RLEnv(
            catalog=catalog,
            config=config,
            market_params=market_params,
            disruption_params=disruption_params,
        )

    n_envs = config.n_envs
    envs = gym.vector.SyncVectorEnv([_env_factory for _ in range(n_envs)])
    print(
        f"[train] SyncVectorEnv: {n_envs} envs, "
        f"obs_dim={envs.single_observation_space.shape[0]}, "
        f"act_dim={envs.single_action_space.shape[0]}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 3. Build TensorBoard SummaryWriter.
    # ------------------------------------------------------------------
    tb_run_dir = os.path.join(config.tb_log_dir, config.experiment_name)
    writer = SummaryWriter(log_dir=tb_run_dir)
    print(f"[train] TensorBoard log dir: {tb_run_dir}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 4. Build the eval callback (or None if --no-eval).
    # ------------------------------------------------------------------
    eval_fn = None
    if not args.no_eval:
        try:
            eval_fn = _make_eval_fn(
                catalog,
                config,
                writer,
                device,
                market_params=market_params,
                disruption_params=disruption_params,
            )
            print(
                f"[train] Eval enabled: {config.n_eval_seeds} seeds, "
                f"cadence={config.eval_cadence_env_steps} env steps.",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[train] WARNING: eval setup failed ({exc}); eval disabled.", file=sys.stderr)
            eval_fn = None

    # ------------------------------------------------------------------
    # 5. Run PPO training.
    # ------------------------------------------------------------------
    print(
        f"[train] Starting PPO: total_env_steps={config.total_env_steps}, "
        f"K_min={config.K_min}, K_max_episode={config.K_max_episode}, "
        f"seed={args.seed}",
        file=sys.stderr,
    )
    actor, critic = train_ppo(
        envs,
        config,
        writer,
        eval_fn=eval_fn,
        device=device,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 6. Save final checkpoint (self-describing bundle via checkpoint module).
    # ------------------------------------------------------------------
    final_path = _save_checkpoint(actor, config.total_env_steps, config)
    print(f"[train] Final checkpoint saved: {final_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 7. Cleanup.
    # ------------------------------------------------------------------
    writer.close()
    envs.close()
    print("[train] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
