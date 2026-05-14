"""Training driver for the supply-chain RL policy framework.

This script wires together every component built in issues 01-07 into a
single end-to-end PPO training run.  It is intentionally thin: all
algorithm logic lives in ``src/rl/agents/ppo.py``; this file only
constructs the pieces and calls ``train_ppo``.

Usage
-----
Quick smoke run (no world cache needed — uses a synthetic catalog)::

    uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2

Full run against a cached LLM-built world::

    uv run python -m src.rl.train \\
        --total-env-steps 1000000 \\
        --experiment-name my_run \\
        --world-cache-path data/worlds/rl_train/world.json

Architecture
------------
1. Load (or synthesise) the product catalog and a base ``StoreTemplate``.
2. Build a ``gymnasium.vector.SyncVectorEnv`` of ``n_envs`` ``RLEnv``
   instances — each env runs the full simulator, so envs are independent.
3. Build a ``SummaryWriter`` at ``runs/<experiment_name>/``.
4. Build the CRN-paired eval callback and pass it to ``train_ppo`` as
   ``eval_fn`` so evaluations are triggered by the PPO loop at the
   configured cadence.
5. Call ``train_ppo`` — it owns all rollout collection, GAE, and
   minibatch updates.
6. On return, save the final Actor / Critic checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from src.rl.agents.ppo import Actor, train_ppo
from src.rl.configs.default import RLConfig
from src.rl.env import RLEnv
from src.rl.episode_sampler import _default_disruption_params
from src.rl.eval import build_eval_seeds, evaluate
from src.sim.scenario import DisruptionParams, MarketParams, StoreTemplate, load_catalog
from src.sim.policy import OrderUpToPolicy


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def _load_world_catalog_and_template(
    config: RLConfig,
) -> tuple[list[Any], StoreTemplate, MarketParams | None, DisruptionParams | None]:
    """Return ``(catalog, base_template, market_params, disruption_params)``.

    Resolution order:
    1. If ``config.world_cache_path`` is set and the file exists, load it.
    2. Else look for ``data/worlds/<config.world_archetype>/world.json``.
    3. Else synthesise a minimal catalog (no LLM required — useful for CI
       smoke runs and quick --total-env-steps experiments).

    For loaded worlds, ``market_params`` is the world's own ``MarketParams``
    and ``disruption_params`` is the default with ``regions`` overridden to
    match the world's market regions — without this, ``Market.market_state``
    would be missing the keys used by the store template's ``region``. For
    the synthetic fallback both are ``None`` so ``sample_episode`` uses its
    own defaults.
    """
    # Try explicit path override first.
    if config.world_cache_path is not None:
        cache = Path(config.world_cache_path)
        if cache.exists():
            return _load_world_from_file(cache, config)
        print(
            f"[train] WARNING: world_cache_path={config.world_cache_path!r} not found; "
            "falling back to auto-lookup.",
            file=sys.stderr,
        )

    # Auto-lookup by archetype.
    auto_path = Path("data/worlds") / config.world_archetype / "world.json"
    if auto_path.exists():
        return _load_world_from_file(auto_path, config)

    # Synthetic fallback.
    print(
        f"[train] No world cache found at {auto_path}. "
        "Generating a synthetic catalog (no LLM calls).",
        file=sys.stderr,
    )
    return _build_synthetic_catalog(config)


def _load_world_from_file(
    path: Path, config: RLConfig
) -> tuple[list[Any], StoreTemplate, MarketParams, DisruptionParams]:
    """Load catalog, StoreTemplate, MarketParams, and DisruptionParams from a cached world.json."""
    from src.llm.world_builder import World

    world = World.from_json(path)
    catalog = world.catalog

    # Pick the first store template (or build a minimal one if none exist).
    if world.store_templates:
        key = next(iter(world.store_templates))
        tmpl = world.store_templates[key]
        # Override episodic knobs so they match the RLConfig.
        tmpl = StoreTemplate(
            id=tmpl.id,
            region=tmpl.region,
            capacity=200,          # overridden per episode by episode_sampler
            init_balance=20000.0,  # overridden per episode by episode_sampler
            init_stock_pct=0.0,
            delivery_lag=config.delivery_lag,
            holding_rate=config.holding_rate,
            order_fee=config.order_fee,
            init_active_count=config.K_active,
        )
    else:
        tmpl = _default_template(config)

    market_params = world.market
    disruption_params = replace(
        _default_disruption_params(),
        regions=list(world.market.regions),
    )

    print(f"[train] Loaded world from {path} ({len(catalog)} products).", file=sys.stderr)
    return catalog, tmpl, market_params, disruption_params


def _build_synthetic_catalog(
    config: RLConfig,
) -> tuple[list[Any], StoreTemplate, None, None]:
    """Build a K_catalog-item synthetic catalog (no LLM) for CI / smoke runs."""
    n = config.K_catalog
    items = [
        {
            "name": f"Product {i}",
            "category": "General",
            "related_products": [],
            "base_price": float(10 + (i % 30)),
            "unit_cost": float(4 + (i % 10)),
            "seasonality": "all_season",
        }
        for i in range(n)
    ]
    catalog = load_catalog(items)
    tmpl = _default_template(config)
    print(f"[train] Synthetic catalog: {n} products.", file=sys.stderr)
    return catalog, tmpl, None, None


def _default_template(config: RLConfig) -> StoreTemplate:
    """Return a minimal StoreTemplate consistent with ``config``."""
    return StoreTemplate(
        id="rl_train_default",
        region="US",
        capacity=200,          # overridden per episode by episode_sampler
        init_balance=20000.0,  # overridden per episode by episode_sampler
        init_stock_pct=0.0,
        delivery_lag=config.delivery_lag,
        holding_rate=config.holding_rate,
        order_fee=config.order_fee,
        init_active_count=config.K_active,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_dir(config: RLConfig) -> Path:
    """Return ``runs/<experiment_name>/checkpoints/``."""
    return Path(config.checkpoint_dir) / config.experiment_name / "checkpoints"


def _save_checkpoint(
    actor: Actor,
    global_step: int,
    config: RLConfig,
) -> Path:
    """Save ``actor`` state dict; return the path written."""
    ckpt_dir = _checkpoint_dir(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"actor_step{global_step:010d}.pt"
    torch.save(actor.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# Eval callback factory
# ---------------------------------------------------------------------------


def _make_eval_fn(
    catalog: list[Any],
    base_template: StoreTemplate,
    config: RLConfig,
    writer: SummaryWriter,
    device: torch.device,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
) -> Any:
    """Return a closure ``(actor) → dict[str, float]`` usable as ``eval_fn``.

    The closure:
    1. Builds the frozen eval specs once on first call.
    2. Wraps the actor in a numpy-friendly policy callable.
    3. Delegates to ``evaluate()`` from ``src.rl.eval``.
    4. Saves a checkpoint each time it is invoked.
    """
    # Build eval specs once — they are deterministic so we cache them.
    eval_specs = build_eval_seeds(
        catalog,
        base_template,
        config,
        market_params=market_params,
        disruption_params=disruption_params,
    )

    checkpoint_counter: list[int] = [0]  # mutable cell for the closure

    def _eval_fn(actor: Actor) -> dict[str, float]:
        actor.eval()

        def _rl_policy(obs_np: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action_and_log_prob(obs_t)
            return action.squeeze(0).cpu().numpy()

        def _baseline_factory():
            return OrderUpToPolicy()

        metrics = evaluate(
            rl_policy_fn=_rl_policy,
            baseline_policy_factory=_baseline_factory,
            eval_specs=eval_specs,
            config=config,
        )

        # Save checkpoint at each eval.
        step = checkpoint_counter[0]
        _save_checkpoint(actor, step, config)
        checkpoint_counter[0] += 1

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
        "--world-cache-path",
        type=str,
        default=None,
        help=(
            "Explicit path to a cached world.json. "
            "When omitted, auto-lookup and synthetic fallback are used."
        ),
    )
    parser.add_argument(
        "--world-archetype",
        type=str,
        default="rl_train",
        help="Archetype label for the world-builder cache auto-lookup.",
    )

    # Episode / env knobs.
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument("--k-active", type=int, default=5, dest="K_active")
    parser.add_argument("--k-catalog", type=int, default=100, dest="K_catalog")
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
        K_active=args.K_active,
        K_catalog=args.K_catalog,
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
        world_archetype=args.world_archetype,
        world_cache_path=args.world_cache_path,
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
    # 1. Load catalog and base template.
    # ------------------------------------------------------------------
    catalog, base_template, market_params, disruption_params = (
        _load_world_catalog_and_template(config)
    )

    # ------------------------------------------------------------------
    # 2. Build SyncVectorEnv.
    # ------------------------------------------------------------------
    def _env_factory():
        return RLEnv(
            catalog=catalog,
            base_template=base_template,
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
                base_template,
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
    # 6. Save final checkpoint.
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
