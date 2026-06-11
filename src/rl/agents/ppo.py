"""CleanRL-style single-file PPO for the supply-chain RL env.

Adapted from CleanRL's ``ppo_continuous_action.py`` with the following
changes:

- ``SetActor`` and ``SetCritic`` (ADR 0021, variable-K shared-weight policy)
  replace the legacy flat MLP networks.  The legacy ``Actor`` and ``Critic``
  classes are kept for backward compatibility with existing tests and
  checkpoints that reference them, but they are no longer used in the
  training loop.
- ``RolloutBuffer`` stores flat ``(K_MAX * F,)`` observations and
  ``(K_MAX * 3,)`` actions as before; the product mask is extracted from the
  observation at column ``ROW_MASK`` when needed by the actor/critic.
- ``train_ppo`` instantiates ``SetActor`` / ``SetCritic`` and passes 2-D
  reshaped obs and the extracted mask at each actor/critic call.
- Checkpoints are saved through ``src.rl.checkpoint.save`` (self-describing
  bundle with layout-version validation).
- All hyperparameters are read from ``RLConfig`` — no magic numbers buried
  inline.
- TensorBoard scalars logged every update:
    train/episodic_return, train/episodic_length,
    losses/value_loss, losses/policy_loss, losses/entropy,
    losses/approx_kl, losses/clipfrac,
    charts/learning_rate, charts/SPS.

Architecture
------------
The SetActor / SetCritic boundary is the *only* place a future heavier model
(transformer, attention-over-SKUs) needs to change.  Everything around it
(rollout collection, GAE, update loop, TensorBoard logging) stays the same.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from src.rl.configs.default import RLConfig
from src.rl.agents.set_actor_critic import SetActor, SetCritic
from src.rl.set_encoder import K_MAX, F, ROW_MASK


# ---------------------------------------------------------------------------
# Legacy network definitions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init + constant bias — standard CleanRL practice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Legacy MLP actor (flat obs/action).  Not used by train_ppo any more.

    Retained so existing tests and stale-checkpoint tests continue to compile.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.net(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_log_prob(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        if action is None:
            raw = dist.rsample()
            action = torch.tanh(raw)
        else:
            raw = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = dist.log_prob(raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class Critic(nn.Module):
    """Legacy MLP critic (flat obs).  Not used by train_ppo any more."""

    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Rollout buffer helpers
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-size buffer for PPO rollout storage.

    Holds ``n_steps × n_envs`` transitions collected from a sync-vec-env.
    Advantages are computed in-place via ``compute_gae`` before the update
    loop reads from the buffer.

    Parameters
    ----------
    n_steps:
        Number of env steps per env per rollout (from ``config.n_steps``).
    n_envs:
        Number of parallel envs.
    obs_dim:
        Flat observation size (``K_MAX * F``).
    act_dim:
        Flat action size (``K_MAX * 3``).
    device:
        PyTorch device for all stored tensors.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, act_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        # 1.0 for real transitions, 0.0 for Gymnasium NEXT_STEP dummy autoreset
        # steps (which carry a placeholder reward/obs and must not enter the
        # PPO loss or advantage normalisation). Defaults to all-real.
        self.masks = torch.ones(n_steps, n_envs, device=device)

        # Filled by compute_gae.
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)

        self._ptr: int = 0

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Store one step's data at the current pointer, then advance.

        ``mask`` is 1.0 for a real transition and 0.0 for a Gymnasium
        NEXT_STEP dummy autoreset step; defaults to all-real.
        """
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        if mask is not None:
            self.masks[t] = mask
        self._ptr += 1

    def compute_gae(
        self,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        critic: SetCritic,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE-λ advantages and Monte-Carlo returns in-place.

        Populates ``self.advantages`` and ``self.returns``.
        """
        with torch.no_grad():
            next_obs_2d = next_obs.reshape(-1, K_MAX, F)
            next_mask = next_obs_2d[:, :, ROW_MASK]
            next_value = critic(next_obs_2d, next_mask).squeeze(-1)  # (n_envs,)
            last_gae = torch.zeros(self.n_envs, device=self.device)
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    v_next = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    v_next = self.values[t + 1]
                delta = self.rewards[t] + gamma * v_next * next_non_terminal - self.values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                self.advantages[t] = last_gae
            self.returns = self.advantages + self.values

    def reset(self) -> None:
        """Reset the write pointer after a full rollout has been consumed."""
        self._ptr = 0

    def get_flat(self) -> tuple[torch.Tensor, ...]:
        """Return all stored data flattened to ``(n_steps * n_envs, ...)``.

        Returns
        -------
        obs, actions, log_probs_old, advantages, returns, masks

        ``masks`` is 1.0 for real transitions and 0.0 for NEXT_STEP dummy
        autoreset steps; callers use it to drop dummy rows from the update.
        """
        b = self.n_steps * self.n_envs
        return (
            self.obs.view(b, self.obs_dim),
            self.actions.view(b, self.act_dim),
            self.log_probs.view(b),
            self.advantages.view(b),
            self.returns.view(b),
            self.masks.view(b),
        )


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------


def train_ppo(
    envs,
    config: RLConfig,
    writer: SummaryWriter,
    eval_fn: Callable[[SetActor, int], dict[str, float]] | None = None,
    *,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[SetActor, SetCritic]:
    """Run a full PPO training loop using the masked set actor-critic.

    The training driver (``train.py``) constructs ``envs``, ``writer``, and
    ``eval_fn`` and then calls this function.  This function owns the rollout
    loop, GAE computation, and minibatch update loop.

    Observations from the env are flat ``(K_MAX * F,)`` vectors.  They are
    reshaped to ``(B, K_MAX, F)`` before each actor/critic call; the product
    mask is extracted from column ``ROW_MASK`` of the reshaped tensor.  Actions
    are ``(B, K_MAX, 3)`` from the actor, flattened back to ``(B, K_MAX * 3)``
    for the env.

    Checkpoints are saved through ``src.rl.checkpoint.save`` (self-describing
    bundle with layout-version validation) rather than bare ``torch.save``.

    Parameters
    ----------
    envs:
        A ``gymnasium.vector.VectorEnv`` (sync or async).  Must expose
        ``envs.single_observation_space`` and ``envs.single_action_space``.
    config:
        ``RLConfig`` instance carrying all hyperparameters.
    writer:
        A ``torch.utils.tensorboard.SummaryWriter`` open for writing.
    eval_fn:
        Optional callable ``(actor: SetActor, global_step: int) → dict[str, float]``.
        Called every ``config.eval_cadence_env_steps`` global env steps; receives
        the global env-step count so checkpoints can be named by it.
    device:
        Torch device.  When ``None``, uses CUDA if available, else CPU.
    seed:
        Seed for torch / numpy RNG used inside the training loop.

    Returns
    -------
    actor, critic
        Trained ``SetActor`` and ``SetCritic`` objects (on ``device``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Derive shapes from the vec-env.
    obs_dim: int = int(np.prod(envs.single_observation_space.shape))  # K_MAX * F
    act_dim: int = int(np.prod(envs.single_action_space.shape))       # K_MAX * 3
    n_envs: int = envs.num_envs

    # Networks (set actor-critic, ADR 0021).
    actor = SetActor().to(device)
    critic = SetCritic().to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.lr,
        eps=1e-5,
    )

    # Rollout buffer (stores flat obs/actions; mask extracted on the fly).
    buffer = RolloutBuffer(config.n_steps, n_envs, obs_dim, act_dim, device)

    # Derived training constants.
    batch_size: int = config.n_steps * n_envs
    minibatch_size: int = batch_size // config.n_minibatches
    n_updates: int = config.total_env_steps // batch_size

    # Episode-return tracking for TensorBoard.
    ep_return_buf: list[float] = []
    ep_length_buf: list[int] = []

    # Initialise vectorised env.
    obs_np, _ = envs.reset(seed=seed)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.zeros(n_envs, device=device)
    ep_ret = [0.0] * n_envs
    ep_len = [0] * n_envs

    next_eval_step: int = config.eval_cadence_env_steps
    global_step: int = 0
    start_time: float = time.time()

    for update in range(1, n_updates + 1):
        # ----------------------------------------------------------------
        # Anneal learning rate (linear decay to 0).
        # ----------------------------------------------------------------
        frac = 1.0 - (update - 1.0) / n_updates
        lr_now = frac * config.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ----------------------------------------------------------------
        # Rollout collection.
        # ----------------------------------------------------------------
        buffer.reset()
        for _step in range(config.n_steps):
            global_step += n_envs

            with torch.no_grad():
                # Reshape flat obs → (n_envs, K_MAX, F); extract product mask.
                obs_2d = obs.reshape(n_envs, K_MAX, F)
                prod_mask = obs_2d[:, :, ROW_MASK]  # (n_envs, K_MAX)

                # SetActor returns action (n_envs, K_MAX, 3), log_prob (n_envs,).
                action_2d, log_prob, _ = actor.get_action_and_log_prob(obs_2d, prod_mask)
                action_flat = action_2d.reshape(n_envs, act_dim)

                # SetCritic returns (n_envs, 1).
                value = critic(obs_2d, prod_mask).squeeze(-1)  # (n_envs,)

            action_np = action_flat.cpu().numpy()
            obs_next_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)
            done_next_np = np.logical_or(terminated_np, truncated_np)

            reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
            done_next = torch.tensor(done_next_np, dtype=torch.float32, device=device)

            # Under Gymnasium's AutoresetMode.NEXT_STEP, the transition that
            # immediately follows a done step is a dummy autoreset step.
            autoreset_np = done.cpu().numpy().astype(bool)
            step_mask = 1.0 - done  # 0.0 for dummy autoreset steps

            buffer.store(obs, action_flat, log_prob, reward, done, value, step_mask)

            obs = torch.tensor(obs_next_np, dtype=torch.float32, device=device)
            done = done_next

            # Track episode returns.
            for i in range(n_envs):
                if autoreset_np[i]:
                    continue
                ep_ret[i] += float(reward_np[i])
                ep_len[i] += 1
                if done_next_np[i]:
                    ep_return_buf.append(ep_ret[i])
                    ep_length_buf.append(ep_len[i])
                    ep_ret[i] = 0.0
                    ep_len[i] = 0

        # ----------------------------------------------------------------
        # GAE computation.
        # ----------------------------------------------------------------
        buffer.compute_gae(obs, done, critic, config.gamma, config.gae_lambda)

        # ----------------------------------------------------------------
        # PPO update.
        # ----------------------------------------------------------------
        b_obs, b_actions, b_log_probs_old, b_advantages, b_returns, b_masks = buffer.get_flat()
        # b_obs: (batch_size, K_MAX * F)
        # b_actions: (batch_size, K_MAX * 3)

        # Drop dummy autoreset steps from the update.
        real_indices = np.flatnonzero(b_masks.cpu().numpy() > 0.5)

        # Normalise advantages over real transitions only.
        b_advantages = b_advantages.clone()
        b_advantages[real_indices] = (
            b_advantages[real_indices] - b_advantages[real_indices].mean()
        ) / (b_advantages[real_indices].std() + 1e-8)

        indices = real_indices

        # Accumulate update statistics for TensorBoard.
        value_losses: list[float] = []
        policy_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        kl_early_stop: bool = False

        for _epoch in range(config.n_epochs):
            if kl_early_stop:
                break
            np.random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                if mb_idx.size == 0:
                    continue

                mb_obs_flat = b_obs[mb_idx]           # (mb, K_MAX * F)
                mb_actions_flat = b_actions[mb_idx]   # (mb, K_MAX * 3)
                mb_log_probs_old = b_log_probs_old[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]

                # Reshape for set actor/critic.
                mb = mb_obs_flat.shape[0]
                mb_obs_2d = mb_obs_flat.reshape(mb, K_MAX, F)
                mb_mask = mb_obs_2d[:, :, ROW_MASK]           # (mb, K_MAX)
                mb_actions_2d = mb_actions_flat.reshape(mb, K_MAX, 3)

                _, new_log_prob, entropy = actor.get_action_and_log_prob(
                    mb_obs_2d, mb_mask, mb_actions_2d
                )
                new_value = critic(mb_obs_2d, mb_mask).squeeze(-1)  # (mb,)

                log_ratio = new_log_prob - mb_log_probs_old
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_frac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                # KL early stop.
                if config.target_kl is not None and approx_kl > config.target_kl:
                    kl_early_stop = True
                    break

                # Policy loss (clipped surrogate).
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped).
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = b_returns[mb_idx] + torch.clamp(
                    new_value - b_returns[mb_idx],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus.
                entropy_loss = entropy.mean()

                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()

                value_losses.append(v_loss.item())
                policy_losses.append(pg_loss.item())
                entropy_losses.append(entropy_loss.item())

        # ----------------------------------------------------------------
        # TensorBoard logging.
        # ----------------------------------------------------------------
        sps = int(global_step / (time.time() - start_time))

        if ep_return_buf:
            writer.add_scalar("train/episodic_return", np.mean(ep_return_buf), global_step)
            writer.add_scalar("train/episodic_length", np.mean(ep_length_buf), global_step)
            ep_return_buf.clear()
            ep_length_buf.clear()

        writer.add_scalar("losses/value_loss", np.mean(value_losses) if value_losses else 0.0, global_step)
        writer.add_scalar("losses/policy_loss", np.mean(policy_losses) if policy_losses else 0.0, global_step)
        writer.add_scalar("losses/entropy", np.mean(entropy_losses) if entropy_losses else 0.0, global_step)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kls) if approx_kls else 0.0, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs) if clip_fracs else 0.0, global_step)
        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.flush()

        # ----------------------------------------------------------------
        # Optional eval pass.
        # ----------------------------------------------------------------
        if eval_fn is not None and global_step >= next_eval_step:
            eval_metrics = eval_fn(actor, global_step)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, global_step)
            writer.flush()
            next_eval_step += config.eval_cadence_env_steps

    return actor, critic


__all__ = ["Actor", "Critic", "RolloutBuffer", "train_ppo"]
